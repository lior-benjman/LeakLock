'use strict';

const toggle = document.getElementById('toggle');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const toggleSublabel = document.getElementById('toggle-sublabel');

let isApplying = false;

function updateUI(enabled, options = {}) {
  toggle.checked = enabled;
  toggle.disabled = Boolean(options.disabled || isApplying);

  if (enabled) {
    statusDot.className = 'status-dot active';
    statusText.textContent = options.statusText || 'Active - monitoring this tab';
    toggleSublabel.textContent = 'Enabled on this tab';
    return;
  }

  statusDot.className = 'status-dot inactive';
  statusText.textContent = options.statusText || 'Off - this tab is not monitored';
  toggleSublabel.textContent = options.sublabel || 'Enable on this tab';
}

function isSupportedTab(tab) {
  return Number.isInteger(tab?.id) && /^https?:\/\//i.test(tab.url || '');
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function sendTabMessage(tabId, message) {
  return chrome.tabs.sendMessage(tabId, message);
}

async function ensureContentScript(tabId) {
  try {
    await sendTabMessage(tabId, { type: 'getState' });
    return;
  } catch (_) {
    // No receiver yet; inject below.
  }

  await chrome.scripting.executeScript({
    target: { tabId },
    files: ['content.js'],
  });
}

async function readCurrentTabState() {
  const tab = await getActiveTab();
  if (!isSupportedTab(tab)) {
    updateUI(false, {
      disabled: true,
      statusText: 'Open a regular website tab to use LeakLock',
      sublabel: 'Unavailable here',
    });
    return;
  }

  try {
    const response = await sendTabMessage(tab.id, { type: 'getState' });
    updateUI(Boolean(response?.enabled));
  } catch (_) {
    updateUI(false);
  }
}

async function setCurrentTabState(enabled) {
  const tab = await getActiveTab();
  if (!isSupportedTab(tab)) {
    updateUI(false, {
      disabled: true,
      statusText: 'Open a regular website tab to use LeakLock',
      sublabel: 'Unavailable here',
    });
    return;
  }

  if (enabled) {
    await ensureContentScript(tab.id);
    await sendTabMessage(tab.id, { type: 'setState', enabled: true });
    updateUI(true);
    return;
  }

  try {
    await sendTabMessage(tab.id, { type: 'setState', enabled: false });
  } catch (_) {
    // If the script was never injected, the tab is already effectively off.
  }
  updateUI(false);
}

toggle.addEventListener('change', async () => {
  if (isApplying) return;

  isApplying = true;
  updateUI(toggle.checked, {
    statusText: toggle.checked ? 'Enabling on this tab...' : 'Turning off...',
  });

  try {
    await setCurrentTabState(toggle.checked);
  } catch (err) {
    console.warn('[LeakLock] Could not update current tab:', err);
    updateUI(false, {
      statusText: 'Could not access this tab',
      sublabel: 'Try another website tab',
    });
  } finally {
    isApplying = false;
    await readCurrentTabState();
  }
});

readCurrentTabState().catch((err) => {
  console.warn('[LeakLock] Could not read current tab state:', err);
  updateUI(false, {
    disabled: true,
    statusText: 'Could not read this tab',
    sublabel: 'Try another website tab',
  });
});
