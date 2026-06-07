'use strict';

const toggle = document.getElementById('toggle');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const toggleSublabel = document.getElementById('toggle-sublabel');

function updateUI(enabled) {
  toggle.checked = enabled;
  if (enabled) {
    statusDot.className = 'status-dot active';
    statusText.textContent = 'Active — monitoring image uploads';
    toggleSublabel.textContent = 'LeakLock is ON';
  } else {
    statusDot.className = 'status-dot inactive';
    statusText.textContent = 'Off — uploads are not monitored';
    toggleSublabel.textContent = 'Click to enable';
  }
}

// Load saved state
chrome.storage.local.get(['leaklockEnabled'], (data) => {
  updateUI(data.leaklockEnabled ?? false);
});

// Handle toggle change
toggle.addEventListener('change', () => {
  const enabled = toggle.checked;
  chrome.storage.local.set({ leaklockEnabled: enabled });
  updateUI(enabled);

  // Notify all content scripts of the new state
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id != null) {
        chrome.tabs.sendMessage(tab.id, { type: 'setState', enabled }).catch(() => {
          // Content script not injected on this tab — safe to ignore
        });
      }
    }
  });
});
