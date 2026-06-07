'use strict';

const API_URL = 'http://localhost:8000/analyze-image';
const OVERLAY_HOST_ID = 'leaklock-overlay-host';

let leaklockEnabled = false;
let isAnalyzing = false;

// ── State init ─────────────────────────────────────────────────────────────
chrome.storage.local.get(['leaklockEnabled'], (data) => {
  leaklockEnabled = data.leaklockEnabled ?? false;
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'setState') {
    leaklockEnabled = msg.enabled;
  }
});

// ── Shadow DOM overlay host ────────────────────────────────────────────────
function getOverlayShadow() {
  let host = document.getElementById(OVERLAY_HOST_ID);
  if (!host) {
    host = document.createElement('div');
    host.id = OVERLAY_HOST_ID;
    // Zero-size host sitting outside the viewport layout
    host.style.cssText = 'all:initial;position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647;';
    document.documentElement.appendChild(host);
  }
  if (!host.shadowRoot) {
    host.attachShadow({ mode: 'open' });
  }
  return host.shadowRoot;
}

function closeOverlay() {
  const sr = getOverlayShadow();
  sr.innerHTML = '';
  isAnalyzing = false;
}

// ── Shared CSS ─────────────────────────────────────────────────────────────
const BASE_CSS = `
  *{box-sizing:border-box;margin:0;padding:0}
  .ll-overlay{
    position:fixed;inset:0;
    background:rgba(0,0,0,0.72);
    display:flex;align-items:center;justify-content:center;
    z-index:2147483647;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    font-size:14px;color:#1e293b;
  }
  .ll-card{
    background:#fff;border-radius:18px;
    padding:28px 26px;max-width:440px;width:90%;
    box-shadow:0 24px 64px rgba(0,0,0,0.45);
  }
`;

// ── "Analyzing…" state ─────────────────────────────────────────────────────
function showAnalyzing() {
  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>
      ${BASE_CSS}
      .ll-logo{font-size:30px;margin-bottom:14px;text-align:center}
      .ll-title{font-size:18px;font-weight:700;color:#0f172a;text-align:center;margin-bottom:18px}
      .ll-spinner{
        width:44px;height:44px;
        border:4px solid #e2e8f0;border-top-color:#3b82f6;
        border-radius:50%;animation:ll-spin 0.9s linear infinite;
        margin:0 auto 14px;
      }
      @keyframes ll-spin{to{transform:rotate(360deg)}}
      .ll-sub{font-size:13px;color:#64748b;text-align:center}
    </style>
    <div class="ll-overlay">
      <div class="ll-card">
        <div class="ll-logo">🔒</div>
        <div class="ll-title">LeakLock</div>
        <div class="ll-spinner"></div>
        <div class="ll-sub">Analyzing image for sensitive content…</div>
      </div>
    </div>`;
}

// ── Result state ───────────────────────────────────────────────────────────
const RISK_COLORS = {
  low:    { bg: '#f0fdf4', border: '#16a34a', text: '#15803d', badge: '#16a34a', icon: '✓' },
  medium: { bg: '#fff7ed', border: '#ea580c', text: '#c2410c', badge: '#ea580c', icon: '⚠️' },
  high:   { bg: '#fef2f2', border: '#dc2626', text: '#b91c1c', badge: '#dc2626', icon: '⚠️' },
};

const RISK_LABELS   = { low: 'LOW RISK',    medium: 'MEDIUM RISK', high: 'HIGH RISK' };
const RISK_MESSAGES = {
  low:    'This image appears safe to upload.',
  medium: 'This image may contain sensitive information.',
  high:   'This image may contain sensitive information!',
};

function showResult(result, inputEl) {
  const { risk_score, risk_level, explanations = [] } = result;
  const c = RISK_COLORS[risk_level] || RISK_COLORS.medium;
  const label   = RISK_LABELS[risk_level]   || risk_level.toUpperCase();
  const message = RISK_MESSAGES[risk_level] || RISK_MESSAGES.medium;

  const exHtml = explanations.length
    ? explanations.map(e => `<li>${e}</li>`).join('')
    : '<li>No specific sensitive items identified</li>';

  const buttonsHtml = risk_level === 'low'
    ? `<button class="ll-btn ll-btn-primary" data-action="proceed">Continue Upload</button>`
    : `
      <button class="ll-btn ll-btn-cancel"  data-action="cancel">Cancel Upload</button>
      <button class="ll-btn ll-btn-primary" data-action="proceed">Upload Anyway</button>
    `;

  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>
      ${BASE_CSS}
      .ll-header{display:flex;align-items:center;gap:10px;margin-bottom:18px}
      .ll-logo{font-size:26px}
      .ll-title{font-size:17px;font-weight:700;color:#0f172a}
      .ll-bar-wrap{background:#f1f5f9;border-radius:99px;height:10px;margin-bottom:10px;overflow:hidden}
      .ll-bar{height:100%;border-radius:99px;background:${c.badge};width:${risk_score}%;transition:width .5s}
      .ll-score-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
      .ll-score{font-size:20px;font-weight:800;color:${c.text}}
      .ll-badge{
        background:${c.badge};color:#fff;
        padding:3px 12px;border-radius:99px;
        font-size:11px;font-weight:700;letter-spacing:.6px;
      }
      .ll-warn{
        background:${c.bg};border:1px solid ${c.border};
        border-radius:10px;padding:13px 15px;margin-bottom:14px;
        color:${c.text};font-weight:600;font-size:14px;
      }
      .ll-explain{
        margin:0 0 20px;padding:0 0 0 18px;
        color:#475569;font-size:13px;line-height:1.65;list-style:disc;
      }
      .ll-explain li{margin-bottom:3px}
      .ll-buttons{display:flex;gap:10px}
      .ll-btn{
        flex:1;padding:11px 8px;border-radius:9px;
        font-size:13px;font-weight:600;cursor:pointer;border:none;
      }
      .ll-btn-cancel{background:#f1f5f9;color:#475569;border:2px solid #e2e8f0}
      .ll-btn-cancel:hover{background:#e2e8f0}
      .ll-btn-primary{background:${c.badge};color:#fff}
      .ll-btn-primary:hover{opacity:.88}
    </style>
    <div class="ll-overlay">
      <div class="ll-card">
        <div class="ll-header">
          <span class="ll-logo">🔒</span>
          <span class="ll-title">LeakLock Analysis</span>
        </div>
        <div class="ll-bar-wrap"><div class="ll-bar"></div></div>
        <div class="ll-score-row">
          <span class="ll-score">Risk Score: ${risk_score}%</span>
          <span class="ll-badge">${label}</span>
        </div>
        <div class="ll-warn">${c.icon} ${message}</div>
        <ul class="ll-explain">${exHtml}</ul>
        <div class="ll-buttons">${buttonsHtml}</div>
      </div>
    </div>`;

  isAnalyzing = false;

  // Wire buttons
  sr.querySelectorAll('.ll-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'cancel' && inputEl) {
        // Clear the file input so the upload won't proceed
        try { inputEl.value = ''; } catch (_) { /* read-only on some browsers */ }
      }
      closeOverlay();
    });
  });
}

// ── Error / backend unavailable ────────────────────────────────────────────
function showError() {
  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>
      ${BASE_CSS}
      .ll-title{font-size:17px;font-weight:700;color:#0f172a;text-align:center;margin-bottom:10px}
      .ll-msg{font-size:13px;color:#64748b;text-align:center;margin-bottom:18px;line-height:1.5}
      .ll-code{font-family:monospace;background:#f1f5f9;border-radius:6px;padding:6px 10px;font-size:12px;color:#0f172a}
      .ll-btn-close{
        display:block;width:100%;padding:10px;border-radius:9px;border:none;
        background:#3b82f6;color:#fff;font-size:13px;font-weight:600;cursor:pointer;
      }
      .ll-btn-close:hover{opacity:.88}
    </style>
    <div class="ll-overlay">
      <div class="ll-card" style="text-align:center">
        <div class="ll-title">🔒 LeakLock</div>
        <div class="ll-msg">
          Backend unavailable — upload will proceed normally.<br><br>
          Start the server with:<br>
          <span class="ll-code">uvicorn leaklock.api:app --port 8000</span>
        </div>
        <button class="ll-btn-close" id="ll-err-close">Close</button>
      </div>
    </div>`;

  sr.getElementById('ll-err-close')?.addEventListener('click', closeOverlay);
  isAnalyzing = false;
}

// ── Core analysis flow ─────────────────────────────────────────────────────
async function analyzeFile(file, inputEl) {
  if (isAnalyzing) return;
  isAnalyzing = true;
  showAnalyzing();

  try {
    const body = new FormData();
    body.append('file', file, file.name || 'upload.jpg');

    const response = await fetch(API_URL, { method: 'POST', body });

    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

    const result = await response.json();
    showResult(result, inputEl);
  } catch (err) {
    console.warn('[LeakLock] Backend error:', err.message);
    showError();
  }
}

// ── File input event handler ───────────────────────────────────────────────
function handleFileChange(event) {
  if (!leaklockEnabled) return;
  if (isAnalyzing) return;

  const input = event.target;
  const file = input.files?.[0];
  if (!file || !file.type.startsWith('image/')) return;

  analyzeFile(file, input);
}

// ── Attach / observe ───────────────────────────────────────────────────────
function attachToInput(input) {
  if (input._llAttached) return;
  input._llAttached = true;
  input.addEventListener('change', handleFileChange);
}

// Attach to file inputs already in DOM
document.querySelectorAll('input[type="file"]').forEach(attachToInput);

// Watch for dynamically added inputs (needed for SPAs like Facebook, Instagram)
const domObserver = new MutationObserver((mutations) => {
  for (const { addedNodes } of mutations) {
    for (const node of addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.tagName === 'INPUT' && node.type === 'file') {
        attachToInput(node);
      }
      node.querySelectorAll?.('input[type="file"]').forEach(attachToInput);
    }
  }
});

const observeTarget = document.body ?? document.documentElement;
domObserver.observe(observeTarget, { childList: true, subtree: true });
