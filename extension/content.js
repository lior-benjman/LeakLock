'use strict';

const API_URL = 'https://leaklock-api-799658247857.us-central1.run.app/analyze-image';
const OVERLAY_HOST_ID = 'leaklock-overlay-host';
const MAX_BATCH_FILES = 10;
const ANALYSIS_TIMEOUT_MS = 30000;

let leaklockEnabled = false;
let isAnalyzing = false;
let currentBatch = null; // set while the results overlay (single image or batch) is open

class AnalysisTimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'AnalysisTimeoutError';
    this.isLeakLockTimeout = true;
  }
}

function isAnalysisTimeoutError(err) {
  return Boolean(
    err?.isLeakLockTimeout ||
    err?.name === 'AbortError' ||
    String(err?.message || '').includes('HTTP 504')
  );
}

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
  if (currentBatch) {
    for (const item of currentBatch.items) {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
      if (item.blurredPreviewUrl) URL.revokeObjectURL(item.blurredPreviewUrl);
    }
  }
  currentBatch = null;
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
  .ll-card.ll-card-wide{max-width:560px;max-height:92vh;overflow:auto}
  .ll-header{display:flex;align-items:center;gap:10px;margin-bottom:18px}
  .ll-logo{font-size:26px}
  .ll-title{font-size:17px;font-weight:700;color:#0f172a}
  .ll-bar-wrap{background:#f1f5f9;border-radius:99px;height:10px;margin-bottom:10px;overflow:hidden}
  .ll-bar{height:100%;border-radius:99px;transition:width .4s}
  .ll-score-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:12px}
  .ll-score{font-size:20px;font-weight:800}
  .ll-badge{
    color:#fff;padding:3px 12px;border-radius:99px;
    font-size:11px;font-weight:700;letter-spacing:.4px;white-space:nowrap;
  }
  .ll-warn{
    border:1px solid;border-radius:10px;padding:13px 15px;margin-bottom:14px;
    font-weight:600;font-size:14px;
  }
  .ll-explain{
    margin:0 0 14px;padding:0 0 0 18px;
    color:#475569;font-size:13px;line-height:1.65;list-style:disc;
  }
  .ll-explain li{margin-bottom:3px}
  .ll-detections-title{font-size:12px;font-weight:800;color:#0f172a;text-transform:uppercase;margin:4px 0 8px}
  .ll-detection-card{
    border:1px solid #e2e8f0;border-radius:10px;
    padding:10px 12px;margin-bottom:8px;background:#fff;
  }
  .ll-det-head{display:flex;justify-content:space-between;gap:10px;align-items:center;color:#0f172a;font-size:13px;font-weight:800}
  .ll-det-meta{color:#64748b;font-size:12px;line-height:1.45;margin-top:4px}
  .ll-det-reason{color:#334155;font-size:12px;line-height:1.45;margin-top:6px}
  .ll-buttons{display:flex;gap:10px;margin-top:16px}
  .ll-btn{
    flex:1;padding:11px 8px;border-radius:9px;
    font-size:13px;font-weight:600;cursor:pointer;border:none;
  }
  .ll-btn-cancel{background:#f1f5f9;color:#475569;border:2px solid #e2e8f0}
  .ll-btn-cancel:hover{background:#e2e8f0}
  .ll-btn-primary{background:#3b82f6;color:#fff}
  .ll-btn-primary:hover{opacity:.88}
  .ll-btn-primary:disabled{background:#cbd5e1;cursor:not-allowed;opacity:1}
  .ll-logo-center{font-size:30px;margin-bottom:14px;text-align:center}
  .ll-title-center{font-size:18px;font-weight:700;color:#0f172a;text-align:center;margin-bottom:18px}
  .ll-spinner{
    width:44px;height:44px;
    border:4px solid #e2e8f0;border-top-color:#3b82f6;
    border-radius:50%;animation:ll-spin 0.9s linear infinite;
    margin:0 auto 14px;
  }
  @keyframes ll-spin{to{transform:rotate(360deg)}}
  .ll-sub{font-size:13px;color:#64748b;text-align:center}
`;

const BATCH_CSS = `
  .ll-batch-note{
    background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
    padding:8px 12px;margin-bottom:14px;color:#1e40af;font-size:12px;line-height:1.5;
  }
  .ll-batch-list{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
  .ll-card-wide .ll-buttons{position:sticky;bottom:-28px;background:#fff;padding-top:12px;padding-bottom:28px;border-top:1px solid #e2e8f0;margin-top:4px}
  .ll-batch-row{border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}
  .ll-batch-row-header{
    width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;
    background:#fff;border:none;padding:12px 14px;cursor:pointer;text-align:left;
  }
  .ll-batch-row-name{font-size:13px;font-weight:600;color:#0f172a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px}
  .ll-batch-row-right{display:flex;align-items:center;gap:8px}
  .ll-chevron{color:#94a3b8;font-size:12px}
  .ll-batch-row-body{padding:0 14px 14px}
  .ll-batch-hint{margin-top:10px;font-size:12px;color:#b91c1c;text-align:center}
  .ll-blur-status{font-size:12px;color:#7c3aed;margin:8px 0;text-align:center}
  .ll-blur-error{
    background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
    padding:8px 10px;margin:8px 0;color:#b91c1c;font-size:12px;line-height:1.5;
  }
  .ll-preview-wrap{
    background:#f1f5f9;border-radius:10px;overflow:hidden;
    margin-bottom:14px;display:flex;align-items:center;justify-content:center;
    max-height:260px;
  }
  .ll-preview-img{display:block;max-width:100%;max-height:260px;object-fit:contain}
  .ll-blur-toggle{
    display:flex;align-items:center;gap:8px;margin:4px 0 6px;
    font-size:13px;color:#334155;cursor:pointer;user-select:none;
  }
  .ll-blur-toggle input{width:16px;height:16px;cursor:pointer}
`;

// ── "Analyzing…" state ─────────────────────────────────────────────────────
function showAnalyzing() {
  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>${BASE_CSS}</style>
    <div class="ll-overlay">
      <div class="ll-card">
        <div class="ll-logo-center">🔒</div>
        <div class="ll-title-center">LeakLock</div>
        <div class="ll-spinner"></div>
        <div class="ll-sub">Analyzing image for sensitive content…</div>
      </div>
    </div>`;
}

function showBatchProgress(current, total, fileName) {
  if (total <= 1) {
    showAnalyzing();
    return;
  }
  const pct = Math.round(((current - 1) / total) * 100);
  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>${BASE_CSS}</style>
    <div class="ll-overlay">
      <div class="ll-card">
        <div class="ll-logo-center">🔒</div>
        <div class="ll-title-center">LeakLock</div>
        <div class="ll-spinner"></div>
        <div class="ll-sub">Analyzing image ${current} of ${total}…</div>
        <div class="ll-bar-wrap" style="margin-top:14px">
          <div class="ll-bar" style="background:#3b82f6;width:${pct}%"></div>
        </div>
        <div class="ll-sub" style="margin-top:6px;font-size:11px;opacity:.75">${escapeHtml(fileName)}</div>
      </div>
    </div>`;
}

// ── Result rendering helpers ───────────────────────────────────────────────
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

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function getAnalyses(payload) {
  if (Array.isArray(payload?.analyses)) return payload.analyses;
  if (Array.isArray(payload?.result?.analyses)) return payload.result.analyses;
  return [];
}

function renderDetectionCards(payload) {
  const analyses = getAnalyses(payload);
  if (analyses.length) {
    return analyses.map((analysis, index) => {
      const detection = analysis.detection || {};
      const risk = analysis.risk || {};
      const evidence = risk.evidence || {};
      const age = evidence.age_estimate || {};
      const className = String(detection.class_name || 'object').replaceAll('_', ' ');
      const confidence = Number(detection.confidence || 0);
      const riskPercent = Number(risk.risk_percent || 0);
      const ageLine = age.age_years !== undefined && age.age_years !== null
        ? `<div class="ll-det-meta">Age estimate: <b>${escapeHtml(age.age_years)}</b>${age.confidence !== undefined && age.confidence !== null ? ` - confidence ${Number(age.confidence).toFixed(2)}` : ''}</div>`
        : Object.keys(age).length
          ? `<div class="ll-det-meta">Age estimate unavailable: ${escapeHtml(age.details || 'no details')}</div>`
          : '';

      return `
        <div class="ll-detection-card">
          <div class="ll-det-head">
            <span>${index + 1}. ${escapeHtml(className)}</span>
            <span>${riskPercent}%</span>
          </div>
          <div class="ll-det-meta">route: ${escapeHtml(analysis.route || '')} - detection confidence ${confidence.toFixed(2)}</div>
          ${ageLine}
          <div class="ll-det-reason">${escapeHtml(risk.reason || '')}</div>
        </div>
      `;
    }).join('');
  }

  const detections = Array.isArray(payload?.detections) ? payload.detections : [];
  if (detections.length) {
    return detections.map((detection, index) => `
      <div class="ll-detection-card">
        <div class="ll-det-head">
          <span>${index + 1}. ${escapeHtml(detection.class_name || 'object')}</span>
          <span>${Number(detection.confidence || 0).toFixed(2)}</span>
        </div>
      </div>
    `).join('');
  }

  return '<div class="ll-detection-card"><div class="ll-det-head">No sensitive objects detected</div></div>';
}

// Pure string-builder: the score bar / explanations / detections block shared
// by every row's expanded body (single image and each batch item alike).
function renderRiskDetail(result) {
  const { risk_score = 0, risk_level, explanations = [] } = result || {};
  const c = RISK_COLORS[risk_level] || RISK_COLORS.medium;
  const label = RISK_LABELS[risk_level] || String(risk_level || 'medium').toUpperCase();
  const message = RISK_MESSAGES[risk_level] || RISK_MESSAGES.medium;
  const exHtml = explanations.length
    ? explanations.map((e) => `<li>${escapeHtml(e)}</li>`).join('')
    : '<li>No specific sensitive risk explanation returned</li>';
  const detectionsHtml = renderDetectionCards(result);

  return `
    <div class="ll-bar-wrap"><div class="ll-bar" style="background:${c.badge};width:${risk_score}%"></div></div>
    <div class="ll-score-row">
      <span class="ll-score" style="color:${c.text}">Risk Score: ${risk_score}%</span>
      <span class="ll-badge" style="background:${c.badge}">${label}</span>
    </div>
    <div class="ll-warn" style="background:${c.bg};border-color:${c.border};color:${c.text}">${c.icon} ${message}</div>
    <ul class="ll-explain">${exHtml}</ul>
    <div class="ll-detections-title">Detections</div>
    ${detectionsHtml}
  `;
}

function renderImagePreview(item) {
  const src = item.blurPreviewOn && item.blurredPreviewUrl ? item.blurredPreviewUrl : item.previewUrl;
  if (!src) return '';
  return `<div class="ll-preview-wrap"><img class="ll-preview-img" src="${src}" alt="preview" /></div>`;
}

// ── Error / backend unavailable ────────────────────────────────────────────
function showError(onProceed) {
  const sr = getOverlayShadow();
  sr.innerHTML = `
    <style>${BASE_CSS}</style>
    <div class="ll-overlay">
      <div class="ll-card" style="text-align:center">
        <div class="ll-title-center">🔒 LeakLock</div>
        <div class="ll-sub" style="margin-bottom:18px;line-height:1.5">
          Backend unavailable - upload will proceed normally.<br><br>
          Configured endpoint:<br>
          <span style="font-family:monospace;background:#f1f5f9;border-radius:6px;padding:6px 10px;font-size:12px;color:#0f172a;display:inline-block;margin-top:6px;word-break:break-all">
            ${API_URL}
          </span>
        </div>
        <button class="ll-btn ll-btn-primary" id="ll-err-close" style="width:100%">Close</button>
      </div>
    </div>`;

  sr.getElementById('ll-err-close')?.addEventListener('click', () => {
    onProceed();
    closeOverlay();
  });
  isAnalyzing = false;
}

// ── Results overlay — used for both a single image and a multi-image batch ─
function decisionLabel(item) {
  if (item.decision === 'exclude') return "Won't Upload";
  if (item.decision === 'upload') return item.usedBlur ? 'Blurred' : 'Upload As-Is';
  return '';
}

function renderDecisionUI(item, idx, isSingle = false) {
  if (item.riskLevel === 'low') return '';

  if (item.blurPending) {
    return `<div class="ll-blur-status">Blurring…</div>`;
  }

  const errorHtml = item.blurError
    ? `<div class="ll-blur-error">Blur failed: ${escapeHtml(item.blurError)}. You can still upload as-is or skip this image.</div>`
    : '';

  // Single image: show the blur toggle (so the user can preview blurred vs original)
  // but omit the per-item action buttons — the global Cancel/Upload Anyway serve that role.
  const buttonsHtml = isSingle ? '' : `
    <div class="ll-buttons">
      <button class="ll-btn ll-btn-cancel" data-decision="exclude" data-row-index="${idx}">Don't Upload</button>
      <button class="ll-btn ll-btn-primary" data-decision="upload" data-row-index="${idx}">Upload This Image</button>
    </div>`;

  return `
    ${errorHtml}
    <label class="ll-blur-toggle">
      <input type="checkbox" data-blur-toggle="${idx}" ${item.blurPreviewOn ? 'checked' : ''} />
      <span>Blur sensitive areas</span>
    </label>
    ${buttonsHtml}`;
}

// Pure: whether every risky (non-low) item has a decision made.
function allRiskyDecided(items) {
  return items.every((item) => item.riskLevel === 'low' || !!item.decision);
}

function renderBatchRow(item, idx, isSingle) {
  const bodyHtml = `
    <div class="ll-batch-row-body">
      ${renderImagePreview(item)}
      ${renderRiskDetail(item.result)}
      ${renderDecisionUI(item, idx, isSingle)}
    </div>`;

  if (isSingle) {
    // Nothing else to compare against — always shown open, no collapse header,
    // and no per-item decision UI (action buttons are the global ones at the bottom).
    return `<div class="ll-batch-row" data-row-index="${idx}">${bodyHtml}</div>`;
  }

  const isLow = item.riskLevel === 'low';
  const c = RISK_COLORS[item.riskLevel] || RISK_COLORS.medium;
  const label = RISK_LABELS[item.riskLevel] || String(item.riskLevel || '').toUpperCase();
  const headerRight = isLow
    ? `<span class="ll-badge" style="background:${c.badge}">${label}</span>`
    : item.decision
      ? `<span class="ll-badge" style="background:${c.badge}">${label} · ${escapeHtml(decisionLabel(item))}</span>`
      : `<span class="ll-badge" style="background:${c.badge}">${label} · Needs decision</span>`;

  return `
    <div class="ll-batch-row" data-row-index="${idx}">
      <button class="ll-batch-row-header" data-toggle-row="${idx}">
        <span class="ll-batch-row-name">${escapeHtml(item.file.name)}</span>
        <span class="ll-batch-row-right">
          ${headerRight}
          <span class="ll-chevron">${item.expanded ? '▾' : '▸'}</span>
        </span>
      </button>
      ${item.expanded ? bodyHtml : ''}
    </div>`;
}

function renderBatchOverlay() {
  const batch = currentBatch;
  if (!batch) return;

  const sr = getOverlayShadow();
  const isSingle = batch.items.length === 1;
  // Single image needs no per-item decision step — the global buttons ARE the decision.
  const decided = isSingle ? true : allRiskyDecided(batch.items);
  const rowsHtml = batch.items.map((item, idx) => renderBatchRow(item, idx, isSingle)).join('');
  const skippedNote = batch.skippedCount > 0
    ? `<div class="ll-batch-note">Only the first ${batch.items.length} images were scanned — ${batch.skippedCount} additional image(s) will be uploaded without analysis (max ${MAX_BATCH_FILES} per batch).</div>`
    : '';
  const titleText = isSingle ? 'LeakLock Analysis' : `LeakLock — ${batch.items.length} images analyzed`;

  sr.innerHTML = `
    <style>${BASE_CSS}${BATCH_CSS}</style>
    <div class="ll-overlay">
      <div class="ll-card ll-card-wide">
        <div class="ll-header">
          <span class="ll-logo">🔒</span>
          <span class="ll-title">${titleText}</span>
        </div>
        ${skippedNote}
        <div class="ll-batch-list">${rowsHtml}</div>
        <div class="ll-buttons">
          <button class="ll-btn ll-btn-cancel" data-batch-action="cancel-all">${isSingle ? 'Cancel Upload' : 'Cancel All'}</button>
          <button class="ll-btn ll-btn-primary" data-batch-action="continue" ${decided ? '' : 'disabled'}>${isSingle && batch.items[0]?.riskLevel !== 'low' ? (batch.items[0]?.blurPreviewOn ? 'Upload' : 'Upload Anyway') : 'Continue Upload'}</button>
        </div>
        ${decided ? '' : '<div class="ll-batch-hint">Decide on every flagged image to continue</div>'}
      </div>
    </div>`;

  wireBatchOverlay(sr, batch);
}

function wireBatchOverlay(sr, batch) {
  sr.querySelectorAll('[data-toggle-row]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.toggleRow);
      batch.items[idx].expanded = !batch.items[idx].expanded;
      renderBatchOverlay();
    });
  });

  sr.querySelectorAll('[data-blur-toggle]').forEach((checkbox) => {
    checkbox.addEventListener('change', async () => {
      const idx = Number(checkbox.dataset.blurToggle);
      const item = batch.items[idx];

      if (!checkbox.checked) {
        item.blurPreviewOn = false;
        renderBatchOverlay();
        return;
      }

      if (item.blurredPreviewUrl) {
        item.blurPreviewOn = true; // already computed earlier — just switch the preview back
        renderBatchOverlay();
        return;
      }

      item.blurPending = true;
      item.blurError = null;
      renderBatchOverlay();
      try {
        item.blurredFile = await blurImage(item.file, item.result);
        item.blurredPreviewUrl = URL.createObjectURL(item.blurredFile);
        item.blurPreviewOn = true;
      } catch (err) {
        item.blurError = err?.message || 'Unknown error';
        item.blurPreviewOn = false; // do not silently show/use an unblurred image instead
      } finally {
        item.blurPending = false;
        renderBatchOverlay();
      }
    });
  });

  sr.querySelectorAll('[data-decision]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.rowIndex);
      const item = batch.items[idx];
      item.decision = btn.dataset.decision; // 'upload' or 'exclude'
      item.usedBlur = item.decision === 'upload' && item.blurPreviewOn;
      item.expanded = false;
      renderBatchOverlay();
    });
  });

  sr.querySelector('[data-batch-action="cancel-all"]')?.addEventListener('click', () => {
    try { if (batch.inputEl) batch.inputEl.value = ''; } catch (_) {}
    closeOverlay();
  });

  sr.querySelector('[data-batch-action="continue"]')?.addEventListener('click', () => {
    const isSingle = batch.items.length === 1;
    if (!isSingle && !allRiskyDecided(batch.items)) return;
    // Single image: respect the blur toggle state when building the file list,
    // but no per-item decision was required.
    // Multi-image: rebuild from per-item decisions as usual.
    let finalFiles;
    if (isSingle) {
      const item = batch.items[0];
      const file = item.blurPreviewOn && item.blurredFile ? item.blurredFile : item.file;
      finalFiles = batch.originalOrder.map((f) => (f === item.file ? file : f));
    } else {
      finalFiles = buildFinalFileList(batch.originalOrder, batch.items);
    }
    resumeFiles(batch.inputEl, finalFiles);
    closeOverlay();
  });
}

// Pure: reconstructs the final ordered file list from the original selection
// and each analyzed item's decision (+ whether blur was on when decided).
// Non-analyzed files (non-images, or images beyond the MAX_BATCH_FILES cap)
// pass through unmodified.
function buildFinalFileList(originalOrder, items) {
  const byFile = new Map(items.map((item) => [item.file, item]));
  const finalFiles = [];

  for (const original of originalOrder) {
    const item = byFile.get(original);
    if (!item) {
      finalFiles.push(original);
      continue;
    }
    if (item.riskLevel === 'low' || item.decision === 'upload') {
      finalFiles.push(item.usedBlur && item.blurredFile ? item.blurredFile : item.file);
    }
    // decision === 'exclude' -> omit entirely
  }

  return finalFiles;
}

// ── Region blurring ─────────────────────────────────────────────────────────
// Pure: extracts bounding boxes for every detection that contributed risk.
function collectRiskyBoxes(result) {
  const analyses = getAnalyses(result);
  const boxes = [];
  for (const analysis of analyses) {
    const risk = analysis?.risk || {};
    if (!(Number(risk.risk_percent) > 0)) continue;
    const box = analysis?.detection?.box;
    if (!box) continue;
    boxes.push({
      x1: Number(box.x1) || 0,
      y1: Number(box.y1) || 0,
      x2: Number(box.x2) || 0,
      y2: Number(box.y2) || 0,
    });
  }
  return boxes;
}

// Pure: computes the source/destination rectangle and downsample size for
// pixelating one region. Clamped to canvas bounds defensively.
function computePixelRegion(box, canvasWidth, canvasHeight, blockSize) {
  const x = Math.max(0, Math.min(canvasWidth - 1, Math.floor(box.x1)));
  const y = Math.max(0, Math.min(canvasHeight - 1, Math.floor(box.y1)));
  const x2 = Math.max(x + 1, Math.min(canvasWidth, Math.ceil(box.x2)));
  const y2 = Math.max(y + 1, Math.min(canvasHeight, Math.ceil(box.y2)));
  const w = x2 - x;
  const h = y2 - y;
  const smallW = Math.max(1, Math.round(w / blockSize));
  const smallH = Math.max(1, Math.round(h / blockSize));
  return { x, y, w, h, smallW, smallH };
}

function pixelateRegion(ctx, box, blockSize) {
  const { x, y, w, h, smallW, smallH } = computePixelRegion(box, ctx.canvas.width, ctx.canvas.height, blockSize);

  const tmp = document.createElement('canvas');
  tmp.width = smallW;
  tmp.height = smallH;
  const tmpCtx = tmp.getContext('2d');
  tmpCtx.drawImage(ctx.canvas, x, y, w, h, 0, 0, smallW, smallH);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, smallW, smallH, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}

// Produces a new File with every risky detection region pixelated. Errors
// are NOT swallowed here — the caller must treat a thrown error as "blurring
// failed," not silently fall back to the original (unblurred) image, since
// the user explicitly asked for this content to be redacted.
async function blurImage(file, result) {
  const boxes = collectRiskyBoxes(result);
  if (!boxes.length) return file; // nothing flagged as risky to blur

  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close?.();

  const blockSize = Math.max(8, Math.round(Math.min(canvas.width, canvas.height) / 30));
  for (const box of boxes) {
    pixelateRegion(ctx, box, blockSize);
  }

  const outType = file.type && file.type.startsWith('image/') ? file.type : 'image/png';
  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('canvas.toBlob returned null'))), outType, 0.92);
  });

  return new File([blob], file.name || 'image.png', { type: blob.type, lastModified: Date.now() });
}

// ── Resume helper — re-fires change with a (possibly modified) file list ───
function resumeFiles(inputEl, files) {
  if (!inputEl) return;
  try {
    const dt = new DataTransfer();
    for (const f of files) dt.items.add(f);
    inputEl.files = dt.files;
    const resumeEvent = new Event('change', { bubbles: true });
    resumeEvent._llResumed = true;
    inputEl.dispatchEvent(resumeEvent);
  } catch (_) {}
}

// ── Core analysis flow ─────────────────────────────────────────────────────
async function fetchAnalysis(file) {
  const body = new FormData();
  body.append('file', file, file.name || 'upload.jpg');
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ANALYSIS_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, { method: 'POST', body, signal: controller.signal });
    if (response.status === 504) {
      throw new AnalysisTimeoutError(`Analysis timed out after ${ANALYSIS_TIMEOUT_MS / 1000}s`);
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    return response.json();
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new AnalysisTimeoutError(`Analysis timed out after ${ANALYSIS_TIMEOUT_MS / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function analyzeBatch(imageFiles, originalOrder, inputEl, skippedCount) {
  if (isAnalyzing) return;
  isAnalyzing = true;

  const items = imageFiles.map((file) => ({
    file,
    result: null,
    riskLevel: null,
    decision: null,
    usedBlur: false,
    blurredFile: null,
    blurPreviewOn: false,
    blurPending: false,
    blurError: null,
    expanded: false,
    previewUrl: URL.createObjectURL(file),
    blurredPreviewUrl: null,
  }));

  for (let i = 0; i < items.length; i++) {
    showBatchProgress(i + 1, items.length, items[i].file.name);
    try {
      items[i].result = await fetchAnalysis(items[i].file);
      items[i].riskLevel = items[i].result.risk_level;
    } catch (err) {
      console.warn('[LeakLock] Backend error:', err.message);
      isAnalyzing = false;
      for (const item of items) URL.revokeObjectURL(item.previewUrl);
      if (isAnalysisTimeoutError(err)) {
        // Fail open on pathological OCR/model hangs. The extension already
        // blocked the host page's change event, so resume the original file
        // selection immediately instead of leaving the upload stuck.
        resumeFiles(inputEl, originalOrder);
        closeOverlay();
        return;
      }
      // Fail open for the whole batch: the backend is down for every file
      // equally, so resume the original selection unmodified.
      showError(() => resumeFiles(inputEl, originalOrder));
      return;
    }
  }

  isAnalyzing = false;
  // Multi-image rows start collapsed; a single image is always shown open
  // since there's only one thing to look at.
  items.forEach((item) => { item.expanded = false; });
  currentBatch = { inputEl, originalOrder, items, skippedCount };
  renderBatchOverlay();
}

// ── File type detection ────────────────────────────────────────────────────
const IMAGE_EXTENSION_RE = /\.(jpe?g|png|webp|gif|bmp|tiff?|heic|heif|avif)$/i;

function isImageFile(file) {
  if (!file) return false;
  if (file.type) return file.type.startsWith('image/');
  // Some OS/browser combinations report an empty MIME type for legitimate
  // images — fall back to sniffing the filename extension instead of
  // silently letting the file skip analysis.
  return IMAGE_EXTENSION_RE.test(file.name || '');
}

// ── File input event handler ───────────────────────────────────────────────
function handleFileChange(event) {
  if (event._llResumed) return; // skip events we dispatched ourselves on proceed
  if (!leaklockEnabled) return;
  if (isAnalyzing) return;

  const input = event.target;
  const allFiles = Array.from(input.files || []);
  const imageFiles = allFiles.filter(isImageFile);
  if (!imageFiles.length) return; // nothing for us to analyze — let the host handle it normally

  // Block the host page from processing this file selection until the user decides
  event.stopImmediatePropagation();

  const capped = imageFiles.slice(0, MAX_BATCH_FILES);
  const skippedCount = imageFiles.length - capped.length;
  analyzeBatch(capped, allFiles, input, skippedCount);
}

// ── Attach / observe ───────────────────────────────────────────────────────
function attachToInput(input) {
  if (input._llAttached) return;
  input._llAttached = true;
  // Capture phase ensures our handler runs before the host page's bubble handlers
  input.addEventListener('change', handleFileChange, true);
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
