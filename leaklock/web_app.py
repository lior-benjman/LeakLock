from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import re
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import PipelineConfig
from .pipeline import LeakLockPipeline


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "uploaded_images"
RESULTS_DIR = REPO_ROOT / "analysis_results"
JSON_RESULTS_DIR = RESULTS_DIR / "json"
OCR_RESULTS_DIR = RESULTS_DIR / "ocr_text"
SUMMARY_CSV_PATH = RESULTS_DIR / "summary.csv"


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JSON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(file_name: str) -> str:
    candidate = Path(file_name or "uploaded_image").name.strip()
    candidate = re.sub(r"[^A-Za-z0-9._ -]+", "_", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    if not candidate:
        candidate = "uploaded_image"
    if "." not in candidate:
        candidate = f"{candidate}.png"
    return candidate


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("Upload payload must be a data URL.")

    header, encoded = data_url.split(",", 1)
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    try:
        content = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001 - base64 raises mixed errors.
        raise ValueError(f"Upload payload is not valid base64: {exc}") from exc

    if not content:
        raise ValueError("Uploaded file was empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file is larger than the 25 MB limit.")
    if not media_type.startswith("image/"):
        raise ValueError("Only image uploads are supported.")

    return media_type, content


def _write_upload(file_name: str, content: bytes) -> Path:
    safe_name = _safe_filename(file_name)
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)
    return file_path


def _make_result_basename(image_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{image_path.stem}_{timestamp}"


def _append_summary_row(result, image_path: Path, json_path: Path) -> None:
    file_exists = SUMMARY_CSV_PATH.exists()
    with SUMMARY_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "image_path",
                "overall_risk_percent",
                "detections_count",
                "json_result_path",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "overall_risk_percent": result.overall_risk_percent,
                "detections_count": len(result.analyses),
                "json_result_path": str(json_path),
            }
        )


def _save_ocr_text_file(result, image_path: Path) -> Path | None:
    text_blocks: list[str] = []
    for analysis in result.analyses:
        if getattr(analysis, "route", None) != "ocr_extraction_layer":
            continue
        evidence = analysis.risk.evidence if hasattr(analysis.risk, "evidence") else {}
        ocr_info = evidence.get("ocr", {}) if isinstance(evidence, dict) else {}
        text = (ocr_info.get("text") or "").strip() if isinstance(ocr_info, dict) else ""
        if text:
            text_blocks.append(f"[{analysis.detection.class_name}]\n{text}")

    if not text_blocks:
        return None

    txt_path = OCR_RESULTS_DIR / f"{_make_result_basename(image_path)}.txt"
    txt_path.write_text("\n\n".join(text_blocks), encoding="utf-8")
    return txt_path


def _save_result_files(result, image_path: Path) -> tuple[Path, Path | None]:
    base_name = _make_result_basename(image_path)
    json_path = JSON_RESULTS_DIR / f"{base_name}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    _append_summary_row(result, image_path, json_path)
    ocr_text_path = _save_ocr_text_file(result, image_path)
    return json_path, ocr_text_path


def _jsonable_result(result, image_path: Path, json_path: Path, ocr_text_path: Path | None) -> dict[str, Any]:
    return {
        "ok": True,
        "image_path": str(image_path),
        "json_result_path": str(json_path),
        "ocr_text_path": str(ocr_text_path) if ocr_text_path is not None else None,
        "summary_csv_path": str(SUMMARY_CSV_PATH),
        "result": result.to_dict(),
    }


def _risk_tone(percent: int) -> str:
    if percent >= 60:
        return "high"
    if percent >= 30:
        return "medium"
    return "low"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeakLock Image Check</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #523147;
      --muted: #7d6472;
      --panel: rgba(255, 255, 255, 0.82);
      --line: #f1cad9;
      --pink: #ffedf5;
      --blue: #eff8ff;
      --mint: #effff6;
      --warn: #fff6dd;
      --danger: #fff0f3;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(135deg, #fff7fb 0%, #f6fbff 48%, #fffdf4 100%);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app {
      width: min(1100px, calc(100vw - 28px));
      margin: 22px auto;
      padding: 22px;
      border: 1px solid #ffd7e5;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.68);
      box-shadow: 0 18px 52px rgba(181, 103, 139, 0.16);
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
      letter-spacing: 0;
      font-weight: 900;
    }

    .sub {
      margin-top: 7px;
      color: var(--muted);
      font-weight: 700;
      font-size: 14px;
    }

    .pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }

    .pill {
      padding: 7px 11px;
      border: 1px solid rgba(216, 123, 159, 0.22);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.74);
      color: #694157;
      font-size: 12px;
      font-weight: 800;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(360px, 1.1fr);
      gap: 16px;
    }

    .panel {
      min-width: 0;
      border: 1px solid rgba(216, 123, 159, 0.18);
      border-radius: 14px;
      background: var(--panel);
      padding: 14px;
      box-shadow: 0 8px 22px rgba(91, 49, 72, 0.07);
    }

    .panel-title {
      margin-bottom: 10px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 900;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 120px 96px;
      gap: 8px;
      margin-bottom: 10px;
    }

    button, .file-label {
      height: 40px;
      border: 1px solid transparent;
      border-radius: 10px;
      background: #fff;
      color: #543248;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
      box-shadow: 0 5px 14px rgba(91, 49, 72, 0.08);
    }

    button:disabled {
      opacity: 0.48;
      cursor: not-allowed;
    }

    .file-label { background: #fff6fa; border-color: #ffd8e6; }
    #analyzeButton { background: #edf8ff; border-color: #cfe9ff; }
    #resetButton { background: #fffdf4; border-color: #f4e7b9; }
    #fileInput { display: none; }

    .status {
      min-height: 42px;
      border-radius: 12px;
      padding: 11px 12px;
      color: #684358;
      background: var(--pink);
      border: 1px solid #ffd5e4;
      font-size: 13px;
      font-weight: 800;
      display: flex;
      align-items: center;
    }

    .status.busy { background: var(--blue); border-color: #cfe6ff; color: #315579; }
    .status.good { background: var(--mint); border-color: #bfe8d0; color: #2e6849; }
    .status.bad { background: #fff2f1; border-color: #ffc4bf; color: #9b4037; }

    .preview-name {
      color: var(--ink);
      font-weight: 900;
      margin: 12px 0 8px;
      word-break: break-word;
    }

    .preview-frame {
      overflow: hidden;
      border-radius: 12px;
      border: 1px dashed #f3bdd1;
      min-height: 260px;
      background: #fffafd;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .preview-frame img {
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      display: none;
    }

    .empty {
      color: #b4879a;
      font-weight: 800;
      text-align: center;
      padding: 32px 18px;
    }

    .result-grid {
      display: grid;
      grid-template-columns: minmax(140px, 0.7fr) minmax(180px, 1.3fr);
      gap: 12px;
      margin-bottom: 12px;
    }

    .risk-card {
      border-radius: 14px;
      padding: 16px;
      color: #4e3443;
      background: #f8fbff;
      border: 1px solid #dbeafe;
    }

    .risk-card.medium { background: var(--warn); border-color: #ffe2a8; }
    .risk-card.high { background: var(--danger); border-color: #ffc5d2; }

    .risk-label {
      color: #77606c;
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 900;
    }

    .risk-score {
      margin-top: 7px;
      font-size: 46px;
      font-weight: 900;
      line-height: 1;
    }

    .mini-stat {
      margin-bottom: 8px;
      border: 1px solid #f0dbe4;
      border-radius: 12px;
      background: #fff;
      padding: 10px 12px;
      color: #5c4150;
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .mini-stat b { color: var(--ink); }

    .detection-card {
      margin-top: 10px;
      border: 1px solid #eed8e2;
      border-radius: 12px;
      background: #fffefe;
      padding: 12px;
    }

    .detection-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      color: var(--ink);
      font-weight: 900;
    }

    .detection-meta {
      color: #806b76;
      font-size: 12px;
      margin-top: 4px;
    }

    .reason {
      color: #593849;
      margin-top: 8px;
      font-size: 13px;
    }

    details {
      margin-top: 12px;
      border-radius: 12px;
      border: 1px solid #ead6df;
      background: #fff;
      padding: 10px 12px;
    }

    summary {
      cursor: pointer;
      font-weight: 900;
    }

    pre {
      max-height: 320px;
      overflow: auto;
      border-radius: 10px;
      background: #2b2230;
      color: #fff7fb;
      padding: 12px;
      font-size: 12px;
    }

    code {
      color: #674057;
      background: #fff3f8;
      border-radius: 6px;
      padding: 2px 5px;
    }

    @media (max-width: 800px) {
      .hero, .grid, .result-grid { display: block; }
      .pills { justify-content: flex-start; margin-top: 12px; }
      .panel { margin-top: 12px; }
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="hero">
      <div>
        <h1>LeakLock Image Check</h1>
        <div class="sub">Pastel privacy scan for faces, IDs, documents, and plates.</div>
      </div>
      <div class="pills" aria-label="Pipeline layers">
        <span class="pill">YOLOv8</span>
        <span class="pill">Face age</span>
        <span class="pill">OCR risk</span>
      </div>
    </section>

    <section class="grid">
      <section class="panel">
        <div class="panel-title">Image</div>
        <div class="controls">
          <label class="file-label" for="fileInput">Choose image</label>
          <input id="fileInput" type="file" accept="image/*">
          <button id="analyzeButton" type="button" disabled>Analyze</button>
          <button id="resetButton" type="button">Reset</button>
        </div>
        <div id="status" class="status">Ready</div>
        <div id="previewName" class="preview-name"></div>
        <div class="preview-frame">
          <div id="previewEmpty" class="empty">No image selected</div>
          <img id="previewImage" alt="Selected image preview">
        </div>
      </section>

      <section class="panel">
        <div class="panel-title">Result</div>
        <div id="resultPanel">
          <div class="detection-card">
            <div class="detection-head">Waiting for analysis</div>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script>
    const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
    const fileInput = document.getElementById("fileInput");
    const analyzeButton = document.getElementById("analyzeButton");
    const resetButton = document.getElementById("resetButton");
    const statusBox = document.getElementById("status");
    const previewName = document.getElementById("previewName");
    const previewImage = document.getElementById("previewImage");
    const previewEmpty = document.getElementById("previewEmpty");
    const resultPanel = document.getElementById("resultPanel");

    let selectedFile = null;
    let selectedDataUrl = null;
    let selectedHash = null;
    let lastAnalyzedHash = null;
    let inFlight = false;

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function sha1Hex(blob) {
      const bytes = await blob.arrayBuffer();
      const hash = await crypto.subtle.digest("SHA-1", bytes);
      return Array.from(new Uint8Array(hash)).map(byte => byte.toString(16).padStart(2, "0")).join("");
    }

    function setStatus(message, tone = "") {
      statusBox.className = `status ${tone}`.trim();
      statusBox.textContent = message;
    }

    function clearSelection() {
      selectedFile = null;
      selectedDataUrl = null;
      selectedHash = null;
      lastAnalyzedHash = null;
      inFlight = false;
      fileInput.value = "";
      analyzeButton.disabled = true;
      previewName.textContent = "";
      previewImage.removeAttribute("src");
      previewImage.style.display = "none";
      previewEmpty.style.display = "block";
      resultPanel.innerHTML = '<div class="detection-card"><div class="detection-head">Waiting for analysis</div></div>';
      setStatus("Ready");
    }

    function readAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    function riskTone(percent) {
      if (percent >= 60) return "high";
      if (percent >= 30) return "medium";
      return "low";
    }

    function renderDetections(result) {
      const analyses = result.analyses || [];
      if (!analyses.length) {
        return '<div class="detection-card"><div class="detection-head">No sensitive objects detected</div></div>';
      }

      return analyses.map((analysis, index) => {
        const detection = analysis.detection || {};
        const risk = analysis.risk || {};
        const evidence = risk.evidence || {};
        const age = evidence.age_estimate || {};
        let ageLine = "";
        if (age.age_years !== undefined && age.age_years !== null) {
          const confidence = age.confidence !== undefined && age.confidence !== null
            ? ` - confidence ${Number(age.confidence).toFixed(2)}`
            : "";
          ageLine = `<div class="detection-meta">Age estimate: <b>${escapeHtml(age.age_years)}</b>${confidence}</div>`;
        } else if (Object.keys(age).length) {
          ageLine = `<div class="detection-meta">Age estimate unavailable: ${escapeHtml(age.details || "no details")}</div>`;
        }

        return `
          <div class="detection-card">
            <div class="detection-head">
              <span>${index + 1}. ${escapeHtml(String(detection.class_name || "object"))}</span>
              <span>${escapeHtml(risk.risk_percent ?? 0)}%</span>
            </div>
            <div class="detection-meta">route: ${escapeHtml(analysis.route || "")} - detection confidence ${Number(detection.confidence || 0).toFixed(2)}</div>
            ${ageLine}
            <div class="reason">${escapeHtml(risk.reason || "")}</div>
          </div>
        `;
      }).join("");
    }

    function renderResult(payload) {
      const result = payload.result;
      const percent = result.overall_risk_percent || 0;
      const tone = riskTone(percent);
      resultPanel.innerHTML = `
        <div class="result-grid">
          <div class="risk-card ${tone}">
            <div class="risk-label">Overall risk</div>
            <div class="risk-score">${percent}%</div>
          </div>
          <div>
            <div class="mini-stat"><b>Image:</b> ${escapeHtml(selectedFile.name)}</div>
            <div class="mini-stat"><b>Detections:</b> ${(result.analyses || []).length}</div>
            <div class="mini-stat"><b>JSON:</b> <code>${escapeHtml(payload.json_result_path)}</code></div>
            <div class="mini-stat"><b>OCR text:</b> <code>${escapeHtml(payload.ocr_text_path || "None")}</code></div>
          </div>
        </div>
        ${renderDetections(result)}
        <details>
          <summary>Raw JSON</summary>
          <pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>
        </details>
      `;
    }

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;

      if (!file.type.startsWith("image/")) {
        clearSelection();
        setStatus("Only image files are supported", "bad");
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        clearSelection();
        setStatus("Image is larger than the 25 MB limit", "bad");
        return;
      }

      try {
        const fileHash = await sha1Hex(file);
        if (fileHash === selectedHash) {
          fileInput.value = "";
          return;
        }

        selectedFile = file;
        selectedHash = fileHash;
        selectedDataUrl = await readAsDataUrl(file);
        previewName.textContent = file.name;
        previewImage.src = selectedDataUrl;
        previewImage.style.display = "block";
        previewEmpty.style.display = "none";
        analyzeButton.disabled = false;
        fileInput.value = "";
        setStatus(`Selected ${file.name}`, "good");
      } catch (error) {
        clearSelection();
        setStatus(`Could not read image: ${error}`, "bad");
      }
    });

    analyzeButton.addEventListener("click", async () => {
      if (inFlight || !selectedFile || !selectedDataUrl) return;
      if (selectedHash === lastAnalyzedHash) {
        analyzeButton.disabled = true;
        setStatus("This image is already analyzed. Reset or choose another image.");
        return;
      }

      inFlight = true;
      analyzeButton.disabled = true;
      setStatus("Analyzing image", "busy");
      resultPanel.innerHTML = '<div class="status busy">Running LeakLock pipeline...</div>';

      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileName: selectedFile.name,
            dataUrl: selectedDataUrl,
            sha1: selectedHash
          })
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || `Request failed with ${response.status}`);
        }

        lastAnalyzedHash = selectedHash;
        renderResult(payload);
        setStatus(`Analysis complete - risk ${payload.result.overall_risk_percent}%`, "good");
      } catch (error) {
        resultPanel.innerHTML = `<div class="status bad">Analysis failed</div><pre>${escapeHtml(error.stack || error)}</pre>`;
        setStatus(`Failed: ${error.message || error}`, "bad");
        analyzeButton.disabled = false;
      } finally {
        inFlight = false;
      }
    });

    resetButton.addEventListener("click", clearSelection);
  </script>
</body>
</html>
"""


class LeakLockWebHandler(BaseHTTPRequestHandler):
    server_version = "LeakLockWeb/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json_body()
            file_name = str(payload.get("fileName") or "uploaded_image.png")
            data_url = str(payload.get("dataUrl") or "")
            media_type, content = _decode_data_url(data_url)
            extension = mimetypes.guess_extension(media_type) or Path(file_name).suffix
            if "." not in Path(file_name).name and extension:
                file_name = f"{file_name}{extension}"

            image_path = _write_upload(file_name, content)
            result = self.server.pipeline.analyze_image(image_path)  # type: ignore[attr-defined]
            json_path, ocr_text_path = _save_result_files(result, image_path)
            self._send_json(_jsonable_result(result, image_path, json_path, ocr_text_path))
        except Exception as exc:  # noqa: BLE001 - web boundary returns JSON errors.
            self._send_json(
                {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[LeakLockWeb] {self.address_string()} - {format % args}")

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body was empty.")
        if content_length > MAX_UPLOAD_BYTES * 2:
            raise ValueError("Request body is too large.")

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - json raises mixed errors.
            raise ValueError(f"Request body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_server(host: str, port: int, weights: Path | None = None) -> ThreadingHTTPServer:
    _ensure_dirs()
    config = PipelineConfig(repo_root=REPO_ROOT)
    if weights is not None:
        config.yolo_weights_path = weights.resolve()
    pipeline = LeakLockPipeline(config=config)
    server = ThreadingHTTPServer((host, port), LeakLockWebHandler)
    server.pipeline = pipeline  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LeakLock local browser UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional YOLO weights path.")
    args = parser.parse_args()

    server = build_server(args.host, args.port, args.weights)
    print(f"LeakLock web app running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping LeakLock web app.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
