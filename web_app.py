from __future__ import annotations

import argparse
import html
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from uvarov_restoration.image_io import load_grayscale
from uvarov_restoration.pipeline import PipelineConfig, run_pipeline
from uvarov_restoration.report import write_html_report
from uvarov_restoration.synthetic import make_sample

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    boundary_token = "boundary="
    if boundary_token not in content_type:
        return {}, {}
    boundary = content_type.split(boundary_token, 1)[1].strip().strip('"').encode()
    marker = b"--" + boundary
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in body.split(marker):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, payload = part.partition(b"\r\n\r\n")
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
        if "name=" not in disposition:
            continue
        name = disposition.split("name=", 1)[1].split(";", 1)[0].strip().strip('"')
        filename = ""
        if "filename=" in disposition:
            filename = disposition.split("filename=", 1)[1].split(";", 1)[0].strip().strip('"')
        payload = payload.rstrip(b"\r\n")
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    server_version = "UvarovRestorationMVP/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.respond_html(index_html())
            return
        if parsed.path.startswith("/runs/"):
            self.serve_file(ROOT / parsed.path.lstrip("/"))
            return
        if parsed.path.startswith("/assets/"):
            self.serve_file(ROOT / parsed.path.lstrip("/"))
            return
        if parsed.path == "/demo":
            qs = parse_qs(parsed.query)
            preset = qs.get("preset", ["microscopy"])[0]
            self.run_demo(preset if preset in {"microscopy", "ct"} else "microscopy")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))
        preset = fields.get("preset", "microscopy")
        method = fields.get("method", "auto").strip()
        run_dir = RUNS / f"web_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        clean = None
        mask = None
        if "image" in files and files["image"][1]:
            filename, payload = files["image"]
            suffix = Path(filename).suffix or ".png"
            upload_path = run_dir / f"upload{suffix}"
            upload_path.write_bytes(payload)
            image = load_grayscale(upload_path)
            preset_name = "custom"
        else:
            clean, image, mask = make_sample(preset if preset in {"microscopy", "ct"} else "microscopy")
            preset_name = preset

        if method == "auto":
            methods = ["none", "trained_prior", "wiener", "rl", "wf_hybrid", "trl_hybrid", "ct_hybrid"]
        else:
            methods = ["none", method]
        config = PipelineConfig(methods=methods, min_area=22 if preset == "microscopy" else 30)
        summary = run_pipeline(image, run_dir, clean=clean, ground_truth_mask=mask, preset=preset_name, config=config)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report = write_html_report(run_dir, summary, title="Результат восстановления")
        rel = "/" + report.relative_to(ROOT).as_posix()
        self.send_response(303)
        self.send_header("Location", rel)
        self.end_headers()

    def run_demo(self, preset: str) -> None:
        run_dir = RUNS / f"{preset}_{int(time.time())}"
        clean, image, mask = make_sample(preset)
        methods = ["none", "trained_prior", "wiener", "rl", "wf_hybrid", "trl_hybrid", "ct_hybrid"]
        summary = run_pipeline(
            image,
            run_dir,
            clean=clean,
            ground_truth_mask=mask,
            preset=preset,
            config=PipelineConfig(methods=methods, min_area=22 if preset == "microscopy" else 30),
        )
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        report_title = "Демо: микроскопия" if preset == "microscopy" else "Демо: спектральная КТ"
        report = write_html_report(run_dir, summary, title=report_title)
        rel = "/" + report.relative_to(ROOT).as_posix()
        self.send_response(303)
        self.send_header("Location", rel)
        self.end_headers()

    def serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            data = resolved.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html; charset=utf-8"
        if path.suffix.lower() == ".png":
            content_type = "image/png"
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            content_type = "image/jpeg"
        elif path.suffix.lower() == ".svg":
            content_type = "image/svg+xml; charset=utf-8"
        elif path.suffix.lower() == ".json":
            content_type = "application/json; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def respond_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def index_html() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>АК БАРС Медицина — восстановление изображений</title>
  <style>
    :root {
      --abm-ink: #10002b;
      --abm-green: #009b3a;
      --abm-teal: #02b1a9;
      --abm-teal-dark: #008f89;
      --abm-muted: #8a959d;
      --abm-line: #dce5e8;
      --abm-soft: #f5f8fa;
      --abm-panel: #ffffff;
      --abm-warning: #b7791f;
      --abm-danger: #b64c57;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Roboto, "Segoe UI", Arial, sans-serif;
      background: var(--abm-soft);
      color: var(--abm-ink);
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--abm-line);
    }
    .service-bar {
      background: var(--abm-ink);
      color: #ffffff;
      font-size: 13px;
    }
    .service-inner {
      max-width: 1240px;
      margin: 0 auto;
      min-height: 36px;
      padding: 8px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .service-inner span {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .topbar {
      max-width: 1240px;
      margin: 0 auto;
      min-height: 74px;
      padding: 16px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
    }
    .brand img {
      width: 152px;
      height: 26px;
      display: block;
      flex: 0 0 auto;
    }
    .divider {
      width: 1px;
      height: 34px;
      background: var(--abm-line);
      flex: 0 0 auto;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 4px 0 0;
      color: var(--abm-muted);
      font-size: 14px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      min-height: 40px;
      padding: 9px 12px;
      border: 1px solid rgba(2,177,169,.32);
      border-radius: 8px;
      background: rgba(2,177,169,.08);
      color: var(--abm-teal-dark);
      font-weight: 800;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--abm-teal);
      box-shadow: 0 0 0 4px rgba(2,177,169,.14);
    }
    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 26px 28px 44px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, 410px) 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--abm-panel);
      border: 1px solid var(--abm-line);
      border-radius: 8px;
      box-shadow: 0 14px 30px rgba(16,0,43,.06);
      overflow: hidden;
    }
    .panel-head {
      min-height: 58px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--abm-line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(2,177,169,.10);
      color: var(--abm-teal-dark);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    form { padding: 18px; }
    label {
      display: block;
      margin: 15px 0 7px;
      font-size: 13px;
      font-weight: 800;
      color: var(--abm-ink);
    }
    select,
    input[type=file] {
      width: 100%;
      min-height: 42px;
      padding: 10px 11px;
      border: 1px solid #c4d0d4;
      border-radius: 8px;
      background: #fbfdfd;
      color: var(--abm-ink);
      font: inherit;
    }
    input.file-input {
      position: absolute;
      width: 1px !important;
      min-height: 1px !important;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    .file-picker {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      width: 100%;
      min-height: 58px;
      padding: 12px 14px;
      border: 1px dashed #b8c9ce;
      border-radius: 8px;
      background: #fbfdfd;
      cursor: pointer;
    }
    .file-picker span {
      color: var(--abm-ink);
      font-weight: 800;
    }
    .file-picker small {
      color: var(--abm-muted);
      font-size: 12px;
      font-weight: 700;
      text-align: right;
    }
    .file-picker:hover {
      border-color: var(--abm-teal);
      background: #f4fdfc;
    }
    select:focus,
    input[type=file]:focus {
      outline: 3px solid rgba(2,177,169,.18);
      border-color: var(--abm-teal);
    }
    .actions {
      margin-top: 18px;
      display: grid;
    }
    button,
    a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 11px 16px;
      border: 1px solid transparent;
      border-radius: 8px;
      text-decoration: none;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    button {
      color: #ffffff;
      background: linear-gradient(90deg, var(--abm-green), var(--abm-teal));
      box-shadow: 0 10px 18px rgba(2,177,169,.18);
    }
    button:hover { filter: brightness(.96); }
    a.button {
      color: var(--abm-ink);
      background: #ffffff;
      border-color: var(--abm-line);
    }
    a.button:hover {
      border-color: var(--abm-teal);
      color: var(--abm-teal-dark);
    }
    .demo-row {
      padding: 0 18px 18px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .metrics {
      padding: 18px;
      border-top: 1px solid var(--abm-line);
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      min-height: 72px;
      padding: 12px;
      border: 1px solid var(--abm-line);
      border-radius: 8px;
      background: #fbfdfd;
    }
    .metric span {
      display: block;
      color: var(--abm-muted);
      font-size: 12px;
      font-weight: 800;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      color: var(--abm-teal-dark);
      font-size: 21px;
      line-height: 1.1;
    }
    .preview { min-width: 0; }
    .preview-grid {
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .image-tile {
      overflow: hidden;
      border: 1px solid var(--abm-line);
      border-radius: 8px;
      background: #10002b;
    }
    .image-tile img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
    }
    .tile-caption {
      min-height: 42px;
      padding: 10px 12px;
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--abm-ink);
      font-size: 13px;
      font-weight: 800;
    }
    .method-strip {
      padding: 0 18px 18px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .method {
      min-height: 64px;
      padding: 10px;
      border: 1px solid var(--abm-line);
      border-left: 4px solid var(--abm-green);
      border-radius: 8px;
      background: #fbfdfd;
      font-weight: 800;
    }
    .method small {
      display: block;
      margin-top: 4px;
      color: var(--abm-muted);
      font-weight: 700;
    }
    .method:nth-child(2) { border-left-color: var(--abm-teal); }
    .method:nth-child(3) { border-left-color: var(--abm-warning); }
    .method:nth-child(4) { border-left-color: var(--abm-danger); }
    @media (max-width: 940px) {
      .workspace { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .status { white-space: normal; }
      h1 { font-size: 22px; }
    }
    @media (max-width: 660px) {
      .service-inner, .topbar, main { padding-left: 16px; padding-right: 16px; }
      .service-inner { align-items: flex-start; flex-direction: column; }
      .brand { align-items: flex-start; flex-direction: column; gap: 10px; }
      .divider { display: none; }
      .preview-grid, .method-strip, .metrics, .demo-row { grid-template-columns: 1fr; }
      h1 { font-size: 21px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="service-bar">
      <div class="service-inner">
        <span>8-800-500-03-03 · круглосуточно, бесплатно по России</span>
        <span>Казань, ул. Меридианная, 1а</span>
      </div>
    </div>
    <div class="topbar">
      <div class="brand">
        <img src="/assets/akbars_logo.svg" alt="АК БАРС Медицина">
        <div class="divider" aria-hidden="true"></div>
        <div>
          <h1>ИИ-восстановление медицинских изображений</h1>
          <p class="subtitle">Микроскопия, спектральная КТ и downstream-оценка распознавания</p>
        </div>
      </div>
      <div class="status"><span class="dot"></span> обученная модель подключена</div>
    </div>
  </header>
  <main>
    <div class="workspace">
      <section class="panel">
        <div class="panel-head">
          <h2>Запуск обработки</h2>
          <span class="tag">локальный контур</span>
        </div>
        <form method="post" action="/process" enctype="multipart/form-data">
          <label for="image">Медицинское изображение</label>
          <label class="file-picker" for="image">
            <span>Выбрать файл PNG/JPEG</span>
            <small>или использовать демо</small>
          </label>
          <input class="file-input" id="image" name="image" type="file" accept="image/*">
          <label for="preset">Модальность</label>
          <select id="preset" name="preset">
            <option value="microscopy">Флуоресцентная микроскопия</option>
            <option value="ct">Спектральная КТ</option>
          </select>
          <label for="method">Алгоритм восстановления</label>
          <select id="method" name="method">
            <option value="auto">Сравнить все методы</option>
            <option value="trained_prior">Обученная модель</option>
            <option value="trl_hybrid">TRL-hybrid: Ричардсон–Люси + prior</option>
            <option value="wf_hybrid">WF-hybrid: адаптивный Винер</option>
            <option value="ct_hybrid">CT-hybrid: подавление полос</option>
            <option value="rl">Классический Ричардсон–Люси</option>
            <option value="wiener">Фильтр Винера</option>
          </select>
          <div class="actions">
            <button type="submit">Запустить обработку</button>
          </div>
        </form>
        <div class="demo-row">
          <a class="button" href="/demo?preset=microscopy">Демо: микроскопия</a>
          <a class="button" href="/demo?preset=ct">Демо: КТ</a>
        </div>
        <div class="metrics">
          <div class="metric"><span>Прирост PSNR, микроскопия</span><strong>+2.64 dB</strong></div>
          <div class="metric"><span>Прирост PSNR, КТ</span><strong>+4.89 dB</strong></div>
        </div>
      </section>
      <section class="panel preview">
        <div class="panel-head">
          <h2>Последний обученный прогон</h2>
          <span class="tag">веса models/*.npz</span>
        </div>
        <div class="preview-grid">
          <div class="image-tile">
            <img src="/runs/microscopy_demo/input.png" alt="Входное микроскопическое изображение">
            <div class="tile-caption"><span>Микроскопия: вход</span><span class="tag">шум + PSF</span></div>
          </div>
          <div class="image-tile">
            <img src="/runs/microscopy_demo/trained_prior.png" alt="Результат обученной модели для микроскопии">
            <div class="tile-caption"><span>Микроскопия: обученная модель</span><span class="tag">26.36 dB</span></div>
          </div>
          <div class="image-tile">
            <img src="/runs/ct_demo/input.png" alt="Входное КТ-изображение">
            <div class="tile-caption"><span>КТ: вход</span><span class="tag">полосы</span></div>
          </div>
          <div class="image-tile">
            <img src="/runs/ct_demo/trained_prior.png" alt="Результат обученной модели для КТ">
            <div class="tile-caption"><span>КТ: обученная модель</span><span class="tag">27.26 dB</span></div>
          </div>
        </div>
        <div class="method-strip">
          <div class="method">Обученная модель<small>локальные веса</small></div>
          <div class="method">TRL-hybrid<small>RL + prior</small></div>
          <div class="method">WF-hybrid<small>адаптивный Винер</small></div>
          <div class="method">CT-hybrid<small>подавление полос</small></div>
        </div>
      </section>
    </div>
  </main>
</body>
</html>"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
