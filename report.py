from __future__ import annotations

import json
from html import escape
from pathlib import Path


METHOD_LABELS = {
    "none": "Вход",
    "trained_prior": "Обученная модель",
    "wiener": "Фильтр Винера",
    "rl": "Ричардсон–Люси",
    "wf_hybrid": "WF-hybrid",
    "trl_hybrid": "TRL-hybrid",
    "ct_hybrid": "CT-hybrid",
}

PRESET_LABELS = {
    "microscopy": "флуоресцентная микроскопия",
    "ct": "спектральная КТ",
    "custom": "загруженное изображение",
}

DISTORTION_LABELS = {
    "blur": "размытие",
    "stripe_artifacts": "полосовые артефакты",
    "strong_noise": "сильный шум",
    "mixed": "смешанные искажения",
    "unknown": "не определено",
}


def _best(summary: dict, key: str) -> str | None:
    rows = [row for row in summary["results"] if key in row]
    if not rows:
        return None
    return max(rows, key=lambda row: row.get(key, 0)).get("name")


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _result_cards(summary: dict) -> str:
    best_psnr = _best(summary, "psnr")
    best_ssim = _best(summary, "ssim")
    best_iou = _best(summary, "iou")
    cards = []
    for item in summary["results"]:
        name = item["name"]
        label = METHOD_LABELS.get(name, name)
        badges = []
        if name == "trained_prior":
            badges.append('<span class="badge trained">обучена</span>')
        if name == best_psnr:
            badges.append('<span class="badge">лучший PSNR</span>')
        if name == best_ssim:
            badges.append('<span class="badge">лучший SSIM</span>')
        if name == best_iou:
            badges.append('<span class="badge alt">лучший IoU</span>')
        badge_html = "".join(badges)
        cards.append(
            f"""
            <article class="result-card {'is-trained' if name == 'trained_prior' else ''}">
              <div class="card-title">
                <h2>{escape(label)}</h2>
                <div class="badges">{badge_html}</div>
              </div>
              <a class="image-link" href="{escape(item['image_file'])}">
                <img src="{escape(item['image_file'])}" alt="{escape(label)} restored image">
              </a>
              <dl class="metrics">
                <div><dt>PSNR</dt><dd>{_fmt(item.get('psnr'), 2)}</dd></div>
                <div><dt>SSIM</dt><dd>{_fmt(item.get('ssim'), 3)}</dd></div>
                <div><dt>IoU</dt><dd>{_fmt(item.get('iou'), 3)}</dd></div>
                <div><dt>Объекты</dt><dd>{int(item.get('objects', 0))}</dd></div>
              </dl>
              <a class="overlay-link" href="{escape(item['overlay_file'])}">Разметка</a>
            </article>
            """
        )
    return "\n".join(cards)


def write_html_report(out_dir: str | Path, summary: dict, title: str = "Uvarov restoration report") -> Path:
    out_dir = Path(out_dir)
    distortion = summary.get("distortion", {})
    best_psnr = _best(summary, "psnr") or "n/a"
    best_iou = _best(summary, "iou") or "n/a"
    preset_raw = str(summary.get("preset", "custom"))
    distortion_raw = str(distortion.get("label", "unknown"))
    preset_label = PRESET_LABELS.get(preset_raw, preset_raw)
    distortion_label = DISTORTION_LABELS.get(distortion_raw, distortion_raw)
    cards = _result_cards(summary)
    json_payload = escape(json.dumps(summary, ensure_ascii=False, indent=2))
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --ink: #10002b;
      --muted: #8a959d;
      --line: #dce5e8;
      --panel: #ffffff;
      --surface: #f5f8fa;
      --green: #009b3a;
      --teal: #02b1a9;
      --teal-dark: #008f89;
      --amber: #b7791f;
      --rose: #b64c57;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--surface);
      color: var(--ink);
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    .service-bar {{
      background: var(--ink);
      color: #ffffff;
      font-size: 13px;
    }}
    .service-inner {{
      max-width: 1320px;
      margin: 0 auto;
      min-height: 36px;
      padding: 8px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    .topbar {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 18px;
      min-width: 0;
    }}
    .brand img {{
      width: 152px;
      height: 26px;
      display: block;
      flex: 0 0 auto;
    }}
    .divider {{
      width: 1px;
      height: 34px;
      background: var(--line);
      flex: 0 0 auto;
    }}
    h1 {{
      margin: 0;
      font-size: 25px;
      letter-spacing: 0;
      line-height: 1.1;
    }}
    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .summary-pill {{
      min-height: 42px;
      display: inline-flex;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid rgba(2,177,169,.32);
      border-radius: 8px;
      background: rgba(2,177,169,.08);
      color: var(--teal-dark);
      font-weight: 800;
      white-space: nowrap;
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px 28px 44px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .meta-item {{
      min-height: 74px;
      padding: 13px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .meta-item span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .meta-item strong {{
      display: block;
      margin-top: 6px;
      font-size: 18px;
      color: var(--ink);
      overflow-wrap: anywhere;
    }}
    .results {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 16px;
    }}
    .result-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 14px 30px rgba(16,0,43,.06);
    }}
    .result-card.is-trained {{
      border-color: rgba(15,118,110,.45);
      box-shadow: 0 16px 32px rgba(15,118,110,.12);
    }}
    .card-title {{
      min-height: 72px;
      padding: 13px 14px 11px;
      border-bottom: 1px solid var(--line);
    }}
    .card-title h2 {{
      margin: 0 0 8px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .badge {{
      display: inline-flex;
      min-height: 22px;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(47,111,159,.11);
      color: var(--ink);
      font-size: 11px;
      font-weight: 800;
    }}
    .badge.trained {{ background: rgba(2,177,169,.12); color: var(--teal-dark); }}
    .badge.alt {{ background: rgba(183,121,31,.12); color: var(--amber); }}
    .image-link {{
      display: block;
      background: var(--ink);
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin: 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .metrics div {{
      padding: 11px 13px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .metrics div:nth-child(even) {{ border-right: 0; }}
    .metrics div:nth-last-child(-n + 2) {{ border-bottom: 0; }}
    dt {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }}
    dd {{
      margin: 4px 0 0;
      font-size: 19px;
      font-weight: 850;
    }}
    .overlay-link {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      text-decoration: none;
      color: var(--teal-dark);
      font-weight: 850;
      background: #fbfdfc;
    }}
    .overlay-link:hover {{ background: #eefdfb; }}
    details {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }}
    summary {{
      padding: 14px 16px;
      cursor: pointer;
      font-weight: 850;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      background: var(--ink);
      color: #d7ede8;
      font-size: 12px;
      line-height: 1.5;
    }}
    @media (max-width: 850px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .brand {{ align-items: flex-start; flex-direction: column; gap: 10px; }}
      .divider {{ display: none; }}
      .meta {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      h1 {{ font-size: 22px; }}
    }}
    @media (max-width: 560px) {{
      main, .topbar, .service-inner {{ padding-left: 16px; padding-right: 16px; }}
      .service-inner {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ grid-template-columns: 1fr; }}
      .summary-pill {{ white-space: normal; }}
      h1 {{ font-size: 21px; }}
    }}
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
          <h1>{escape(title)}</h1>
          <p class="subtitle">Сравнение восстановления, детекции и результатов разметки</p>
        </div>
      </div>
      <div class="summary-pill">обученная модель активна</div>
    </div>
  </header>
  <main>
    <section class="meta">
      <div class="meta-item"><span>Модальность</span><strong>{escape(preset_label)}</strong></div>
      <div class="meta-item"><span>Искажение</span><strong>{escape(distortion_label)}</strong></div>
      <div class="meta-item"><span>Лучший PSNR</span><strong>{escape(METHOD_LABELS.get(best_psnr, best_psnr))}</strong></div>
      <div class="meta-item"><span>Лучший IoU</span><strong>{escape(METHOD_LABELS.get(best_iou, best_iou))}</strong></div>
    </section>
    <section class="results">
      {cards}
    </section>
    <details>
      <summary>Технические данные JSON</summary>
      <pre>{json_payload}</pre>
    </details>
  </main>
</body>
</html>"""
    report = out_dir / "report.html"
    report.write_text(html, encoding="utf-8")
    return report
