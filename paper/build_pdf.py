"""Render paper/weight_space_router.md to paper/weight_space_router.pdf.

    ~/ai_to_ai_loops/.venv/bin/python -I paper/build_pdf.py        # needs markdown-it-py and a Chromium headless shell

The Markdown stays the source of truth; this script only adds layout (cover block, boxed sections, figure captions).
"""
import glob
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
SRC, HTML, PDF = HERE / "weight_space_router.md", HERE / "weight_space_router.html", HERE / "weight_space_router.pdf"

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400&display=block');
:root { --ink:#141413; --ink2:#55534e; --muted:#8a8984; --rule:#e4e3df; --accent:#2a78d6; --accent-dk:#1b4f91;
        --tint:#eef4fb; --warm:#fbf6ee; --warm-rule:#eb6834; }
@page { size: Letter; margin: 0.85in 0.95in 0.9in 0.95in;
  @bottom-left { content: "Claude, please fix my model · Hinkel · November 2026"; font: 500 7.8pt Inter, sans-serif; color: #8a8984; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 500 7.8pt Inter, sans-serif; color: #8a8984; } }
@page :first { @bottom-left { content: none; } }
html { font-size: 11.6pt; }
body { font-family: "Source Serif 4", Georgia, serif; color: var(--ink); line-height: 1.52; margin: 0; background: #fff;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; hyphens: auto; font-optical-sizing: auto; }
p { margin: 0 0 0.62em; orphans: 3; widows: 3; }
a { color: var(--accent-dk); text-decoration: none; word-break: break-word; }
strong { font-weight: 600; }
code { font-family: "JetBrains Mono", monospace; font-size: 0.84em; background: #f3f2ef; padding: 0.05em 0.3em; border-radius: 3px; }
h1, h2, h3 { font-family: Inter, sans-serif; color: var(--ink); break-after: avoid; }
h2 { font-size: 15pt; font-weight: 700; margin: 1.5em 0 0.55em; padding-top: 0.5em; border-top: 2px solid var(--ink); letter-spacing: -0.01em; }
h3 { font-size: 11.8pt; font-weight: 600; margin: 1.25em 0 0.4em; color: var(--accent-dk); }
ul, ol { margin: 0 0 0.7em; padding-left: 1.35em; }
li { margin: 0.18em 0; }
li > ul, li > ol { margin: 0.15em 0 0.2em; }
hr { display: none; }

/* cover */
.kicker { font: 600 8.5pt Inter, sans-serif; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.9em; }
h1 { font-size: 25pt; line-height: 1.12; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 0.45em; }
h1 .sub { display: block; font-size: 15pt; font-weight: 500; color: var(--ink2); letter-spacing: -0.01em; margin-top: 0.35em; line-height: 1.25; }
.byline { font: 600 11pt Inter, sans-serif; margin: 0 0 0.15em; }
.status { font: 400 9pt Inter, sans-serif; color: var(--ink2); margin: 0 0 1.4em; padding-bottom: 1.1em; border-bottom: 1px solid var(--rule); }
.status em { font-style: normal; }
.highlights { background: var(--tint); border-radius: 6px; padding: 0.9em 1.2em 0.6em; margin: 0 0 1.1em; break-inside: avoid; }
.highlights h3, .inbrief h3, .bigpic h3 { font: 700 8.5pt Inter, sans-serif; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent-dk); margin: 0 0 0.6em; }
.highlights ul { margin: 0; padding-left: 1.1em; }
.highlights li { margin: 0 0 0.45em; font-size: 10.8pt; line-height: 1.45; }
.highlights li::marker { color: var(--accent); }
.inbrief { border-left: 3px solid var(--accent); padding: 0.1em 0 0.1em 1em; margin: 0 0 1em; font-size: 11pt; color: var(--ink2); break-inside: avoid; }
.inbrief p { margin: 0; }
.bigpic { background: var(--warm); border-left: 3px solid var(--warm-rule); border-radius: 0 6px 6px 0; padding: 0.95em 1.2em 0.5em; margin: 1.4em 0; break-inside: avoid; }
.bigpic h3 { color: #a8431c; }
.bigpic h2 { all: unset; }

/* figures and tables */
figure { margin: 1.3em 0 1.4em; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure.narrow img { width: 82%; margin: 0 auto; }
figcaption, .tcap { font-family: Inter, sans-serif; font-size: 8.9pt; line-height: 1.45; color: var(--ink2); margin-top: 0.55em; }
figcaption strong, .tcap strong { color: var(--ink); }
table { border-collapse: collapse; width: 100%; font-family: Inter, sans-serif; font-size: 8.8pt; margin: 1em 0 0.3em; break-inside: avoid; font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 600; border-bottom: 1.5px solid var(--ink); padding: 0.4em 0.5em; vertical-align: bottom; }
td { border-bottom: 1px solid var(--rule); padding: 0.38em 0.5em; }
tr:nth-child(n+4) td:first-child { font-weight: 500; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
.tcap { margin-bottom: 1.2em; }

/* back matter */
.backmatter { font-size: 10.2pt; color: var(--ink2); }
.backmatter h2 { font-size: 12.5pt; border-top-width: 1px; }
.backmatter h3 { font-size: 10.5pt; }
.refs ol { font-size: 9.6pt; padding-left: 1.6em; }
.refs li { margin-bottom: 0.35em; }
"""


def build():
    md = SRC.read_text()
    html = MarkdownIt("commonmark").enable("table").render(md)

    # Title: split "Main: subtitle" and add kicker, byline, status line.
    m = re.search(r"<h1>(.*?): (.*?)</h1>\s*<p><strong>(.*?)</strong></p>\s*<p><em>(.*?)</em></p>", html, re.S)
    status = m.group(4).replace(" An informal report in the style of a Patterns article.", "")
    cover = (f'<div class="kicker">Experiment report · Weight-space programming</div>'
             f'<h1>{m.group(1)}<span class="sub">{m.group(2)[0].upper() + m.group(2)[1:]}</span></h1>'
             f'<p class="byline">{m.group(3)}</p><p class="status">{status}</p>')
    html = html[:m.start()] + cover + html[m.end():]

    # Boxes: Highlights, In brief, The bigger picture.
    html = re.sub(r"(<h3>Highlights</h3>\s*<ul>.*?</ul>)", r'<section class="highlights">\1</section>', html, flags=re.S)
    html = re.sub(r"(<h3>In brief</h3>\s*<p>.*?</p>)", r'<section class="inbrief">\1</section>', html, flags=re.S)
    html = re.sub(r"<h2>The bigger picture</h2>(.*?)(?=<hr />|<h2>)",
                  r'<section class="bigpic"><h3>The bigger picture</h3>\1</section>', html, flags=re.S)

    # Figures: image paragraph + following bold "Figure N." caption paragraph.
    def fig(mm):
        cls = ' class="narrow"' if "fig1" not in mm.group(1) else ""
        return f'<figure{cls}><img src="{mm.group(1)}" alt="{mm.group(2)}"><figcaption>{mm.group(3)}</figcaption></figure>'
    html = re.sub(r'<p><img src="(.*?)" alt="(.*?)" /></p>\s*<p>(<strong>Figure.*?)</p>', fig, html, flags=re.S)
    html = re.sub(r"<p>(<strong>Table \d.*?)</p>", r'<p class="tcap">\1</p>', html, flags=re.S)

    # Back matter (Resource availability onward) in a smaller register; references styled.
    i = html.index("<h2>Resource availability</h2>")
    j = html.index("<h2>References</h2>")
    html = html[:i] + '<div class="backmatter">' + html[i:j] + '<section class="refs">' + html[j:] + "</section></div>"

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>Claude, please fix my model</title><style>{CSS}</style></head><body>{html}</body></html>')
    HTML.write_text(page)

    chrome = sorted(glob.glob(str(Path.home() / ".cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")))[-1]
    subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)
    print("wrote", PDF, PDF.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    build()
