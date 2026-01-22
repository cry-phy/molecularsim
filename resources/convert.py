#!/usr/bin/env python3
import argparse
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    /* Make the iframe content clean + centered */
    html, body {{
      margin: 0;
      height: 100%;
      width: 100%;
      background: transparent;
    }}
    .wrap {{
      height: 100%;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: transparent;
    }}
    /* Scale SVG to fit iframe while preserving aspect ratio */
    svg {{
      width: 100%;
      height: 100%;
      max-width: 100%;
      max-height: 100%;
      display: block;
    }}
  </style>
</head>
<body>
  <div class="wrap">
{svg}
  </div>
</body>
</html>
"""


def strip_xml_preamble(s: str) -> str:
    # Remove common XML/DOCTYPE preambles that can be annoying when inlining
    lines = s.splitlines()
    out = []
    for line in lines:
        l = line.lstrip()
        if l.startswith("<?xml"):
            continue
        if l.startswith("<!DOCTYPE"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def force_transparent_background(svg: str) -> str:
    # If there is a root <svg ...> tag with a style that sets background, remove it.
    # We keep it simple: just remove occurrences of "background:" in the root style attribute.
    # (Doesn't try to be a full XML parser; works for typical Inkscape exports.)
    import re

    # remove background / background-color inside style="..."
    def _clean_style(m):
        style = m.group(1)
        style = re.sub(r"background(?:-color)?\s*:\s*[^;\"']+;?", "", style, flags=re.I)
        # collapse redundant spaces/semicolons
        style = re.sub(r";{2,}", ";", style)
        style = style.strip().strip(";")
        return f'style="{style}"' if style else ""

    svg = re.sub(r'style="([^"]*)"', _clean_style, svg, count=1)

    # Also remove any explicit white rect as a first child if it looks like a background.
    # This is heuristic: first <rect ... width="100%" height="100%" ... fill="#fff or white">
    svg = re.sub(
        r'(<svg\b[^>]*>\s*)(<rect\b[^>]*(?:width="100%")[^>]*(?:height="100%")[^>]*\bfill="(?:#fff(?:fff)?|white)"[^>]*/?>\s*)',
        r"\1",
        svg,
        flags=re.I,
    )
    return svg


def main():
    ap = argparse.ArgumentParser(
        description="Convert an SVG file into a self-contained HTML file for iframes (Marp-friendly)."
    )
    ap.add_argument("svg", help="Input .svg path")
    ap.add_argument(
        "-o",
        "--out",
        help="Output .html path (default: same name next to svg)",
        default=None,
    )
    ap.add_argument(
        "--no-transparent",
        action="store_true",
        help="Do not try to force a transparent background",
    )
    args = ap.parse_args()

    in_path = Path(args.svg)
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    svg = in_path.read_text(encoding="utf-8", errors="replace")
    svg = strip_xml_preamble(svg)
    if not args.no_transparent:
        svg = force_transparent_background(svg)

    out_path = Path(args.out) if args.out else in_path.with_suffix(".html")
    html = HTML_TEMPLATE.format(title=in_path.stem, svg=svg)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
