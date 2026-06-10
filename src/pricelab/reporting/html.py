from __future__ import annotations

import html


def markdown_to_basic_html(markdown_text: str) -> str:
    body_lines: list[str] = []
    in_list = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("# "):
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                body_lines.append("<ul>")
                in_list = True
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            if in_list:
                body_lines.append("</ul>")
                in_list = False
            body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_list:
        body_lines.append("</ul>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>PriceLab Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;line-height:1.5;color:#17202a}"
        "h1,h2{color:#0f172a}li{margin:4px 0}code{background:#eef2f7;padding:2px 4px;border-radius:4px}</style>"
        "</head><body>"
        + "\n".join(body_lines)
        + "</body></html>"
    )

