from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    source = Path("POLYMIND_V2_COLAB_GUIDE.md")
    target = Path("POLYMIND_V2_COLAB_GUIDE.pdf")
    header = Path("scripts/pandoc_header.tex")
    template = Path("scripts/pandoc_template.tex")
    variables = Path("scripts/pandoc_variables.yaml")
    if not source.exists():
        raise FileNotFoundError(f"Guide source not found: {source}")
    if not header.exists():
        raise FileNotFoundError(f"Pandoc header not found: {header}")
    if not template.exists():
        raise FileNotFoundError(f"Pandoc template not found: {template}")
    if not variables.exists():
        raise FileNotFoundError(f"Pandoc variables not found: {variables}")

    pandoc = shutil.which("pandoc")
    engine = shutil.which("tectonic")
    if pandoc is not None and engine is not None:
        cmd = [
            pandoc,
            str(source),
            "--from",
            "gfm",
            "--toc",
            "--top-level-division=chapter",
            "--pdf-engine",
            engine,
            "--template",
            str(template),
            "--metadata-file",
            str(variables),
            "--include-in-header",
            str(header),
            "--syntax-highlighting",
            "tango",
            "--output",
            str(target),
        ]
        subprocess.run(cmd, check=True)
        print(f"Rendered {target} with pandoc + tectonic")
        return

    from markdown import markdown
    from weasyprint import HTML

    text = source.read_text(encoding="utf-8")
    html_body = markdown(
        text,
        extensions=["fenced_code", "tables", "toc"],
    )
    css = """
    @page { size: A4; margin: 24mm 22mm 24mm 24mm; }
    body {
      font-family: Georgia, serif;
      line-height: 1.6;
      color: #1f2c36;
    }
    h1, h2, h3 {
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #11263c;
      page-break-after: avoid;
    }
    h1 {
      margin-top: 2.2em;
      padding-top: 0.4em;
      border-top: 2px solid #a67c52;
    }
    code, pre {
      font-family: Menlo, monospace;
    }
    code {
      background: #f7f1e7;
      padding: 0.1em 0.25em;
    }
    pre {
      padding: 12px 14px;
      border: 1px solid #d7c9b3;
      border-left: 4px solid #a67c52;
      background: #fbf8f2;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0;
    }
    th, td {
      border-bottom: 1px solid #d9d3c7;
      padding: 7px 9px;
      vertical-align: top;
    }
    th {
      color: #11263c;
      background: #f6f1e8;
    }
    blockquote {
      border-left: 4px solid #a67c52;
      padding-left: 14px;
      color: #425464;
      margin-left: 0;
    }
    """
    HTML(
        string=f"<html><head><style>{css}</style></head><body>{html_body}</body></html>"
    ).write_pdf(str(target))
    print(f"Rendered {target} with markdown + weasyprint fallback")


if __name__ == "__main__":
    main()
