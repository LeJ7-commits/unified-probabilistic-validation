"""
build_thesis.py
================
Assembles the thesis Word document from markdown chapter files.

Usage (from repo root):
    python build_thesis.py

Output:
    thesis_output/Unified_Probabilistic_Validation_Le_Askarova_2026.docx

Requirements:
    pip install python-docx
    pandoc must be installed (https://pandoc.org/installing.html)

Chapter order:
    papers/00_abstract.md
    papers/01_introduction.md
    papers/02_methodology.md
    papers/03_results.md
    papers/04_discussion.md
    papers/05_governance_implications.md
    papers/06_references_and_ai_statement.md
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent
PAPERS_DIR  = REPO_ROOT / "papers"
OUTPUT_DIR  = REPO_ROOT / "thesis_output"
OUTPUT_FILE = OUTPUT_DIR / "Unified_Probabilistic_Validation_Le_Askarova_2026.docx"
TEMP_FILE   = OUTPUT_DIR / "_thesis_combined.md"

TITLE = "Unified Probabilistic Validation for Energy Markets: A Production-Grade Architecture for Simulation and Forecasting Models"
AUTHORS = "Jia Yang Le & Komila Askarova"
SUPERVISOR = "Luca Margaritella"
INDUSTRY_SUPERVISOR = "Rikard Green, Energy Quant Solutions Sweden AB"
PROGRAMME = "Data Analytics and Business Economics"
INSTITUTION = "Lund University School of Economics and Management"
YEAR = "2026"

JEL_CODES = "C12, C15, C52, Q47, G32"
KEYWORDS = "probabilistic validation, energy markets, PIT diagnostics, conformal prediction, governance classification, Basel traffic-light, model risk"

# ── Chapter order ─────────────────────────────────────────────────────────────

CHAPTERS = [
    "01_introduction.md",
    "02_methodology.md",
    "03_results.md",
    "04_discussion.md",
    "05_governance_implications.md",
    "06_references_and_ai_statement.md",
]


# ── Title page block ──────────────────────────────────────────────────────────

TITLE_PAGE = f"""---
title: |
  {TITLE}
subtitle: "Master's Thesis"
author:
  - "Authors: {AUTHORS}"
  - "Study programme: {PROGRAMME}"
  - "Supervisor: {SUPERVISOR}"
  - "Industry Supervisor: {INDUSTRY_SUPERVISOR}"
  - "Year of defence: {YEAR}"
date: ""
mainfont: "Arial"
fontsize: 12pt
geometry: "a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm"
linestretch: 1.5
toc: true
toc-depth: 2
numbersections: true
header-includes: |
  \\usepackage{{setspace}}
  \\usepackage{{fancyhdr}}
  \\pagestyle{{fancy}}
  \\fancyhf{{}}
  \\fancyfoot[C]{{\\thepage}}
---

\\newpage

"""

ABSTRACT_HEADER = f"""# Abstract

"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_chapter(path: Path) -> str:
    """Read a markdown chapter file, return content as string."""
    if not path.exists():
        print(f"  [WARN] Chapter not found: {path.name} — skipping")
        return f"\n\n# {path.stem.replace('_', ' ').title()} (MISSING)\n\n*File not found: {path}*\n\n"
    content = path.read_text(encoding="utf-8")
    # Ensure chapter starts with a page break in the combined doc
    return f"\n\n\\newpage\n\n{content}\n\n"


def fix_heading_levels(content: str, chapter_num: int) -> str:
    """
    Ensure chapter headings are H1 and sub-headings are H2/H3.
    Chapter files use ## for top-level sections — promote to # for the
    combined document so pandoc TOC works correctly.
    """
    # Already has a # title — don't change
    if re.match(r'^# ', content.lstrip()):
        return content

    # Promote ## → # (top-level sections become chapter headings)
    # Only if no # exists
    lines = content.split('\n')
    result = []
    for line in lines:
        if line.startswith('## '):
            result.append('# ' + line[3:])
        elif line.startswith('### '):
            result.append('## ' + line[4:])
        elif line.startswith('#### '):
            result.append('### ' + line[5:])
        else:
            result.append(line)
    return '\n'.join(result)


def check_pandoc() -> bool:
    """Check pandoc is available."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True, text=True
        )
        version = result.stdout.split('\n')[0]
        print(f"  pandoc: {version}")
        return True
    except FileNotFoundError:
        print("  [ERROR] pandoc not found.")
        print("  Install from: https://pandoc.org/installing.html")
        print("  Windows: winget install JohnMacFarlane.Pandoc")
        return False


def build_combined_markdown() -> Path:
    """Combine all chapters into a single markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = [TITLE_PAGE]

    # Abstract — special handling (no chapter number)
    abstract_path = PAPERS_DIR / "00_abstract.md"
    if abstract_path.exists():
        abstract_content = abstract_path.read_text(encoding="utf-8")
        # Remove leading # Abstract if present (we add it manually)
        abstract_content = re.sub(r'^#+\s*Abstract\s*\n', '', abstract_content, flags=re.IGNORECASE)
        parts.append(f"\\newpage\n\n# Abstract\n\n{abstract_content}\n\n")
        parts.append(f"**JEL Classification:** {JEL_CODES}\n\n")
        parts.append(f"**Keywords:** {KEYWORDS}\n\n")
    else:
        print("  [WARN] 00_abstract.md not found — insert abstract manually")
        parts.append("\\newpage\n\n# Abstract\n\n*Insert abstract here.*\n\n")
        parts.append(f"**JEL Classification:** {JEL_CODES}\n\n")
        parts.append(f"**Keywords:** {KEYWORDS}\n\n")

    # Acknowledgements placeholder
    parts.append("\\newpage\n\n# Acknowledgements\n\n*Insert acknowledgements here.*\n\n")

    # Chapters 1–5 + references
    chapter_labels = [
        "Introduction",
        "Methodology",
        "Results",
        "Discussion",
        "Governance Implications",
        "References and AI Statement",
    ]

    for i, (filename, label) in enumerate(zip(CHAPTERS, chapter_labels), start=1):
        path = PAPERS_DIR / filename
        content = read_chapter(path)

        # For chapters 1–5, add chapter heading if not present
        if i <= 5:
            # Check if content already starts with a proper chapter heading
            stripped = content.lstrip()
            if not stripped.startswith('# '):
                # Add chapter heading
                content = f"\\newpage\n\n# {label}\n\n{content}"
            else:
                content = f"\\newpage\n\n{content}"

        parts.append(content)

    combined = ''.join(parts)

    TEMP_FILE.write_text(combined, encoding="utf-8")
    print(f"  Combined markdown: {TEMP_FILE}")
    return TEMP_FILE


def convert_to_docx(md_path: Path, output_path: Path) -> bool:
    """Convert combined markdown to Word document using pandoc."""

    cmd = [
        "pandoc",
        str(md_path),
        "--from", "markdown+raw_tex+yaml_metadata_block",
        "--to", "docx",
        "--output", str(output_path),
        "--toc",
        "--toc-depth=2",
        "--standalone",
        "--highlight-style=tango",
        # Reference doc for consistent styling (generate one if needed)
    ]

    print(f"  Running: {' '.join(cmd[:6])} ...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] pandoc failed:")
        print(result.stderr)
        return False

    if result.stderr:
        # Warnings only
        for line in result.stderr.split('\n'):
            if line.strip():
                print(f"  [WARN] {line}")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Thesis Word Document Builder")
    print(f"  {TITLE[:55]}...")
    print("=" * 60)

    print("\n[1/3] Checking dependencies...")
    if not check_pandoc():
        sys.exit(1)

    print("\n[2/3] Combining chapters...")
    md_path = build_combined_markdown()

    # Report what was found
    for filename in CHAPTERS:
        path = PAPERS_DIR / filename
        status = "✓" if path.exists() else "✗ MISSING"
        size = f"({path.stat().st_size:,} bytes)" if path.exists() else ""
        print(f"  {status} {filename} {size}")

    print("\n[3/3] Converting to Word...")
    success = convert_to_docx(md_path, OUTPUT_FILE)

    if success:
        size_kb = OUTPUT_FILE.stat().st_size // 1024
        print(f"\n  ✓ Output: {OUTPUT_FILE}")
        print(f"  ✓ Size:   {size_kb} KB")
        print("\n  Post-processing steps:")
        print("  1. Open the .docx and accept the TOC update prompt")
        print("  2. Add acknowledgements text (marked as placeholder)")
        print("  3. Review table formatting — complex tables may need manual adjustment")
        print("  4. Set font to Times New Roman 12pt if LUSEM requires it")
        print("  5. Add page numbers in header/footer if not present")
        print("  6. Add title page logo if required by LUSEM template")
    else:
        print("\n  [FAILED] Check errors above.")
        sys.exit(1)

    # Clean up temp file
    TEMP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
