GeoCert Paper

This repo contains the LaTeX source and the minimal figure assets to build the GeoCert paper PDF.

Build
- Windows (PowerShell): latexmk -pdf -halt-on-error -interaction=nonstopmode .\paper\geocert_paper.tex
- Or use pdflatex twice.

Reproduce figures
- The paper/figs directory contains pre-rendered images.

Repository hygiene
- Large outputs and caches are excluded by .gitignore. Only figures referenced by the paper are tracked.
