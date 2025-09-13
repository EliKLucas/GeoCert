# GeoCert: Geometric Consistency Tests for Neural Representations

This repository contains the LaTeX source, figures, data, and code to reproduce the GeoCert paper.

## Reproducibility Statement

All artifacts (CSV/PNGs) are produced by a single command:
```bash
python scripts/bundle_camera_ready.py
```

We release the code and the prompt set, with seeds {0,1,2}. Environment: Python 3.11, PyTorch 2.1, transformers 4.35.

We record the Git commit hash, HF model revisions, and SHA256 of prompt files in the bundle metadata.

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Build Paper PDF
```bash
# Windows (PowerShell)
latexmk -pdf -halt-on-error -interaction=nonstopmode .\paper\geocert_paper.tex

# Linux/macOS
latexmk -pdf -halt-on-error -interaction=nonstopmode paper/geocert_paper.tex
```

### 3. Reproduce All Artifacts
```bash
python scripts/bundle_camera_ready.py
```

## Repository Structure

```
paper/
├── geocert_paper.tex          # Main LaTeX document
├── figs/                      # All figures referenced in paper
├── prompts/
│   └── balanced256_v1.jsonl   # Prompt set (SHA256: 229A54D2...)
├── artifacts.json             # Reproducibility metadata
├── paper_aggregate.csv        # Main results CSV
├── T1_main_metrics.csv        # Table 1 data
├── T2_edge_attribution.csv    # Table 2 data
├── T3_triangle_isometry.csv   # Table 3 data
└── T7_prompt_robustness.csv   # Table 7 data

scripts/
├── bundle_camera_ready.py     # Main reproducibility script
└── ...                        # Other analysis scripts

requirements.txt               # Python dependencies
```

## Experimental Details

- **Seeds**: 0, 1, 2
- **Prompts**: 256 prompts per experiment
- **ρ-steps**: 21 mixing fractions
- **Bootstrap CIs**: Computed across seeds
- **Models**: OPT-125M, GPT-Neo-125M, DistilGPT2, OPT-350M, Pythia-1B

## Repository Hygiene

Large outputs and caches are excluded by .gitignore. Only figures and data referenced by the paper are tracked.
