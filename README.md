# GeoCert: Geometric Consistency Tests for Neural Representations

This repository contains the LaTeX source, figures, data, and code to reproduce the GeoCert paper.

## Reproducibility Statement

This repository contains the complete paper source and all data needed to reproduce the GeoCert paper results.

**Core Method**: GeoCert tests geometric consistency of LLM representations using EDM (Euclidean Distance Matrix) theory, distinguishing convex blends vs edge-mixes, and explaining failures via K=2 disentangling.

**Data**: Seeds {0,1,2}, 256 prompts, 21 ρ-steps, bootstrap CIs. Environment: Python 3.11, PyTorch 2.1, transformers 4.35.

**Traceability**: Git commit hash, HF model revisions, and SHA256 of prompt files recorded in artifacts.json.

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

### 3. Data and Results
All figures and tables are pre-computed and included in the `paper/` directory. The paper presents results from the GeoCert method applied to OPT-125M, GPT-Neo-125M, and DistilGPT2 models.

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

geocert/                       # Core GeoCert implementation
scripts/                       # Analysis scripts (for reference)
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
