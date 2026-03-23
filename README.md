# Mapping an MLP in Terms of Tokens

Research investigating whether transformer MLP layers can be described as simple programs that detect which token directions are present in the residual stream input and produce output token directions.

## Key Findings

- **Partial support for hypothesis**: A shallow linear program over token directions explains 10-54% of MLP output variance (layer-dependent), with the last layer being most amenable (R²=0.54, or 0.70 with 500 token directions)
- **Token directions are NOT privileged**: Random orthogonal directions perform comparably, indicating the approximability comes from MLPs being partially linear, not from token directions being special
- **U-shaped linearity**: Early layers (0-2) and late layers (10-11) are approximately linear; middle layers (3-8) are highly nonlinear and resist any simple program description
- **MLP activations are distributed**: 50-66% of neurons are active; top-100 neurons explain only 15-23% of total activation
- **Layer 11 best supports the hypothesis** but the effect is not token-direction-specific

## Repository Structure

```
├── REPORT.md                    # Full research report with all results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── experiment.py            # Main experiments (6 experiments)
│   └── analysis_extra.py        # Additional analysis (linearity, next-token)
├── results/
│   ├── metrics.json             # All quantitative results
│   ├── additional_analysis.json # Linearity and next-token results
│   ├── case_studies.json        # Detailed token-level case studies
│   └── plots/                   # All visualizations
├── papers/                      # Downloaded research papers
├── code/                        # Cloned reference repositories
└── datasets/                    # Dataset documentation
```

## How to Reproduce

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv add torch numpy matplotlib scipy pandas tqdm transformer-lens einops jaxtyping scikit-learn

# Run experiments
CUDA_VISIBLE_DEVICES=0 python src/experiment.py
CUDA_VISIBLE_DEVICES=0 python src/analysis_extra.py
```

Requires a GPU with ~8+ GB memory. Runtime: ~15 minutes on RTX A6000.

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, analysis, and discussion.
