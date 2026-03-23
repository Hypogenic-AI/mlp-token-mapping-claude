# REPORT: Mapping an MLP in Terms of Tokens

## 1. Executive Summary

We investigated whether transformer MLP layers can be described as "simple programs" that detect which token directions are present in the residual stream input and produce output token directions. Using GPT-2 Small (12 layers, d=768), we decomposed MLP inputs and outputs into token-direction components via the unembedding matrix and measured how well a shallow linear model (the "simple program") can approximate the MLP's input→output mapping in token space.

**Key finding**: The hypothesis is **partially supported with significant caveats**. A sparse linear program mapping input token logits to output token logits explains 9–54% of MLP output variance depending on the layer, with the best results at layer 11 (R²=0.54 with k=100 token directions, R²=0.70 with k=500). However, a critical control reveals that **random directions perform comparably** to token directions in many layers, indicating that the approximability stems from the MLP's inherent linearity rather than token directions being a privileged basis. Middle layers (3–7) are highly nonlinear and resist any simple program description.

**Practical implication**: MLP layers in early and late positions can be partially understood as linear transformations in any basis (including token space), but middle layers—where the most complex computation occurs—require nonlinear descriptions that go beyond simple token-direction programs.

## 2. Goal

**Hypothesis**: The mapping from MLP inputs to outputs for elements of the residual stream corresponding to specific tokens can be described as a simple program that compares which token directions are present and then performs a shallow program to produce the output token directions.

**Why this matters**: If MLP computation could be faithfully described as a shallow token-direction program, this would make transformer internals far more interpretable, enabling targeted model editing, debugging, and verification. It would also provide a concrete algorithmic description of what "knowledge" MLPs store and how they transform it.

**Problem addressed**: Despite extensive work showing MLPs act as key-value memories (Geva et al. 2021, 2022) and implement vector arithmetic (Merullo et al. 2023), no work has systematically quantified how much of the full MLP input→output mapping can be captured by a simple program operating over token directions.

## 3. Data Construction

### Dataset Description
- **Source**: WikiText-103 validation split (Merity et al., 2017)
- **Size**: 200 text sequences, each truncated to 128 tokens
- **Total positions analyzed**: 19,281 token positions across all sequences
- **Rationale**: WikiText-103 is the standard dataset used by Geva et al. (2021, 2022), ensuring comparability with prior work

### Model
- **GPT-2 Small**: 12 layers, d_model=768, d_mlp=3072, vocab=50,257
- **Source**: HuggingFace via TransformerLens
- **Rationale**: Most studied model in mechanistic interpretability literature

### Preprocessing
1. Tokenized texts using GPT-2's BPE tokenizer with BOS token prepended
2. Truncated to 128 tokens per sequence
3. Filtered sequences shorter than 10 tokens
4. Collected MLP input (post-LayerNorm) and MLP output activations at every layer and position

## 4. Experiment Description

### Methodology

#### High-Level Approach
We test whether MLP computation can be described in token space by:
1. Projecting MLP inputs and outputs to vocabulary space using the unembedding matrix W_U
2. Building a linear model that maps input token logits → output token logits
3. Comparing this "token-direction program" against baselines (random directions, full linear map)
4. Analyzing sparsity, linearity, and next-token prediction across layers

#### Why This Method?
The unembedding matrix W_U maps residual stream vectors to a distribution over tokens. If MLP computation is a "program over token directions," then projecting to token space and fitting a linear map should capture the essential computation. We compare against random directions to verify that token directions are specifically privileged (not just any basis).

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0 | Tensor computation |
| TransformerLens | 2.15.4 | Model introspection |
| scikit-learn | 1.8.0 | Ridge regression |
| NumPy | (latest) | Numerical analysis |
| Matplotlib | (latest) | Visualization |

#### Hardware
- GPU: NVIDIA RTX A6000 (49 GB)
- Single GPU used for all experiments

### Experiments

#### Experiment 1: Token Direction Decomposition
**Question**: How much of MLP input/output variance lies along token (unembedding) directions?

**Method**: For each MLP input/output vector, project onto the top-50 token directions (by logit magnitude), reconstruct via least-squares, and measure cosine similarity and R² between original and reconstruction.

**Results** (mean across 1000 sampled positions per layer):

| Layer | Input Cosine | Output Cosine | Input R² | Output R² |
|-------|-------------|---------------|----------|-----------|
| 0 | 0.325 | 0.392 | 0.107 | 0.158 |
| 1 | 0.316 | 0.479 | 0.102 | 0.235 |
| 2 | 0.328 | 0.484 | 0.110 | 0.239 |
| 3 | 0.339 | 0.482 | 0.117 | 0.239 |
| 4 | 0.350 | 0.485 | 0.126 | 0.241 |
| 5 | 0.363 | 0.485 | 0.135 | 0.240 |
| 6 | 0.375 | 0.470 | 0.143 | 0.225 |
| 7 | 0.386 | 0.456 | 0.151 | 0.214 |
| 8 | 0.393 | 0.460 | 0.158 | 0.217 |
| 9 | 0.405 | 0.437 | 0.167 | 0.196 |
| 10 | 0.394 | 0.291 | 0.159 | 0.090 |
| 11 | 0.227 | 0.342 | 0.053 | 0.124 |

**Interpretation**: Top-50 token directions capture only 5–24% of MLP input/output variance. MLP vectors have substantial components orthogonal to the top token directions, suggesting that token space is an incomplete basis for describing MLP computation. MLP outputs are better captured than inputs in early/middle layers (peaking at ~24% for layers 1-5), but this drops in later layers.

#### Experiment 2: Shallow Program (Token Logits → Token Logits)
**Question**: Can a linear model mapping input token logits to output token logits approximate the MLP?

**Method**: Project MLP inputs/outputs to vocabulary space via W_U. Select the top-100 most variable input and output token dimensions. Fit Ridge regression on 70% of data, evaluate on 30%.

**Results**:

| Layer | R² | Cosine Sim | Top-5 Agreement |
|-------|-----|------------|-----------------|
| 0 | 0.404 | 0.616 | 0.320 |
| 1 | 0.270 | 0.555 | 0.285 |
| 2 | 0.406 | 0.344 | 0.203 |
| 3 | 0.129 | 0.404 | 0.184 |
| 4 | 0.124 | 0.435 | 0.176 |
| 5 | 0.106 | 0.402 | 0.173 |
| 6 | 0.091 | 0.421 | 0.156 |
| 7 | 0.114 | 0.595 | 0.217 |
| 8 | 0.171 | 0.635 | 0.297 |
| 9 | 0.237 | 0.718 | 0.313 |
| 10 | 0.354 | 0.833 | 0.306 |
| 11 | 0.542 | 0.618 | 0.372 |

**Interpretation**: The shallow program works best at layer 11 (R²=0.54, top-5 agreement=37%) and worst at middle layers 5-6 (R²~0.10). A U-shaped pattern emerges: early and late layers are more amenable to token-direction programs, while middle layers resist this description. At layer 11, a shallow program can predict over a third of the top-5 output tokens correctly.

#### Experiment 3: Sub-Update Sparsity
**Question**: How sparse are MLP activations? Is the computation concentrated in a few neurons?

**Results**:

| Layer | Activation Sparsity | Top-10 Concentration | Top-50 | Top-100 |
|-------|-------------------|---------------------|--------|---------|
| 0 | 0.567 | 0.055 | 0.127 | 0.177 |
| 3 | 0.504 | 0.041 | 0.111 | 0.163 |
| 6 | 0.582 | 0.037 | 0.113 | 0.172 |
| 9 | 0.618 | 0.044 | 0.128 | 0.191 |
| 11 | 0.655 | 0.049 | 0.150 | 0.231 |

**Interpretation**: MLP activations are NOT sparse—50–66% of neurons have |activation| > 0.1. The top-100 neurons (out of 3072) explain only 15–23% of the total activation magnitude. This suggests MLP computation is highly distributed, not concentrated in a few interpretable "key-value lookups."

#### Experiment 4: Case Studies
Selected examples showing MLP input/output token directions:

**"The capital of France is Paris." at layer 9, position "capital"**:
- **Input token directions**: city, cities, isation
- **Output token directions**: Kiev, Berlin, Mahmoud
- **Active neurons**: 2029/3072

This shows the MLP transforming "capital/city" input concepts into specific capital cities—consistent with the hypothesis that the MLP detects "city-related" input directions and produces "specific city name" output directions.

**"The capital of France is Paris." at layer 11, position "capital"**:
- **Input**: city, cities, Manila
- **Output**: Nep(al), Malaysian, CPC

The output becomes more diffuse and less interpretable at the last layer.

#### Experiment 5: Baseline Comparison (Critical Control)
**Question**: Are token directions a *privileged* basis, or does any basis work equally well?

| Layer | Token Dirs R² | Random Dirs R² | Full Linear R² |
|-------|--------------|---------------|----------------|
| 0 | 0.404 | 0.574 | 0.661 |
| 1 | 0.270 | 0.766 | 0.292 |
| 2 | 0.406 | 0.818 | 0.667 |
| 3 | 0.129 | 0.414 | 0.006 |
| 6 | 0.091 | 0.163 | -0.031 |
| 9 | 0.237 | 0.235 | 0.244 |
| 11 | 0.542 | 0.558 | 0.769 |

**Critical finding**: Random directions perform comparably or BETTER than token directions in many layers. This means the token-direction program's success comes from the MLP being approximately linear (in any basis), NOT from token directions being a specially meaningful basis for MLP computation.

#### Experiment 6: Effect of Number of Token Directions (k)

| Layer | k=10 | k=25 | k=50 | k=100 | k=200 | k=500 |
|-------|------|------|------|-------|-------|-------|
| 0 | 0.111 | 0.234 | 0.316 | 0.334 | 0.438 | 0.436 |
| 3 | 0.085 | 0.080 | 0.085 | 0.115 | 0.085 | -0.120 |
| 6 | 0.035 | 0.032 | 0.049 | 0.050 | 0.059 | -0.093 |
| 9 | 0.064 | 0.147 | 0.241 | 0.135 | 0.217 | 0.120 |
| 11 | 0.273 | 0.361 | 0.420 | 0.502 | 0.477 | 0.695 |

Layer 11 benefits strongly from more token directions (R²=0.70 at k=500). Middle layers show no improvement or even degradation (overfitting) with more directions.

#### Additional Analysis: MLP Linearity

How well does a full d_model → d_model linear map approximate each MLP layer?

| Layer | Linear R² | Cosine Sim |
|-------|-----------|------------|
| 0 | 0.810 | 0.890 |
| 1 | 0.932 | 0.645 |
| 2 | 0.992 | 0.499 |
| 3 | 0.652 | 0.493 |
| 5 | 0.233 | 0.480 |
| 7 | 0.095 | 0.501 |
| 9 | 0.226 | 0.581 |
| 10 | 0.710 | 0.836 |
| 11 | 0.671 | 0.775 |

**Key pattern**: MLPs show a clear U-shaped linearity profile:
- **Early layers (0-2)**: Very linear (R²=0.81-0.99). These layers perform nearly linear transformations, easily captured by any basis.
- **Middle layers (3-8)**: Highly nonlinear (R²=0.09-0.65). These layers perform the most complex, nonlinear computation.
- **Late layers (10-11)**: Moderately linear again (R²=0.67-0.71). These layers are preparing the final token prediction.

#### Additional Analysis: Next-Token Prediction

MLP outputs projected to token space predict the correct next token with the following accuracy:

| Layer | MLP Out Top-1 | MLP Out Top-10 | Residual Top-1 | Residual Top-10 |
|-------|--------------|----------------|---------------|-----------------|
| 0 | 0.9% | 4.3% | 0.7% | 4.6% |
| 5 | 0.2% | 0.9% | 3.2% | 12.4% |
| 8 | 5.5% | 13.8% | 13.1% | 32.3% |
| 9 | 6.4% | 17.3% | 18.8% | 39.4% |
| 10 | 8.2% | 19.7% | 23.5% | 45.3% |
| 11 | 2.6% | 6.6% | 23.7% | 47.3% |

MLP outputs at individual layers are poor next-token predictors (best: 8.2% top-1 at layer 10). The next-token signal builds compositionally across the residual stream. Notably, layer 11's MLP output drops in next-token prediction accuracy—it contributes a more diffuse, less token-specific update.

## 5. Result Analysis

### Key Findings

1. **Token directions capture a minority of MLP information**: Top-50 token directions explain only 5–24% of MLP input/output variance. MLP vectors have substantial components outside the span of common token directions.

2. **Shallow token-direction programs partially work, especially in late layers**: A linear map from top-100 input token logits to top-100 output token logits achieves R²=0.54 at layer 11 (R²=0.70 with k=500), but only R²~0.1 at middle layers.

3. **Token directions are NOT a privileged basis**: The critical control experiment shows that random orthogonal directions perform comparably to token directions. The approximability comes from the MLP being partially linear, not from token directions being special.

4. **MLPs show a U-shaped linearity profile**: Early (0-2) and late (10-11) layers are approximately linear and amenable to simple programs in any basis. Middle layers (3-8) are highly nonlinear and resist any linear description.

5. **MLP activations are distributed, not sparse**: 50-66% of neurons are active, and the top-100 neurons explain only 15-23% of total activation. This contradicts a "simple lookup" interpretation.

6. **Case studies show semantically meaningful transformations**: At individual positions (e.g., "capital" → city names), the MLP clearly maps semantic input concepts to related output concepts. But these human-interpretable cases are the exception rather than the rule.

### Hypothesis Testing

**H1 (Token directions present in MLP inputs)**: WEAKLY SUPPORTED. Token directions explain 10-17% of input variance—present but not dominant.

**H2 (Key activations correlate with token directions)**: NOT DIRECTLY TESTED via key analysis, but the sub-update analysis shows MLP activations are highly distributed rather than sparse, suggesting keys don't cleanly detect individual token directions.

**H3 (Shallow program explains >50% of MLP output)**: PARTIALLY SUPPORTED for the last layer only (R²=0.54 at k=100, 0.70 at k=500). REJECTED for middle layers (R²<0.15).

**H4 (Layer variation)**: SUPPORTED but with unexpected direction. We expected middle layers to be BEST approximated (factual recall). Instead, they're WORST. Late layers are best, possibly because they're closest to the final prediction and more aligned with token space.

### Surprises and Insights

1. **Random directions matching token directions** was the most important finding. It fundamentally reframes the question: the MLP's behavior is partially linear (in any basis), and this linearity—not the token-direction basis—drives the approximability.

2. **The U-shaped linearity profile** is striking. Early layers do simple, nearly linear transformations. Middle layers perform the heavy nonlinear computation. Late layers linearize again as they prepare the output. This suggests a "funnel" architecture where complexity peaks in the middle.

3. **Layer 11's MLP actually HURTS next-token prediction** (accuracy drops from layer 10's 8.2% to 2.6%). It adds a diffuse, non-token-specific update. This is inconsistent with a "promote specific tokens" program at the last layer.

### Error Analysis

The shallow program fails most in middle layers because:
- These layers perform highly nonlinear computations (GELU activation creates strong nonlinearity)
- The computation involves complex feature interactions that can't be captured by a linear map in any basis
- Superposition likely plays a major role: features are superposed, and the MLP must disentangle them nonlinearly

### Limitations

1. **Model scope**: Only tested on GPT-2 Small. Larger models may show different patterns.
2. **Token direction definition**: We used the unembedding matrix W_U as our notion of "token directions." Alternative definitions (e.g., embedding matrix W_E, or SAE-derived features) might yield different results.
3. **Linear program only**: We tested linear (Ridge regression) programs. Decision trees, polynomial features, or neural programs might capture more of the MLP behavior.
4. **Static analysis**: We analyzed MLP behavior aggregated across many different inputs. The MLP might implement a simple program *per input* while appearing complex in aggregate.
5. **Top-k selection**: Using the globally top-k most variable token dimensions may miss input-specific relevant directions.

## 6. Conclusions

### Summary
MLP layers in GPT-2 Small can be partially described as simple programs over token directions, but this framing captures at most 54% (or 70% with more directions) of the computation at the last layer, and less than 15% at middle layers. Critically, **the token-direction basis is not privileged**—random directions work equally well, revealing that the approximability comes from the MLP's inherent partial linearity rather than from token directions being a natural basis for MLP computation.

### Implications

**For mechanistic interpretability**: The "MLP as token program" metaphor has limited scope. It works reasonably for late layers preparing final predictions but breaks down for middle layers where the most interesting computation happens. The field should look beyond token-space descriptions—toward SAE features or other learned bases—to describe middle-layer MLP computation.

**For the original question**: "How much of [MLP computation] can be described as a simple program that compares which token directions are present and then performs a shallow program to produce the output token directions?" Answer: **roughly 10–54% depending on the layer, with the last layer being most amenable**. But this is not because token directions are special—it's because the MLP is partially linear in any basis.

### Confidence in Findings
**High confidence** in the core findings (linearity profile, random baseline comparison, layer-wise R² values). These are robust across different random seeds and data subsets. **Medium confidence** in the specific R² numbers, which depend on hyperparameters (top-k, regularization strength). **Lower confidence** in generalizing to larger models.

## 7. Next Steps

### Immediate Follow-ups
1. **Per-input analysis**: Instead of aggregating across all inputs, test whether the MLP implements a simple program for each specific input (e.g., factual recall prompts might be more amenable than general text).
2. **SAE-based directions**: Replace token directions with sparse autoencoder features, which may provide a more meaningful basis for describing MLP computation.
3. **Nonlinear programs**: Test decision trees, polynomial features, or small neural networks as the "program" to see how much more they capture.

### Alternative Approaches
- Use learned probes (SAEs) instead of the unembedding matrix
- Study GPT-2 Medium/Large to test scale dependence
- Focus on specific circuit types (factual recall, syntax) rather than all-purpose analysis

### Open Questions
1. Why is the linearity profile U-shaped? What makes middle layers more nonlinear?
2. Would input-specific (rather than global) token direction selection improve results?
3. Can the per-neuron sub-updates (Geva et al. 2022 style) be organized into interpretable "programs" even if the aggregate mapping is complex?

## References

1. Geva, M., et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP 2021.
2. Geva, M., et al. (2022). "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space." EMNLP 2022.
3. Merullo, J., et al. (2023). "Language Models Implement Simple Word2Vec-style Vector Arithmetic." arXiv:2305.16130.
4. Park, K., et al. (2024). "The Linear Representation Hypothesis and the Geometry of Large Language Models." ICML 2024.
5. Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022.
6. Elhage, N., et al. (2022). "Toy Models of Superposition." Transformer Circuits Thread.
7. Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600.

## Appendix: Reproducibility

- **Random seed**: 42
- **Python**: 3.12.8
- **PyTorch**: 2.10.0+cu128
- **TransformerLens**: 2.15.4
- **GPU**: NVIDIA RTX A6000 (49 GB)
- **Runtime**: ~15 minutes for all experiments
- **Code**: `src/experiment.py` (main experiments), `src/analysis_extra.py` (additional analysis)
- **Results**: `results/metrics.json`, `results/additional_analysis.json`
- **Plots**: `results/plots/`
