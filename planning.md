# Research Plan: Mapping an MLP in Terms of Tokens

## Motivation & Novelty Assessment

### Why This Research Matters
Transformer MLPs are the primary mechanism for storing and transforming knowledge, yet we lack a unified description of what they compute in terms of interpretable token-level operations. If MLP computation can be described as a simple program over token directions, this would make transformer internals far more interpretable and could enable targeted model editing, debugging, and verification.

### Gap in Existing Work
Prior work has shown that (1) MLP keys detect input patterns (Geva 2021), (2) MLP values promote token concepts (Geva 2022), (3) MLPs implement vector arithmetic for factual recall (Merullo 2023), and (4) representations are linear directions (Park 2024). However, no work has systematically tested whether the *full* MLP input→output mapping can be faithfully described as a **shallow program**: "check which token directions are present in input, then produce a weighted combination of output token directions."

### Our Novel Contribution
We directly test whether MLP layers can be approximated by a simple interpretable program:
1. Decompose MLP inputs and outputs into token-direction components
2. Measure how much of the MLP output variance is explained by a linear/shallow model operating on token-direction features of the input
3. Quantify this across layers, finding where the "simple program" approximation works and where it breaks down

### Experiment Justification
- **Experiment 1 (Token Direction Decomposition)**: Establish that MLP inputs/outputs can be meaningfully decomposed into token directions using the unembedding matrix. This is the foundational step.
- **Experiment 2 (Key Activation as Token Detection)**: Test whether MLP key activations (the intermediate hidden layer) can be predicted from which token directions are present in the input.
- **Experiment 3 (Shallow Program Approximation)**: Build a simple program (sparse linear model) that maps detected input token directions to output token directions, and measure how faithfully it approximates the actual MLP.
- **Experiment 4 (Layer-wise Analysis)**: Compare early vs. middle vs. late layers to see where the approximation is most/least faithful.

## Research Question
Can the input-to-output mapping of transformer MLP layers be described as a simple program that detects which token directions are present in the residual stream input and produces a shallow combination of output token directions?

## Hypothesis Decomposition
1. **H1**: MLP inputs at each token position have significant components along token embedding directions (i.e., the residual stream encodes "which tokens are relevant" as directions).
2. **H2**: MLP key activations (hidden layer) correlate with the presence of specific token directions in the input.
3. **H3**: A sparse linear model mapping top-k input token directions to output token directions can explain a significant fraction (>50%) of the MLP output variance.
4. **H4**: This approximation quality varies by layer, with middle/upper layers being better approximated (where factual/semantic processing dominates).

## Proposed Methodology

### Approach
Use GPT-2 Small (12 layers, d=768, d_m=3072) via TransformerLens. For a corpus of text (WikiText-103 validation), collect MLP inputs and outputs at each layer and token position. Decompose these into token-direction components using the unembedding matrix. Build shallow approximation models and measure faithfulness.

### Experimental Steps
1. Load GPT-2 Small, collect MLP inputs/outputs on ~1000 text sequences
2. Project MLP inputs and outputs to vocabulary space (via unembedding matrix W_U)
3. For each MLP layer, measure: (a) how much input/output variance is along token directions, (b) which tokens are most represented
4. Build a "simple program" approximation: take top-k input token logits → sparse linear map → predict top-k output token logits
5. Measure R² and cosine similarity between approximation and actual MLP output
6. Analyze by layer and by input type

### Baselines
- Random direction projections (control for projection artifacts)
- Full-rank linear map (upper bound on linear approximability)
- Identity baseline (MLP output ≈ MLP input, i.e., no transformation)

### Evaluation Metrics
- **R²**: Fraction of MLP output variance explained by approximation
- **Cosine similarity**: Between predicted and actual MLP output vectors
- **Top-k agreement**: Whether the top predicted tokens match between approximation and actual output
- **Faithfulness**: KL divergence between token distributions induced by approximation vs. actual MLP output

### Statistical Analysis Plan
- Bootstrap confidence intervals for all metrics (1000 resamples)
- Compare across layers using paired tests
- Report effect sizes

## Expected Outcomes
- H1 likely supported: residual stream is known to encode token information
- H2 partially supported: keys detect patterns but may not cleanly map to individual token directions
- H3: the key question - we expect >30% variance explained for upper layers, less for lower layers
- H4: middle layers (factual recall) should be best approximated; early layers (syntactic) may be harder

## Timeline and Milestones
1. Data collection & projection: 20 min
2. Token direction analysis: 20 min
3. Shallow program approximation: 30 min
4. Layer-wise analysis & visualization: 20 min
5. Documentation: 20 min

## Potential Challenges
- Memory: Storing all MLP activations across many sequences. Mitigation: batch processing.
- Superposition: Token directions may be superposed, making decomposition noisy. Mitigation: use top-k projections.
- Unembedding matrix may not perfectly capture all token directions. Mitigation: also try embedding matrix.

## Success Criteria
The research succeeds if we can quantify how much of MLP computation is describable as a shallow program over token directions, even if the answer is "not much." The key deliverable is concrete numbers (R², cosine sim, top-k agreement) across layers.
