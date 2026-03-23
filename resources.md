# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Mapping an MLP in Terms of Tokens". The hypothesis is that MLP input-to-output mappings for specific tokens can be described as a simple program that compares which token directions are present and then performs a shallow program to produce output token directions.

## Papers
Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| FFN as Key-Value Memories | Geva et al. | 2021 | papers/2012.14913_ffn_key_value_memories.pdf | Keys=patterns, Values=vocab distributions |
| FFN Builds Predictions via Vocab Space | Geva et al. | 2022 | papers/2203.14680_ffn_vocab_space.pdf | Sub-updates promote interpretable concepts |
| Word2Vec-style Vector Arithmetic | Merullo et al. | 2023 | papers/2305.16130_word2vec_arithmetic.pdf | FFNs implement simple additive updates |
| Toy Models of Superposition | Elhage et al. | 2022 | papers/2209.10652_toy_models_superposition.pdf | Features as directions in superposition |
| Linear Representation Hypothesis | Park et al. | 2024 | papers/2311.03658_linear_representations_hypothesis.pdf | Formalizes concept directions |
| ROME | Meng et al. | 2022 | papers/2202.05262_rome.pdf | Factual associations in MLP layers |
| Knowledge Neurons | Dai et al. | 2022 | papers/2104.08696_knowledge_neurons.pdf | Individual neurons for facts |
| Geometry of Truth | Marks & Tegmark | 2023 | papers/2310.06824_geometry_of_truth.pdf | Linear truth representations |
| Tuned Lens | Belrose et al. | 2023 | papers/2303.08112_tuned_lens.pdf | Improved intermediate decoding |
| SAEs for Interpretable Features | Cunningham et al. | 2023 | papers/2309.08600_sparse_autoencoders_features.pdf | SAE feature decomposition |
| Factual Recall Mechanisms | (2024) | 2024 | papers/2403.14537_factual_recall_mechanisms.pdf | Mechanism of factual recall |
| Neuron2Graph | (2023) | 2023 | papers/2305.19911_neuron2graph.pdf | Network analysis of transformers |
| Computation in Superposition | (2023) | 2023 | papers/2310.10248_computation_in_superposition.pdf | Computing in superposition |
| Dictionary Learning (Anthropic) | Anthropic | 2023 | papers/2310.17230_dictionary_learning_anthropic.pdf | SAE for MLP features |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets documented: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GPT-2 Models | HuggingFace | 124M-774M params | MLP weight analysis | HuggingFace Hub | Primary data source |
| WikiText-103 | HuggingFace | 103M tokens | Probing text | HuggingFace Datasets | Used by Geva et al. |
| CounterFact | ROME repo | 21K facts | Factual recall | code/rome/data/ | Already available locally |
| BIG-Bench tasks | HuggingFace | Various | Relational tasks | HuggingFace Datasets | Past tense, capitals, etc. |

See datasets/README.md for download instructions.

## Code Repositories
Total repositories cloned: 7

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ff-layers | github.com/mega002/ff-layers | FFN key-value memory analysis | code/ff-layers/ | Geva et al. 2021 |
| ffn-values | github.com/aviclu/ffn-values | FFN sub-update concept analysis | code/ffn-values/ | Geva et al. 2022 |
| lm_vector_arithmetic | github.com/jmerullo/lm_vector_arithmetic | Vector arithmetic in LMs | code/lm_vector_arithmetic/ | Merullo et al. 2023 |
| linear_rep_geometry | github.com/KihoPark/linear_rep_geometry | Linear representation geometry | code/linear_rep_geometry/ | Park et al. 2024 |
| rome | github.com/kmeng01/rome | Factual editing via MLPs | code/rome/ | Meng et al. 2022 |
| sparse_autoencoder | github.com/openai/sparse_autoencoder | SAE for feature decomposition | code/sparse_autoencoder/ | OpenAI |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability library | code/TransformerLens/ | Essential tool |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Started with paper-finder service (diligent mode) for "MLP mechanistic interpretability transformer token directions"
2. Used domain knowledge to identify seminal papers (Geva et al., Elhage et al., Meng et al.)
3. Searched for additional papers on sparse autoencoders, linear representations, and computation in superposition
4. Identified code repositories from paper links and known mechanistic interpretability tools

### Selection Criteria
- Papers directly analyzing FFN/MLP layer computation (highest priority)
- Papers formalizing "token directions" or "linear representations" (high priority)
- Papers providing tools for analyzing model internals (medium priority)
- Focus on recent work (2021-2024) with established citations

### Challenges Encountered
- Some Anthropic papers (Mathematical Framework, Scaling Monosemanticity, SoLU) are published on transformer-circuits.pub rather than arXiv, so PDFs couldn't be downloaded via standard arXiv links
- Semantic Scholar API rate limits prevented automated arXiv ID lookups for some papers

### Gaps and Workarounds
- Anthropic Transformer Circuits Thread papers are not available as standard PDFs; referenced in literature review via URLs
- No single paper has attempted the full "MLP as program over token directions" framing - this is the research gap

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary model**: GPT-2 Small (12 layers, d=768, d_m=3072, vocab=50257)
   - Most studied model in the literature
   - Small enough for detailed analysis
   - TransformerLens provides easy access to all internals
   - Upgrade to GPT-2 Medium for validation

2. **Primary tool**: TransformerLens
   - Provides `HookedTransformer` with access to all activations
   - Built-in logit lens functionality
   - Easy activation patching and ablation
   - Supports all GPT-2 variants

3. **Experimental approach**:
   - **Phase 1**: For each MLP layer, project all value vectors to vocabulary space (Geva et al. 2022 method). Characterize what "concepts" (token groups) each value vector promotes.
   - **Phase 2**: For specific inputs, analyze which keys activate and what the resulting sub-update composition looks like in token space. Can this be described as a simple program?
   - **Phase 3**: Test whether the MLP computation can be faithfully approximated by a simple rule-based program that checks token direction presence and outputs token direction updates.
   - **Phase 4**: Validate across different types of inputs (factual recall, syntactic, semantic).

4. **Evaluation metrics**:
   - Faithfulness: How well does the "program" approximation match the actual MLP output?
   - Interpretability: Can the extracted programs be understood by humans?
   - Generalization: Do the programs transfer across similar inputs?
