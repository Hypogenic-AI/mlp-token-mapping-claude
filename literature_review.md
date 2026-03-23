# Literature Review: Mapping an MLP in Terms of Tokens

## Research Area Overview

This research investigates whether the input-to-output mapping of transformer MLP (feed-forward) layers can be described as a simple program operating over token directions in the residual stream. The core hypothesis is that MLP computations can be decomposed into: (1) detecting which token directions are present in the input, and (2) executing a shallow program to produce output token directions.

This sits at the intersection of **mechanistic interpretability** (understanding transformer internals), **linear representation theory** (how concepts are encoded as directions), and **MLP/FFN function analysis** (what feed-forward layers compute).

## Key Papers

### Paper 1: Transformer Feed-Forward Layers Are Key-Value Memories
- **Authors**: Geva, Schuster, Berant, Levy
- **Year**: 2021
- **Source**: arXiv:2012.14913, EMNLP 2021
- **Key Contribution**: Establishes that FFN layers operate as key-value memories where FF(x) = f(x·K^T)·V
- **Methodology**: Analyze a 16-layer transformer (WikiLM, d=1024, d_m=4096) trained on WikiText-103. For each key vector k_i, retrieve top-t training prefixes with highest memory coefficient ReLU(x·k_i), then have human experts annotate patterns.
- **Key Findings**:
  - Each key k_i correlates with specific human-interpretable input patterns (n-grams or semantic topics)
  - Lower layers (1-9) capture shallow patterns (e.g., specific last words); upper layers (10-16) capture semantic patterns (e.g., "TV shows", "time ranges")
  - Each value v_i induces a distribution over the vocabulary via softmax(v_i · E); in upper layers, this distribution predicts the next token following the key's pattern
  - Layer outputs are compositional: ~68% of the time, the layer's prediction differs from any individual memory's prediction
  - Residual connections act as a refinement mechanism across layers
- **Datasets Used**: WikiText-103
- **Code Available**: Yes, https://github.com/mega002/ff-layers/
- **Relevance**: **CRITICAL** - Directly supports viewing MLPs as pattern-matching over inputs (keys detecting token patterns) and producing token-level outputs (values as vocab distributions). This is the foundational paper for the research hypothesis.

### Paper 2: Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space
- **Authors**: Geva, Caciularu, Wang, Goldberg
- **Year**: 2022
- **Source**: arXiv:2203.14680, EMNLP 2022
- **Key Contribution**: Shows FFN sub-updates (individual value vectors weighted by coefficients) encode human-interpretable concepts and build predictions via token promotion
- **Methodology**: Decompose FFN output as sum of sub-updates m_i·v_i. Project each v_i to vocabulary space via E·v_i to get a ranking over tokens. Have experts annotate concepts in top-30 tokens per value vector. Analyze on WikiLM and GPT-2.
- **Key Findings**:
  - Sub-update projections to vocabulary are meaningful: 37-55% of top tokens associated with concepts vs 16-23% for random vectors
  - Value vectors across ALL layers encode small sets of well-defined concepts (e.g., "food and drinks", "WH-relativizers", "pronouns", "measurement")
  - FFN updates work via **promotion** mechanism: top candidates are those pushed strongly by dominant sub-updates, not those surviving elimination
  - Each sub-update m_i·v_i introduces a scaling factor exp(e_w · m_i·v_i) to token probability - positive dot product promotes, negative suppresses
  - Practical application: increasing weight of just 10 sub-updates in GPT-2 reduces toxicity by ~50%
- **Datasets Used**: WikiText-103, WebText (GPT-2)
- **Code Available**: Yes, https://github.com/aviclu/ffn-values
- **Relevance**: **CRITICAL** - Directly demonstrates that MLP computations can be interpreted as promoting/suppressing specific tokens (or token groups representing concepts) in the vocabulary space. This is exactly the "shallow program producing output token directions" part of the hypothesis.

### Paper 3: Toy Models of Superposition
- **Authors**: Elhage, Trenton, Henighan, et al. (Anthropic)
- **Year**: 2022
- **Source**: arXiv:2209.10652, Transformer Circuits Thread
- **Key Contribution**: Demonstrates that neural networks represent more features than they have dimensions by using superposition - features are encoded as nearly-orthogonal directions
- **Methodology**: Study toy models (1-2 layer ReLU networks) with varying sparsity and importance of features. Analyze when and how features are represented in superposition.
- **Key Findings**:
  - Networks can store more features than dimensions when features are sparse
  - Features are represented as directions (not individual neurons)
  - Superposition creates a tradeoff: more features stored but with interference
  - Phase transitions exist between dedicated neurons, superposition, and no representation
  - The geometry of superposition follows specific patterns (antipodal, polytope structures)
- **Relevance**: **HIGH** - Explains why token directions in the residual stream may be superimposed, and why MLPs might need to "compare which token directions are present" - they must disentangle superposed representations. The key-value memory framework of MLPs is how models read from superposed representations.

### Paper 4: Language Models Implement Simple Word2Vec-style Vector Arithmetic
- **Authors**: Merullo, Eickhoff, Pavlick
- **Year**: 2023
- **Source**: arXiv:2305.16130
- **Key Contribution**: Shows LMs use simple additive vector arithmetic (similar to word2vec) implemented primarily by FFN layers for relational tasks
- **Methodology**: Early decoding (logit lens) across layers to track prediction evolution. FFN output patching experiments on GPT-2 Medium across world capitals, past-tense, uppercasing tasks. Models from 124M to 176B parameters.
- **Key Findings**:
  - Distinct processing stages visible via early decoding: (A) argument surfacing (e.g., "Poland" appears), (B) function application (FFN transforms Poland→Warsaw), (C) saturation (answer stabilizes)
  - The FFN outputs a **content-independent update** that can be patched between examples (e.g., the Poland→Warsaw update works to produce Beijing from China)
  - This mechanism is specific to **factual recall from pretraining memory**, not retrieval from local context
  - FFNs and attention specialize: FFNs for factual recall, attention for copying from context
  - Mechanism only works for one-to-one relations (not many-to-many)
- **Code Available**: Yes, https://github.com/jmerullo/lm_vector_arithmetic
- **Relevance**: **HIGH** - Directly demonstrates that MLP/FFN layers implement simple programs (vector addition) that map input token directions to output token directions. The "content-independent update" is exactly the kind of shallow program the hypothesis describes.

### Paper 5: The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors**: Park, Choe, Veitch
- **Year**: 2024
- **Source**: arXiv:2311.03658, ICML 2024
- **Key Contribution**: Formalizes the linear representation hypothesis using counterfactuals; identifies a "causal inner product" that unifies embedding and unembedding representations
- **Methodology**: Formal framework using causal/counterfactual reasoning. Three formalizations of linear representation: subspace, measurement (probing), intervention (steering). Experiments on LLaMA-2.
- **Key Findings**:
  - Concepts have two linear representations: embedding (input/context) space and unembedding (output/word) space
  - Unembedding representations connect to measurement (linear probing); embedding representations connect to intervention (steering)
  - A "causal inner product" exists where causally separable concepts are orthogonal
  - This inner product unifies the two representation spaces
- **Code Available**: Yes, https://github.com/KihoPark/linear_rep_geometry
- **Relevance**: **HIGH** - Provides the theoretical foundation for what "token directions" means in the hypothesis. The formal framework for linear representations justifies analyzing MLP operations in terms of directions in the residual stream.

### Paper 6: Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Meng, Bau, Andonian, Belinkov
- **Year**: 2022
- **Source**: arXiv:2202.05262, NeurIPS 2022
- **Key Contribution**: Localizes factual associations to specific MLP layers and develops ROME method for editing facts by modifying MLP weights
- **Methodology**: Causal tracing to identify which layers store factual associations. Rank-one model editing (ROME) modifies MLP value matrices to change stored facts.
- **Key Findings**:
  - Factual associations are concentrated in middle-layer MLPs (layers 15-25 in GPT-2 XL)
  - The "subject's last token" position is critical for factual recall
  - MLP layers act as key-value stores where keys select subjects and values store associated attributes
  - Facts can be edited by rank-one updates to MLP value matrices: v_new = v_old + Δ
- **Code Available**: Yes, https://github.com/kmeng01/rome
- **Relevance**: **HIGH** - Demonstrates that MLP layers store and retrieve factual associations in a way consistent with key-value memory interpretation. The ability to edit facts by modifying value vectors supports the view that MLPs map token inputs to token outputs via simple programs.

### Paper 7: Knowledge Neurons in Pretrained Transformers
- **Authors**: Dai, Dong, Hao, Sui, Chang, Wei
- **Year**: 2022
- **Source**: arXiv:2104.08696, ACL 2022
- **Key Contribution**: Identifies specific FFN neurons that express factual knowledge and shows they can be manipulated
- **Methodology**: Attribute factual knowledge expression to individual FFN neurons via integrated gradients. Validate by suppressing/amplifying identified neurons.
- **Key Findings**:
  - Specific neurons ("knowledge neurons") in FFN layers activate for particular facts
  - These neurons are concentrated in upper layers
  - Suppressing knowledge neurons erases facts; amplifying them strengthens recall
  - Knowledge neurons are shared across semantically related prompts
- **Relevance**: **MEDIUM** - Shows individual MLP neurons correspond to factual associations, supporting the view that MLP computation is decomposable into interpretable units operating on token-level information.

### Paper 8: The Geometry of Truth
- **Authors**: Marks, Tegmark
- **Year**: 2023
- **Source**: arXiv:2310.06824
- **Key Contribution**: Shows that truth values of factual statements are linearly represented in LLM activations
- **Relevance**: **MEDIUM** - Demonstrates that semantic properties (truth/falsehood) have linear representations, supporting the broader framework of linear concept directions in residual streams.

### Paper 9: Sparse Autoencoders Find Highly Interpretable Features in Language Models
- **Authors**: Cunningham, Ewart, Riggs, Huben, Sharkey
- **Year**: 2023
- **Source**: arXiv:2309.08600
- **Key Contribution**: Uses sparse autoencoders (SAEs) to decompose MLP activations into interpretable features
- **Methodology**: Train sparse autoencoders on MLP hidden activations to find a sparse overcomplete basis of interpretable features.
- **Key Findings**:
  - SAE features are more interpretable than individual neurons
  - Features often correspond to specific token patterns or semantic concepts
  - Provides a method to disentangle superposed features in MLP layers
- **Relevance**: **HIGH** - SAEs provide a practical tool for identifying the "token directions" that MLPs operate on. If MLP computation can be described as a program over token directions, SAEs help identify those directions.

### Paper 10: Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models
- **Authors**: (2024)
- **Source**: arXiv:2403.14537
- **Key Contribution**: Detailed analysis of the mechanisms by which transformers recall facts, focusing on the interplay between attention and MLP layers
- **Relevance**: **MEDIUM** - Provides additional evidence for how MLP layers participate in factual recall circuits, complementing the ROME and knowledge neurons findings.

### Paper 11: The Tuned Lens
- **Authors**: Belrose et al.
- **Year**: 2023
- **Source**: arXiv:2303.08112
- **Key Contribution**: Improves on the logit lens by learning affine transformations per layer to better decode intermediate representations
- **Relevance**: **MEDIUM** - Provides better tools for reading intermediate residual stream states as vocabulary distributions, useful for analyzing MLP input/output in token space.

### Paper 12: Computation in Superposition
- **Authors**: (2023)
- **Source**: arXiv:2310.10248
- **Key Contribution**: Studies how neural networks perform computation (not just representation) in superposition
- **Relevance**: **HIGH** - Directly relevant to understanding how MLPs execute "programs" when features are superposed. If token directions are superposed, the MLP must perform computation in superposition to map inputs to outputs.

## Common Methodologies

1. **Logit Lens / Early Decoding**: Project intermediate representations to vocabulary space via unembedding matrix E. Used in Geva et al. (2021, 2022), Merullo et al. (2023), nostalgebraist (2020). Core tool for interpreting MLP outputs as token distributions.

2. **Causal Tracing / Activation Patching**: Corrupt model inputs and restore activations at specific layers/positions to identify causal contributions. Used in ROME (Meng et al.), knowledge neurons (Dai et al.).

3. **FFN Decomposition**: Decompose FFN output as weighted sum of value vectors: FFN(x) = Σ m_i · v_i. Used in Geva et al. (2022), extended by SAE approaches.

4. **Sparse Autoencoders**: Train overcomplete dictionaries on MLP activations to find interpretable features beyond individual neurons. Used in Cunningham et al. (2023), Anthropic's work.

5. **Linear Probing**: Train linear classifiers on intermediate representations to detect concept presence. Foundation for linear representation hypothesis.

## Standard Baselines

- **Random projections**: Compare MLP value vector projections to random vector projections (Geva et al. 2022)
- **Individual neurons vs. directions**: Compare neuron-level interpretability to direction-level (SAEs)
- **Attention-only models**: Ablate FFN layers entirely to measure their contribution (Merullo et al. 2023)
- **Random sub-updates**: Compare dominant sub-updates to random ones (Geva et al. 2022)

## Evaluation Metrics

- **Agreement rate**: Fraction of cases where value vector's top prediction matches the actual next token
- **Concept coverage**: Percentage of top-scoring tokens associated with human-interpretable concepts
- **Patching success rate**: Whether patching an FFN output from one context produces correct output in another
- **Rank of target token**: Where the correct token ranks in the distribution induced by a value vector
- **Cosine similarity**: Between value vector projections and token embeddings

## Datasets in the Literature

- **WikiText-103**: Used by Geva et al. (2021, 2022) - primary corpus for WikiLM analysis
- **WebText/OpenWebText**: GPT-2's training data, used for GPT-2 analysis
- **CounterFact**: Used by Meng et al. (2022) for factual editing evaluation
- **BIG-Bench tasks**: Used by Merullo et al. (2023) for vector arithmetic experiments
- **Model weights themselves**: The MLP weight matrices (W_K, W_V) and unembedding matrix E are primary data

## Gaps and Opportunities

1. **No systematic "program extraction"**: While individual papers show MLPs act as key-value memories, promote concepts, and implement vector arithmetic, no one has systematically described the full input→output mapping as a program over token directions.

2. **Beyond individual neurons/features**: Most work either looks at individual neurons (knowledge neurons) or uses SAEs. The hypothesis suggests a higher-level description as a program combining multiple direction comparisons.

3. **Shallow vs. deep programs**: It's unclear how "shallow" the MLP program actually is. Is it a simple lookup table? A set of if-then rules? A composition of simple operations?

4. **Quantifying token direction presence**: While linear probing shows concepts are linearly detectable, the specific mechanism by which MLPs "compare which token directions are present" is not well-characterized.

5. **Cross-layer composition**: How do the simple per-layer MLP programs compose across the full network to produce complex behavior?

## Recommendations for Experiment Design

Based on literature review:

- **Recommended models**: GPT-2 Small (12 layers, d=768, most studied) and GPT-2 Medium (24 layers, used by Merullo et al.) as primary models. Both have well-understood architectures and extensive prior analysis.

- **Recommended datasets**: WikiText-103 validation set for probing (consistent with Geva et al.), CounterFact for factual recall analysis, BIG-Bench for relational tasks.

- **Recommended baselines**: (1) Random direction comparison, (2) Full FFN output vs. decomposed sub-updates, (3) Attention-only ablation.

- **Recommended metrics**: Agreement rate between MLP output and token direction predictions, concept coverage of value vectors, patching success rate across contexts.

- **Recommended tools**: TransformerLens for model inspection, the logit lens for intermediate decoding, SAEs for feature decomposition.

- **Methodological considerations**:
  - Start with individual MLP layers before analyzing cross-layer composition
  - Use the unembedding matrix E to project all vectors to vocabulary space
  - Compare upper vs. lower layer behavior (Geva et al. showed qualitative differences)
  - Consider both the key (input pattern) and value (output distribution) sides of the MLP
  - The hypothesis is most testable on factual recall tasks where the "program" should be simplest
