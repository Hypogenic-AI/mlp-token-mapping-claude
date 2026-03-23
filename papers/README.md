# Downloaded Papers

## Core Papers (Directly support hypothesis)

1. **Transformer Feed-Forward Layers Are Key-Value Memories** (2012.14913_ffn_key_value_memories.pdf)
   - Authors: Geva, Schuster, Berant, Levy
   - Year: 2021
   - arXiv: 2012.14913
   - Why relevant: Foundational paper showing FFN layers as key-value memories with keys detecting token patterns and values encoding output distributions

2. **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space** (2203.14680_ffn_vocab_space.pdf)
   - Authors: Geva, Caciularu, Wang, Goldberg
   - Year: 2022
   - arXiv: 2203.14680
   - Why relevant: Shows FFN sub-updates encode human-interpretable concepts and build predictions via token promotion

3. **Language Models Implement Simple Word2Vec-style Vector Arithmetic** (2305.16130_word2vec_arithmetic.pdf)
   - Authors: Merullo, Eickhoff, Pavlick
   - Year: 2023
   - arXiv: 2305.16130
   - Why relevant: Demonstrates FFN layers implement simple additive vector arithmetic for relational tasks (mapping token directions)

4. **Toy Models of Superposition** (2209.10652_toy_models_superposition.pdf)
   - Authors: Elhage et al. (Anthropic)
   - Year: 2022
   - arXiv: 2209.10652
   - Why relevant: Explains superposition of features as directions, fundamental to understanding "token directions" in residual stream

## Supporting Papers (Theoretical/methodological foundations)

5. **The Linear Representation Hypothesis and the Geometry of Large Language Models** (2311.03658_linear_representations_hypothesis.pdf)
   - Authors: Park, Choe, Veitch
   - Year: 2024
   - arXiv: 2311.03658
   - Why relevant: Formalizes what "linear directions" means, provides causal framework for token representations

6. **Locating and Editing Factual Associations in GPT (ROME)** (2202.05262_rome.pdf)
   - Authors: Meng, Bau, Andonian, Belinkov
   - Year: 2022
   - arXiv: 2202.05262
   - Why relevant: Localizes factual storage to MLP layers, enables editing via value matrix modification

7. **Knowledge Neurons in Pretrained Transformers** (2104.08696_knowledge_neurons.pdf)
   - Authors: Dai, Dong, Hao, Sui, Chang, Wei
   - Year: 2022
   - arXiv: 2104.08696
   - Why relevant: Identifies specific FFN neurons for factual knowledge

8. **The Geometry of Truth** (2310.06824_geometry_of_truth.pdf)
   - Authors: Marks, Tegmark
   - Year: 2023
   - arXiv: 2310.06824
   - Why relevant: Linear representations of truth values in LLMs

9. **Eliciting Latent Predictions from Transformers with the Tuned Lens** (2303.08112_tuned_lens.pdf)
   - Authors: Belrose et al.
   - Year: 2023
   - arXiv: 2303.08112
   - Why relevant: Improved method for reading intermediate representations as vocab distributions

10. **Sparse Autoencoders Find Highly Interpretable Features in Language Models** (2309.08600_sparse_autoencoders_features.pdf)
    - Authors: Cunningham, Ewart, Riggs, Huben, Sharkey
    - Year: 2023
    - arXiv: 2309.08600
    - Why relevant: Tool for decomposing MLP activations into interpretable features/directions

11. **Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models** (2403.14537_factual_recall_mechanisms.pdf)
    - Authors: (2024)
    - arXiv: 2403.14537
    - Why relevant: Detailed mechanism analysis of factual recall involving MLP layers

12. **Neuron2Graph** (2305.19911_neuron2graph.pdf)
    - Authors: (2023)
    - arXiv: 2305.19911
    - Why relevant: Network analysis approach to understanding transformer internals

13. **Computation in Superposition** (2310.10248_computation_in_superposition.pdf)
    - Authors: (2023)
    - arXiv: 2310.10248
    - Why relevant: How networks compute (not just represent) in superposition - relevant to MLP program execution

14. **Dictionary Learning with Sparse Autoencoders** (2310.17230_dictionary_learning_anthropic.pdf)
    - Authors: Anthropic
    - Year: 2023
    - arXiv: 2310.17230
    - Why relevant: Anthropic's SAE approach to finding interpretable features in MLP layers
