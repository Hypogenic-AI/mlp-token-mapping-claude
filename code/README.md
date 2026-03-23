# Cloned Repositories

## Repo 1: ff-layers
- **URL**: https://github.com/mega002/ff-layers/
- **Purpose**: Code for "Transformer Feed-Forward Layers Are Key-Value Memories" (Geva et al., 2021)
- **Location**: code/ff-layers/
- **Key files**: Scripts for retrieving top trigger examples per key, computing memory coefficients, analyzing key-value patterns
- **Notes**: Designed for WikiLM model on WikiText-103

## Repo 2: ffn-values
- **URL**: https://github.com/aviclu/ffn-values
- **Purpose**: Code for "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" (Geva et al., 2022)
- **Location**: code/ffn-values/
- **Key files**: Scripts for projecting value vectors to vocabulary space, analyzing sub-updates, toxicity reduction
- **Notes**: Works with WikiLM and GPT-2 models

## Repo 3: lm_vector_arithmetic
- **URL**: https://github.com/jmerullo/lm_vector_arithmetic
- **Purpose**: Code for "Language Models Implement Simple Word2Vec-style Vector Arithmetic" (Merullo et al., 2023)
- **Location**: code/lm_vector_arithmetic/
- **Key files**: Early decoding scripts, FFN output patching experiments, vector arithmetic analysis
- **Notes**: Primarily uses GPT-2 Medium

## Repo 4: linear_rep_geometry
- **URL**: https://github.com/KihoPark/linear_rep_geometry
- **Purpose**: Code for "The Linear Representation Hypothesis and the Geometry of Large Language Models" (Park et al., 2024)
- **Location**: code/linear_rep_geometry/
- **Key files**: Causal inner product computation, linear probing, steering vector experiments
- **Notes**: Uses LLaMA-2

## Repo 5: rome
- **URL**: https://github.com/kmeng01/rome
- **Purpose**: Code for "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
- **Location**: code/rome/
- **Key files**: Causal tracing scripts, ROME editing method, CounterFact dataset
- **Notes**: Contains CounterFact dataset in data/ directory

## Repo 6: sparse_autoencoder
- **URL**: https://github.com/openai/sparse_autoencoder
- **Purpose**: OpenAI's sparse autoencoder implementation for finding interpretable features in language models
- **Location**: code/sparse_autoencoder/
- **Key files**: SAE training and analysis code
- **Notes**: Can be used to decompose MLP activations into interpretable features

## Repo 7: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Library for mechanistic interpretability of transformer models
- **Location**: code/TransformerLens/
- **Key files**: HookedTransformer class for accessing all intermediate activations
- **Notes**: Essential tool for running experiments - provides easy access to all model internals including MLP inputs/outputs, residual stream states, attention patterns. Supports GPT-2, GPT-Neo, Pythia, and many other models.
