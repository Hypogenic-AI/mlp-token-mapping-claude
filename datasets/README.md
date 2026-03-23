# Datasets for MLP Token Mapping Research

This directory contains datasets and data access scripts for the research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Overview

For this research, the primary "data" consists of:
1. **Pre-trained transformer model weights** (GPT-2 family) - the MLP weight matrices ARE the object of study
2. **Text corpora** for running inputs through models and collecting activations
3. **Token embeddings** extracted from model weights

## Dataset 1: GPT-2 Models (Primary)

### Overview
- **Source**: HuggingFace Hub (`gpt2`, `gpt2-medium`, `gpt2-large`)
- **Size**: 124M / 355M / 774M parameters
- **Format**: HuggingFace Transformers model
- **Key Properties**:
  - GPT-2 Small: 12 layers, d=768, d_m=3072, vocab=50257
  - GPT-2 Medium: 24 layers, d=1024, d_m=4096, vocab=50257
- **License**: MIT

### Download Instructions
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 Small (recommended starting point)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 Medium (used in Merullo et al. 2023)
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
```

### Key Weight Matrices
```python
# For layer l:
W_K = model.transformer.h[l].mlp.c_fc.weight      # Keys: (d, d_m)
W_V = model.transformer.h[l].mlp.c_proj.weight     # Values: (d_m, d)
E = model.lm_head.weight                            # Unembedding: (vocab, d)
token_emb = model.transformer.wte.weight            # Token embeddings: (vocab, d)
```

## Dataset 2: WikiText-103

### Overview
- **Source**: HuggingFace Datasets (`wikitext`, `wikitext-103-v1`)
- **Size**: ~500MB, 103M tokens
- **Format**: Text corpus
- **Task**: Probing text for MLP activation patterns
- **Splits**: train (1.8M lines), validation (3.7K lines), test (4.3K lines)
- **License**: CC BY-SA 3.0

### Download Instructions
```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext-103")
```

### Notes
- Used by Geva et al. (2021, 2022) for their FFN key-value memory analysis
- WikiLM model in those papers was trained on this corpus
- Good for analyzing what patterns MLP keys detect

## Dataset 3: CounterFact (for factual association probing)

### Overview
- **Source**: Part of ROME codebase (`https://github.com/kmeng01/rome`)
- **Size**: ~21K factual statements
- **Format**: JSON
- **Task**: Testing factual recall via MLPs
- **License**: MIT

### Download Instructions
```python
# Available via the cloned ROME repository
import json
with open("code/rome/data/counterfact.json") as f:
    counterfact = json.load(f)
```

## Dataset 4: BIG-Bench Tasks (for vector arithmetic validation)

### Overview
- **Source**: `https://github.com/google/BIG-bench`
- **Size**: Various small datasets
- **Format**: JSON
- **Tasks**: Past tense mapping, colored objects, world capitals
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
# Past tense task
past_tense = load_dataset("bigbench", "past_tense")
```

## Sample Script

See `sample_load_model.py` for a complete example of loading a model and accessing MLP weights.
