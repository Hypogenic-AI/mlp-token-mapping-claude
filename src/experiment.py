"""
Mapping an MLP in Terms of Tokens
=================================
Main experiment: analyze whether MLP layers in GPT-2 Small can be described as
simple programs that detect input token directions and produce output token directions.

Approach:
1. Collect MLP inputs/outputs across text sequences
2. Project to vocabulary space using the unembedding matrix
3. Build shallow approximations and measure faithfulness
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspaces/mlp-token-mapping-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─── Step 1: Load Model ─────────────────────────────────────────────────────

print("\n=== Loading GPT-2 Small ===")
import transformer_lens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

n_layers = model.cfg.n_layers  # 12
d_model = model.cfg.d_model    # 768
d_mlp = model.cfg.d_mlp        # 3072
n_vocab = model.cfg.d_vocab    # 50257

print(f"Model: GPT-2 Small")
print(f"Layers: {n_layers}, d_model: {d_model}, d_mlp: {d_mlp}, vocab: {n_vocab}")

# Get unembedding matrix W_U: (d_model, n_vocab)
W_U = model.W_U.detach()  # (d_model, vocab)
# Get embedding matrix W_E: (n_vocab, d_model)
W_E = model.W_E.detach()  # (vocab, d_model)

# Normalize W_U columns for cosine-style projections
W_U_normalized = W_U / W_U.norm(dim=0, keepdim=True)  # (d_model, vocab)


# ─── Step 2: Collect Text Data ──────────────────────────────────────────────

print("\n=== Loading Text Data ===")
from datasets import load_dataset

# Use WikiText-103 validation set (consistent with literature)
try:
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:200]
    print(f"Loaded {len(texts)} text samples from WikiText-103 validation")
except Exception as e:
    print(f"WikiText-103 load error: {e}, using fallback texts")
    texts = [
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "Albert Einstein developed the theory of relativity in the early 20th century.",
        "Machine learning algorithms can learn patterns from data without being explicitly programmed.",
    ] * 50

# Tokenize
MAX_SEQ_LEN = 128
NUM_SEQUENCES = min(len(texts), 200)
all_tokens = []
for text in texts[:NUM_SEQUENCES]:
    toks = model.to_tokens(text, prepend_bos=True)
    if toks.shape[1] > MAX_SEQ_LEN:
        toks = toks[:, :MAX_SEQ_LEN]
    if toks.shape[1] >= 10:  # Skip very short sequences
        all_tokens.append(toks)

print(f"Prepared {len(all_tokens)} sequences for analysis")


# ─── Step 3: Collect MLP Inputs/Outputs ─────────────────────────────────────

print("\n=== Collecting MLP Activations ===")

# We'll store per-layer statistics rather than all activations (memory efficient)
# For each layer, collect MLP input and output projections to vocab space

# Data structures for results
layer_results = {}

# Process in batches
BATCH_SIZE = 8

def collect_mlp_activations(tokens_list, layers_to_analyze=None):
    """Collect MLP input/output for all layers across all token positions."""
    if layers_to_analyze is None:
        layers_to_analyze = list(range(n_layers))

    # Accumulate per-layer data
    all_mlp_in = {l: [] for l in layers_to_analyze}
    all_mlp_out = {l: [] for l in layers_to_analyze}
    all_input_tokens = []

    for batch_start in tqdm(range(0, len(tokens_list), BATCH_SIZE), desc="Collecting activations"):
        batch = tokens_list[batch_start:batch_start + BATCH_SIZE]
        # Pad to same length within batch
        max_len = max(t.shape[1] for t in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        masks = torch.zeros(len(batch), max_len, dtype=torch.bool, device=DEVICE)
        for i, t in enumerate(batch):
            padded[i, :t.shape[1]] = t[0]
            masks[i, :t.shape[1]] = True

        # Run model with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(padded, return_type="logits")

        for layer in layers_to_analyze:
            # MLP input = output of layer norm before MLP (ln2)
            mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"]  # (batch, seq, d_model)
            # MLP output
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]  # (batch, seq, d_model)

            # Flatten valid positions
            for i in range(len(batch)):
                valid = masks[i]
                all_mlp_in[layer].append(mlp_in[i, valid].cpu())
                all_mlp_out[layer].append(mlp_out[i, valid].cpu())

        # Collect input tokens too
        for i in range(len(batch)):
            valid = masks[i]
            all_input_tokens.append(padded[i, valid].cpu())

        del cache
        torch.cuda.empty_cache()

    # Concatenate
    for layer in layers_to_analyze:
        all_mlp_in[layer] = torch.cat(all_mlp_in[layer], dim=0)
        all_mlp_out[layer] = torch.cat(all_mlp_out[layer], dim=0)

    all_input_tokens = torch.cat(all_input_tokens, dim=0)
    return all_mlp_in, all_mlp_out, all_input_tokens


mlp_inputs, mlp_outputs, input_tokens = collect_mlp_activations(all_tokens)
n_positions = mlp_inputs[0].shape[0]
print(f"Collected {n_positions} token positions across all sequences")


# ─── Step 4: Experiment 1 - Token Direction Decomposition ───────────────────

print("\n=== Experiment 1: Token Direction Decomposition ===")
print("How much of MLP input/output lies along token (unembedding) directions?")

def project_to_vocab(vectors, W_U_mat, top_k=50):
    """Project vectors to vocabulary space. Returns logits for all vocab tokens.
    vectors: (n, d_model), W_U_mat: (d_model, vocab)
    Returns: (n, vocab) logits
    """
    return vectors @ W_U_mat  # (n, vocab)

def compute_token_direction_stats(vectors, W_U_mat, top_k=50):
    """Measure how concentrated vectors are along token directions."""
    # Full projection to vocab space
    logits = vectors.float() @ W_U_mat.cpu().float()  # (n, vocab)

    # Get top-k token logits for each position
    top_vals, top_ids = logits.topk(top_k, dim=-1)

    # Reconstruct from top-k token directions only
    # For each position, sum of top-k token direction components
    W_U_cpu = W_U_mat.cpu().float()

    # Sample a subset for detailed reconstruction analysis (memory efficient)
    sample_size = min(1000, vectors.shape[0])
    sample_idx = torch.randperm(vectors.shape[0])[:sample_size]

    reconstruction_cosines = []
    variance_explained = []

    for idx in sample_idx:
        v = vectors[idx].float()
        top_token_ids = top_ids[idx]  # (top_k,)
        top_token_dirs = W_U_cpu[:, top_token_ids].T  # (top_k, d_model)

        # Project v onto span of top-k token directions (least squares)
        # v ≈ sum(alpha_i * u_i) where u_i = W_U[:, token_i]
        # Solve: top_token_dirs @ alpha = v (in least-squares sense)
        # top_token_dirs: (top_k, d_model), v: (d_model,)
        coeffs, _, _, _ = torch.linalg.lstsq(top_token_dirs.T, v.unsqueeze(1))
        reconstructed = (top_token_dirs.T @ coeffs).squeeze()

        cos = torch.nn.functional.cosine_similarity(v.unsqueeze(0), reconstructed.unsqueeze(0)).item()
        var_exp = 1 - ((v - reconstructed).norm() ** 2 / v.norm() ** 2).item()

        reconstruction_cosines.append(cos)
        variance_explained.append(var_exp)

    return {
        "mean_cosine_top_k_reconstruction": np.mean(reconstruction_cosines),
        "std_cosine": np.std(reconstruction_cosines),
        "mean_variance_explained": np.mean(variance_explained),
        "std_variance_explained": np.std(variance_explained),
        "top_k": top_k,
    }

# Analyze each layer
exp1_results = {"input": {}, "output": {}}
for layer in tqdm(range(n_layers), desc="Exp1: Token direction analysis"):
    # MLP input
    stats_in = compute_token_direction_stats(mlp_inputs[layer], W_U, top_k=50)
    exp1_results["input"][layer] = stats_in

    # MLP output
    stats_out = compute_token_direction_stats(mlp_outputs[layer], W_U, top_k=50)
    exp1_results["output"][layer] = stats_out

    print(f"  Layer {layer}: Input cos={stats_in['mean_cosine_top_k_reconstruction']:.3f}, "
          f"Output cos={stats_out['mean_cosine_top_k_reconstruction']:.3f}, "
          f"Input R²={stats_in['mean_variance_explained']:.3f}, "
          f"Output R²={stats_out['mean_variance_explained']:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

layers_range = list(range(n_layers))
in_cosines = [exp1_results["input"][l]["mean_cosine_top_k_reconstruction"] for l in layers_range]
out_cosines = [exp1_results["output"][l]["mean_cosine_top_k_reconstruction"] for l in layers_range]
in_var = [exp1_results["input"][l]["mean_variance_explained"] for l in layers_range]
out_var = [exp1_results["output"][l]["mean_variance_explained"] for l in layers_range]

axes[0].plot(layers_range, in_cosines, 'o-', label='MLP Input')
axes[0].plot(layers_range, out_cosines, 's-', label='MLP Output')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Cosine Similarity')
axes[0].set_title('Top-50 Token Direction Reconstruction\n(Cosine Similarity)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(layers_range, in_var, 'o-', label='MLP Input')
axes[1].plot(layers_range, out_var, 's-', label='MLP Output')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Variance Explained (R²)')
axes[1].set_title('Top-50 Token Direction Reconstruction\n(Variance Explained)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp1_token_direction_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'exp1_token_direction_decomposition.png'}")


# ─── Step 5: Experiment 2 - MLP as Token-to-Token Mapping ───────────────────

print("\n=== Experiment 2: MLP as Token-to-Token Mapping ===")
print("Can we predict MLP output token logits from input token logits?")

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

def evaluate_shallow_program(mlp_in_layer, mlp_out_layer, W_U_mat,
                              top_k_in=100, top_k_out=100, n_samples=2000):
    """
    Test whether a shallow program (sparse linear map) can map
    input token logits → output token logits.

    The "program" is:
    1. Project input to top-k token logits (which token directions are present?)
    2. Apply linear map
    3. Compare predicted output token logits to actual
    """
    W_U_cpu = W_U_mat.cpu().float()

    # Sample positions
    n_total = mlp_in_layer.shape[0]
    sample_idx = torch.randperm(n_total)[:n_samples]

    # Project input and output to vocab space
    inp = mlp_in_layer[sample_idx].float() @ W_U_cpu  # (n_samples, vocab)
    out = mlp_out_layer[sample_idx].float() @ W_U_cpu  # (n_samples, vocab)

    # Strategy 1: Use top-k input token logits to predict full output logits
    # Find the top-k most commonly activated input tokens across all samples
    mean_abs_inp = inp.abs().mean(dim=0)  # (vocab,)
    top_in_tokens = mean_abs_inp.topk(top_k_in).indices  # (top_k_in,)

    # Similarly for output
    mean_abs_out = out.abs().mean(dim=0)
    top_out_tokens = mean_abs_out.topk(top_k_out).indices  # (top_k_out,)

    # Extract features: input logits for top input tokens
    X = inp[:, top_in_tokens].numpy()  # (n_samples, top_k_in)
    # Target: output logits for top output tokens
    Y = out[:, top_out_tokens].numpy()  # (n_samples, top_k_out)

    # Train/test split
    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Fit Ridge regression (shallow linear program)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)

    # Metrics
    r2_per_token = []
    for j in range(top_k_out):
        r2 = r2_score(Y_test[:, j], Y_pred[:, j])
        r2_per_token.append(r2)

    r2_overall = r2_score(Y_test.ravel(), Y_pred.ravel())

    # Cosine similarity between predicted and actual output vectors (in token space)
    cos_sims = []
    for i in range(Y_test.shape[0]):
        cos = np.dot(Y_test[i], Y_pred[i]) / (np.linalg.norm(Y_test[i]) * np.linalg.norm(Y_pred[i]) + 1e-8)
        cos_sims.append(cos)

    # Also measure: can we predict the TOP output tokens from input tokens?
    # For each test sample, check if top-5 predicted output tokens match actual top-5
    top5_agreement = []
    for i in range(Y_test.shape[0]):
        actual_top5 = set(np.argsort(-Y_test[i])[:5])
        pred_top5 = set(np.argsort(-Y_pred[i])[:5])
        agreement = len(actual_top5 & pred_top5) / 5
        top5_agreement.append(agreement)

    return {
        "r2_overall": float(r2_overall),
        "r2_per_token_mean": float(np.mean(r2_per_token)),
        "r2_per_token_std": float(np.std(r2_per_token)),
        "cosine_sim_mean": float(np.mean(cos_sims)),
        "cosine_sim_std": float(np.std(cos_sims)),
        "top5_agreement_mean": float(np.mean(top5_agreement)),
        "top5_agreement_std": float(np.std(top5_agreement)),
        "n_train": n_train,
        "n_test": n_samples - n_train,
        "top_k_in": top_k_in,
        "top_k_out": top_k_out,
    }


exp2_results = {}
for layer in tqdm(range(n_layers), desc="Exp2: Shallow program"):
    result = evaluate_shallow_program(mlp_inputs[layer], mlp_outputs[layer], W_U,
                                       top_k_in=100, top_k_out=100, n_samples=3000)
    exp2_results[layer] = result
    print(f"  Layer {layer}: R²={result['r2_overall']:.3f}, "
          f"cos={result['cosine_sim_mean']:.3f}, "
          f"top5_agree={result['top5_agreement_mean']:.3f}")

# Plot Experiment 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

r2_vals = [exp2_results[l]["r2_overall"] for l in layers_range]
cos_vals = [exp2_results[l]["cosine_sim_mean"] for l in layers_range]
top5_vals = [exp2_results[l]["top5_agreement_mean"] for l in layers_range]

axes[0].bar(layers_range, r2_vals, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('R² (Overall)')
axes[0].set_title('Shallow Program Faithfulness\n(R² of Input→Output Token Logits)')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(layers_range, cos_vals, color='darkorange', alpha=0.8)
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Cosine Similarity')
axes[1].set_title('Shallow Program Faithfulness\n(Cosine Sim in Token Space)')
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(layers_range, top5_vals, color='forestgreen', alpha=0.8)
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('Top-5 Agreement')
axes[2].set_title('Shallow Program Faithfulness\n(Top-5 Output Token Agreement)')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp2_shallow_program.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'exp2_shallow_program.png'}")


# ─── Step 6: Experiment 3 - Sub-Update Decomposition (Geva-style) ───────────

print("\n=== Experiment 3: Sub-Update Decomposition ===")
print("Decompose MLP into individual neuron contributions and analyze in token space")

def analyze_sub_updates(tokens_list, layer, W_U_mat, n_sequences=20, max_seq_len=64):
    """
    Decompose MLP output as sum of sub-updates m_i * v_i (Geva et al. 2022 style).
    Analyze whether each sub-update corresponds to promoting specific tokens.
    """
    W_U_cpu = W_U_mat.cpu().float()

    # Get MLP weight matrices for this layer
    W_in = model.blocks[layer].mlp.W_in.detach().cpu().float()   # (d_model, d_mlp)
    W_out = model.blocks[layer].mlp.W_out.detach().cpu().float()  # (d_mlp, d_model)
    b_in = model.blocks[layer].mlp.b_in.detach().cpu().float()    # (d_mlp,)
    b_out = model.blocks[layer].mlp.b_out.detach().cpu().float()  # (d_model,)

    # Value vectors: columns of W_out, projected to vocab space
    # Each value vector v_i = W_out[i, :] is (d_model,)
    # Its vocab projection = v_i @ W_U = (vocab,)
    value_vectors = W_out  # (d_mlp, d_model)
    value_logits = value_vectors @ W_U_cpu  # (d_mlp, vocab)

    # For each value vector, get top tokens it promotes
    top_k_per_neuron = 10
    top_neuron_tokens = value_logits.topk(top_k_per_neuron, dim=-1)

    # Analyze sparsity of activation patterns
    activation_sparsities = []
    sub_update_concentration = []

    for seq_idx in range(min(n_sequences, len(tokens_list))):
        toks = tokens_list[seq_idx][:, :max_seq_len]

        with torch.no_grad():
            _, cache = model.run_with_cache(toks.to(DEVICE), return_type="logits")

        # Get MLP hidden activations (after GELU)
        mlp_pre = cache[f"blocks.{layer}.mlp.hook_pre"]  # (1, seq, d_mlp)
        # Apply GELU to get activations
        mlp_post = cache[f"blocks.{layer}.mlp.hook_post"]  # (1, seq, d_mlp) - after activation

        activations = mlp_post[0].cpu().float()  # (seq, d_mlp)

        # Sparsity: fraction of neurons with |activation| > threshold
        threshold = 0.1
        sparsity = (activations.abs() > threshold).float().mean(dim=-1)  # (seq,)
        activation_sparsities.extend(sparsity.tolist())

        # Concentration: what fraction of MLP output is explained by top-k neurons?
        for pos in range(activations.shape[0]):
            act = activations[pos]  # (d_mlp,)
            abs_act = act.abs()
            sorted_abs, _ = abs_act.sort(descending=True)
            total_contribution = sorted_abs.sum()
            for k in [10, 50, 100]:
                top_k_contribution = sorted_abs[:k].sum()
                concentration = (top_k_contribution / (total_contribution + 1e-8)).item()
                sub_update_concentration.append((k, concentration))

        del cache
        torch.cuda.empty_cache()

    # Aggregate concentration results
    concentration_by_k = {}
    for k, c in sub_update_concentration:
        if k not in concentration_by_k:
            concentration_by_k[k] = []
        concentration_by_k[k].append(c)

    return {
        "mean_sparsity": float(np.mean(activation_sparsities)),
        "std_sparsity": float(np.std(activation_sparsities)),
        "concentration": {k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v))
        } for k, v in concentration_by_k.items()},
        "n_neurons": d_mlp,
    }


exp3_results = {}
for layer in tqdm(range(n_layers), desc="Exp3: Sub-update analysis"):
    result = analyze_sub_updates(all_tokens, layer, W_U, n_sequences=30)
    exp3_results[layer] = result
    conc = result["concentration"]
    print(f"  Layer {layer}: sparsity={result['mean_sparsity']:.3f}, "
          f"top10={conc[10]['mean']:.3f}, top50={conc[50]['mean']:.3f}, "
          f"top100={conc[100]['mean']:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sparsities = [exp3_results[l]["mean_sparsity"] for l in layers_range]
axes[0].bar(layers_range, sparsities, color='purple', alpha=0.7)
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Activation Sparsity')
axes[0].set_title('MLP Activation Sparsity per Layer\n(Fraction of neurons with |act| > 0.1)')
axes[0].grid(True, alpha=0.3, axis='y')

for k, color, marker in [(10, 'red', 'o'), (50, 'blue', 's'), (100, 'green', '^')]:
    conc_vals = [exp3_results[l]["concentration"][k]["mean"] for l in layers_range]
    axes[1].plot(layers_range, conc_vals, f'{marker}-', color=color, label=f'Top-{k}', alpha=0.8)

axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Fraction of Total Activation')
axes[1].set_title('MLP Output Concentration\n(Fraction explained by top-k neurons)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp3_sub_update_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'exp3_sub_update_analysis.png'}")


# ─── Step 7: Experiment 4 - Direct Residual Stream Analysis ────────────────

print("\n=== Experiment 4: Token-Specific Case Studies ===")
print("For specific token positions, trace the MLP program in detail")

def trace_mlp_program(text, layer, W_U_mat, top_k=10):
    """For a specific input, trace the MLP's token-to-token program in detail."""
    tokens = model.to_tokens(text, prepend_bos=True)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens.to(DEVICE), return_type="logits")

    W_U_cpu = W_U_mat.cpu().float()

    mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"][0].cpu().float()  # (seq, d_model)
    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0].cpu().float()
    mlp_post = cache[f"blocks.{layer}.mlp.hook_post"][0].cpu().float()  # (seq, d_mlp)

    str_tokens = model.to_str_tokens(tokens[0])

    results = []
    for pos in range(len(str_tokens)):
        # Input token logits
        in_logits = mlp_in[pos] @ W_U_cpu  # (vocab,)
        out_logits = mlp_out[pos] @ W_U_cpu  # (vocab,)

        # Top input token directions
        top_in_vals, top_in_ids = in_logits.topk(top_k)
        top_in_tokens = [model.tokenizer.decode(int(tid)) for tid in top_in_ids]

        # Top output token directions
        top_out_vals, top_out_ids = out_logits.topk(top_k)
        top_out_tokens = [model.tokenizer.decode(int(tid)) for tid in top_out_ids]

        # Which neurons fired most
        act = mlp_post[pos]
        top_neuron_vals, top_neuron_ids = act.abs().topk(top_k)

        results.append({
            "position": pos,
            "current_token": str_tokens[pos],
            "top_input_tokens": list(zip(top_in_tokens, top_in_vals.tolist())),
            "top_output_tokens": list(zip(top_out_tokens, top_out_vals.tolist())),
            "top_neuron_ids": top_neuron_ids.tolist(),
            "top_neuron_activations": top_neuron_vals.tolist(),
            "n_active_neurons": int((act.abs() > 0.1).sum()),
        })

    del cache
    torch.cuda.empty_cache()
    return results


# Analyze a few representative sentences
case_studies = [
    "The capital of France is Paris.",
    "Einstein developed the theory of relativity.",
    "The cat sat on the mat.",
]

exp4_results = {}
for text in case_studies:
    exp4_results[text] = {}
    for layer in [0, 3, 6, 9, 11]:  # Sample layers
        trace = trace_mlp_program(text, layer, W_U, top_k=10)
        exp4_results[text][layer] = trace

        # Print a few interesting positions
        print(f"\n  Text: '{text}', Layer {layer}")
        for t in trace[:3]:  # First 3 tokens
            print(f"    Token '{t['current_token']}': "
                  f"Input→[{', '.join(x[0].strip() for x in t['top_input_tokens'][:3])}], "
                  f"Output→[{', '.join(x[0].strip() for x in t['top_output_tokens'][:3])}], "
                  f"Active neurons: {t['n_active_neurons']}/{d_mlp}")


# ─── Step 8: Experiment 5 - Residual Stream vs. Random Baseline ─────────────

print("\n\n=== Experiment 5: Baseline Comparison ===")
print("Compare shallow program to random direction baseline")

def random_baseline(mlp_in_layer, mlp_out_layer, d_model, top_k_in=100, top_k_out=100, n_samples=2000):
    """Baseline: use random directions instead of token directions."""
    # Random orthogonal directions
    random_dirs = torch.randn(d_model, top_k_in + top_k_out)
    random_dirs = torch.linalg.qr(random_dirs)[0]  # Orthogonalize

    R_in = random_dirs[:, :top_k_in]   # (d_model, top_k_in)
    R_out = random_dirs[:, top_k_in:]  # (d_model, top_k_out)

    sample_idx = torch.randperm(mlp_in_layer.shape[0])[:n_samples]

    X = (mlp_in_layer[sample_idx].float() @ R_in).numpy()
    Y = (mlp_out_layer[sample_idx].float() @ R_out).numpy()

    n_train = int(0.7 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)

    r2 = r2_score(Y_test.ravel(), Y_pred.ravel())

    cos_sims = []
    for i in range(Y_test.shape[0]):
        cos = np.dot(Y_test[i], Y_pred[i]) / (np.linalg.norm(Y_test[i]) * np.linalg.norm(Y_pred[i]) + 1e-8)
        cos_sims.append(cos)

    return {
        "r2_overall": float(r2),
        "cosine_sim_mean": float(np.mean(cos_sims)),
    }


exp5_results = {"token_dirs": {}, "random_dirs": {}, "full_linear": {}}
for layer in tqdm(range(n_layers), desc="Exp5: Baselines"):
    # Token direction results (from exp2)
    exp5_results["token_dirs"][layer] = {
        "r2": exp2_results[layer]["r2_overall"],
        "cos": exp2_results[layer]["cosine_sim_mean"],
    }

    # Random baseline
    rand_result = random_baseline(mlp_inputs[layer], mlp_outputs[layer], d_model)
    exp5_results["random_dirs"][layer] = {
        "r2": rand_result["r2_overall"],
        "cos": rand_result["cosine_sim_mean"],
    }

    # Full linear baseline (project to full d_model, linear map)
    sample_idx = torch.randperm(mlp_inputs[layer].shape[0])[:3000]
    X_full = mlp_inputs[layer][sample_idx].float().numpy()
    Y_full = mlp_outputs[layer][sample_idx].float().numpy()
    n_train = int(0.7 * 3000)

    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_full[:n_train], Y_full[:n_train])
    Y_pred_full = ridge_full.predict(X_full[n_train:])

    # Project to token space for comparable metrics
    Y_test_tok = Y_full[n_train:] @ W_U.cpu().float().numpy()
    Y_pred_tok = Y_pred_full @ W_U.cpu().float().numpy()

    # Use top-100 tokens for comparison
    mean_abs = np.abs(Y_test_tok).mean(axis=0)
    top_tokens = np.argsort(-mean_abs)[:100]

    r2_full = r2_score(Y_test_tok[:, top_tokens].ravel(), Y_pred_tok[:, top_tokens].ravel())

    cos_sims_full = []
    for i in range(Y_test_tok.shape[0]):
        a = Y_test_tok[i, top_tokens]
        b = Y_pred_tok[i, top_tokens]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims_full.append(cos)

    exp5_results["full_linear"][layer] = {
        "r2": float(r2_full),
        "cos": float(np.mean(cos_sims_full)),
    }

    print(f"  Layer {layer}: Token R²={exp5_results['token_dirs'][layer]['r2']:.3f}, "
          f"Random R²={exp5_results['random_dirs'][layer]['r2']:.3f}, "
          f"Full Linear R²={exp5_results['full_linear'][layer]['r2']:.3f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for method, color, label in [
    ("token_dirs", "steelblue", "Token Directions (k=100)"),
    ("random_dirs", "gray", "Random Directions (k=100)"),
    ("full_linear", "darkorange", "Full Linear (d=768)"),
]:
    r2_vals = [exp5_results[method][l]["r2"] for l in layers_range]
    cos_vals = [exp5_results[method][l]["cos"] for l in layers_range]
    axes[0].plot(layers_range, r2_vals, 'o-', color=color, label=label)
    axes[1].plot(layers_range, cos_vals, 'o-', color=color, label=label)

axes[0].set_xlabel('Layer')
axes[0].set_ylabel('R²')
axes[0].set_title('Shallow Program Faithfulness: R²\nToken vs Random vs Full Linear')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Cosine Similarity')
axes[1].set_title('Shallow Program Faithfulness: Cosine Sim\nToken vs Random vs Full Linear')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp5_baseline_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'exp5_baseline_comparison.png'}")


# ─── Step 9: Experiment 6 - Varying top-k ──────────────────────────────────

print("\n=== Experiment 6: Effect of top-k on Approximation Quality ===")

exp6_results = {}
k_values = [10, 25, 50, 100, 200, 500]
test_layers = [0, 3, 6, 9, 11]

for layer in test_layers:
    exp6_results[layer] = {}
    for k in tqdm(k_values, desc=f"Exp6: Layer {layer}"):
        result = evaluate_shallow_program(mlp_inputs[layer], mlp_outputs[layer], W_U,
                                           top_k_in=k, top_k_out=min(k, 100), n_samples=2000)
        exp6_results[layer][k] = result
        print(f"  Layer {layer}, k={k}: R²={result['r2_overall']:.3f}, cos={result['cosine_sim_mean']:.3f}")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for layer in test_layers:
    r2_vals = [exp6_results[layer][k]["r2_overall"] for k in k_values]
    ax.plot(k_values, r2_vals, 'o-', label=f'Layer {layer}')

ax.set_xlabel('Number of Token Directions (k)')
ax.set_ylabel('R² (Overall)')
ax.set_title('Effect of Number of Token Directions\non Shallow Program Faithfulness')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp6_topk_effect.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'exp6_topk_effect.png'}")


# ─── Step 10: Save All Results ──────────────────────────────────────────────

print("\n=== Saving Results ===")

# Convert numpy types for JSON serialization
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj

all_results = {
    "experiment_1_token_direction_decomposition": make_serializable(exp1_results),
    "experiment_2_shallow_program": make_serializable(exp2_results),
    "experiment_3_sub_update_analysis": make_serializable(exp3_results),
    "experiment_5_baseline_comparison": make_serializable(exp5_results),
    "experiment_6_topk_effect": make_serializable(exp6_results),
    "config": {
        "model": "gpt2",
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "n_vocab": n_vocab,
        "seed": SEED,
        "n_sequences": len(all_tokens),
        "max_seq_len": MAX_SEQ_LEN,
        "n_positions_analyzed": n_positions,
        "device": str(DEVICE),
    }
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'metrics.json'}")

# Save case study results separately (they contain strings)
case_study_serializable = {}
for text, layers_data in exp4_results.items():
    case_study_serializable[text] = {}
    for layer, traces in layers_data.items():
        case_study_serializable[text][str(layer)] = traces

with open(RESULTS_DIR / "case_studies.json", "w") as f:
    json.dump(case_study_serializable, f, indent=2, default=str)
print(f"Saved: {RESULTS_DIR / 'case_studies.json'}")

print("\n=== ALL EXPERIMENTS COMPLETE ===")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Plots saved to: {PLOTS_DIR}")
