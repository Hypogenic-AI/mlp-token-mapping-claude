"""
Additional analysis: understand why random directions sometimes match/exceed token directions.
Test whether token directions have a SEMANTIC advantage (predict meaningful tokens)
even if R² is similar.
"""
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = "cuda:0"
RESULTS_DIR = Path("/workspaces/mlp-token-mapping-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Load model
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
W_U = model.W_U.detach()

# Key question: Does the MLP output, projected to token space, actually predict
# the NEXT TOKEN well? This is the real test of "token direction" relevance.

print("=== Analysis: MLP Output → Next Token Prediction ===")

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
texts = [t for t in dataset["text"] if len(t.strip()) > 100][:100]

MAX_SEQ_LEN = 128
all_tokens = []
for text in texts:
    toks = model.to_tokens(text, prepend_bos=True)
    if toks.shape[1] > MAX_SEQ_LEN and toks.shape[1] >= 10:
        all_tokens.append(toks[:, :MAX_SEQ_LEN])

print(f"Using {len(all_tokens)} sequences")

# For each layer, measure how well the MLP output predicts the correct next token
# compared to: (a) the MLP input, (b) random vectors

results = {}
for layer in range(12):
    correct_rank_mlp_out = []
    correct_rank_mlp_in = []
    correct_rank_residual = []
    mlp_out_logit_of_correct = []

    for seq_idx in range(min(50, len(all_tokens))):
        toks = all_tokens[seq_idx].to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, return_type="logits")

        mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"][0]  # (seq, d)
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0]  # (seq, d)
        residual = cache[f"blocks.{layer}.hook_resid_post"][0]  # (seq, d)

        # Next tokens (shifted by 1)
        next_tokens = toks[0, 1:]  # (seq-1,)

        # Project to vocab
        mlp_out_logits = mlp_out[:-1] @ W_U  # (seq-1, vocab)
        mlp_in_logits = mlp_in[:-1] @ W_U
        residual_logits = residual[:-1] @ W_U

        for pos in range(next_tokens.shape[0]):
            target = next_tokens[pos].item()

            # Rank of correct next token in MLP output distribution
            out_vals = mlp_out_logits[pos]
            rank_out = (out_vals > out_vals[target]).sum().item()
            correct_rank_mlp_out.append(rank_out)
            mlp_out_logit_of_correct.append(out_vals[target].item())

            # Rank in MLP input
            in_vals = mlp_in_logits[pos]
            rank_in = (in_vals > in_vals[target]).sum().item()
            correct_rank_mlp_in.append(rank_in)

            # Rank in full residual stream
            res_vals = residual_logits[pos]
            rank_res = (res_vals > res_vals[target]).sum().item()
            correct_rank_residual.append(rank_res)

        del cache
        torch.cuda.empty_cache()

    results[layer] = {
        "mlp_out_median_rank": float(np.median(correct_rank_mlp_out)),
        "mlp_out_mean_rank": float(np.mean(correct_rank_mlp_out)),
        "mlp_out_top1_pct": float(np.mean([r == 0 for r in correct_rank_mlp_out])),
        "mlp_out_top10_pct": float(np.mean([r < 10 for r in correct_rank_mlp_out])),
        "mlp_in_median_rank": float(np.median(correct_rank_mlp_in)),
        "mlp_in_top10_pct": float(np.mean([r < 10 for r in correct_rank_mlp_in])),
        "residual_median_rank": float(np.median(correct_rank_residual)),
        "residual_top1_pct": float(np.mean([r == 0 for r in correct_rank_residual])),
        "residual_top10_pct": float(np.mean([r < 10 for r in correct_rank_residual])),
    }

    print(f"Layer {layer}: MLP_out median_rank={results[layer]['mlp_out_median_rank']:.0f}, "
          f"top1={results[layer]['mlp_out_top1_pct']:.3f}, "
          f"top10={results[layer]['mlp_out_top10_pct']:.3f} | "
          f"Residual top1={results[layer]['residual_top1_pct']:.3f}, "
          f"top10={results[layer]['residual_top10_pct']:.3f}")


# Plot: how well does each layer's MLP output predict next token?
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

layers = list(range(12))
mlp_out_top1 = [results[l]["mlp_out_top1_pct"] for l in layers]
mlp_out_top10 = [results[l]["mlp_out_top10_pct"] for l in layers]
mlp_in_top10 = [results[l]["mlp_in_top10_pct"] for l in layers]
res_top1 = [results[l]["residual_top1_pct"] for l in layers]
res_top10 = [results[l]["residual_top10_pct"] for l in layers]

axes[0].plot(layers, mlp_out_top1, 'o-', color='steelblue', label='MLP Output')
axes[0].plot(layers, res_top1, 's-', color='darkorange', label='Full Residual')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Top-1 Accuracy')
axes[0].set_title('Next Token Prediction Accuracy\n(Top-1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(layers, mlp_out_top10, 'o-', color='steelblue', label='MLP Output')
axes[1].plot(layers, mlp_in_top10, '^-', color='gray', label='MLP Input')
axes[1].plot(layers, res_top10, 's-', color='darkorange', label='Full Residual')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Top-10 Accuracy')
axes[1].set_title('Next Token Prediction Accuracy\n(Top-10)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

mlp_out_med = [results[l]["mlp_out_median_rank"] for l in layers]
mlp_in_med = [results[l]["mlp_in_median_rank"] for l in layers]
res_med = [results[l]["residual_median_rank"] for l in layers]

axes[2].plot(layers, mlp_out_med, 'o-', color='steelblue', label='MLP Output')
axes[2].plot(layers, mlp_in_med, '^-', color='gray', label='MLP Input')
axes[2].plot(layers, res_med, 's-', color='darkorange', label='Full Residual')
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('Median Rank of Correct Token')
axes[2].set_title('Next Token Rank\n(Lower = Better)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "analysis_next_token_prediction.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {PLOTS_DIR / 'analysis_next_token_prediction.png'}")


# ─── Analysis 2: How "simple" are MLP programs? ────────────────────────────
# Measure nonlinearity: compare MLP(x) to a linear approximation W*x+b

print("\n=== Analysis: MLP Nonlinearity by Layer ===")

nonlinearity_results = {}
for layer in range(12):
    # Sample positions
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    n_sample = 3000
    all_positions = mlp_inputs_count = None

    # Collect activations for this layer
    mlp_ins = []
    mlp_outs = []
    for seq_idx in range(min(50, len(all_tokens))):
        toks = all_tokens[seq_idx].to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, return_type="logits")
        mlp_in = cache[f"blocks.{layer}.ln2.hook_normalized"][0].cpu()
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0].cpu()
        mlp_ins.append(mlp_in)
        mlp_outs.append(mlp_out)
        del cache
        torch.cuda.empty_cache()

    mlp_ins = torch.cat(mlp_ins, dim=0).float().numpy()
    mlp_outs = torch.cat(mlp_outs, dim=0).float().numpy()

    # Subsample
    idx = np.random.permutation(mlp_ins.shape[0])[:n_sample]
    X = mlp_ins[idx]
    Y = mlp_outs[idx]

    n_train = int(0.7 * n_sample)

    # Linear approximation in full d_model space
    ridge = Ridge(alpha=1.0)
    ridge.fit(X[:n_train], Y[:n_train])
    Y_pred = ridge.predict(X[n_train:])

    r2_linear = r2_score(Y[n_train:].ravel(), Y_pred.ravel())

    # Per-dimension R²
    r2_per_dim = []
    for d in range(Y.shape[1]):
        r2_d = r2_score(Y[n_train:, d], Y_pred[:, d])
        r2_per_dim.append(r2_d)

    # Cosine similarity
    cos_sims = []
    for i in range(Y[n_train:].shape[0]):
        a, b = Y[n_train:][i], Y_pred[i]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims.append(cos)

    nonlinearity_results[layer] = {
        "r2_linear_full": float(r2_linear),
        "r2_per_dim_mean": float(np.mean(r2_per_dim)),
        "cosine_sim_mean": float(np.mean(cos_sims)),
        "cosine_sim_std": float(np.std(cos_sims)),
    }

    print(f"Layer {layer}: Linear R²={r2_linear:.3f}, "
          f"Per-dim R²={np.mean(r2_per_dim):.3f}, "
          f"Cos={np.mean(cos_sims):.3f}±{np.std(cos_sims):.3f}")

# Plot nonlinearity
fig, ax = plt.subplots(figsize=(8, 5))
r2_vals = [nonlinearity_results[l]["r2_linear_full"] for l in range(12)]
cos_vals = [nonlinearity_results[l]["cosine_sim_mean"] for l in range(12)]

ax.plot(range(12), cos_vals, 'o-', color='steelblue', label='Cosine Sim (Linear Approx)')
ax.plot(range(12), r2_vals, 's-', color='darkorange', label='R² (Linear Approx)')
ax.set_xlabel('Layer')
ax.set_ylabel('Score')
ax.set_title('How Linear is Each MLP Layer?\n(Full d_model → d_model linear map)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "analysis_mlp_linearity.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'analysis_mlp_linearity.png'}")

# Save additional results
with open(RESULTS_DIR / "additional_analysis.json", "w") as f:
    json.dump({
        "next_token_prediction": {str(k): v for k, v in results.items()},
        "mlp_linearity": {str(k): v for k, v in nonlinearity_results.items()},
    }, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'additional_analysis.json'}")

print("\n=== Additional Analysis Complete ===")
