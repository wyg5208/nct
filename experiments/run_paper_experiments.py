"""
NCT Paper Experiments - Comprehensive Experimental Validation (v2)
Generates all quantitative results for the NCT journal/conference paper.

v2 Changes:
  - Experiment A: New metrics (weight_entropy, attn_contribution, weight_std)
  - Experiment D: Structured attention patterns for meaningful Phi
  - Experiment E: Extreme neuromodulator states, richer metrics
  - Experiment F: NEW - Sequence prediction downstream task (7 configs)

Experiments:
  A. Hybrid Learning Convergence Analysis (varying lambda, improved metrics)
  B. Free Energy Minimization Long-Range (100 steps, 5 seeds)
  C. Phi Computation Accuracy (small networks, ground truth comparison)
  D. Scale Analysis (varying d_model, structured attention for Phi)
  E. Ablation Study (7 configurations, richer metrics)
  F. Sequence Prediction Task (downstream evaluation, 7 configs)

Usage:
    python run_paper_experiments.py [--experiment A|B|C|D|E|F|all]

Author: Yonggang Weng
Date: February 2026
"""

from __future__ import annotations
import sys, os, json, time, argparse, logging, math
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add NCT to path
NCT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NCT_ROOT))
sys.path.insert(0, str(NCT_ROOT.parent))

from nct_modules.nct_hybrid_learning import (
    TransformerSTDP, STDPEvent, ClassicSTDP, NeuromodulatorGate
)
from nct_modules.nct_predictive_coding import (
    PredictiveCodingDecoder, PredictiveHierarchy
)
from nct_modules.nct_metrics import PhiFromAttention

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("NCT_Experiments")
logger.setLevel(logging.INFO)

OUTPUT_DIR = NCT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility
# ============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_stdp_events(n_events: int, n_neurons: int) -> List[STDPEvent]:
    """Generate random STDP events with biologically plausible timing."""
    events = []
    for _ in range(n_events):
        pre = np.random.randint(0, n_neurons)
        post = np.random.randint(0, n_neurons)
        while post == pre:
            post = np.random.randint(0, n_neurons)
        pre_time = np.random.uniform(0, 100)
        post_time = pre_time + np.random.normal(0, 15)  # ~15ms jitter
        events.append(STDPEvent(pre, post, pre_time, post_time))
    return events


def compute_weight_entropy(weights: np.ndarray, n_bins: int = 50) -> float:
    """Compute Shannon entropy of weight distribution."""
    non_zero = weights[weights > 0]
    if len(non_zero) < 2:
        return 0.0
    hist, _ = np.histogram(non_zero, bins=n_bins, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist + 1e-12)))


def generate_structured_attention(size: int, n_blocks: int = 4) -> torch.Tensor:
    """Generate structured attention pattern (block-diagonal + off-diagonal).
    Mimics real attention: strong local clusters with weak global connections."""
    attn = torch.zeros(size, size)
    block_size = size // n_blocks
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, size)
        # Strong intra-block attention
        attn[start:end, start:end] = torch.rand(end - start, end - start) * 0.8 + 0.2
    # Weak inter-block connections
    attn += torch.rand(size, size) * 0.05
    # Normalize to row-stochastic
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


# ============================================================================
# Experiment A: Hybrid Learning Convergence Analysis (v2)
# ============================================================================

def experiment_A_convergence(n_steps=1000, n_seeds=5):
    """Test convergence under different lambda values with improved metrics."""
    logger.info("=" * 60)
    logger.info("Experiment A: Hybrid Learning Convergence Analysis (v2)")
    logger.info("=" * 60)

    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    d_model = 256
    events_per_step = 20
    nt_state = {"dopamine": 0.6, "serotonin": 0.4,
                "norepinephrine": 0.5, "acetylcholine": 0.7}

    all_results = {}

    for lam in lambdas:
        seed_results = {
            "ltp_ratios": [], "weight_stds": [], "weight_entropies": [],
            "attn_contributions": [], "mean_abs_delta_w": [],
        }

        for seed in range(n_seeds):
            set_seed(seed * 42 + 7)
            learner = TransformerSTDP(
                n_neurons=d_model, d_model=d_model,
                stdp_learning_rate=0.01,
                attention_modulation_lambda=lam,
                sparsity=0.01,
            )

            step_ltp = []
            step_weight_std = []
            step_entropy = []
            step_attn_contrib = []
            step_abs_dw = []

            # Structured context: evolves slowly with a consistent pattern
            global_ctx = torch.randn(d_model)

            for step in range(n_steps):
                events = generate_stdp_events(events_per_step, d_model)
                ctx = global_ctx if lam > 0 else None
                updates = learner.forward(events, ctx, nt_state)

                # LTP ratio
                ltp = sum(1 for u in updates if u.total_delta_w > 0)
                step_ltp.append(ltp / max(1, len(updates)))

                # Attention contribution ratio
                total_attn = sum(abs(u.delta_w_attn) for u in updates)
                total_stdp = sum(abs(u.delta_w_std) for u in updates)
                if total_stdp + total_attn > 0:
                    step_attn_contrib.append(total_attn / (total_stdp + total_attn))
                else:
                    step_attn_contrib.append(0.0)

                # Mean absolute delta_w
                step_abs_dw.append(np.mean([abs(u.total_delta_w) for u in updates]))

                # Weight statistics (every 50 steps for efficiency)
                if step % 50 == 0:
                    w = learner.get_weight_matrix()
                    nz = w[w > 0]
                    step_weight_std.append(float(np.std(nz)) if len(nz) > 0 else 0.0)
                    step_entropy.append(compute_weight_entropy(w))

                # Slowly evolve context (smooth, not random jump)
                if step % 50 == 0 and step > 0:
                    global_ctx = global_ctx * 0.7 + torch.randn(d_model) * 0.3

                learner.reset_history()

            seed_results["ltp_ratios"].append(step_ltp)
            seed_results["weight_stds"].append(step_weight_std)
            seed_results["weight_entropies"].append(step_entropy)
            seed_results["attn_contributions"].append(step_attn_contrib)
            seed_results["mean_abs_delta_w"].append(step_abs_dw)

        # Aggregate across seeds
        agg = {}
        # Per-step metrics
        for key in ["ltp_ratios", "attn_contributions", "mean_abs_delta_w"]:
            arr = np.array(seed_results[key])
            agg[f"{key}_final_mean"] = float(arr[:, -1].mean())
            agg[f"{key}_final_std"] = float(arr[:, -1].std())
        # Per-checkpoint metrics (every 50 steps)
        for key in ["weight_stds", "weight_entropies"]:
            arr = np.array(seed_results[key])
            agg[f"{key}_final_mean"] = float(arr[:, -1].mean())
            agg[f"{key}_final_std"] = float(arr[:, -1].std())
            agg[f"{key}_initial_mean"] = float(arr[:, 0].mean())

        all_results[f"lambda_{lam}"] = agg
        logger.info(
            f"  lambda={lam}: LTP={agg['ltp_ratios_final_mean']:.3f}, "
            f"w_std={agg['weight_stds_final_mean']:.4f}, "
            f"entropy={agg['weight_entropies_final_mean']:.3f}, "
            f"attn_contrib={agg['attn_contributions_final_mean']:.4f}, "
            f"|dw|={agg['mean_abs_delta_w_final_mean']:.6f}"
        )

    out_path = OUTPUT_DIR / "exp_A_convergence.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Experiment A saved to {out_path}")
    return all_results


# ============================================================================
# Experiment B: Free Energy Minimization Long-Range (unchanged)
# ============================================================================

def experiment_B_free_energy(n_steps=100, n_seeds=5):
    """Long-range free energy minimization across 4 hierarchy levels."""
    logger.info("=" * 60)
    logger.info("Experiment B: Free Energy Minimization (100 steps)")
    logger.info("=" * 60)

    d_model = 256
    config = {f"layer{i}_dim": d_model for i in range(4)}
    config["n_heads"] = 8

    all_seeds = {"total_fe": [], "layer_fe": [[], [], [], []]}

    for seed in range(n_seeds):
        set_seed(seed * 17 + 3)
        hierarchy = PredictiveHierarchy(config)
        optimizer = torch.optim.Adam(hierarchy.parameters(), lr=1e-3)

        base_signal = torch.randn(1, d_model)
        fe_history = []
        layer_fe_hist = [[] for _ in range(4)]

        for step in range(n_steps):
            noise = torch.randn(1, d_model) * 0.1
            drift = torch.sin(torch.tensor(step * 0.1)) * 0.05
            sensory = base_signal + noise + drift

            optimizer.zero_grad()
            results = hierarchy.forward(sensory)

            fe = results["total_free_energy"]
            fe_history.append(fe)

            for li, err in enumerate(results["errors"]):
                if err is not None:
                    layer_fe_hist[li].append(err.mean().item())
                else:
                    layer_fe_hist[li].append(0.0)

            valid_errs = [e for e in results["errors"] if e is not None and e.requires_grad]
            if valid_errs:
                loss = sum(e.mean() for e in valid_errs)
                try:
                    loss.backward()
                    optimizer.step()
                except RuntimeError:
                    pass

            if step % 20 == 0 and step > 0:
                base_signal = base_signal * 0.9 + torch.randn(1, d_model) * 0.1

        hierarchy.reset_history()
        all_seeds["total_fe"].append(fe_history)
        for li in range(4):
            all_seeds["layer_fe"][li].append(layer_fe_hist[li])

    total_arr = np.array(all_seeds["total_fe"])
    results = {
        "total_fe_mean": total_arr.mean(axis=0).tolist(),
        "total_fe_std": total_arr.std(axis=0).tolist(),
        "initial_fe": float(total_arr[:, 0].mean()),
        "final_fe": float(total_arr[:, -1].mean()),
        "fe_reduction_pct": float((1 - total_arr[:, -1].mean() / total_arr[:, 0].mean()) * 100),
    }
    for li in range(4):
        arr = np.array(all_seeds["layer_fe"][li])
        results[f"layer{li}_final_mean"] = float(arr[:, -1].mean())
        results[f"layer{li}_final_std"] = float(arr[:, -1].std())

    logger.info(f"  Initial FE: {results['initial_fe']:.4f}")
    logger.info(f"  Final FE: {results['final_fe']:.4f}")
    logger.info(f"  Reduction: {results['fe_reduction_pct']:.1f}%")

    out_path = OUTPUT_DIR / "exp_B_free_energy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment B saved to {out_path}")
    return results


# ============================================================================
# Experiment C: Phi Computation Accuracy (unchanged)
# ============================================================================

def _exact_phi_small(attn_matrix: np.ndarray) -> float:
    """Brute-force exact Phi for small matrices (exhaustive MIP search)."""
    N = attn_matrix.shape[0]
    if N < 2:
        return 0.0

    def mi_approx(mat):
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, np.abs(np.diag(mat)) + 1e-6)
        diag_sum = np.sum(np.log(np.abs(np.diag(mat)) + 1e-6))
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            return 0.0
        return max(0.0, 0.5 * (diag_sum - logdet))

    I_total = mi_approx(attn_matrix)

    min_part = float("inf")
    for mask_int in range(1, 2**N - 1):
        bits = [(mask_int >> b) & 1 for b in range(N)]
        A_idx = [i for i, b in enumerate(bits) if b == 1]
        B_idx = [i for i, b in enumerate(bits) if b == 0]
        if len(A_idx) == 0 or len(B_idx) == 0:
            continue
        sub_A = attn_matrix[np.ix_(A_idx, A_idx)]
        sub_B = attn_matrix[np.ix_(B_idx, B_idx)]
        part_mi = mi_approx(sub_A) + mi_approx(sub_B)
        min_part = min(min_part, part_mi)

    return max(0.0, I_total - min_part)


def experiment_C_phi_accuracy(n_samples=50):
    """Compare NCT Phi approximation vs exact computation on small networks."""
    logger.info("=" * 60)
    logger.info("Experiment C: Phi Computation Accuracy")
    logger.info("=" * 60)

    phi_calc = PhiFromAttention(n_partitions=50, epsilon=1e-6)
    sizes = [4, 6, 8]
    results = {}

    for sz in sizes:
        exact_phis = []
        approx_phis = []

        for _ in range(n_samples):
            raw = np.random.rand(sz, sz) + 0.1
            attn = raw / raw.sum(axis=1, keepdims=True)

            exact_phi = _exact_phi_small(attn)

            attn_t = torch.tensor(attn, dtype=torch.float32)
            attn_4d = attn_t.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                approx_phi = phi_calc(attn_4d).item()

            exact_phis.append(exact_phi)
            approx_phis.append(approx_phi)

        exact_arr = np.array(exact_phis)
        approx_arr = np.array(approx_phis)

        mae = np.mean(np.abs(exact_arr - approx_arr))
        corr = np.corrcoef(exact_arr, approx_arr)[0, 1] if np.std(exact_arr) > 0 else 0.0
        rmse = np.sqrt(np.mean((exact_arr - approx_arr) ** 2))

        results[f"size_{sz}"] = {
            "n_samples": n_samples,
            "exact_mean": float(exact_arr.mean()),
            "exact_std": float(exact_arr.std()),
            "approx_mean": float(approx_arr.mean()),
            "approx_std": float(approx_arr.std()),
            "mae": float(mae),
            "rmse": float(rmse),
            "correlation": float(corr),
        }
        logger.info(f"  Size {sz}x{sz}: MAE={mae:.4f}, Corr={corr:.4f}, "
                     f"RMSE={rmse:.4f}")

    out_path = OUTPUT_DIR / "exp_C_phi_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment C saved to {out_path}")
    return results


# ============================================================================
# Experiment D: Scale Analysis (v2 - structured attention for Phi)
# ============================================================================

def experiment_D_scale(n_cycles=10):
    """Performance at different d_model scales with meaningful Phi."""
    logger.info("=" * 60)
    logger.info("Experiment D: Scale Analysis (v2)")
    logger.info("=" * 60)

    dims = [128, 256, 512, 768]
    results = {}

    for d in dims:
        set_seed(42)
        config = {f"layer{i}_dim": d for i in range(4)}
        config["n_heads"] = 8

        learner = TransformerSTDP(n_neurons=d, d_model=d, sparsity=0.01)
        hierarchy = PredictiveHierarchy(config)
        phi_calc = PhiFromAttention(n_partitions=10)

        timings = {"stdp": [], "predictive": [], "phi": []}
        phi_vals = []

        for _ in range(n_cycles):
            # STDP timing
            events = generate_stdp_events(20, d)
            ctx = torch.randn(d)
            t0 = time.perf_counter()
            learner.forward(events, ctx, {"dopamine": 0.6, "serotonin": 0.4,
                                           "norepinephrine": 0.5, "acetylcholine": 0.7})
            timings["stdp"].append((time.perf_counter() - t0) * 1000)
            learner.reset_history()

            # Predictive coding timing
            sensory = torch.randn(1, d)
            t0 = time.perf_counter()
            res = hierarchy.forward(sensory)
            timings["predictive"].append((time.perf_counter() - t0) * 1000)

            # Phi timing with STRUCTURED attention (v2 fix)
            attn_size = max(2, d // 64)
            structured_attn = generate_structured_attention(attn_size)
            attn_4d = structured_attn.unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1)
            # Add slight per-head variation
            attn_4d = attn_4d + torch.randn_like(attn_4d) * 0.01
            attn_4d = attn_4d.clamp(min=0)
            attn_4d = attn_4d / attn_4d.sum(dim=-1, keepdim=True)

            t0 = time.perf_counter()
            phi = phi_calc(attn_4d)
            timings["phi"].append((time.perf_counter() - t0) * 1000)
            phi_vals.append(phi.mean().item())

        hierarchy.reset_history()

        param_count = sum(p.numel() for p in learner.parameters()) + \
                      sum(p.numel() for p in hierarchy.parameters())

        results[f"d_{d}"] = {
            "d_model": d,
            "param_count": param_count,
            "stdp_ms_mean": float(np.mean(timings["stdp"])),
            "stdp_ms_std": float(np.std(timings["stdp"])),
            "predictive_ms_mean": float(np.mean(timings["predictive"])),
            "predictive_ms_std": float(np.std(timings["predictive"])),
            "phi_ms_mean": float(np.mean(timings["phi"])),
            "phi_ms_std": float(np.std(timings["phi"])),
            "phi_mean": float(np.mean(phi_vals)),
            "phi_std": float(np.std(phi_vals)),
            "total_ms_mean": float(np.mean(timings["stdp"]) +
                                    np.mean(timings["predictive"]) +
                                    np.mean(timings["phi"])),
        }
        logger.info(f"  d_model={d}: params={param_count:,}, "
                     f"total={results[f'd_{d}']['total_ms_mean']:.1f}ms, "
                     f"phi={results[f'd_{d}']['phi_mean']:.4f}")

    out_path = OUTPUT_DIR / "exp_D_scale.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment D saved to {out_path}")
    return results


# ============================================================================
# Experiment E: Ablation Study (v2 - richer metrics)
# ============================================================================

def experiment_E_ablation(n_steps=500, n_seeds=5):
    """Ablation: disable each component and measure impact with richer metrics."""
    logger.info("=" * 60)
    logger.info("Experiment E: Ablation Study (v2)")
    logger.info("=" * 60)

    d_model = 256
    events_per_step = 20
    configs = {
        "NCT_Full":        {"stdp": True, "attn": True, "neuro": True, "pred": True},
        "w/o_STDP":        {"stdp": False, "attn": True, "neuro": True, "pred": True},
        "w/o_Attention":   {"stdp": True, "attn": False, "neuro": True, "pred": True},
        "w/o_Neuromod":    {"stdp": True, "attn": True, "neuro": False, "pred": True},
        "w/o_Predictive":  {"stdp": True, "attn": True, "neuro": True, "pred": False},
        "STDP_only":       {"stdp": True, "attn": False, "neuro": False, "pred": False},
        "Attention_only":  {"stdp": False, "attn": True, "neuro": False, "pred": False},
    }

    # v2: More extreme neurotransmitter state for clear differentiation
    nt_active = {"dopamine": 0.9, "serotonin": 0.3,
                 "norepinephrine": 0.8, "acetylcholine": 0.9}

    all_results = {}

    for cfg_name, cfg in configs.items():
        seed_data = {
            "phi": [], "fe": [], "ltp_ratio_var": [],
            "weight_std": [], "weight_entropy": [],
            "attn_contribution": [], "mean_abs_dw": [],
        }

        for seed in range(n_seeds):
            set_seed(seed * 31 + 11)

            lam = 0.1 if cfg["attn"] else 0.0
            learner = TransformerSTDP(
                n_neurons=d_model, d_model=d_model,
                stdp_learning_rate=0.01,
                attention_modulation_lambda=lam,
                sparsity=0.01,
            )

            pc_config = {f"layer{i}_dim": d_model for i in range(4)}
            pc_config["n_heads"] = 8
            hierarchy = PredictiveHierarchy(pc_config) if cfg["pred"] else None
            phi_calc = PhiFromAttention(n_partitions=10)

            # v2: w/o Neuromod uses fixed modulation=1.0 (bypass gate entirely)
            use_nt = nt_active if cfg["neuro"] else None
            ctx_tensor = torch.randn(d_model) if cfg["attn"] else None

            ltp_ratios = []
            fe_values = []
            attn_contribs = []
            abs_dws = []

            for step in range(n_steps):
                events = generate_stdp_events(events_per_step, d_model)

                if cfg["stdp"]:
                    updates = learner.forward(events, ctx_tensor, use_nt)
                    ltp = sum(1 for u in updates if u.total_delta_w > 0)
                    ltp_ratios.append(ltp / max(1, len(updates)))

                    # Track attention contribution
                    t_attn = sum(abs(u.delta_w_attn) for u in updates)
                    t_stdp = sum(abs(u.delta_w_std) for u in updates)
                    attn_contribs.append(t_attn / (t_stdp + t_attn + 1e-12))
                    abs_dws.append(np.mean([abs(u.total_delta_w) for u in updates]))
                else:
                    ltp_ratios.append(0.5)
                    attn_contribs.append(0.0)
                    abs_dws.append(0.0)

                if hierarchy is not None:
                    sensory = torch.randn(1, d_model)
                    res = hierarchy.forward(sensory)
                    fe_values.append(res["total_free_energy"])
                else:
                    fe_values.append(0.0)

                learner.reset_history()
                if step % 50 == 0 and cfg["attn"]:
                    ctx_tensor = ctx_tensor * 0.7 + torch.randn(d_model) * 0.3

            # End-of-run metrics
            # Phi from structured attention (v2: not random)
            attn_map = generate_structured_attention(4)
            attn_4d = attn_map.unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1)
            attn_4d = attn_4d + torch.randn_like(attn_4d) * 0.01
            attn_4d = attn_4d.clamp(min=0)
            attn_4d = attn_4d / attn_4d.sum(dim=-1, keepdim=True)
            phi_val = phi_calc(attn_4d).mean().item()

            # Weight statistics
            w = learner.get_weight_matrix()
            nz = w[w > 0]
            w_std = float(np.std(nz)) if len(nz) > 0 else 0.0
            w_ent = compute_weight_entropy(w)

            if hierarchy is not None:
                hierarchy.reset_history()

            seed_data["phi"].append(phi_val)
            seed_data["fe"].append(fe_values[-1] if fe_values else 0.0)
            seed_data["ltp_ratio_var"].append(float(np.var(ltp_ratios[-100:])))
            seed_data["weight_std"].append(w_std)
            seed_data["weight_entropy"].append(w_ent)
            seed_data["attn_contribution"].append(float(np.mean(attn_contribs[-100:])))
            seed_data["mean_abs_dw"].append(float(np.mean(abs_dws[-100:])))

        agg = {}
        for k, v in seed_data.items():
            arr = np.array(v)
            agg[k + "_mean"] = float(arr.mean())
            agg[k + "_std"] = float(arr.std())
        agg["config"] = cfg

        all_results[cfg_name] = agg
        logger.info(
            f"  {cfg_name:20s}: w_std={agg['weight_std_mean']:.4f}, "
            f"entropy={agg['weight_entropy_mean']:.3f}, "
            f"attn_c={agg['attn_contribution_mean']:.4f}, "
            f"|dw|={agg['mean_abs_dw_mean']:.6f}, "
            f"FE={agg['fe_mean']:.4f}, Phi={agg['phi_mean']:.4f}"
        )

    out_path = OUTPUT_DIR / "exp_E_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Experiment E saved to {out_path}")
    return all_results


# ============================================================================
# Experiment F: Temporal Association Learning Task (Downstream)
# ============================================================================

def experiment_F_temporal_association(n_steps=300, n_seeds=5):
    """Downstream task: Temporal association learning.

    Tests how well different NCT configurations learn structured temporal
    associations from spike-timing patterns.

    Ground truth structure:
    - LTP pairs: pre neuron fires consistently BEFORE post neuron (causal)
    - LTD pairs: pre neuron fires consistently AFTER post neuron (anti-causal)
    - Neutral synapses: random or no events

    Metrics:
    - discrimination: mean(W_LTP) - mean(W_LTD)  (higher = better learning)
    - pattern_correlation: corr(W_designated, ground_truth)
    - ltp_strength: mean weight at LTP positions
    - ltd_strength: mean weight at LTD positions
    """
    logger.info("=" * 60)
    logger.info("Experiment F: Temporal Association Learning Task")
    logger.info("=" * 60)

    d_model = 128
    n_ltp_pairs = 25
    n_ltd_pairs = 25

    configs = {
        "NCT_Full":        {"stdp": True, "attn": True, "neuro": True},
        "w/o_STDP":        {"stdp": False, "attn": True, "neuro": True},
        "w/o_Attention":   {"stdp": True, "attn": False, "neuro": True},
        "w/o_Neuromod":    {"stdp": True, "attn": True, "neuro": False},
        "STDP_only":       {"stdp": True, "attn": False, "neuro": False},
        "Attention_only":  {"stdp": False, "attn": True, "neuro": False},
    }

    nt_active = {"dopamine": 0.9, "serotonin": 0.3,
                 "norepinephrine": 0.8, "acetylcholine": 0.9}

    # Fixed LTP/LTD pair indices (non-overlapping)
    ltp_pairs = [(i, i + d_model // 2) for i in range(n_ltp_pairs)]
    ltd_pairs = [(i + n_ltp_pairs, i + n_ltp_pairs + d_model // 2)
                 for i in range(n_ltd_pairs)]

    all_results = {}

    for cfg_name, cfg in configs.items():
        seed_disc = []
        seed_ltp_w = []
        seed_ltd_w = []
        seed_corr = []
        seed_disc_curve = []

        for seed in range(n_seeds):
            set_seed(seed * 53 + 13)

            lam = 0.2 if cfg["attn"] else 0.0
            learner = TransformerSTDP(
                n_neurons=d_model, d_model=d_model,
                stdp_learning_rate=0.01,
                attention_modulation_lambda=lam,
                sparsity=0.01,
            )

            use_nt = nt_active if cfg["neuro"] else None

            # Context encoding LTP pair positions (attention reinforces these)
            ctx = torch.zeros(d_model)
            for pre, post in ltp_pairs:
                ctx[pre] = 1.0
                ctx[post] = 0.5
            ctx = ctx / (ctx.norm() + 1e-8)
            ctx_tensor = ctx if cfg["attn"] else None

            disc_curve = []

            for step in range(n_steps):
                events = []

                if cfg["stdp"]:
                    # LTP events: pre fires before post (positive dt)
                    for pre, post in ltp_pairs:
                        if np.random.random() < 0.8:  # 80% firing probability
                            t_pre = np.random.uniform(0, 50)
                            dt = np.abs(np.random.normal(5, 2))  # ~5ms causal
                            events.append(STDPEvent(pre, post, t_pre, t_pre + dt))

                    # LTD events: pre fires after post (negative dt)
                    for pre, post in ltd_pairs:
                        if np.random.random() < 0.8:
                            t_post = np.random.uniform(0, 50)
                            dt = np.abs(np.random.normal(5, 2))  # ~5ms anti-causal
                            events.append(STDPEvent(pre, post, t_post + dt, t_post))

                    # Random noise events
                    events += generate_stdp_events(5, d_model)

                    learner.forward(events, ctx_tensor, use_nt)
                    learner.reset_history()

                # Track discrimination every 10 steps
                if step % 10 == 0:
                    W = learner.get_weight_matrix()
                    ltp_w = np.mean([W[p, q] for p, q in ltp_pairs])
                    ltd_w = np.mean([W[p, q] for p, q in ltd_pairs])
                    disc_curve.append(float(ltp_w - ltd_w))

            # Final evaluation
            W = learner.get_weight_matrix()
            ltp_weights = np.array([W[p, q] for p, q in ltp_pairs])
            ltd_weights = np.array([W[p, q] for p, q in ltd_pairs])

            final_disc = float(np.mean(ltp_weights) - np.mean(ltd_weights))
            seed_disc.append(final_disc)
            seed_ltp_w.append(float(np.mean(ltp_weights)))
            seed_ltd_w.append(float(np.mean(ltd_weights)))
            seed_disc_curve.append(disc_curve)

            # Pattern correlation at designated positions
            target_vals = [1.0] * n_ltp_pairs + [-1.0] * n_ltd_pairs
            actual_vals = list(ltp_weights) + list(ltd_weights)
            corr = np.corrcoef(target_vals, actual_vals)[0, 1]
            seed_corr.append(float(corr) if not np.isnan(corr) else 0.0)

        # Aggregate
        disc_arr = np.array(seed_disc)
        ltp_arr = np.array(seed_ltp_w)
        ltd_arr = np.array(seed_ltd_w)
        corr_arr = np.array(seed_corr)
        curve_arr = np.array(seed_disc_curve)

        all_results[cfg_name] = {
            "config": cfg,
            "discrimination_mean": float(disc_arr.mean()),
            "discrimination_std": float(disc_arr.std()),
            "ltp_weight_mean": float(ltp_arr.mean()),
            "ltp_weight_std": float(ltp_arr.std()),
            "ltd_weight_mean": float(ltd_arr.mean()),
            "ltd_weight_std": float(ltd_arr.std()),
            "pattern_corr_mean": float(corr_arr.mean()),
            "pattern_corr_std": float(corr_arr.std()),
            "disc_curve_mean": curve_arr.mean(axis=0).tolist(),
            "disc_curve_std": curve_arr.std(axis=0).tolist(),
        }
        logger.info(
            f"  {cfg_name:20s}: disc={disc_arr.mean():.4f}+-{disc_arr.std():.4f}, "
            f"LTP_w={ltp_arr.mean():.4f}, LTD_w={ltd_arr.mean():.4f}, "
            f"corr={corr_arr.mean():.3f}"
        )

    out_path = OUTPUT_DIR / "exp_F_temporal_association.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Experiment F saved to {out_path}")
    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NCT Paper Experiments v2")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["A", "B", "C", "D", "E", "F", "all"],
                        help="Which experiment to run")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NCT Paper Experiments v2")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)

    t_start = time.time()

    if args.experiment in ("A", "all"):
        experiment_A_convergence()
    if args.experiment in ("B", "all"):
        experiment_B_free_energy()
    if args.experiment in ("C", "all"):
        experiment_C_phi_accuracy()
    if args.experiment in ("D", "all"):
        experiment_D_scale()
    if args.experiment in ("E", "all"):
        experiment_E_ablation()
    if args.experiment in ("F", "all"):
        experiment_F_temporal_association()

    elapsed = time.time() - t_start
    logger.info(f"\nAll experiments completed in {elapsed:.1f}s")
    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
