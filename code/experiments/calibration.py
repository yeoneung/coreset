"""Condition-level calibration of the truncation certificate.

For every held-out condition c the residual second moment around the fixed
nominal is estimated from that condition's oracle modes, giving the
per-condition log-determinant certificate
B_s(c) = 0.5 * logdet(I + rho_s M_c) and the trace certificate
T_s(c) = 0.5 * rho_s * tr M_c.  The same frozen score model is then
warm-started at several truncation points from two Gaussian sources:

- ``isotropic``: the mean-anchored VP source with covariance (1-a) I, whose
  coupling certificate is the trace quantity T_s(c) (Remark on the fixed
  diffusion covariance);
- ``matched``: the per-condition covariance-matched source with covariance
  (1-a) I + a M_c, whose certificate is B_s(c).

Per-condition mode recall under each source is compared with a full-chain
reference, so each sampler is calibrated against the certificate that
actually applies to it.

Outputs results/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from nab import env  # noqa: E402
from nab import eval as eval_mod  # noqa: E402
from nab.diffusion import Diffusion  # noqa: E402
from nab.model import UNet1D  # noqa: E402


SAMPLES_PER_CONDITION = 16


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (avoids a scipy dependency)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0])
    sorted_values = values[order]
    base = np.arange(1, values.shape[0] + 1, dtype=float)
    i = 0
    while i < values.shape[0]:
        j = i
        while j + 1 < values.shape[0] and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = base[i : j + 1].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x) - rankdata(x).mean()
    ry = rankdata(y) - rankdata(y).mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def load_model(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UNet1D(use_nominal=ck["use_nominal"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, Diffusion(T=ck["T"], shift=ck["shift"], device=str(device))


def per_condition_eigensystems(raw, instance_order):
    """Eigenvalues and trajectory-space eigenvectors of each condition's
    residual second moment M_c.

    With n_c oracle modes the moment has rank at most n_c, so the nonzero
    spectrum comes from the n_c x n_c Gram matrix G = B B^T / n_c; the
    trajectory-space eigenvector for (lambda, u) is B^T u / sqrt(n_c lambda).
    """
    residuals = (raw["traj"] - raw["nominal"]).double().flatten(1).cpu()
    groups: dict[int, list[int]] = {}
    for row, inst_id in enumerate(raw["inst_id"].tolist()):
        groups.setdefault(inst_id, []).append(row)
    systems = []
    for inst_id in instance_order:
        block = residuals[groups[inst_id]]
        gram = block @ block.T / block.shape[0]
        lam, u = torch.linalg.eigh(gram)
        lam = lam.clamp_min(0.0)
        denom = (lam * block.shape[0]).sqrt().clamp_min(1e-12)
        vec = block.T @ u / denom
        systems.append((lam, vec))
    return systems


def matched_perturbation(systems, alpha_bar, s, sample_shape, generator, device):
    """Zero-mean source perturbation with covariance (1-a) I + a M_c,
    ordered to match repeat_interleave(s) over conditions."""
    a = float(alpha_bar)
    d = int(np.prod(sample_shape))
    n_total = len(systems) * s
    out = math.sqrt(1.0 - a) * torch.randn(
        n_total, d, generator=generator, device=device, dtype=torch.float32
    )
    blocks = []
    for lam, vec in systems:
        scale = lam.clamp_min(0.0).sqrt().to(device=device, dtype=torch.float32)
        v = vec.to(device=device, dtype=torch.float32)
        z = torch.randn(s, lam.shape[0], generator=generator, device=device,
                        dtype=torch.float32)
        blocks.append(math.sqrt(a) * (z * scale) @ v.T)
    out = out + torch.cat(blocks, dim=0)
    return out.reshape((n_total,) + tuple(sample_shape))


def per_condition_recall(x, inst, s):
    n_inst = inst["start"].shape[0]
    cond = eval_mod.rep_cond(inst, s=s)
    feasible = env.feasible(x, cond["obs"], cond["mask"], tol=1e-2).view(n_inst, s)
    signature = env.signature(
        x, cond["start"], cond["goal"], cond["obs"], cond["mask"]
    ).view(n_inst, s, -1)
    recall = np.zeros(n_inst)
    for i in range(n_inst):
        oracle = inst["modes"][i]
        sampled = {
            tuple(signature[i, j].tolist())
            for j in range(s)
            if feasible[i, j]
        }
        recall[i] = len(oracle & sampled) / len(oracle)
    return recall


@torch.no_grad()
def sample_run(model, diffusion, cond, nfe, t_start, anchor, seed, chunk,
               device, perturbation=None):
    gen = torch.Generator(device=device).manual_seed(20_000 + seed)
    outputs = []
    n = cond["start"].shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sub = {k: v[lo:hi] for k, v in cond.items()}
        outputs.append(
            diffusion.sample(
                model, sub, n_steps=nfe, eta=0.0,
                t_start=t_start, anchor=anchor, gen=gen,
                source_perturbation=None if perturbation is None
                else perturbation[lo:hi],
            )
        )
    return torch.cat(outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfe", type=int, default=16,
                        help="Warm grid-interval count, not the actual score-call count.")
    parser.add_argument("--nfe-ref", type=int, default=128,
                        help="Reference grid-interval count; includes an extra final score call.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--t-starts", type=int, nargs="+", default=[64, 123, 192, 234])
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "calibration.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_mod.DEVICE = str(device)
    eval_mod.S = SAMPLES_PER_CONDITION
    eval_mod.CHUNK = args.chunk

    test_path = PROJECT_ROOT / "artifacts" / "data" / "test.pt"
    checkpoint = PROJECT_ROOT / "artifacts" / "checkpoints" / "base_nom.pt"
    model, diffusion = load_model(checkpoint, device)
    inst, raw = eval_mod.load_instances(str(test_path))
    cond = eval_mod.rep_cond(inst, s=SAMPLES_PER_CONDITION)
    n_inst = inst["start"].shape[0]
    sample_shape = tuple(inst["nominal"].shape[1:])

    # instance order used by load_instances: first occurrence in file order
    seen, instance_order = set(), []
    for inst_id in raw["inst_id"].tolist():
        if inst_id not in seen:
            seen.add(inst_id)
            instance_order.append(inst_id)
    assert len(instance_order) == n_inst

    systems = per_condition_eigensystems(raw, instance_order)
    eigenvalues = [lam.numpy() for lam, _ in systems]
    n_modes = np.array([len(m) for m in inst["modes"]])

    certificates = {}
    for t_start in args.t_starts:
        a = float(diffusion.abar[t_start])
        rho = a / (1.0 - a)
        certificates[str(t_start)] = {
            "abar": a,
            "logdet": [
                float(0.5 * np.log1p(rho * lam).sum()) for lam in eigenvalues
            ],
            "trace": [float(0.5 * rho * lam.sum()) for lam in eigenvalues],
        }

    reference_runs = np.stack([
        per_condition_recall(
            sample_run(model, diffusion, cond, args.nfe_ref, None, "none",
                       seed, args.chunk, device),
            inst, SAMPLES_PER_CONDITION,
        )
        for seed in range(args.seeds)
    ])
    print("reference recall", reference_runs.mean(), flush=True)

    warm = {"isotropic": {}, "matched": {}}
    for t_start in args.t_starts:
        alpha_bar = float(diffusion.abar[t_start])
        for source in ("isotropic", "matched"):
            runs = []
            for seed in range(args.seeds):
                if source == "matched":
                    source_gen = torch.Generator(device=device).manual_seed(
                        10_000 + seed
                    )
                    perturbation = matched_perturbation(
                        systems, alpha_bar, SAMPLES_PER_CONDITION,
                        sample_shape, source_gen, device,
                    )
                else:
                    perturbation = None
                runs.append(per_condition_recall(
                    sample_run(model, diffusion, cond, args.nfe, t_start,
                               "mean", seed, args.chunk, device,
                               perturbation=perturbation),
                    inst, SAMPLES_PER_CONDITION,
                ))
            warm[source][str(t_start)] = np.stack(runs)
            print("t_start", t_start, source, "recall",
                  warm[source][str(t_start)].mean(), flush=True)

    reference_mean = reference_runs.mean(axis=0)
    multimodal = n_modes > 1
    correlations = {}
    per_condition = {
        "n_modes": n_modes.tolist(),
        "recall_reference_mean": reference_mean.tolist(),
        "warm": {},
    }
    for t_start in args.t_starts:
        key = str(t_start)
        correlations[key] = {}
        per_condition["warm"][key] = {}
        for source in ("isotropic", "matched"):
            warm_mean = warm[source][key].mean(axis=0)
            loss = reference_mean - warm_mean
            entry = {"mean_loss_mm": float(loss[multimodal].mean()),
                     "mean_loss_all": float(loss.mean())}
            for cert in ("trace", "logdet"):
                b = np.array(certificates[key][cert])
                entry[f"spearman_{cert}_mm"] = spearman(
                    b[multimodal], loss[multimodal]
                )
                entry[f"spearman_{cert}_all"] = spearman(b, loss)
            correlations[key][source] = entry
            per_condition["warm"][key][source] = {
                "recall_mean": warm_mean.tolist(),
                "recall_std": warm[source][key].std(axis=0).tolist(),
            }
            print(key, source, entry, flush=True)

    payload = {
        "metadata": {
            "experiment": "condition-level certificate calibration",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "checkpoint": "artifacts/checkpoints/base_nom.pt",
            "sources": {
                "isotropic": "mean-anchored VP, covariance (1-a)I; "
                             "certificate T_s(c) = 0.5 rho_s tr M_c",
                "matched": "mean-anchored, covariance (1-a)I + a M_c; "
                           "certificate B_s(c) = 0.5 logdet(I + rho_s M_c)",
            },
            "nfe_warm": args.nfe,
            "nfe_reference": args.nfe_ref,
            "nfe_semantics": "Interpolation-interval settings",
            "actual_score_nfe_warm_by_start": {
                str(t): min(args.nfe + 1, t + 1) for t in args.t_starts
            },
            "actual_score_nfe_reference": min(args.nfe_ref + 1, diffusion.T),
            "seeds": args.seeds,
            "samples_per_condition": SAMPLES_PER_CONDITION,
            "n_conditions": int(n_inst),
            "n_multimodal": int(multimodal.sum()),
            "t_starts": args.t_starts,
            "T": diffusion.T,
        },
        "certificates": certificates,
        "correlations": correlations,
        "per_condition": per_condition,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
