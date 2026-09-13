"""Gaussian-source warm-start ablation on obstacle-avoidance trajectories.

This experiment uses one fixed conditional VP score model and changes only the
Gaussian source supplied at an intermediate reverse time.  The residual
second-moment models are estimated from the training split; the WSD-style
baseline predicts condition-dependent diagonal moments.  All sources are
evaluated on a held-out split.  Outputs go to covariance_ablation.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from nab import eval as eval_mod  # noqa: E402
from nab.covariance import ConditionalDiagonalAnchor, ResidualMomentModel  # noqa: E402
from nab.diffusion import Diffusion  # noqa: E402
from nab.model import UNet1D  # noqa: E402


METHODS = ("vp", "diagonal", "wsd", "lowrank", "full")
METHOD_LABELS = {
    "vp": "VP isotropic",
    "diagonal": "pooled diagonal",
    "wsd": "WSD-style cond. diagonal",
    "lowrank": "rank-4",
    "full": "full",
}
SAMPLES_PER_CONDITION = 16


def load_model(checkpoint: Path, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UNet1D(use_nominal=ck["use_nominal"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diffusion = Diffusion(T=ck["T"], shift=ck["shift"], device=str(device))
    return model, diffusion, ck


def summarize(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in runs[0]:
        values = [float(run[key]) for run in runs]
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


@torch.no_grad()
def sample_configuration(
    model,
    diffusion,
    conditions,
    moment: ResidualMomentModel,
    wsd_anchor: ConditionalDiagonalAnchor,
    method: str,
    t_start: int,
    nfe: int,
    seed: int,
    rank: int,
    chunk: int,
    device: torch.device,
):
    source_generator = torch.Generator(device=device).manual_seed(10_000 + seed)
    reverse_generator = torch.Generator(device=device).manual_seed(20_000 + seed)
    alpha_bar = float(diffusion.abar[t_start])
    outputs = []
    n = conditions["start"].shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sub = {key: value[lo:hi] for key, value in conditions.items()}
        if method == "wsd":
            predicted_residual, _ = wsd_anchor(sub)
            perturbation = wsd_anchor.sample_perturbation(
                sub, alpha_bar, generator=source_generator
            )
            anchor_traj = sub["nominal"] + predicted_residual
        else:
            perturbation = moment.sample_perturbation(
                method,
                alpha_bar,
                hi - lo,
                device=device,
                generator=source_generator,
                rank=rank,
                dtype=sub["nominal"].dtype,
            )
            anchor_traj = None
        outputs.append(
            diffusion.sample(
                model,
                sub,
                n_steps=nfe,
                eta=0.0,
                t_start=t_start,
                anchor="mean",
                gen=reverse_generator,
                anchor_traj=anchor_traj,
                source_perturbation=perturbation,
            )
        )
    return torch.cat(outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfe", type=int, default=4,
                        help="Grid-interval count; score calls are min(nfe+1, t_start+1).")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--t-starts", type=int, nargs="+", default=[64, 123, 192, 234])
    parser.add_argument(
        "--wsd-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "wsd_diag.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "covariance_ablation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    eval_mod.DEVICE = str(device)
    eval_mod.S = SAMPLES_PER_CONDITION
    eval_mod.CHUNK = args.chunk

    train_path = PROJECT_ROOT / "artifacts" / "data" / "train.pt"
    test_path = PROJECT_ROOT / "artifacts" / "data" / "test.pt"
    checkpoint = PROJECT_ROOT / "artifacts" / "checkpoints" / "base_nom.pt"
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    test_raw_cpu = torch.load(test_path, map_location="cpu", weights_only=True)
    train_residuals = train["traj"] - train["nominal"]
    test_residuals = test_raw_cpu["traj"] - test_raw_cpu["nominal"]
    moment = ResidualMomentModel.fit(train_residuals)

    model, diffusion, ck = load_model(checkpoint, device)
    wsd_checkpoint = torch.load(args.wsd_checkpoint, map_location="cpu", weights_only=True)
    wsd_anchor = ConditionalDiagonalAnchor.from_checkpoint(wsd_checkpoint, device)
    instances, _ = eval_mod.load_instances(str(test_path))
    conditions = eval_mod.rep_cond(instances, s=SAMPLES_PER_CONDITION)

    cross_entropy = {}
    reverse_sampling = {}
    for t_start in args.t_starts:
        alpha_bar = float(diffusion.abar[t_start])
        ce = {
            method: moment.heldout_cross_entropy(
                method, alpha_bar, test_residuals, rank=args.rank
            )
            for method in METHODS
            if method != "wsd"
        }
        test_conditions = {
            key: value.to(device)
            for key, value in test_raw_cpu.items()
            if key in ("start", "goal", "obs", "mask", "nominal")
        }
        ce["wsd"] = wsd_anchor.heldout_cross_entropy(
            test_conditions,
            alpha_bar,
            test_residuals,
        )
        cross_entropy[str(t_start)] = {
            method: {
                "nll": value,
                "nll_per_dim": value / moment.dimension,
                "excess_vs_full_per_dim": (value - ce["full"]) / moment.dimension,
            }
            for method, value in ce.items()
        }

        reverse_sampling[str(t_start)] = {}
        for method in METHODS:
            runs = []
            for seed in range(args.seeds):
                samples = sample_configuration(
                    model,
                    diffusion,
                    conditions,
                    moment,
                    wsd_anchor,
                    method,
                    t_start,
                    args.nfe,
                    seed,
                    args.rank,
                    args.chunk,
                    device,
                )
                metrics = eval_mod.metrics(samples, instances)
                runs.append(metrics)
                print(
                    f"t={t_start:3d} {METHOD_LABELS[method]:>12s} seed={seed} "
                    f"feas={metrics['feasible']:.3f} "
                    f"recall-mm={metrics['mode_recall_mm']:.3f} "
                    f"div={metrics['diversity']:.3f}",
                    flush=True,
                )
            reverse_sampling[str(t_start)][method] = {
                "label": METHOD_LABELS[method],
                "runs": runs,
                "summary": summarize(runs),
            }

    output = {
        "metadata": {
            "experiment": "Gaussian warm-start ablation with WSD-style conditional diagonal",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "checkpoint": "artifacts/checkpoints/base_nom.pt",
            "wsd_checkpoint": str(args.wsd_checkpoint.relative_to(PROJECT_ROOT)),
            "wsd_training": wsd_checkpoint.get("training", {}),
            "train_data": "artifacts/data/train.pt",
            "test_data": "artifacts/data/test.pt",
            "n_train_residuals": int(train_residuals.shape[0]),
            "n_test_residuals": int(test_residuals.shape[0]),
            "n_test_conditions": int(instances["start"].shape[0]),
            "samples_per_condition": SAMPLES_PER_CONDITION,
            "nfe": args.nfe,
            "nfe_semantics": "Interpolation-interval setting; actual score calls include clean reconstruction",
            "actual_score_nfe_by_start": {
                str(t): min(args.nfe + 1, t + 1) for t in args.t_starts
            },
            "cost_reference": "artifacts/data/test_ref.pt",
            "rank": args.rank,
            "seeds": list(range(args.seeds)),
            "t_starts": args.t_starts,
            "T": int(ck["T"]),
        },
        "spectrum": {
            "trace": moment.trace,
            "top_eigenvalues": [float(v) for v in moment.eigenvalues[:12]],
            "explained_fraction": moment.explained_fraction([1, 2, 4, 8, 16, 32]),
        },
        "source_cross_entropy": cross_entropy,
        "reverse_sampling": reverse_sampling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
