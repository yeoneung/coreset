"""Train the WSD-style conditional diagonal Gaussian anchor.

The score network is not touched.  A small deterministic context network is
trained by Gaussian NLL to predict the mean and marginal variance of the clean
trajectory residual relative to the fixed nominal.  Splitting by ``inst_id``
prevents trajectories from the same planning condition entering both training
and validation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from nab.covariance import ConditionalDiagonalAnchor, condition_features  # noqa: E402


def subset(data, mask):
    return {
        key: value[mask]
        for key, value in data.items()
        if torch.is_tensor(value) and value.shape[:1] == mask.shape
    }


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for lo in range(0, n, batch_size):
        yield order[lo : lo + batch_size]


@torch.no_grad()
def evaluate(model, data, residuals, batch_size):
    total, count = 0.0, 0
    for lo in range(0, residuals.shape[0], batch_size):
        hi = min(lo + batch_size, residuals.shape[0])
        cond = {key: value[lo:hi] for key, value in data.items()}
        loss = model.clean_nll(cond, residuals[lo:hi])
        total += float(loss) * (hi - lo)
        count += hi - lo
    return total / count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "wsd_diag.pt",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "wsd_training.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    raw = torch.load(
        PROJECT_ROOT / "artifacts" / "data" / "train.pt",
        map_location="cpu",
        weights_only=True,
    )
    # A deterministic condition-level split: no multimodal siblings leak.
    validation_mask = raw["inst_id"].remainder(10).eq(0)
    training_mask = ~validation_mask
    train = subset(raw, training_mask)
    validation = subset(raw, validation_mask)
    train_residuals = train["traj"] - train["nominal"]
    validation_residuals = validation["traj"] - validation["nominal"]

    features = condition_features(train)
    input_mean = features.mean(0)
    input_scale = features.std(0).clamp_min(1e-3)
    residual_scale = train_residuals.flatten(1).square().mean(0).sqrt().clamp_min(1e-3)
    model = ConditionalDiagonalAnchor(
        input_mean=input_mean,
        input_scale=input_scale,
        residual_scale=residual_scale,
        sample_shape=tuple(train_residuals.shape[1:]),
        hidden=args.hidden,
    ).to(device)
    train = {key: value.to(device) for key, value in train.items()}
    validation = {key: value.to(device) for key, value in validation.items()}
    train_residuals = train_residuals.to(device)
    validation_residuals = validation_residuals.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    order_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    best_state = None
    best_validation = float("inf")
    best_epoch = -1
    training_log = []
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        running, count = 0.0, 0
        for index_cpu in batches(train_residuals.shape[0], args.batch_size, order_generator):
            index = index_cpu.to(device)
            cond = {key: value[index] for key, value in train.items()}
            residual = train_residuals[index]
            optimizer.zero_grad(set_to_none=True)
            loss = model.clean_nll(cond, residual)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * index.numel()
            count += index.numel()
        scheduler.step()
        model.eval()
        validation_nll = evaluate(
            model, validation, validation_residuals, args.batch_size
        )
        training_nll = running / count
        training_log.append(
            {
                "epoch": epoch + 1,
                "train_nll_per_dim": training_nll,
                "validation_nll_per_dim": validation_nll,
            }
        )
        if validation_nll < best_validation - 1e-6:
            best_validation = validation_nll
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"epoch={epoch + 1:4d} train={training_nll:.6f} "
                f"validation={validation_nll:.6f} best={best_validation:.6f}",
                flush=True,
            )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_training = evaluate(model, train, train_residuals, args.batch_size)
    final_validation = evaluate(
        model, validation, validation_residuals, args.batch_size
    )
    checkpoint = model.checkpoint()
    checkpoint["training"] = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "train_nll_per_dim": final_training,
        "validation_nll_per_dim": final_validation,
        "n_train": int(train_residuals.shape[0]),
        "n_validation": int(validation_residuals.shape[0]),
        "split": "inst_id modulo 10",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "metadata": {
            "experiment": "WSD-style conditional diagonal Gaussian anchor",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "seed": args.seed,
            "hidden": args.hidden,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
        "best_epoch": best_epoch,
        "train_nll_per_dim": final_training,
        "validation_nll_per_dim": final_validation,
        "training_log": training_log,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.metrics}")


if __name__ == "__main__":
    main()
