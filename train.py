#!/usr/bin/env python3
"""Train a KhaosChess net on packed data from prepare.py.

    python3 train.py data/train.bin --epochs 30 --out nets/run1

Writes nets/run1/epoch_N.pt checkpoints plus nets/run1/best.pt, and exports a
loadable nets/run1/best.nnue after every improvement so the net can be tested in
the engine while training continues.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khaosnnue import dataset as ds  # noqa: E402
from khaosnnue import format as netformat  # noqa: E402
from khaosnnue import model as netmodel  # noqa: E402
from khaosnnue import quant  # noqa: E402


def export_checkpoint(model, path):
    (feature_weights, feature_bias, l2_weights, l2_bias,
     output_weights, output_bias) = netmodel.quantized_tensors(model)

    worst = quant.accumulator_headroom(
        max(abs(w) for w in feature_weights),
        max(abs(b) for b in feature_bias),
    )
    if worst > 32767:
        raise SystemExit(
            f"refusing to export: worst-case accumulator {worst} exceeds int16 "
            f"(32767). Lower quant.FEATURE_WEIGHT_CLAMP and retrain."
        )

    netformat.write_net(path, feature_weights, feature_bias, l2_weights,
                        l2_bias, output_weights, output_bias)
    return worst


def run_epoch(model, loader, optimizer, device, blend, train):
    model.train(train)
    total_loss = 0.0
    total_samples = 0

    for own, their, scores, results in loader:
        own = own.to(device, non_blocking=True)
        their = their.to(device, non_blocking=True)
        scores = scores.to(device, non_blocking=True)
        results = results.to(device, non_blocking=True)

        target = ds.make_target(scores, results, blend)

        with torch.set_grad_enabled(train):
            predicted = torch.sigmoid(model(own, their))
            loss = torch.nn.functional.mse_loss(predicted, target)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Enforce the int16 accumulator budget after every step.
            model.clamp_weights()

        total_loss += loss.item() * own.shape[0]
        total_samples += own.shape[0]

    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="packed .bin from prepare.py")
    parser.add_argument("--out", default="nets/run", help="output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--blend", type=float, default=0.7,
                        help="weight on the search score vs the game result "
                             "(1.0 = score only, 0.0 = result only)")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--resume", help="checkpoint to continue from")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"device: {device}")

    full = ds.PackedDataset(args.data)
    val_size = max(1, int(len(full) * args.val_fraction))
    train_size = len(full) - val_size
    if train_size <= 0:
        raise SystemExit(f"{args.data} has too few records to split")

    train_set, val_set = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"positions: {train_size} train, {val_size} validation")

    common = dict(
        batch_size=args.batch,
        collate_fn=ds.collate_batch,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    model = netmodel.KhaosNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", best_val)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device,
                               args.blend, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device,
                             args.blend, train=False)
        scheduler.step()

        elapsed = time.time() - started
        marker = ""

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_loss),
        }
        torch.save(checkpoint, out_dir / f"epoch_{epoch}.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")
            worst = export_checkpoint(model, out_dir / "best.nnue")
            marker = f"  <- best, exported (accumulator peak {worst})"

        print(f"epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s{marker}")

    print(f"\ndone. best validation loss {best_val:.6f}")
    print(f"net: {out_dir / 'best.nnue'}")


if __name__ == "__main__":
    main()
