#!/usr/bin/env python3
"""Export a training checkpoint to a .nnue the engine can load.

train.py already exports its best checkpoint automatically; this is for
exporting any other epoch, or re-exporting after changing quantization.

    python3 export.py nets/run1/epoch_12.pt -o nets/run1/epoch_12.nnue
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khaosnnue import format as netformat  # noqa: E402
from khaosnnue import model as netmodel  # noqa: E402
from khaosnnue import quant  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help=".pt file from train.py")
    parser.add_argument("-o", "--out", required=True, help="output .nnue path")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = netmodel.KhaosNet()
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    (feature_weights, feature_bias, l2_weights, l2_bias,
     output_weights, output_bias) = netmodel.quantized_tensors(model)

    worst = quant.accumulator_headroom(
        max(abs(w) for w in feature_weights),
        max(abs(b) for b in feature_bias),
    )
    print(f"worst-case accumulator magnitude: {worst} (int16 limit 32767)")
    if worst > 32767:
        raise SystemExit(
            "refusing to export: this net can overflow the int16 accumulator. "
            "Lower quant.FEATURE_WEIGHT_CLAMP and retrain."
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    netformat.write_net(args.out, feature_weights, feature_bias, l2_weights,
                        l2_bias, output_weights, output_bias)

    print(f"wrote {args.out} ({netformat.expected_size()} bytes)")
    print("verify it against the engine with:")
    print(f"  python3 scripts/check_engine.py <path-to>/KhaosChess {args.out}")


if __name__ == "__main__":
    main()
