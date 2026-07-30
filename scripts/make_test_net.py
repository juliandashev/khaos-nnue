#!/usr/bin/env python3
"""Write a deterministic pseudo-random .nnue. Pure standard library.

Not a useful chess net -- its purpose is to exercise the format and the engine's
loader, and to give scripts/check_engine.py something whose evaluations both
sides can be compared on.

    python3 scripts/make_test_net.py nets/test.nnue --seed 1
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khaosnnue import format as netformat  # noqa: E402
from khaosnnue import quant  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="path to write the .nnue to")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Small feature weights, matching the accumulator-headroom constraint the
    # real trainer enforces with its weight clamp.
    feature_weights = [rng.randint(-64, 64)
                       for _ in range(quant.INPUTS * quant.HIDDEN)]
    feature_bias = [rng.randint(-64, 64) for _ in range(quant.HIDDEN)]
    output_weights = [rng.randint(-32, 32) for _ in range(2 * quant.HIDDEN)]
    output_bias = rng.randint(-1000, 1000)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    netformat.write_net(args.out, feature_weights, feature_bias,
                        output_weights, output_bias)

    print(f"wrote {args.out} ({netformat.expected_size()} bytes, seed {args.seed})")


if __name__ == "__main__":
    main()
