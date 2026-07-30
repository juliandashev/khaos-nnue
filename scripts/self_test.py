#!/usr/bin/env python3
"""Invariant checks for the dependency-free core. Pure standard library.

Run this after touching khaosnnue/features.py, format.py or quant.py. It does
not need torch, numpy, or the engine binary.

    python3 scripts/self_test.py
"""

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khaosnnue import features, quant  # noqa: E402
from khaosnnue import format as netformat  # noqa: E402

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/2k5/1q6/K7 b - - 0 1",
    # datagen's get_fen() emits a double space before the halfmove clock; the
    # parser must tolerate it since every generated line looks like this.
    "rnb1kb1r/pppp1pp1/7n/4p1qp/8/N1P3P1/PP1PPP1P/1RBQKBNR w Kkq -  3 5",
]

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def main():
    print("feature indexing")

    # A8 = 0 in the engine, so the first FEN character maps to square 0.
    pieces, stm = features.parse_fen(FENS[0])
    check("startpos has 32 pieces", len(pieces) == 32, f"got {len(pieces)}")
    check("startpos side to move is white", stm == features.WHITE)
    check(
        "black rook sits on square 0 (a8)",
        (features.BLACK, 4, 0) in pieces,
        f"pieces[0]={pieces[0]}",
    )
    check(
        "white rook sits on square 56 (a1)",
        (features.WHITE, 4, 56) in pieces,
    )

    # Index bounds.
    all_in_range = True
    for fen in FENS:
        white, black, _ = features.features_from_fen(fen)
        for idx in white + black:
            if not (0 <= idx < quant.INPUTS):
                all_in_range = False
    check(f"all indices within [0, {quant.INPUTS})", all_in_range)

    # Own-piece features live in the low half from their own perspective.
    own_half_ok = True
    for fen in FENS:
        pieces, _ = features.parse_fen(fen)
        for colour, pt, sq in pieces:
            if features.feature_index(colour, colour, pt, sq) >= 384:
                own_half_ok = False
    check("own pieces index into the low half", own_half_ok)

    print("\nmirror_index (used by prepare.py to halve the dataset)")

    involution = all(
        features.mirror_index(features.mirror_index(i)) == i
        for i in range(quant.INPUTS)
    )
    check("mirror is an involution", involution)

    bijection = len({features.mirror_index(i) for i in range(quant.INPUTS)}) == \
        quant.INPUTS
    check("mirror is a bijection", bijection)

    # The load-bearing claim: storing only white-perspective indices loses
    # nothing, because mirror_index recovers the black-perspective ones.
    derivation_ok = True
    mismatch = None
    for fen in FENS:
        white, black, _ = features.features_from_fen(fen)
        derived = [features.mirror_index(i) for i in white]
        if derived != black:
            derivation_ok = False
            mismatch = fen
    check("mirror(white indices) == black indices", derivation_ok,
          f"first mismatch: {mismatch}")

    print("\n.nnue format round-trip")

    rng = random.Random(7)
    feature_weights = [rng.randint(-500, 500)
                       for _ in range(quant.INPUTS * quant.HIDDEN)]
    feature_bias = [rng.randint(-500, 500) for _ in range(quant.HIDDEN)]
    output_weights = [rng.randint(-300, 300) for _ in range(2 * quant.HIDDEN)]
    output_bias = rng.randint(-100000, 100000)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "round_trip.nnue"
        netformat.write_net(path, feature_weights, feature_bias,
                            output_weights, output_bias)

        check("file size matches expected_size()",
              path.stat().st_size == netformat.expected_size(),
              f"{path.stat().st_size} vs {netformat.expected_size()}")

        back = netformat.read_net(path)
        check("feature_weights survive", back["feature_weights"] == feature_weights)
        check("feature_bias survives", back["feature_bias"] == feature_bias)
        check("output_weights survive", back["output_weights"] == output_weights)
        check("output_bias survives", back["output_bias"] == output_bias)

        # Out-of-range weights must be refused rather than silently truncated.
        try:
            netformat.write_net(path, [70000] + feature_weights[1:],
                                feature_bias, output_weights, output_bias)
            check("rejects out-of-int16 weights", False, "no exception raised")
        except netformat.NetFormatError:
            check("rejects out-of-int16 weights", True)

        # Wrong-length inputs must be refused too.
        try:
            netformat.write_net(path, feature_weights[:-1], feature_bias,
                                output_weights, output_bias)
            check("rejects wrong-length weights", False, "no exception raised")
        except netformat.NetFormatError:
            check("rejects wrong-length weights", True)

    print("\nFEN parser rejects malformed input")
    for bad in ["", "not a fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"]:
        try:
            features.parse_fen(bad)
            check(f"rejects {bad[:32]!r}", False, "parsed successfully")
        except features.FenError:
            check(f"rejects {bad[:32]!r}", True)

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed")


if __name__ == "__main__":
    main()
