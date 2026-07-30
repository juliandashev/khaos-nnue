#!/usr/bin/env python3
"""Cross-check the engine's NNUE evaluation against the Python reference.

Pure standard library: this runs anywhere the engine binary does, with no
training environment installed.

Both sides load the same .nnue and evaluate the same FENs. They must agree
exactly -- the arithmetic is integer end to end, so any difference means the
feature indexing, the perspective handling or the quantization scales have
drifted between src/nnue.cpp and khaosnnue/. That is the failure mode this
pipeline is most likely to hit and the one playing strength hides best: a net
with mismatched indexing still plays legal chess, just badly.

    python3 scripts/check_engine.py ../chess-engine/bin/KhaosChess nets/test.nnue
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khaosnnue import format as netformat  # noqa: E402
from khaosnnue import refeval  # noqa: E402

# A spread of material, castling states, promotion-heavy and endgame positions.
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/2k5/1q6/K7 b - - 0 1",
    "8/3k4/8/8/3K4/8/8/8 w - - 0 1",
    "2r3k1/5pp1/p3p2p/1p1qP3/3P4/1QP2P2/P5PP/4R1K1 b - - 0 1",
    "8/5k2/8/8/8/8/2R5/4K3 w - - 0 1",
]


def engine_evals(engine, net, fens):
    """Run one engine process over every FEN, returning its static evals."""
    lines = [f"setoption name EvalFile value {net}"]
    for fen in fens:
        lines.append(f"position fen {fen}")
        lines.append("eval")
    lines.append("quit")

    proc = subprocess.run(
        [engine],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    )

    loaded = [l for l in proc.stdout.splitlines()
              if "loaded net" in l or "EvalFile error" in l]
    if not loaded or "error" in loaded[0]:
        raise SystemExit(
            f"engine did not load the net.\n"
            f"  {loaded[0] if loaded else '(no EvalFile response at all)'}"
        )

    evals = []
    for line in proc.stdout.splitlines():
        if line.startswith("static eval:"):
            value = line.split(":", 1)[1].strip()
            score, source = value.split()
            evals.append((int(score), source.strip("()")))

    return evals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", help="path to the KhaosChess binary")
    parser.add_argument("net", help="path to a .nnue")
    args = parser.parse_args()

    net = netformat.read_net(args.net)

    headroom = refeval.max_accumulator_magnitude(net)
    print(f"worst-case accumulator magnitude: {headroom} (int16 limit 32767)")
    if headroom > 32767:
        print("  WARNING: this net can saturate the int16 accumulator")

    evals = engine_evals(args.engine, args.net, POSITIONS)
    if len(evals) != len(POSITIONS):
        raise SystemExit(
            f"expected {len(POSITIONS)} static evals from the engine, "
            f"got {len(evals)}"
        )

    failures = 0
    skipped = 0
    print()
    print(f"{'engine':>10}  {'python':>10}  {'':4} position")

    for fen, (engine_score, source) in zip(POSITIONS, evals):
        reference = refeval.evaluate(net, fen)

        if source != "nnue":
            # The engine short-circuited to exact endgame knowledge (the KPK
            # bitbase, known mates) before consulting the net, so there is no
            # network evaluation to compare against here.
            skipped += 1
            mark = "skip"
            shown = "-"
        elif engine_score == reference:
            mark = "ok"
            shown = str(reference)
        else:
            failures += 1
            mark = "FAIL"
            shown = str(reference)

        print(f"{engine_score:>10}  {shown:>10}  {mark:4} {fen}")

    print()
    compared = len(POSITIONS) - skipped
    if failures:
        raise SystemExit(
            f"{failures}/{compared} positions disagree -- the Python and C++ "
            f"implementations have diverged"
        )

    print(f"all {compared} compared positions agree exactly"
          + (f" ({skipped} skipped: engine used exact endgame knowledge)"
             if skipped else ""))


if __name__ == "__main__":
    main()
