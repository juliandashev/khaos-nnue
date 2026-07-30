#!/usr/bin/env python3
"""Pack datagen text shards into a fixed-record binary for fast training.

Parsing FENs in Python is far too slow to do once per epoch, so it happens once
here and training then memory-maps the result.

Record layout (numpy structured dtype, 72 bytes):

    count    uint8       number of pieces on the board, <= 32
    feat     uint16[32]  white-perspective feature indices, padded with 65535
    score    int32       search score in engine units, side-to-move relative
    stm      uint8       0 = white to move, 1 = black
    result   uint8       0 = stm lost, 1 = draw, 2 = stm won
    _pad     uint8

Only white-perspective indices are stored; the black-perspective ones are
recovered with features.mirror_index() at load time, which halves the file.

    python3 prepare.py data/*.txt -o data/train.bin --jobs 16
"""

import argparse
import glob
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khaosnnue import features  # noqa: E402

MAX_PIECES = 32
PAD = 65535

RECORD = np.dtype([
    ("count", np.uint8),
    ("feat", np.uint16, (MAX_PIECES,)),
    ("score", np.int32),
    ("stm", np.uint8),
    ("result", np.uint8),
    ("_pad", np.uint8),
])

# Discard positions whose score is beyond this (engine units, pawn ~= 410).
# Such positions are already decided; their exact score carries no signal and
# they drag the sigmoid into its flat tails.
SCORE_LIMIT = 32 * 410

_RESULT_CODE = {"0.0": 0, "0.5": 1, "1.0": 2}


class Stats:
    __slots__ = ("kept", "bad_line", "bad_fen", "too_many_pieces",
                 "score_clipped", "bad_result")

    def __init__(self):
        self.kept = 0
        self.bad_line = 0
        self.bad_fen = 0
        self.too_many_pieces = 0
        self.score_clipped = 0
        self.bad_result = 0

    def merge(self, other):
        for slot in self.__slots__:
            setattr(self, slot, getattr(self, slot) + getattr(other, slot))

    def report(self):
        return (
            f"kept {self.kept}, dropped: "
            f"{self.bad_line} malformed, {self.bad_fen} bad FEN, "
            f"{self.too_many_pieces} over {MAX_PIECES} pieces, "
            f"{self.score_clipped} beyond score limit, "
            f"{self.bad_result} bad result"
        )


def parse_chunk(lines):
    """Convert a list of text lines into a record array plus stats."""
    stats = Stats()
    out = np.zeros(len(lines), dtype=RECORD)
    n = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) != 3:
            stats.bad_line += 1
            continue

        fen = parts[0].strip()
        score_text = parts[1].strip()
        result_text = parts[2].strip()

        result_code = _RESULT_CODE.get(result_text)
        if result_code is None:
            stats.bad_result += 1
            continue

        try:
            white_score = int(score_text)
        except ValueError:
            stats.bad_line += 1
            continue

        try:
            white_idx, stm = features.white_features_from_fen(fen)
        except features.FenError:
            stats.bad_fen += 1
            continue

        if len(white_idx) > MAX_PIECES:
            stats.too_many_pieces += 1
            continue

        # Both labels are stored relative to the side to move, matching what the
        # network predicts, so the training loop needs no conditionals.
        if stm == features.BLACK:
            score = -white_score
            result_code = 2 - result_code
        else:
            score = white_score

        if abs(score) > SCORE_LIMIT:
            stats.score_clipped += 1
            continue

        rec = out[n]
        rec["count"] = len(white_idx)
        rec["feat"][:len(white_idx)] = white_idx
        if len(white_idx) < MAX_PIECES:
            rec["feat"][len(white_idx):] = PAD
        rec["score"] = score
        rec["stm"] = stm
        rec["result"] = result_code
        n += 1

    stats.kept = n
    return out[:n], stats


def chunked_lines(paths, chunk_size):
    chunk = []
    for path in paths:
                # datagen appends, so a shard can be reopened and grown; read
                # lazily rather than slurping a multi-GB file.
        with open(path, "r") as f:
            for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+",
                        help="datagen text shards (globs are expanded)")
    parser.add_argument("-o", "--out", required=True, help="output .bin path")
    parser.add_argument("--jobs", type=int, default=mp.cpu_count())
    parser.add_argument("--chunk", type=int, default=200_000,
                        help="lines per worker task")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle the packed records before writing "
                             "(needs the whole set in RAM)")
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"no such input: {missing[0]}")

    print(f"packing {len(paths)} shard(s) with {args.jobs} worker(s)")

    total = Stats()
    blocks = []

    with mp.Pool(args.jobs) as pool:
        for records, stats in pool.imap(
            parse_chunk, chunked_lines(paths, args.chunk), chunksize=1
        ):
            total.merge(stats)
            if len(records):
                blocks.append(records)
            print(f"\r  {total.kept} positions", end="", flush=True)

    print()
    if not blocks:
        raise SystemExit("no usable positions found")

    packed = np.concatenate(blocks)

    if args.shuffle:
        # Consecutive positions come from the same game and are highly
        # correlated; shuffling once here means training can read sequentially.
        print("shuffling...")
        rng = np.random.default_rng(0)
        rng.shuffle(packed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(args.out)

    print(total.report())
    print(f"wrote {len(packed)} records to {args.out} "
          f"({packed.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
