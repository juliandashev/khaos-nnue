"""Memory-mapped dataset over the packed records prepare.py writes.

Requires numpy and torch.

Batches are built as dense (batch, 768) tensors. That is more memory traffic
than a sparse gather, but the matmul is tiny at this topology and dense keeps
the model a plain nn.Linear, which in turn keeps the exporter obvious. If this
ever becomes the bottleneck, the drop-in fix is an EmbeddingBag over the index
lists -- the file format already stores them.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from . import features, quant

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

# result code -> win probability from the side to move's point of view
_RESULT_VALUE = np.array([0.0, 0.5, 1.0], dtype=np.float32)

# Lookup table for features.mirror_index over the whole input range, so the
# per-batch perspective flip is a single numpy gather instead of Python calls.
_MIRROR = np.array(
    [features.mirror_index(i) for i in range(quant.INPUTS)], dtype=np.uint16
)


class PackedDataset(Dataset):
    def __init__(self, path):
        self.data = np.memmap(path, dtype=RECORD, mode="r")
        if len(self.data) == 0:
            raise ValueError(f"{path} contains no records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Returned as raw arrays; collate_batch does the expansion so the work
        # happens once per batch rather than once per sample.
        rec = self.data[index]
        return (
            np.asarray(rec["feat"]),
            int(rec["count"]),
            int(rec["score"]),
            int(rec["stm"]),
            int(rec["result"]),
        )


def collate_batch(samples):
    """Build dense perspective tensors and the training targets for a batch."""
    batch = len(samples)

    own = torch.zeros((batch, quant.INPUTS), dtype=torch.float32)
    their = torch.zeros((batch, quant.INPUTS), dtype=torch.float32)
    scores = torch.empty(batch, dtype=torch.float32)
    results = torch.empty(batch, dtype=torch.float32)

    for row, (feat, count, score, stm, result) in enumerate(samples):
        white_idx = feat[:count].astype(np.int64)
        black_idx = _MIRROR[white_idx].astype(np.int64)

        # `own` is always the side to move's perspective, which is the
        # convention the engine's forward() uses.
        if stm == features.WHITE:
            own_idx, their_idx = white_idx, black_idx
        else:
            own_idx, their_idx = black_idx, white_idx

        own[row, own_idx] = 1.0
        their[row, their_idx] = 1.0
        scores[row] = score
        results[row] = float(_RESULT_VALUE[result])

    return own, their, scores, results


def make_target(scores, results, blend):
    """Blend the search score and the game result into one training target.

    Both are converted to win probabilities from the side to move's point of
    view. `blend` is the weight on the score: 1.0 trains purely on the search's
    opinion, 0.0 purely on how the game actually ended. Somewhere in between is
    standard -- the score is dense and low-variance but inherits the search's
    blind spots, the result is unbiased but extremely noisy per position.
    """
    score_prob = torch.sigmoid(scores / quant.EVAL_SCALE)
    return blend * score_prob + (1.0 - blend) * results
