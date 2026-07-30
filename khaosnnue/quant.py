"""Quantization constants. Must match include/nnue.h in the engine exactly.

Every one of these appears in the .nnue header and the engine refuses to load a
file whose values disagree, so a mismatch is a loud failure rather than a
silently mis-scaled evaluation.
"""

# Topology
INPUTS = 768
HIDDEN = 256

# Feature weights/biases are stored as round(w * QA); the accumulator therefore
# lives on the same scale as a clipped ReLU whose ceiling is QA.
QA = 255

# Output weights are stored as round(w * QB). The output bias is pre-scaled by
# QA * QB so it can be added straight onto the int32 accumulation.
QB = 64

# Engine eval units per 1.0 of raw network output. The engine's tuned pawn is
# ~410, so this is the usual "4 pawns" sigmoid scale expressed in engine units
# instead of centipawns. Used in both directions: the training target is
# sigmoid(engine_score / EVAL_SCALE), and inference multiplies back by it.
EVAL_SCALE = 1640

# Clamp applied to feature weights during training. The int16 accumulator must
# not saturate: with at most 32 pieces on the board a single neuron accumulates
# the bias plus 32 feature weights, so keeping |w| under this bound leaves
# 32 * 1.98 * 255 ~= 16.2k of headroom inside int16's 32767.
FEATURE_WEIGHT_CLAMP = 1.98

# Worst-case number of pieces contributing to one accumulator.
MAX_PIECES = 32


def accumulator_headroom(max_abs_feature_weight_int: int,
                         max_abs_feature_bias_int: int) -> int:
    """Worst-case magnitude a single accumulator neuron can reach.

    Returns the bound so callers can compare it against int16's range before
    shipping a net; export.py treats exceeding it as a hard error.
    """
    return max_abs_feature_bias_int + MAX_PIECES * max_abs_feature_weight_int
