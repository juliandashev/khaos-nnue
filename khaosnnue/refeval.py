"""Reference integer forward pass. Pure standard library.

Recomputes exactly what nnue::forward() does in the engine, so the two can be
compared on the same net and FEN. If they ever disagree, the feature indexing or
the quantization arithmetic has drifted apart -- which is the single most likely
way this pipeline breaks, and the hardest to notice from playing strength alone.
"""

from . import features, quant


def accumulators(net, fen):
    """Return (white_accumulator, black_accumulator) for a position."""
    white_idx, black_idx, _stm = features.features_from_fen(fen)

    fw = net["feature_weights"]
    bias = net["feature_bias"]
    hidden = quant.HIDDEN

    acc = {
        features.WHITE: list(bias),
        features.BLACK: list(bias),
    }

    for perspective, indices in ((features.WHITE, white_idx),
                                (features.BLACK, black_idx)):
        target = acc[perspective]
        for index in indices:
            base = index * hidden
            for i in range(hidden):
                target[i] += fw[base + i]

    return acc[features.WHITE], acc[features.BLACK]


def evaluate(net, fen):
    """Static evaluation in engine units, from the side to move's point of view."""
    _w, _b, stm = features.features_from_fen(fen)
    white_acc, black_acc = accumulators(net, fen)

    own = white_acc if stm == features.WHITE else black_acc
    their = black_acc if stm == features.WHITE else white_acc

    qa, qb = quant.QA, quant.QB
    hidden, l2 = quant.HIDDEN, quant.L2

    # Concatenated clipped activations: own then their.
    x = [min(max(v, 0), qa) for v in own] + [min(max(v, 0), qa) for v in their]

    # Hidden layer 2: // qb returns to the qa scale, then clip. Negatives clip
    # to 0 either way, so // matches the engine's truncating divide here.
    l2w, l2b = net["l2_weights"], net["l2_bias"]
    h = []
    for j in range(l2):
        s = l2b[j]
        base = j * 2 * hidden
        for i in range(2 * hidden):
            s += x[i] * l2w[base + i]
        h.append(min(max(s // qb, 0), qa))

    # Output layer.
    ow = net["output_weights"]
    total = net["output_bias"]
    for j in range(l2):
        total += h[j] * ow[j]

    # Integer division truncating toward zero, matching C++ '/' on int64.
    scaled = total * quant.EVAL_SCALE
    denom = qa * qb
    result = abs(scaled) // denom
    return -result if scaled < 0 else result


def max_accumulator_magnitude(net):
    """Worst-case accumulator value, for an int16 saturation check."""
    return quant.accumulator_headroom(
        max(abs(w) for w in net["feature_weights"]),
        max(abs(b) for b in net["feature_bias"]),
    )
