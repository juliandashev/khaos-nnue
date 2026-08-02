"""Read and write the engine's .nnue file format. Pure standard library.

This module is the contract with src/nnue.cpp. It is deliberately free of numpy
and torch so it can be exercised (and the engine cross-checked) on a machine
with nothing installed. export.py converts torch tensors to plain Python lists
and hands them here.

Layout, all little-endian:

    char     magic[8]     "KHAOSNN1"
    uint32   version       2
    uint32   inputs        768
    uint32   hidden        256
    uint32   l2            32
    int32    qa            255
    int32    qb            64
    int32    eval_scale    1640
    int32    reserved[3]
    int16    feature_weights[inputs * hidden]     feature-major
    int16    feature_bias[hidden]
    int16    l2_weights[l2 * (2 * hidden)]         output-major
    int32    l2_bias[l2]
    int16    output_weights[l2]
    int32    output_bias
"""

import struct

from . import quant

MAGIC = b"KHAOSNN1"
FORMAT_VERSION = 2

_HEADER = struct.Struct("<8sIIIIiii3i")

INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647


class NetFormatError(ValueError):
    """Raised for a file that is not a loadable net."""


def _check_int16(values, what):
    for i, v in enumerate(values):
        if not (INT16_MIN <= v <= INT16_MAX):
            raise NetFormatError(
                f"{what}[{i}] = {v} does not fit in int16; the trainer's weight "
                f"clamp is too loose or quantization scales are wrong"
            )


def _check_int32(values, what):
    for i, v in enumerate(values):
        if not (INT32_MIN <= v <= INT32_MAX):
            raise NetFormatError(f"{what}[{i}] = {v} does not fit in int32")


def expected_size():
    """Byte size of a well-formed net file."""
    return (
        _HEADER.size
        + 2 * quant.INPUTS * quant.HIDDEN         # feature_weights
        + 2 * quant.HIDDEN                         # feature_bias
        + 2 * quant.L2 * 2 * quant.HIDDEN          # l2_weights
        + 4 * quant.L2                             # l2_bias (int32)
        + 2 * quant.L2                             # output_weights
        + 4                                        # output_bias (int32)
    )


def write_net(path, feature_weights, feature_bias, l2_weights, l2_bias,
              output_weights, output_bias):
    """Write a net. All weight lists are flat; l2_weights is output-major
    (neuron 0's 2*HIDDEN inputs, then neuron 1's, ...). l2_bias/output_bias are
    int32, pre-scaled by QA * QB."""
    def _len(name, values, expected):
        if len(values) != expected:
            raise NetFormatError(
                f"{name} has {len(values)} entries, expected {expected}"
            )

    _len("feature_weights", feature_weights, quant.INPUTS * quant.HIDDEN)
    _len("feature_bias", feature_bias, quant.HIDDEN)
    _len("l2_weights", l2_weights, quant.L2 * 2 * quant.HIDDEN)
    _len("l2_bias", l2_bias, quant.L2)
    _len("output_weights", output_weights, quant.L2)
    if not (INT32_MIN <= output_bias <= INT32_MAX):
        raise NetFormatError(f"output_bias {output_bias} does not fit in int32")

    _check_int16(feature_weights, "feature_weights")
    _check_int16(feature_bias, "feature_bias")
    _check_int16(l2_weights, "l2_weights")
    _check_int16(output_weights, "output_weights")
    _check_int32(l2_bias, "l2_bias")

    with open(path, "wb") as f:
        f.write(
            _HEADER.pack(
                MAGIC,
                FORMAT_VERSION,
                quant.INPUTS,
                quant.HIDDEN,
                quant.L2,
                quant.QA,
                quant.QB,
                quant.EVAL_SCALE,
                0, 0, 0,
            )
        )
        f.write(struct.pack(f"<{len(feature_weights)}h", *feature_weights))
        f.write(struct.pack(f"<{len(feature_bias)}h", *feature_bias))
        f.write(struct.pack(f"<{len(l2_weights)}h", *l2_weights))
        f.write(struct.pack(f"<{len(l2_bias)}i", *l2_bias))
        f.write(struct.pack(f"<{len(output_weights)}h", *output_weights))
        f.write(struct.pack("<i", output_bias))


def read_net(path):
    """Read a net, returning a dict with the same keys write_net() takes.

    Validates the header against this build's constants, so loading a net
    exported by a differently configured trainer fails here rather than
    producing wrong evaluations.
    """
    with open(path, "rb") as f:
        blob = f.read()

    if len(blob) < _HEADER.size:
        raise NetFormatError(f"{path}: too short to hold a header")

    (magic, version, inputs, hidden, l2, qa, qb, eval_scale,
     _r0, _r1, _r2) = _HEADER.unpack_from(blob, 0)

    if magic != MAGIC:
        raise NetFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise NetFormatError(
            f"{path}: format version {version}, this tool writes {FORMAT_VERSION}"
        )
    if (inputs, hidden, l2) != (quant.INPUTS, quant.HIDDEN, quant.L2):
        raise NetFormatError(
            f"{path}: topology {inputs}x{hidden}x{l2}, expected "
            f"{quant.INPUTS}x{quant.HIDDEN}x{quant.L2}"
        )
    if (qa, qb, eval_scale) != (quant.QA, quant.QB, quant.EVAL_SCALE):
        raise NetFormatError(
            f"{path}: quantization {qa}/{qb}/{eval_scale}, expected "
            f"{quant.QA}/{quant.QB}/{quant.EVAL_SCALE}"
        )
    if len(blob) != expected_size():
        raise NetFormatError(
            f"{path}: {len(blob)} bytes, expected {expected_size()}"
        )

    offset = _HEADER.size
    n_fw = inputs * hidden
    feature_weights = list(struct.unpack_from(f"<{n_fw}h", blob, offset))
    offset += 2 * n_fw
    feature_bias = list(struct.unpack_from(f"<{hidden}h", blob, offset))
    offset += 2 * hidden
    n_l2w = l2 * 2 * hidden
    l2_weights = list(struct.unpack_from(f"<{n_l2w}h", blob, offset))
    offset += 2 * n_l2w
    l2_bias = list(struct.unpack_from(f"<{l2}i", blob, offset))
    offset += 4 * l2
    output_weights = list(struct.unpack_from(f"<{l2}h", blob, offset))
    offset += 2 * l2
    (output_bias,) = struct.unpack_from("<i", blob, offset)

    return {
        "feature_weights": feature_weights,
        "feature_bias": feature_bias,
        "l2_weights": l2_weights,
        "l2_bias": l2_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
    }
