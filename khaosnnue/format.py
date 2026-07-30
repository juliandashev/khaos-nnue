"""Read and write the engine's .nnue file format. Pure standard library.

This module is the contract with src/nnue.cpp. It is deliberately free of numpy
and torch so it can be exercised (and the engine cross-checked) on a machine
with nothing installed. export.py converts torch tensors to plain Python lists
and hands them here.

Layout, all little-endian:

    char     magic[8]     "KHAOSNN1"
    uint32   version       1
    uint32   inputs        768
    uint32   hidden        256
    int32    qa            255
    int32    qb            64
    int32    eval_scale    1640
    int32    reserved[4]
    int16    feature_weights[inputs * hidden]   feature-major
    int16    feature_bias[hidden]
    int16    output_weights[2 * hidden]         [own half][their half]
    int32    output_bias                        pre-scaled by qa * qb
"""

import struct

from . import quant

MAGIC = b"KHAOSNN1"
FORMAT_VERSION = 1

_HEADER = struct.Struct("<8sIIIiii4i")

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


def expected_size():
    """Byte size of a well-formed net file."""
    return (
        _HEADER.size
        + 2 * quant.INPUTS * quant.HIDDEN
        + 2 * quant.HIDDEN
        + 2 * 2 * quant.HIDDEN
        + 4
    )


def write_net(path, feature_weights, feature_bias, output_weights, output_bias):
    """Write a net.

    feature_weights: flat list of INPUTS * HIDDEN ints, feature-major
                     (feature 0's HIDDEN values, then feature 1's, ...)
    feature_bias:    HIDDEN ints
    output_weights:  2 * HIDDEN ints, own half first
    output_bias:     one int, already scaled by QA * QB
    """
    if len(feature_weights) != quant.INPUTS * quant.HIDDEN:
        raise NetFormatError(
            f"feature_weights has {len(feature_weights)} entries, expected "
            f"{quant.INPUTS * quant.HIDDEN}"
        )
    if len(feature_bias) != quant.HIDDEN:
        raise NetFormatError(
            f"feature_bias has {len(feature_bias)} entries, expected {quant.HIDDEN}"
        )
    if len(output_weights) != 2 * quant.HIDDEN:
        raise NetFormatError(
            f"output_weights has {len(output_weights)} entries, expected "
            f"{2 * quant.HIDDEN}"
        )
    if not (INT32_MIN <= output_bias <= INT32_MAX):
        raise NetFormatError(f"output_bias {output_bias} does not fit in int32")

    _check_int16(feature_weights, "feature_weights")
    _check_int16(feature_bias, "feature_bias")
    _check_int16(output_weights, "output_weights")

    with open(path, "wb") as f:
        f.write(
            _HEADER.pack(
                MAGIC,
                FORMAT_VERSION,
                quant.INPUTS,
                quant.HIDDEN,
                quant.QA,
                quant.QB,
                quant.EVAL_SCALE,
                0, 0, 0, 0,
            )
        )
        f.write(struct.pack(f"<{len(feature_weights)}h", *feature_weights))
        f.write(struct.pack(f"<{len(feature_bias)}h", *feature_bias))
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

    (magic, version, inputs, hidden, qa, qb, eval_scale,
     _r0, _r1, _r2, _r3) = _HEADER.unpack_from(blob, 0)

    if magic != MAGIC:
        raise NetFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        raise NetFormatError(
            f"{path}: format version {version}, this tool writes {FORMAT_VERSION}"
        )
    if (inputs, hidden) != (quant.INPUTS, quant.HIDDEN):
        raise NetFormatError(
            f"{path}: topology {inputs}x{hidden}, expected "
            f"{quant.INPUTS}x{quant.HIDDEN}"
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
    output_weights = list(struct.unpack_from(f"<{2 * hidden}h", blob, offset))
    offset += 4 * hidden
    (output_bias,) = struct.unpack_from("<i", blob, offset)

    return {
        "feature_weights": feature_weights,
        "feature_bias": feature_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
    }
