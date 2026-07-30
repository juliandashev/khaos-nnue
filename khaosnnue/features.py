"""FEN -> NNUE feature indices. Pure standard library, no numpy or torch.

This is the single source of truth for feature indexing on the Python side and
must mirror nnue::feature_index() in the engine's include/nnue.h:

    index(p, pc, pt, s) = (pc != p) * 384 + (pt - PAWN) * 64 + s ^ (p * 56)

Square numbering follows the engine, which is unusual: A8 = 0 and the index
runs rank 8 down to rank 1. That happens to be exactly the order a FEN lists
squares, so parsing is a straight left-to-right walk with no flipping.
"""

WHITE = 0
BLACK = 1

# Piece type ordering matches the engine's PieceType enum offsets (PAWN = 1).
_PIECE_TO_TYPE = {
    "p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6,
}


class FenError(ValueError):
    """Raised for a FEN this module cannot parse."""


def parse_fen(fen: str):
    """Return (pieces, side_to_move).

    `pieces` is a list of (colour, piece_type, square) with the engine's
    A8 = 0 square numbering. `side_to_move` is WHITE or BLACK.
    """
    parts = fen.split()
    if len(parts) < 2:
        raise FenError(f"FEN needs at least a board and a side to move: {fen!r}")

    board, stm_field = parts[0], parts[1]

    pieces = []
    square = 0
    for ch in board:
        if ch == "/":
            continue
        if ch.isdigit():
            square += int(ch)
            continue

        piece_type = _PIECE_TO_TYPE.get(ch.lower())
        if piece_type is None:
            raise FenError(f"unknown piece {ch!r} in {fen!r}")
        if square > 63:
            raise FenError(f"too many squares in {fen!r}")

        colour = WHITE if ch.isupper() else BLACK
        pieces.append((colour, piece_type, square))
        square += 1

    if square != 64:
        raise FenError(f"board describes {square} squares, expected 64: {fen!r}")

    if stm_field not in ("w", "b"):
        raise FenError(f"side to move must be 'w' or 'b', got {stm_field!r}")

    return pieces, (WHITE if stm_field == "w" else BLACK)


def feature_index(perspective: int, colour: int, piece_type: int,
                  square: int) -> int:
    """Mirror of nnue::feature_index()."""
    own_or_their = 384 if colour != perspective else 0
    relative_square = square ^ (56 if perspective == BLACK else 0)
    return own_or_their + (piece_type - 1) * 64 + relative_square


def mirror_index(white_index: int) -> int:
    """Black-perspective index for a piece given its white-perspective index.

    Flipping perspective flips the own/their half and rank-flips the square.
    The square occupies the low 6 bits of the within-half offset and 56 < 64,
    so xor-ing the offset with 56 touches only the square. This lets the packed
    dataset store one index per piece instead of two.
    """
    half, offset = divmod(white_index, 384)
    return (half ^ 1) * 384 + (offset ^ 56)


def features_from_fen(fen: str):
    """Return (white_indices, black_indices, side_to_move) for one position."""
    pieces, stm = parse_fen(fen)
    white = [feature_index(WHITE, c, pt, s) for (c, pt, s) in pieces]
    black = [feature_index(BLACK, c, pt, s) for (c, pt, s) in pieces]
    return white, black, stm


def white_features_from_fen(fen: str):
    """Return (white_perspective_indices, side_to_move).

    The black-perspective indices are recoverable with mirror_index(), so the
    packed dataset only needs these.
    """
    pieces, stm = parse_fen(fen)
    return [feature_index(WHITE, c, pt, s) for (c, pt, s) in pieces], stm
