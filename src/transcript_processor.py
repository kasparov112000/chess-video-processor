"""
Transcript Processing Module

Processes audio transcripts and extracts chess move references.
"""

import re
import logging
from typing import Optional
import chess

from .models import TranscriptSegment, DetectedMove
from .config import settings

logger = logging.getLogger(__name__)


# Chess move patterns in natural language
# Order matters - more specific patterns first
MOVE_PATTERNS = [
    # Castling (check first)
    r'\b(O-O-O|O-O|0-0-0|0-0)\b',
    # Descriptive: "knight to f3", "bishop takes e5" (before SAN to avoid partial matches)
    r'\b(knight|bishop|rook|queen|king|pawn)\s+(?:to\s+)?([a-h][1-8])\b',
    r'\b(knight|bishop|rook|queen|king|pawn)\s+takes\s+(?:on\s+)?([a-h][1-8])\b',
    # Piece + square: "the move Nf3", "playing e4"
    r'\b(?:move|play|playing|plays)\s+([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])\b',
    # Standard algebraic notation: "e4", "Nf3", "Bxe5" - but NOT bare squares after piece names
    # This requires the move to not be preceded by piece names
    r'(?<!\bknight\s)(?<!\bbishop\s)(?<!\brook\s)(?<!\bqueen\s)(?<!\bking\s)(?<!\bpawn\s)\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b',
    # Pawn moves: e4, d5, exd5 (lowercase, no piece letter)
    r'(?<!\bknight\s)(?<!\bbishop\s)(?<!\brook\s)(?<!\bqueen\s)(?<!\bking\s)(?<!\bpawn\s)\b([a-h]x?[a-h]?[1-8](?:=[QRBN])?[+#]?)\b',
]

PIECE_MAP = {
    'knight': 'N',
    'bishop': 'B',
    'rook': 'R',
    'queen': 'Q',
    'king': 'K',
    'pawn': '',
}


class TranscriptProcessor:
    """Processes chess tutorial transcripts to extract moves"""

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in MOVE_PATTERNS]

    def extract_moves_from_text(
        self, text: str, timestamp: float = 0.0
    ) -> list[dict]:
        """
        Extract potential chess moves from text.

        Returns list of dicts with move info and position in text.
        """
        moves = []
        # Track matched character positions to avoid overlapping matches
        matched_positions: set[int] = set()

        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                # Skip if this position overlaps with an already matched range
                match_range = set(range(match.start(), match.end()))
                if matched_positions & match_range:
                    continue

                move_text = match.group(0)
                san = self._normalize_to_san(move_text, match.groups())

                if san:
                    moves.append({
                        'raw': move_text,
                        'san': san,
                        'position': match.start(),
                        'timestamp': timestamp,
                    })
                    # Mark these positions as matched
                    matched_positions.update(match_range)

        # Sort by position in text and deduplicate
        moves.sort(key=lambda x: x['position'])
        return self._deduplicate_moves(moves)

    def _normalize_to_san(self, raw: str, groups: tuple) -> Optional[str]:
        """Convert various move formats to Standard Algebraic Notation"""
        raw = raw.strip()

        # Already SAN format
        if re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?$', raw):
            return raw

        # Castling
        if raw.upper() in ['O-O', '0-0']:
            return 'O-O'
        if raw.upper() in ['O-O-O', '0-0-0']:
            return 'O-O-O'

        # Descriptive: "knight to f3" -> "Nf3"
        if len(groups) >= 2:
            piece_name = groups[0].lower() if groups[0] else ''
            square = groups[1].lower() if len(groups) > 1 and groups[1] else ''

            if piece_name in PIECE_MAP and square:
                piece = PIECE_MAP[piece_name]
                # Check for "takes"
                if 'takes' in raw.lower():
                    return f"{piece}x{square}"
                return f"{piece}{square}"

        return None

    def _deduplicate_moves(self, moves: list[dict]) -> list[dict]:
        """Remove duplicate moves that are close together"""
        if not moves:
            return []

        result = [moves[0]]
        for move in moves[1:]:
            # Skip if same move within 50 chars
            if move['san'] == result[-1]['san'] and \
               move['position'] - result[-1]['position'] < 50:
                continue
            result.append(move)

        return result

    def process_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[dict]:
        """Process transcript segments and extract moves with timestamps"""
        all_moves = []

        for segment in segments:
            segment_moves = self.extract_moves_from_text(
                segment.text,
                timestamp=segment.start
            )

            # Interpolate timestamps within segment
            if segment_moves:
                duration = segment.end - segment.start
                for i, move in enumerate(segment_moves):
                    # Estimate timestamp based on position in text
                    text_progress = move['position'] / max(len(segment.text), 1)
                    move['timestamp'] = segment.start + (duration * text_progress)

            all_moves.extend(segment_moves)

        return all_moves

    def validate_move_sequence(
        self, moves: list[dict], starting_fen: str = chess.STARTING_FEN
    ) -> list[DetectedMove]:
        """
        Validate a sequence of moves against chess rules.

        Returns only valid moves with FEN positions.
        """
        board = chess.Board(starting_fen)
        validated_moves: list[DetectedMove] = []
        move_number = 1

        for move_data in moves:
            san = move_data['san']
            fen_before = board.fen()

            try:
                # Try to parse and apply the move
                chess_move = board.parse_san(san)

                # Get the proper SAN before pushing the move
                proper_san = board.san(chess_move)

                board.push(chess_move)

                validated_moves.append(
                    DetectedMove(
                        move_number=move_number,
                        move_san=proper_san,
                        move_uci=chess_move.uci(),
                        timestamp=move_data['timestamp'],
                        source='transcript',
                        fen_before=fen_before,
                        fen_after=board.fen(),
                    )
                )

                if board.turn == chess.WHITE:
                    move_number += 1

            except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError) as e:
                logger.debug(f"Invalid move '{san}': {e}")
                # Move is invalid in current position - might be from a variation
                continue

        return validated_moves


# Singleton instance
transcript_processor = TranscriptProcessor()
