from pydantic import BaseModel
from typing import Optional
from datetime import timedelta


class BoardPosition(BaseModel):
    """A chess board position detected from video"""
    timestamp: float  # seconds into video
    fen: str  # FEN notation
    confidence: float  # 0-1 detection confidence
    frame_number: int


class TranscriptSegment(BaseModel):
    """A segment of transcribed speech"""
    start: float  # seconds
    end: float  # seconds
    text: str
    confidence: float


class DetectedMove(BaseModel):
    """A chess move extracted from transcript or inferred from positions"""
    move_number: int
    move_san: str  # Standard Algebraic Notation (e.g., "Nf3")
    move_uci: str  # UCI format (e.g., "g1f3")
    timestamp: float
    source: str  # "transcript", "visual", or "inferred"
    fen_before: str
    fen_after: str
    commentary: Optional[str] = None


class ProcessingResult(BaseModel):
    """Complete result from video processing"""
    video_id: str
    duration: float

    # Extracted data
    positions: list[BoardPosition]
    transcript_segments: list[TranscriptSegment]
    moves: list[DetectedMove]

    # Final output
    pgn: str
    opening_name: Optional[str] = None

    # Metadata
    total_moves: int
    moves_from_visual: int
    moves_from_transcript: int
    moves_inferred: int
    processing_time: float


class VideoProcessRequest(BaseModel):
    """Request to process a chess video"""
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    transcript: Optional[str] = None  # Pre-existing transcript
    transcript_segments: Optional[list[TranscriptSegment]] = None

    # Options
    extract_audio: bool = True
    sample_rate: int = 1  # frames per second


class ChessQuestionRequest(BaseModel):
    """Request to generate questions from processed video"""
    processing_result: ProcessingResult
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    difficulty: str = "intermediate"
    num_questions: int = 5
