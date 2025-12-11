"""
FastAPI Application

Exposes endpoints for video processing and chess extraction.
Connect from your webapp via Tailscale.
"""

import logging
import time
from pathlib import Path
from typing import Optional
import tempfile
import httpx
import chess

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import (
    VideoProcessRequest,
    ProcessingResult,
    BoardPosition,
    TranscriptSegment,
    DetectedMove,
)
from .video_processor import board_detector
from .transcript_processor import transcript_processor
from .fusion_agent import fusion_agent

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Chess Video Processor",
    description="Extract chess games from tutorial videos using AI",
    version="1.0.0",
)

# CORS for webapp access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:4202",
        "https://app.learnbytesting.ai",
        "https://app.domyhomework.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chess-video-processor"}


@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong", "tailscale_ip": settings.tailscale_ip}


@app.post("/process-video", response_model=ProcessingResult)
async def process_video(
    video: UploadFile = File(None),
    video_url: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    sample_rate: int = Form(1),
):
    """
    Process a chess tutorial video.

    Either upload a video file or provide a URL.
    Optionally provide pre-existing transcript.
    """
    start_time = time.time()

    if not video and not video_url:
        raise HTTPException(400, "Either video file or video_url required")

    # Save uploaded video to temp file
    video_path = None
    if video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            video_path = tmp.name
    elif video_url:
        # Download video from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(video_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(response.content)
                video_path = tmp.name

    try:
        # Step 1: Extract board positions from video
        # Convert sample_rate: API uses "seconds between frames", internally use fps
        # e.g., sample_rate=30 means 1 frame every 30 seconds = 0.033 fps
        effective_sample_rate = 1.0 / sample_rate if sample_rate > 0 else 1
        logger.info(f"Extracting board positions from video (1 frame every {sample_rate}s)...")
        positions = board_detector.process_video(video_path, effective_sample_rate)
        logger.info(f"Found {len(positions)} unique positions")

        # Check if positions are reliable enough to use for fusion
        # If only starting position with low confidence, fall back to transcript-only
        if len(positions) <= 1:
            if positions and positions[0].confidence < 0.5:
                logger.info("Single low-confidence position detected, using transcript-only mode")
                positions = []  # Clear positions to trigger transcript-only fusion

        # Step 2: Process transcript
        logger.info("Processing transcript...")
        transcript_segments = []
        transcript_moves = []

        if transcript:
            # Simple segmentation if raw transcript provided
            # In production, use Whisper with word timestamps
            transcript_segments = [
                TranscriptSegment(
                    start=0,
                    end=len(transcript) / 10,  # Rough estimate
                    text=transcript,
                    confidence=1.0,
                )
            ]
            transcript_moves = transcript_processor.extract_moves_from_text(transcript)

        # Step 3: Fusion - combine visual and transcript data
        logger.info("Running fusion agent...")
        validated_moves = fusion_agent.process(
            positions=positions,
            transcript_segments=transcript_segments,
            transcript_moves=transcript_moves,
        )

        # Step 4: Generate PGN
        pgn = generate_pgn(validated_moves)

        # Calculate stats
        processing_time = time.time() - start_time
        moves_from_visual = sum(1 for m in validated_moves if m.source == "visual")
        moves_from_transcript = sum(1 for m in validated_moves if m.source == "transcript")
        moves_inferred = sum(1 for m in validated_moves if m.source == "inferred")

        return ProcessingResult(
            video_id=video.filename if video else video_url or "unknown",
            duration=0,  # Would get from video metadata
            positions=positions,
            transcript_segments=transcript_segments,
            moves=validated_moves,
            pgn=pgn,
            total_moves=len(validated_moves),
            moves_from_visual=moves_from_visual,
            moves_from_transcript=moves_from_transcript,
            moves_inferred=moves_inferred,
            processing_time=processing_time,
        )

    finally:
        # Cleanup temp file
        if video_path:
            Path(video_path).unlink(missing_ok=True)


@app.post("/extract-visual-positions")
async def extract_visual_positions(
    video: UploadFile = File(None),
    video_url: Optional[str] = Form(None),
    sample_rate: int = Form(10),
    provider: str = Form("yolo"),
):
    """
    Extract visual board positions from video using YOLO.

    Returns positions array to be saved to question document.

    Args:
        video: Video file upload
        video_url: URL to video file
        sample_rate: Seconds between sampled frames (default 10)
        provider: "yolo" (fast, free, default)
    """
    start_time = time.time()

    if not video and not video_url:
        raise HTTPException(400, "Either video file or video_url required")

    # Set provider to YOLO (only supported provider now)
    original_provider = settings.llm_provider
    settings.llm_provider = "yolo"

    # Save uploaded video to temp file
    video_path = None
    try:
        if video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                content = await video.read()
                tmp.write(content)
                video_path = tmp.name
        elif video_url:
            # Download video from URL
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.get(video_url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(response.content)
                    video_path = tmp.name

        # Extract positions
        effective_sample_rate = 1.0 / sample_rate if sample_rate > 0 else 0.1
        logger.info(f"Extracting positions with {provider} (1 frame every {sample_rate}s)...")

        positions = board_detector.process_video(video_path, effective_sample_rate)

        processing_time = time.time() - start_time
        logger.info(f"Extracted {len(positions)} positions in {processing_time:.1f}s")

        # Convert positions to serializable format
        positions_data = [
            {
                "timestamp": p.timestamp,
                "fen": p.fen,
                "confidence": p.confidence,
                "frame_number": p.frame_number,
            }
            for p in positions
        ]

        return {
            "provider": provider,
            "positions_count": len(positions),
            "positions": positions_data,
            "processing_time": processing_time,
            "sample_rate": sample_rate,
        }

    finally:
        # Restore original provider
        settings.llm_provider = original_provider
        # Cleanup temp file
        if video_path:
            Path(video_path).unlink(missing_ok=True)


@app.post("/process-transcript")
async def process_transcript_only(
    transcript: str = Form(...),
    starting_fen: str = Form(chess.STARTING_FEN),
):
    """
    Process transcript text only (no video).

    Useful when you already have the transcript from Whisper.
    """
    # Extract moves from transcript
    moves = transcript_processor.extract_moves_from_text(transcript)

    # Validate against chess rules
    validated = transcript_processor.validate_move_sequence(moves, starting_fen)

    # Generate PGN
    pgn = generate_pgn(validated)

    return {
        "raw_moves_found": len(moves),
        "valid_moves": len(validated),
        "moves": validated,
        "pgn": pgn,
    }


@app.post("/forward-to-chess-ai")
async def forward_to_chess_ai(result: ProcessingResult):
    """
    Forward processing result to chess-ai service for question generation.

    This connects to your existing NestJS chess-ai service.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.chess_ai_url}/chess-ai/generate-questions",
            json={
                "gameText": result.pgn,
                "categoryName": result.opening_name or "Chess Tutorial",
                "difficulty": "intermediate",
                "numberOfQuestions": 5,
            },
            timeout=60.0,
        )

        if response.status_code != 200:
            raise HTTPException(
                response.status_code,
                f"Chess AI service error: {response.text}"
            )

        return response.json()


def generate_pgn(moves: list[DetectedMove]) -> str:
    """Generate PGN string from validated moves"""
    if not moves:
        return ""

    pgn_parts = []
    current_move_num = 0

    for move in moves:
        if move.move_number != current_move_num:
            current_move_num = move.move_number
            pgn_parts.append(f"{current_move_num}.")

        pgn_parts.append(move.move_san)

    return " ".join(pgn_parts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
