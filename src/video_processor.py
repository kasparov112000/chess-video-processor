"""
Video Board Detection Module

Extracts chess board positions from video frames using:
- YOLOv8 with screenshot_reader model (fast, local, free) - default
- Moondream (local, experimental fallback)
"""

import cv2
import numpy as np
import base64
import httpx
import asyncio
from pathlib import Path
from typing import Generator
import chess
import logging
import re

from .models import BoardPosition
from .config import settings

logger = logging.getLogger(__name__)

# YOLO models (lazy loaded)
_yolo_model = None
_screenshot_reader_model = None

def get_screenshot_reader_model():
    """Lazy load screenshot reader model (best for Lichess/Chess.com screenshots)"""
    global _screenshot_reader_model
    if _screenshot_reader_model is None:
        try:
            from ultralytics import YOLO
            model_path = Path(__file__).parent.parent / "chess_screenshot_reader" / "model" / "screenshotReader_endToEnd_model2.pt"
            if model_path.exists():
                _screenshot_reader_model = YOLO(str(model_path))
                logger.info(f"Loaded screenshot reader model from {model_path}")
            else:
                logger.warning(f"Screenshot reader model not found at {model_path}")
        except ImportError:
            logger.warning("ultralytics not installed, YOLO detection unavailable")
    return _screenshot_reader_model

def get_yolo_model():
    """Lazy load YOLO model (original trained model, fallback)"""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = Path(__file__).parent.parent / "runs" / "chess" / "chess_pieces" / "weights" / "best.pt"
            if model_path.exists():
                _yolo_model = YOLO(str(model_path))
                logger.info(f"Loaded YOLO chess model from {model_path}")
            else:
                logger.warning(f"YOLO model not found at {model_path}")
        except ImportError:
            logger.warning("ultralytics not installed, YOLO detection unavailable")
    return _yolo_model


class ChessBoardDetector:
    """Detects and extracts chess positions from video frames"""

    def __init__(self):
        self.previous_fen: str | None = None
        self.frame_count = 0

    def extract_frames(
        self, video_path: str, sample_rate: float | None = None
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames from video at specified sample rate.

        Args:
            video_path: Path to video file
            sample_rate: Frames per second to extract (default from settings).
                        Use small values like 0.033 for 1 frame per 30 seconds.

        Yields:
            Tuple of (frame_number, timestamp_seconds, frame)
        """
        sample_rate = sample_rate or settings.frame_sample_rate
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return  # Don't raise, just return empty

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"Video: {duration:.0f}s, {fps:.0f} fps, {total_frames} frames")

        # Calculate frame interval (how many frames to skip)
        frame_interval = max(1, int(fps / sample_rate)) if sample_rate > 0 and sample_rate < fps else 1
        expected_samples = total_frames // frame_interval
        logger.info(f"Sampling every {frame_interval} frames (~{expected_samples} samples)")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                timestamp = frame_num / fps
                yield frame_num, timestamp, frame

            frame_num += 1

        cap.release()

    def detect_board(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """
        Detect chess board in frame and extract it.

        Returns:
            Tuple of (extracted_board_image, confidence)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find chessboard corners (works for physical boards)
        # For screen captures, we'll use contour detection
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        if ret:
            # Physical board detected
            return self._extract_board_from_corners(frame, corners), 0.9

        # Try contour-based detection for screen captures
        board, confidence = self._detect_board_contours(frame)
        return board, confidence

    def _detect_board_contours(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Detect board using contour detection (for screen captures)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find largest square-ish contour (likely the board)
        best_contour = None
        best_area = 0

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > best_area and area > 10000:  # Min area threshold
                    # Check if roughly square
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.8 < aspect_ratio < 1.2:
                        best_contour = approx
                        best_area = area

        if best_contour is not None:
            # Extract and warp the board to a standard size
            board = self._warp_perspective(frame, best_contour)
            return board, 0.7

        return None, 0.0

    def _extract_board_from_corners(
        self, frame: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Extract board image using detected corners"""
        # Get bounding rectangle
        corners = corners.reshape(-1, 2)
        x_min, y_min = corners.min(axis=0).astype(int)
        x_max, y_max = corners.max(axis=0).astype(int)

        # Add margin for full board
        margin = int((x_max - x_min) / 7)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(frame.shape[1], x_max + margin)
        y_max = min(frame.shape[0], y_max + margin)

        board = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(board, (400, 400))

    def _warp_perspective(
        self, frame: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        """Warp detected board to standard square"""
        pts = contour.reshape(4, 2).astype(np.float32)

        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(pts)

        dst = np.array(
            [[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.float32
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (400, 400))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def board_to_fen(self, board_image: np.ndarray) -> tuple[str, float]:
        """
        Convert board image to FEN notation.

        Provider selection:
        - provider=yolo: Use YOLO detection (fast, free) - default
        - provider=moondream: Use Moondream (experimental fallback)

        Returns:
            Tuple of (fen_string, confidence)
        """
        try:
            # YOLO is the primary provider (fast and free)
            if settings.llm_provider == "yolo" or get_screenshot_reader_model() is not None or get_yolo_model() is not None:
                return self._board_to_fen_yolo(board_image)

            # Fallback to async Moondream
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._board_to_fen_async(board_image))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error in board_to_fen: {e}")
            return chess.STARTING_FEN, 0.1

    async def _board_to_fen_async(self, board_image: np.ndarray) -> tuple[str, float]:
        """
        Async implementation of board to FEN conversion using Moondream.
        Fallback when YOLO model is not available.
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.png', board_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare prompt for FEN extraction
            prompt = """Look at this chess board image and identify the position of all pieces.
Then output the FEN (Forsyth-Edwards Notation) for this position.

FEN format rules:
- Start from rank 8 (top) to rank 1 (bottom)
- Use: K=white king, Q=white queen, R=white rook, B=white bishop, N=white knight, P=white pawn
- Use: k=black king, q=black queen, r=black rook, b=black bishop, n=black knight, p=black pawn
- Numbers 1-8 represent empty squares
- Ranks separated by /
- Add " w - - 0 1" at end for standard suffix

Example starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Output ONLY the FEN string, nothing else."""

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.ollama_url}/api/generate",
                    json={
                        "model": settings.ollama_vision_model,
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False
                    }
                )

                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return chess.STARTING_FEN, 0.1

                result = response.json()
                raw_response = result.get("response", "").strip()
                logger.debug(f"Moondream raw response: {raw_response}")

                # Extract FEN from response
                fen = self._extract_fen_from_response(raw_response)

                if fen and self._validate_fen(fen):
                    logger.info(f"Successfully extracted FEN: {fen}")
                    return fen, 0.85
                else:
                    logger.warning(f"Invalid FEN extracted: {fen}")
                    return chess.STARTING_FEN, 0.3

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Make sure Ollama is running.")
            return chess.STARTING_FEN, 0.1
        except Exception as e:
            logger.error(f"Error calling Moondream: {e}")
            return chess.STARTING_FEN, 0.1

    def _extract_fen_from_response(self, response: str) -> str | None:
        """Extract FEN string from model response."""
        # Try to find FEN pattern in response
        # FEN pattern: pieces/pieces/pieces/pieces/pieces/pieces/pieces/pieces [w|b] ...
        fen_pattern = r'([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+(\s+[wb]\s+[\-KQkq]+\s+[\-a-h1-8]+\s+\d+\s+\d+)?'

        match = re.search(fen_pattern, response)
        if match:
            fen = match.group(0)
            # If no suffix, add standard one
            if ' ' not in fen:
                fen = f"{fen} w - - 0 1"
            return fen

        # Try to extract just the piece placement part
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if '/' in line and len(line.split('/')) == 8:
                # Looks like piece placement
                parts = line.split()
                piece_placement = parts[0]
                if self._is_valid_piece_placement(piece_placement):
                    return f"{piece_placement} w - - 0 1"

        return None

    def _is_valid_piece_placement(self, placement: str) -> bool:
        """Check if piece placement string is valid."""
        ranks = placement.split('/')
        if len(ranks) != 8:
            return False

        valid_chars = set('rnbqkpRNBQKP12345678/')
        if not all(c in valid_chars for c in placement):
            return False

        # Check each rank sums to 8
        for rank in ranks:
            count = 0
            for c in rank:
                if c.isdigit():
                    count += int(c)
                else:
                    count += 1
            if count != 8:
                return False

        return True

    def _validate_fen(self, fen: str) -> bool:
        """
        Validate FEN string using python-chess.

        Uses lenient validation - just checks if FEN can be parsed,
        not if position is reachable (which is too strict for video frames
        where detection might make minor errors).
        """
        try:
            # Just check if python-chess can parse it
            board = chess.Board(fen)
            # Basic checks: has both kings, not too many pieces
            white_kings = len(board.pieces(chess.KING, chess.WHITE))
            black_kings = len(board.pieces(chess.KING, chess.BLACK))
            return white_kings == 1 and black_kings == 1
        except (ValueError, Exception):
            return False

    def _board_to_fen_yolo(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Extract FEN using local YOLO model.

        Steps:
        1. Detect all pieces with YOLO (prefers screenshot_reader model for Lichess/Chess.com)
        2. Find board boundaries from piece positions
        3. Map each piece to a square (8x8 grid)
        4. Generate FEN string
        """
        # Try screenshot reader model first (better for Lichess/Chess.com screenshots)
        model = get_screenshot_reader_model()
        if model is None:
            # Fall back to original YOLO model
            model = get_yolo_model()
        if model is None:
            logger.warning("No YOLO model available")
            return chess.STARTING_FEN, 0.1

        # Run detection
        results = model(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes

        # Collect piece detections (exclude 'board' class)
        detections = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            name = model.names[cls]
            # Skip 'board' class from screenshot reader model
            if name == 'board':
                continue
            detections.append({
                'name': name,
                'conf': conf,
                'cx': cx,
                'cy': cy,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

        if len(detections) < 4:
            logger.debug(f"Too few pieces detected ({len(detections)}), skipping")
            return chess.STARTING_FEN, 0.1

        # Find board boundaries from piece positions
        all_x = [d['cx'] for d in detections]
        all_y = [d['cy'] for d in detections]

        # Estimate board boundaries with margin
        min_x = min(all_x) - 40
        max_x = max(all_x) + 40
        min_y = min(all_y) - 40
        max_y = max(all_y) + 40

        # Make it square for better accuracy
        size = max(max_x - min_x, max_y - min_y)
        center_x = (min(all_x) + max(all_x)) / 2
        center_y = (min(all_y) + max(all_y)) / 2
        min_x = center_x - size / 2
        max_x = center_x + size / 2
        min_y = center_y - size / 2
        max_y = center_y + size / 2

        board_width = max_x - min_x
        board_height = max_y - min_y

        if board_width < 50 or board_height < 50:
            return chess.STARTING_FEN, 0.1

        # Map pieces to 8x8 grid
        board = [[None for _ in range(8)] for _ in range(8)]

        # YOLO class names to FEN pieces (both naming conventions)
        piece_map = {
            # Screenshot reader model uses underscores
            'white_king': 'K', 'white_queen': 'Q', 'white_rook': 'R',
            'white_bishop': 'B', 'white_knight': 'N', 'white_pawn': 'P',
            'black_king': 'k', 'black_queen': 'q', 'black_rook': 'r',
            'black_bishop': 'b', 'black_knight': 'n', 'black_pawn': 'p',
            # Original YOLO model uses hyphens
            'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
            'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
            'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
            'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
            'bishop': 'B',  # Generic bishop treated as white
        }

        for det in detections:
            # Calculate grid position (0-7)
            col = int((det['cx'] - min_x) / board_width * 8)
            row = int((det['cy'] - min_y) / board_height * 8)

            # Clamp to valid range
            col = max(0, min(7, col))
            row = max(0, min(7, row))

            piece = piece_map.get(det['name'])
            if piece:
                # If square already occupied, keep higher confidence
                if board[row][col] is None or det['conf'] > board[row][col][1]:
                    board[row][col] = (piece, det['conf'])

        # Generate FEN
        fen_rows = []
        for row in board:
            fen_row = ""
            empty = 0
            for cell in row:
                if cell is None:
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell[0]
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)

        piece_placement = "/".join(fen_rows)
        fen = f"{piece_placement} w - - 0 1"

        # Calculate confidence as average of piece confidences
        confs = [d['conf'] for d in detections]
        avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Validate the FEN
        if self._validate_fen(fen):
            logger.info(f"YOLO extracted FEN ({len(detections)} pieces): {piece_placement[:40]}...")
            return fen, avg_conf
        else:
            logger.warning(f"Invalid YOLO FEN: {fen}")
            return chess.STARTING_FEN, 0.2

    def process_video(self, video_path: str, sample_rate: float | None = None) -> list[BoardPosition]:
        """
        Process entire video and extract board positions.

        Args:
            video_path: Path to video file
            sample_rate: Frames per second to extract.
                        - YOLO is fast, can use higher rates (1-5 fps)

        Only records positions when board state changes.
        YOLO works on full frames (no OpenCV board detection needed).
        """
        # Reset state for new video
        self.previous_fen = None

        positions: list[BoardPosition] = []

        # YOLO works on full frames
        use_full_frame = get_screenshot_reader_model() is not None or get_yolo_model() is not None or settings.llm_provider == "yolo"

        for frame_num, timestamp, frame in self.extract_frames(video_path, sample_rate):
            if use_full_frame:
                # YOLO: send full frame, it finds the board
                board_img = frame
                detect_confidence = 1.0
            else:
                # OpenCV detection for Moondream
                board_img, detect_confidence = self.detect_board(frame)
                if board_img is None:
                    continue

            fen, fen_confidence = self.board_to_fen(board_img)
            overall_confidence = detect_confidence * fen_confidence

            # Skip unclear frames
            if fen_confidence < 0.3:
                logger.debug(f"Skipping unclear frame at {timestamp:.2f}s")
                continue

            # Only record if position changed
            if fen != self.previous_fen:
                positions.append(
                    BoardPosition(
                        timestamp=timestamp,
                        fen=fen,
                        confidence=overall_confidence,
                        frame_number=frame_num,
                    )
                )
                self.previous_fen = fen
                logger.info(f"New position at {timestamp:.2f}s: {fen[:30]}...")

        return positions


# Singleton instance
board_detector = ChessBoardDetector()
