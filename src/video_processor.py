"""
Video Board Detection Module

Extracts chess board positions from video frames using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator
import chess
import logging

from .models import BoardPosition
from .config import settings

logger = logging.getLogger(__name__)


class ChessBoardDetector:
    """Detects and extracts chess positions from video frames"""

    def __init__(self):
        self.previous_fen: str | None = None
        self.frame_count = 0

    def extract_frames(
        self, video_path: str, sample_rate: int | None = None
    ) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Extract frames from video at specified sample rate.

        Args:
            video_path: Path to video file
            sample_rate: Frames per second to extract (default from settings)

        Yields:
            Tuple of (frame_number, timestamp_seconds, frame)
        """
        sample_rate = sample_rate or settings.frame_sample_rate
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate) if sample_rate < fps else 1

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

        This is a placeholder - in production, you'd use:
        1. A trained CNN model for piece recognition
        2. Or a service like chess-recognition

        For now, returns a basic detection confidence.
        """
        # TODO: Implement actual piece recognition
        # Options:
        # 1. Train a CNN on piece images
        # 2. Use template matching for known piece sets
        # 3. Use an external API/model

        # Placeholder: Return starting position
        # In real implementation, analyze each square
        logger.warning("board_to_fen: Using placeholder - implement piece recognition")

        return chess.STARTING_FEN, 0.5

    def process_video(self, video_path: str) -> list[BoardPosition]:
        """
        Process entire video and extract board positions.

        Only records positions when board state changes.
        """
        positions: list[BoardPosition] = []

        for frame_num, timestamp, frame in self.extract_frames(video_path):
            board_img, detect_confidence = self.detect_board(frame)

            if board_img is None:
                continue

            fen, fen_confidence = self.board_to_fen(board_img)
            overall_confidence = detect_confidence * fen_confidence

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
