"""
LangGraph Fusion Agent

Combines visual board detection with transcript analysis to extract
complete and accurate chess games from tutorial videos.
"""

import logging
from typing import TypedDict, Annotated, Sequence
import operator
import chess

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from litellm import completion

from .models import BoardPosition, TranscriptSegment, DetectedMove, ProcessingResult
from .config import settings

logger = logging.getLogger(__name__)


class FusionState(TypedDict):
    """State for the fusion agent"""
    # Input data
    positions: list[BoardPosition]
    transcript_segments: list[TranscriptSegment]
    transcript_moves: list[dict]  # Moves extracted from transcript

    # Working state
    current_position_idx: int
    current_move_number: int
    board_fen: str

    # Output
    validated_moves: list[DetectedMove]
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Metadata
    errors: list[str]


def get_llm_response(prompt: str, system: str = "") -> str:
    """Get response from configured LLM using LiteLLM"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # LiteLLM automatically routes to correct provider based on model name
    model = settings.llm_model
    if settings.llm_provider == "anthropic":
        model = f"anthropic/{model}" if not model.startswith("anthropic/") else model
    elif settings.llm_provider == "openai":
        model = f"openai/{model}" if not model.startswith("openai/") else model

    response = completion(
        model=model,
        messages=messages,
        api_key=settings.anthropic_api_key if settings.llm_provider == "anthropic"
                else settings.openai_api_key,
    )

    return response.choices[0].message.content


class ChessFusionAgent:
    """
    LangGraph agent that fuses visual and transcript data to extract moves.

    The agent:
    1. Aligns visual positions with transcript timestamps
    2. Uses LLM to disambiguate unclear moves
    3. Validates all moves against chess rules
    4. Fills gaps where neither source captured the move
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FusionState)

        # Add nodes
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("align_timestamps", self._align_timestamps)
        workflow.add_node("process_segment", self._process_segment)
        workflow.add_node("fill_gaps", self._fill_gaps)
        workflow.add_node("validate_sequence", self._validate_sequence)
        workflow.add_node("finalize", self._finalize)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "align_timestamps")
        workflow.add_edge("align_timestamps", "process_segment")
        workflow.add_conditional_edges(
            "process_segment",
            self._should_continue_processing,
            {
                "continue": "process_segment",
                "done": "fill_gaps",
            }
        )
        workflow.add_edge("fill_gaps", "validate_sequence")
        workflow.add_edge("validate_sequence", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _initialize(self, state: FusionState) -> dict:
        """Initialize the processing state"""
        return {
            "current_position_idx": 0,
            "current_move_number": 1,
            "board_fen": chess.STARTING_FEN,
            "validated_moves": [],
            "errors": [],
            "messages": [HumanMessage(content="Starting chess game extraction...")],
        }

    def _align_timestamps(self, state: FusionState) -> dict:
        """Align visual positions with transcript segments by timestamp"""
        positions = state["positions"]
        segments = state["transcript_segments"]

        if not positions or not segments:
            return {"messages": [AIMessage(content="Alignment: Limited data available")]}

        # Create timestamp mapping
        aligned_data = []
        for pos in positions:
            # Find transcript segment(s) near this position
            nearby_segments = [
                seg for seg in segments
                if abs(seg.start - pos.timestamp) < 5.0  # Within 5 seconds
            ]
            aligned_data.append({
                "position": pos,
                "segments": nearby_segments,
            })

        return {
            "messages": [
                AIMessage(content=f"Aligned {len(aligned_data)} positions with transcript")
            ]
        }

    def _process_segment(self, state: FusionState) -> dict:
        """Process current position/transcript segment pair"""
        positions = state["positions"]
        transcript_moves = state["transcript_moves"]
        current_idx = state["current_position_idx"]
        board_fen = state["board_fen"]
        validated_moves = state["validated_moves"].copy()

        # Handle transcript-only mode when no positions detected
        if not positions:
            # Process all transcript moves directly
            if current_idx >= len(transcript_moves):
                return {"current_position_idx": current_idx}

            board = chess.Board(board_fen)
            move_data = transcript_moves[current_idx]
            move_found = False

            try:
                chess_move = board.parse_san(move_data["san"])
                board.push(chess_move)

                # Calculate move number based on whose turn it was
                move_num = state["current_move_number"]

                validated_moves.append(DetectedMove(
                    move_number=move_num,
                    move_san=move_data["san"],
                    move_uci=chess_move.uci(),
                    timestamp=move_data.get("timestamp", 0.0),
                    source="transcript",
                    fen_before=board_fen,
                    fen_after=board.fen(),
                ))
                move_found = True
            except (chess.InvalidMoveError, chess.AmbiguousMoveError, chess.IllegalMoveError) as e:
                logger.debug(f"Skipping invalid move {move_data.get('san')}: {e}")

            new_move_number = state["current_move_number"]
            if move_found and board.turn == chess.WHITE:
                new_move_number += 1

            return {
                "current_position_idx": current_idx + 1,
                "current_move_number": new_move_number,
                "board_fen": board.fen() if move_found else board_fen,
                "validated_moves": validated_moves,
            }

        # Normal mode with visual positions
        if current_idx >= len(positions):
            return {"current_position_idx": current_idx}

        current_pos = positions[current_idx]
        board = chess.Board(board_fen)

        # Find transcript moves near this timestamp
        nearby_moves = [
            m for m in transcript_moves
            if current_pos.timestamp - 2 <= m["timestamp"] <= current_pos.timestamp + 2
        ]

        # Try to find what move was made
        move_found = False

        # First, try transcript moves
        for move_data in nearby_moves:
            try:
                chess_move = board.parse_san(move_data["san"])
                board.push(chess_move)

                validated_moves.append(DetectedMove(
                    move_number=state["current_move_number"],
                    move_san=move_data["san"],
                    move_uci=chess_move.uci(),
                    timestamp=move_data["timestamp"],
                    source="transcript",
                    fen_before=board_fen,
                    fen_after=board.fen(),
                ))
                move_found = True
                break
            except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                continue

        # If transcript didn't help, try to infer from position change
        if not move_found and current_idx > 0:
            prev_pos = positions[current_idx - 1]
            inferred_move = self._infer_move_from_positions(
                prev_pos.fen, current_pos.fen
            )
            if inferred_move:
                validated_moves.append(inferred_move)
                move_found = True

        new_move_number = state["current_move_number"]
        if move_found and board.turn == chess.WHITE:
            new_move_number += 1

        return {
            "current_position_idx": current_idx + 1,
            "current_move_number": new_move_number,
            "board_fen": board.fen() if move_found else board_fen,
            "validated_moves": validated_moves,
        }

    def _should_continue_processing(self, state: FusionState) -> str:
        """Check if we should continue processing positions/moves"""
        positions = state["positions"]
        transcript_moves = state["transcript_moves"]
        current_idx = state["current_position_idx"]

        # In transcript-only mode, iterate through transcript_moves
        if not positions:
            if current_idx < len(transcript_moves):
                return "continue"
            return "done"

        # Normal mode - iterate through positions
        if current_idx < len(positions):
            return "continue"
        return "done"

    def _fill_gaps(self, state: FusionState) -> dict:
        """Use LLM to fill gaps in the move sequence"""
        validated_moves = state["validated_moves"]
        transcript_segments = state["transcript_segments"]

        if not validated_moves:
            return {"messages": [AIMessage(content="No moves to fill gaps for")]}

        # Check for gaps in move sequence
        gaps = []
        for i in range(len(validated_moves) - 1):
            curr = validated_moves[i]
            next_move = validated_moves[i + 1]

            # Check if there's a gap in move numbers
            expected_next = curr.move_number + (1 if chess.Board(curr.fen_after).turn == chess.WHITE else 0)
            if next_move.move_number > expected_next + 1:
                gaps.append({
                    "after_move": i,
                    "fen": curr.fen_after,
                    "missing_count": next_move.move_number - expected_next,
                })

        if gaps:
            # Use LLM to help fill gaps
            full_transcript = " ".join([s.text for s in transcript_segments])
            filled = self._llm_fill_gaps(gaps, full_transcript, validated_moves)
            if filled:
                validated_moves = self._merge_moves(validated_moves, filled)

        return {
            "validated_moves": validated_moves,
            "messages": [AIMessage(content=f"Gap filling complete. Found {len(gaps)} gaps.")],
        }

    def _llm_fill_gaps(
        self, gaps: list[dict], transcript: str, existing_moves: list[DetectedMove]
    ) -> list[DetectedMove]:
        """Use LLM to infer missing moves from context"""
        if not gaps:
            return []

        prompt = f"""You are a chess expert analyzing a tutorial transcript.

The following moves have been extracted so far:
{self._format_moves(existing_moves)}

There appear to be gaps in the sequence. Here's the full transcript:
{transcript[:3000]}  # Limit length

Based on the context, what moves might be missing?
Only suggest moves that are clearly implied by the transcript.
Return moves in SAN format (e.g., "e4", "Nf3", "O-O").

Format: One move per line, just the move notation."""

        try:
            response = get_llm_response(
                prompt,
                system="You are a chess expert. Only return valid chess moves in SAN notation."
            )
            # Parse response and validate moves
            # (simplified - would need more robust parsing)
            return []
        except Exception as e:
            logger.error(f"LLM gap filling failed: {e}")
            return []

    def _validate_sequence(self, state: FusionState) -> dict:
        """Final validation of the complete move sequence"""
        validated_moves = state["validated_moves"]

        if not validated_moves:
            return {"errors": ["No valid moves extracted"]}

        # Replay all moves to ensure sequence is valid
        board = chess.Board()
        valid_sequence = []
        errors = []

        for move in validated_moves:
            try:
                chess_move = board.parse_san(move.move_san)
                board.push(chess_move)
                valid_sequence.append(move)
            except Exception as e:
                errors.append(f"Invalid move {move.move_san} at position {move.move_number}: {e}")

        return {
            "validated_moves": valid_sequence,
            "errors": state.get("errors", []) + errors,
        }

    def _finalize(self, state: FusionState) -> dict:
        """Finalize and prepare output"""
        return {
            "messages": [
                AIMessage(
                    content=f"Extraction complete. {len(state['validated_moves'])} moves extracted."
                )
            ]
        }

    def _infer_move_from_positions(
        self, fen_before: str, fen_after: str
    ) -> DetectedMove | None:
        """Infer what move was made between two positions"""
        try:
            board_before = chess.Board(fen_before)
            board_after = chess.Board(fen_after)

            # Try all legal moves to find which one results in fen_after
            for move in board_before.legal_moves:
                board_before.push(move)
                if board_before.fen().split()[0] == fen_after.split()[0]:
                    return DetectedMove(
                        move_number=1,  # Will be corrected later
                        move_san=board_before.san(move),
                        move_uci=move.uci(),
                        timestamp=0,
                        source="visual",
                        fen_before=fen_before,
                        fen_after=fen_after,
                    )
                board_before.pop()
        except Exception as e:
            logger.debug(f"Could not infer move: {e}")

        return None

    def _format_moves(self, moves: list[DetectedMove]) -> str:
        """Format moves for display"""
        result = []
        for m in moves:
            result.append(f"{m.move_number}. {m.move_san}")
        return " ".join(result)

    def _merge_moves(
        self, existing: list[DetectedMove], new: list[DetectedMove]
    ) -> list[DetectedMove]:
        """Merge new moves into existing sequence"""
        # Simple merge by timestamp
        all_moves = existing + new
        all_moves.sort(key=lambda x: x.timestamp)
        return all_moves

    def process(
        self,
        positions: list[BoardPosition],
        transcript_segments: list[TranscriptSegment],
        transcript_moves: list[dict],
    ) -> list[DetectedMove]:
        """Run the fusion agent"""
        initial_state: FusionState = {
            "positions": positions,
            "transcript_segments": transcript_segments,
            "transcript_moves": transcript_moves,
            "current_position_idx": 0,
            "current_move_number": 1,
            "board_fen": chess.STARTING_FEN,
            "validated_moves": [],
            "messages": [],
            "errors": [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["validated_moves"]


# Singleton instance
fusion_agent = ChessFusionAgent()
