# Chess Video Processor

AI-powered service to extract chess games from tutorial videos by combining:
- **Visual analysis** (OpenCV board detection)
- **Audio transcription** (Whisper integration)
- **LLM fusion** (LangGraph agent to combine sources)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Chess Tutorial Video                        │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Video Frames│     │    Audio    │     │ Transcript  │
   │  (OpenCV)   │     │  (Whisper)  │     │   (Text)    │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────────────────────────────────────────────┐
   │              LangGraph Fusion Agent                 │
   │  - Aligns timestamps                                │
   │  - Matches visual positions to spoken moves         │
   │  - Validates with python-chess                      │
   │  - Fills gaps using LLM                             │
   └─────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Complete Game PGN  │
                    │  + Move Timestamps  │
                    └─────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd chess-video-processor

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Development
python -m src.api

# Or with uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 3025 --reload
```

### 4. Test

```bash
# Health check
curl http://localhost:3025/health

# Process transcript only
curl -X POST http://localhost:3025/process-transcript \
  -F "transcript=Today we're looking at the Italian Game. White plays e4, black responds e5. Knight f3, knight c6. And now the key move, bishop c4."
```

## API Endpoints

### `GET /health`
Health check endpoint.

### `GET /ping`
Simple ping, returns Tailscale IP if configured.

### `POST /process-video`
Process a chess tutorial video.

**Parameters:**
- `video` (file): Video file upload
- `video_url` (string): Or URL to video
- `transcript` (string): Optional pre-existing transcript
- `sample_rate` (int): Frames per second to analyze (default: 1)

**Response:**
```json
{
  "video_id": "tutorial.mp4",
  "positions": [...],
  "moves": [...],
  "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4",
  "total_moves": 6,
  "moves_from_visual": 2,
  "moves_from_transcript": 4,
  "processing_time": 5.2
}
```

### `POST /process-transcript`
Process transcript text only (faster, no video needed).

**Parameters:**
- `transcript` (string): Transcript text
- `starting_fen` (string): Starting position (default: standard)

### `POST /forward-to-chess-ai`
Forward results to chess-ai service for question generation.

## Tailscale Setup (Production Connection)

This service runs locally and connects to your production K8s cluster via Tailscale.

### 1. Install Tailscale

Download from https://tailscale.com/download

### 2. Join Your Network

```bash
tailscale up
```

### 3. Configure Service

Add your Tailscale IP to `.env`:
```
TAILSCALE_IP=100.x.x.x
```

### 4. Access from Production

Your webapp can call:
```
http://100.x.x.x:3025/process-video
```

### 5. Orchestrator Integration (Optional)

Add to orchestrator's configured-services.ts:
```typescript
{
  serviceName: 'chess-video-processor',
  url: ENV_NAME === 'LOCAL'
    ? 'http://localhost:3025'
    : 'http://100.x.x.x:3025',  // Your Tailscale IP
  port: 3025
}
```

## Integration with Existing Services

### webapp (Angular)

```typescript
// In a service
processChessVideo(videoFile: File, transcript?: string) {
  const formData = new FormData();
  formData.append('video', videoFile);
  if (transcript) {
    formData.append('transcript', transcript);
  }

  return this.http.post(
    `${environment.chessVideoProcessorUrl}/process-video`,
    formData
  );
}
```

### chess-ai (NestJS)

After processing, forward to chess-ai for question generation:
```bash
curl -X POST http://localhost:3025/forward-to-chess-ai \
  -H "Content-Type: application/json" \
  -d '{"pgn": "1. e4 e5 2. Nf3 Nc6", ...}'
```

## Development

### Project Structure

```
chess-video-processor/
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI endpoints
│   ├── config.py           # Settings/configuration
│   ├── models.py           # Pydantic models
│   ├── video_processor.py  # OpenCV board detection
│   ├── transcript_processor.py  # Move extraction from text
│   └── fusion_agent.py     # LangGraph multi-source fusion
├── requirements.txt
├── .env.example
└── README.md
```

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format
black src/

# Lint
ruff check src/
```

## TODO / Future Improvements

- [ ] Train CNN model for piece recognition (currently placeholder)
- [ ] Add Whisper integration for direct audio processing
- [ ] Support for multiple games in single video
- [ ] Add opening book lookup
- [ ] Cache processed videos
- [ ] Add WebSocket for progress updates
- [ ] Docker containerization

## License

MIT
