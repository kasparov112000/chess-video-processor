"""
Quick test script for Moondream vision board detection.
"""
import asyncio
import base64
import httpx
import sys

# Test with a chess starting position image URL
# We'll use a simple test with base64 encoded board

async def test_moondream():
    """Test Moondream vision with a chess board description."""

    print("Testing Moondream connection...")

    # First test basic connection
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get("http://localhost:11434/api/version")
            print(f"Ollama version: {response.json()}")
        except Exception as e:
            print(f"Cannot connect to Ollama: {e}")
            return False

    # Test vision capability with a simple prompt (no image first)
    print("\nTesting text-only prompt...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "What is the starting position in chess in FEN notation?",
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()
            resp_text = result.get('response', 'No response')[:500]
            # Handle encoding issues on Windows
            print(f"Response: {resp_text.encode('ascii', 'replace').decode('ascii')}")
        else:
            print(f"Error: {response.status_code}")
            return False

    print("\n[OK] Moondream is working!")
    print("\nTo test with an actual chess board image:")
    print("1. Save a chess board screenshot as 'test_board.png'")
    print("2. Run the chess-video-processor service")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_moondream())
    sys.exit(0 if success else 1)
