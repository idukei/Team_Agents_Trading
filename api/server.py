"""
TeamTrade Dashboard — FastAPI server.

Endpoints:
  GET  /              → frontend/index.html
  WS   /ws            → real-time state updates (WebSocket)
  GET  /api/health    → {"status": "ok"}
  GET  /api/state     → current state snapshot (JSON)
  POST /api/chat      → SSE streaming chat (OpenRouter qwen3.6-plus:free)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.state_broadcaster import broadcaster
from agents.chat.agent import chat_agent

log = structlog.get_logger(__name__)

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

app = FastAPI(
    title="TeamTrade Dashboard",
    description="Real-time trading system dashboard",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)


# ── Static frontend ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the single-page dashboard."""
    if not FRONTEND_PATH.exists():
        return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)
    return FileResponse(FRONTEND_PATH, media_type="text/html")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ws_clients": len(broadcaster._clients),
    }


# ── State snapshot ────────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state() -> JSONResponse:
    """Return the latest broadcasted state snapshot."""
    return JSONResponse(broadcaster._latest_state)


# ── WebSocket — real-time state updates ───────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await broadcaster.connect(ws)
    try:
        while True:
            # Keep connection alive; state is pushed from broadcaster
            await asyncio.sleep(1)
            # Handle any incoming ping messages from client
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        await broadcaster.disconnect(ws)


# ── Chat — SSE streaming via OpenRouter ───────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream a chat response from OpenRouter qwen/qwen3.6-plus:free.

    Returns Server-Sent Events (SSE). Each event is a JSON object:
      data: {"delta": "text chunk"}
      data: {"done": true}
    """
    current_state = broadcaster._latest_state

    async def event_generator():
        try:
            async for chunk in chat_agent.stream(request.messages, current_state):
                # Escape newlines in delta for SSE format
                payload = json.dumps({"delta": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as exc:
            log.warning("chat_sse_error", error=str(exc))
            yield f"data: {json.dumps({'delta': f' [Error: {exc}]'})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
