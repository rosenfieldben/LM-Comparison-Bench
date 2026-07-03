"""Helpers for mocking OpenRouter SSE streams through respx."""

import json

import httpx


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


DONE_MARKER = b"data: [DONE]\n\n"


class ChunkStream(httpx.AsyncByteStream):
    """Byte stream that can die mid-flight, like a dropped connection."""

    def __init__(self, chunks: list[bytes], exc: Exception | None = None):
        self._chunks = chunks
        self._exc = exc

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._exc is not None:
            raise self._exc
