"""HTTP middleware shared across all bounded contexts.

Currently provides :class:`CorrelationIdMiddleware`, which stamps every
request with a correlation ID so that all log lines emitted during that
request (via ``structlog``) can be traced end-to-end in Loki/Grafana.
"""

import uuid
from typing import Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assigns a unique request ID to each incoming HTTP request.

    The ID is taken from the ``X-Request-ID`` request header if present
    (allowing upstream proxies/gateways to propagate their own trace ID),
    otherwise a fresh UUID4 is generated.

    The ID is then:
        1. Bound to ``structlog.contextvars`` — the ``merge_contextvars``
           processor in ``app/shared/logging.py`` injects it into every
           log line produced during this request.
        2. Stored on ``request.state.request_id`` for ad-hoc access.
        3. Echoed back in the ``X-Request-ID`` response header so clients
           can correlate a response with their logs.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Reuse an upstream-provided ID if present, else mint a new one.
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Reset per-request context so IDs from a previous request on a
        # reused worker don't leak in.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        request.state.request_id = request_id

        response = await call_next(request)

        # Echo the ID back so clients can correlate response ↔ logs.
        response.headers["X-Request-ID"] = request_id
        return response
