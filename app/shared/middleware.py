import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


from typing import Callable, Awaitable

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique request ID to each incoming HTTP request 
    and binds it to the structlog context variables for comprehensive logging.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Generate a unique request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Bind the request ID to structlog's contextvars
        # This will automatically be included in all logs within this request's lifecycle
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        # Add to request state so other parts of the app can access it if needed
        request.state.request_id = request_id

        # Proceed with the request
        response = await call_next(request)
        
        # Return the request ID in the headers
        response.headers["X-Request-ID"] = request_id
        return response
