"""Logging configuration module using structlog."""

import logging
import logging.handlers
import sys

import structlog

from app.core.config import get_logger_settings


class JSONTCPHandler(logging.handlers.SocketHandler):
    """Sends JSON formatted log records over TCP."""
    def makePickle(self, record: logging.LogRecord) -> bytes:
        return (self.format(record) + '\n').encode('utf-8')


def setup_logging() -> None:
    """Configures production-grade structured JSON logging."""
    settings = get_logger_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Add TCP handler for Vector
    root_logger = logging.getLogger()
    tcp_handler = JSONTCPHandler(settings.vector_host, settings.vector_port)
    tcp_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(tcp_handler)


    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
