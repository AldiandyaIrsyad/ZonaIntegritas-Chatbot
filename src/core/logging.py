"""
Logging setup and configuration.

Provides structured JSON logging via TCP to Vector, and standard output logging.
"""
import logging
import os
import queue
import socket
import sys
import threading
import time
from logging.handlers import QueueHandler, QueueListener

from pythonjsonlogger.json import JsonFormatter

from .config import get_vector_settings

_logging_initialized = False
_logging_init_lock = threading.Lock()

class TCPJSONHandler(logging.Handler):
    """A custom logging handler that sends JSON-formatted logs over TCP.
    
    Args:
        host (str): The host address of the TCP receiver.
        port (int): The port of the TCP receiver.
    """
    def __init__(self, host, port):
        super().__init__()
        self.address = (host, port)
        self.sock = None
        self.last_retry = 0.0
        self.retry_cooldown = 5.0

    def _connect(self):
        now = time.time()
        if now - self.last_retry < self.retry_cooldown:
            return
        self.last_retry = now
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            self.sock.connect(self.address)
            self.sock.settimeout(None)
        except Exception:
            self.sock = None

    def emit(self, record):
        try:
            # TCP requires a delimiter, so we add a newline.
            msg = self.format(record) + "\n"
            if not self.sock:
                self._connect()
            if self.sock:
                self.sock.sendall(msg.encode('utf-8'))
        except Exception:
            self.sock = None
            self.handleError(record)


def setup_logging():
    """Initializes the logging configuration for the application.
    
    Sets up JSON formatting, a TCP handler for remote logging (e.g., Vector),
    and a stream handler for stdout, all processed asynchronously via a QueueHandler.
    """
    global _logging_initialized
    with _logging_init_lock:
        if _logging_initialized:
            return
        _logging_initialized = True

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Explicitly enable INFO logging for our application code
    logging.getLogger("src").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)

    formatter = JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={"levelname": "level", "asctime": "timestamp"}
    )

    # TCP Handler — sends logs directly to the Vector container
    vector_settings = get_vector_settings()
    tcp_handler = TCPJSONHandler(vector_settings.host, vector_settings.port)
    tcp_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    listener = QueueListener(
        log_queue,
        tcp_handler,
        stream_handler,
        respect_handler_level=True
    )
    listener.start()



def get_logger(name):
    """Gets a logger instance configured with the application's logging setup.
    
    Args:
        name (str): The name of the logger (usually __name__).
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)

