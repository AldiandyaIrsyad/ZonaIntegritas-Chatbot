import logging
import sys
import os
from pythonjsonlogger.json import JsonFormatter

import socket

class TCPJSONHandler(logging.Handler):
    def __init__(self, host, port):
        super().__init__()
        self.address = (host, port)
        self.sock = None

    def _connect(self):
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

_logging_initialized = False

def setup_logging():
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={"levelname": "level", "asctime": "timestamp"}
    )

    # TCP Handler — sends logs directly to the Vector container
    tcp_handler = TCPJSONHandler('127.0.0.1', 9000)
    tcp_handler.setFormatter(formatter)
    root_logger.addHandler(tcp_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

def get_logger(name):
    setup_logging()
    return logging.getLogger(name)

