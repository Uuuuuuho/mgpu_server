"""
Network utilities for Multi-GPU Scheduler
"""

import socket
import json
import time
from typing import Any, Dict, Optional
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def send_json_message(sock: socket.socket, message: Dict[str, Any], timeout: float = 30.0) -> bool:
    """Send a JSON message over a socket"""
    try:
        sock.settimeout(timeout)
        data = json.dumps(message).encode('utf-8')
        sock.send(data)
        return True
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return False


def receive_json_message(sock: socket.socket, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Receive a JSON message from a socket"""
    try:
        sock.settimeout(timeout)
        data = sock.recv(8192)
        if not data:
            return None
        return json.loads(data.decode('utf-8'))
    except socket.timeout:
        logger.debug("Socket timeout while receiving message")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON message: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to receive message: {e}")
        return None


def create_server_socket(host: str, port: int, backlog: int = 5) -> Optional[socket.socket]:
    """Create and bind a server socket"""
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(backlog)
        logger.info(f"Server socket created and listening on {host}:{port}")
        return server_socket
    except Exception as e:
        logger.error(f"Failed to create server socket on {host}:{port}: {e}")
        return None


def connect_to_server(host: str, port: int, timeout: float = 30.0, retries: int = 3) -> Optional[socket.socket]:
    """Connect to a server with retry logic"""
    for attempt in range(retries):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            client_socket.connect((host, port))
            logger.debug(f"Connected to {host}:{port}")
            return client_socket
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to connect to {host}:{port} after {retries} attempts")
    return None


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except:
        return False
