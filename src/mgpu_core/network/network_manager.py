"""
Network utilities for Multi-GPU Scheduler
"""

import socket
import json
import logging
from typing import Dict, Optional, Any


logger = logging.getLogger(__name__)


class NetworkManager:
    """Handles network communication with timeout and error handling"""
    
    @staticmethod
    def connect_to_server(host: str, port: int, timeout: Optional[float] = 10.0) -> Optional[socket.socket]:
        """Create connection to server with timeout"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if timeout is not None:
                sock.settimeout(timeout)
            sock.connect((host, port))
            return sock
        except Exception as e:
            logger.error(f"Failed to connect to {host}:{port}: {e}")
            return None
    
    @staticmethod
    def send_json_message(sock: socket.socket, message: Dict[str, Any], timeout: Optional[float] = 10.0) -> bool:
        """Send JSON message with timeout"""
        try:
            if timeout is not None:
                sock.settimeout(timeout)
            data = json.dumps(message).encode()
            sock.send(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    @staticmethod
    def receive_json_message(sock: socket.socket, timeout: Optional[float] = 10.0) -> Optional[Dict[str, Any]]:
        """Receive JSON message with timeout"""
        try:
            if timeout is not None:
                sock.settimeout(timeout)
            data = sock.recv(8192).decode()
            if not data:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    @staticmethod
    def send_to_node(node_info, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to node and get response"""
        try:
            sock = NetworkManager.connect_to_server(node_info.host, node_info.port)
            if not sock:
                return None
            
            if not NetworkManager.send_json_message(sock, message):
                sock.close()
                return None
            
            response = NetworkManager.receive_json_message(sock)
            sock.close()
            
            # Reset failure count on successful communication
            node_info.failure_count = 0
            return response
            
        except Exception as e:
            logger.error(f"Failed to send to node {node_info.node_id}: {e}")
            
            # Track failure count
            node_info.failure_count = getattr(node_info, 'failure_count', 0) + 1
            logger.warning(f"Node {node_info.node_id} failure count: {node_info.failure_count}")
            
            return None
