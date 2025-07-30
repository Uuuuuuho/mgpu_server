"""
Node Manager for Multi-GPU Master Server
"""

import socket
import time
import sys
import os
from typing import Dict, Optional, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu_core.models.job_models import NodeInfo
from mgpu_core.network.network_manager import NetworkManager
from mgpu_core.utils.logging_utils import setup_logger
from mgpu_core.utils.system_utils import IPManager


logger = setup_logger(__name__)


class NodeManager:
    """Manages node registration and communication"""
    
    def __init__(self):
        self.nodes = {}  # node_id -> NodeInfo
    
    def register_node(self, request: Dict) -> Dict:
        """Handle node registration"""
        try:
            node_id = request.get('node_id')
            host = request.get('host', '127.0.0.1')
            port = request.get('port', 8081)
            gpu_count = request.get('gpu_count', 1)
            gpu_info = request.get('gpu_info', [])
            
            if not node_id:
                return {'status': 'error', 'message': 'node_id required'}
            
            # Enhanced logging with node registration details
            logger.info(f"Node registration request:")
            logger.info(f"   ├─ Node ID: {node_id}")
            logger.info(f"   ├─ Host IP: {host}")
            logger.info(f"   ├─ Port: {port}")
            logger.info(f"   └─ GPU Count: {gpu_count}")
            
            # Add or update node
            self.add_node(node_id, host, port, gpu_count)
            
            # Reset failure count on successful registration
            if node_id in self.nodes:
                self.nodes[node_id].failure_count = 0
            
            # Enhanced logging with GPU information
            if gpu_info:
                logger.info(f"Node {node_id} connected from {host}:{port}")
                logger.info(f"   └─ {gpu_count} GPU(s) detected:")
                for gpu in gpu_info:
                    logger.info(f"      ├─ GPU {gpu['id']}: {gpu['name']} ({gpu['memory']})")
            else:
                logger.info(f"Node {node_id} connected from {host}:{port}")
                logger.info(f"   └─ {gpu_count} GPU(s) available")
            
            # Test connectivity back to node
            if self.test_node_connectivity(node_id):
                logger.info(f"Master can connect back to {node_id} at {host}:{port}")
            else:
                logger.warning(f"Master cannot connect back to {node_id} at {host}:{port}")
            
            return {'status': 'ok', 'message': f'Node {node_id} registered successfully'}
            
        except Exception as e:
            logger.error(f"Node registration error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def add_node(self, node_id: str, host: str, port: int, gpu_count: int):
        """Add a node to the cluster"""
        self.nodes[node_id] = NodeInfo(node_id, host, port, gpu_count)
        logger.info(f"Node {node_id} added to cluster (total nodes: {len(self.nodes)})")
    
    def test_node_connectivity(self, node_id: str) -> bool:
        """Test if master can connect back to node"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(5.0)
            test_sock.connect((node.host, node.port))
            test_sock.close()
            return True
        except Exception as e:
            logger.debug(f"Connectivity test failed for {node_id}: {e}")
            return False
    
    def get_node_health_status(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive health status of a node"""
        if node_id not in self.nodes:
            return {'status': 'not_found', 'healthy': False}
        
        node = self.nodes[node_id]
        current_time = time.time()
        
        # Calculate health metrics
        failure_count = getattr(node, 'failure_count', 0)
        last_heartbeat = getattr(node, 'last_heartbeat', 0)
        time_since_heartbeat = current_time - last_heartbeat
        
        # Determine health status
        is_healthy = (
            failure_count < 3 and 
            time_since_heartbeat < 300  # 5 minutes
        )
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'healthy': is_healthy,
            'failure_count': failure_count,
            'time_since_heartbeat': time_since_heartbeat,
            'available_gpus': node.available_gpus,
            'running_jobs': len(node.running_jobs),
            'last_heartbeat_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_heartbeat))
        }
    
    def handle_node_status(self, request: Dict) -> Dict:
        """Handle node status update"""
        node_id = request.get('node_id')
        if not node_id or node_id not in self.nodes:
            return {'status': 'error', 'message': 'Invalid node_id'}
        
        try:
            node = self.nodes[node_id]
            
            # Update heartbeat
            node.last_heartbeat = time.time()
            
            # Update available GPUs if provided
            if 'available_gpus' in request:
                node.available_gpus = request['available_gpus']
            
            # Update running jobs if provided
            if 'running_jobs' in request:
                node.running_jobs = request['running_jobs']
            
            return {'status': 'ok', 'message': 'Status updated'}
            
        except Exception as e:
            logger.error(f"Node status error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_all_nodes(self) -> Dict[str, NodeInfo]:
        """Get all registered nodes"""
        return self.nodes
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get specific node by ID"""
        return self.nodes.get(node_id)
