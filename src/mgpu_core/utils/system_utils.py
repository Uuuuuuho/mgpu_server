"""
Utility functions for Multi-GPU Scheduler
"""

import subprocess
import socket
import logging
from typing import List, Dict, Any, Optional


logger = logging.getLogger(__name__)


class GPUManager:
    """GPU information and management utilities"""
    
    @staticmethod
    def get_gpu_info(gpu_count: int) -> List[Dict[str, Any]]:
        """Get detailed GPU information using nvidia-smi"""
        try:
            # Try to get GPU info from nvidia-smi
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_info = []
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[:gpu_count]):
                    parts = line.split(', ')
                    gpu_info.append({
                        'id': i,
                        'name': parts[0].strip() if len(parts) > 0 else 'Unknown GPU',
                        'memory': f"{parts[1].strip()} MB" if len(parts) > 1 else 'Unknown'
                    })
                return gpu_info
            else:
                # Fallback if nvidia-smi fails
                return [{'id': i, 'name': 'GPU', 'memory': 'Unknown'} for i in range(gpu_count)]
                
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return [{'id': i, 'name': 'GPU', 'memory': 'Unknown'} for i in range(gpu_count)]
    
    @staticmethod
    def get_gpu_utilization(gpu_id: int, available_gpus: List[int]) -> float:
        """Get GPU utilization (simplified - returns 0 if available)"""
        try:
            # For now, just check if GPU is in use
            return 0.0 if gpu_id in available_gpus else 100.0
        except:
            return 0.0


class IPManager:
    """IP address detection and management"""
    
    @staticmethod
    def get_actual_ip_address(master_host: str, master_port: int) -> str:
        """Get actual IP address using multiple detection methods"""
        # Method 1: Try connecting to master server to get local IP
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(5.0)
            test_sock.connect((master_host, master_port))
            actual_ip = test_sock.getsockname()[0]
            test_sock.close()
            if actual_ip and actual_ip != "127.0.0.1":
                logger.info(f"IP detected via master connection: {actual_ip}")
                return actual_ip
        except Exception as e:
            logger.debug(f"Method 1 (master connection) failed: {e}")
        
        # Method 2: Use hostname -I command
        try:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                for ip in ips:
                    if ip and not ip.startswith("127."):
                        logger.info(f"IP detected via hostname -I: {ip}")
                        return ip
        except Exception as e:
            logger.debug(f"Method 2 (hostname -I) failed: {e}")
        
        # Method 3: Connect to external IP to find interface IP
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(5.0)
            test_sock.connect(("8.8.8.8", 80))
            actual_ip = test_sock.getsockname()[0]
            test_sock.close()
            if actual_ip and actual_ip != "127.0.0.1":
                logger.info(f"IP detected via external connection: {actual_ip}")
                return actual_ip
        except Exception as e:
            logger.debug(f"Method 3 (external connection) failed: {e}")
        
        # Fallback to configured host or localhost
        logger.warning("Could not detect actual IP, using fallback")
        return "127.0.0.1"


class TimeoutConfig:
    """Timeout configuration management"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default timeout configuration"""
        return {
            'session_timeout': 7200,  # 2 hours
            'connection_timeout': 30,  # 30 seconds
            'max_wait_time': 300,  # 5 minutes
            'max_consecutive_timeouts': 30  # 30 consecutive timeouts
        }
