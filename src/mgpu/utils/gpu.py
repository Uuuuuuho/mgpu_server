"""
GPU utilities for Multi-GPU Scheduler
"""

import subprocess
import re
from typing import List, Dict, Optional
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def get_gpu_count() -> int:
    """Get the number of available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
        else:
            logger.warning("nvidia-smi not available or failed")
            return 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("nvidia-smi not found or timed out")
        return 0


def get_gpu_utilization() -> Dict[int, float]:
    """Get GPU utilization for all GPUs"""
    utilization = {}
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,utilization.gpu', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        gpu_id = int(parts[0].strip())
                        util = float(parts[1].strip())
                        utilization[gpu_id] = util
    except Exception as e:
        logger.warning(f"Failed to get GPU utilization: {e}")
    
    return utilization


def get_gpu_memory_usage() -> Dict[int, Dict[str, float]]:
    """Get GPU memory usage for all GPUs"""
    memory_info = {}
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_id = int(parts[0].strip())
                        used = float(parts[1].strip())
                        total = float(parts[2].strip())
                        memory_info[gpu_id] = {
                            'used': used,
                            'total': total,
                            'free': total - used,
                            'utilization': (used / total) * 100 if total > 0 else 0
                        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory usage: {e}")
    
    return memory_info


def is_gpu_available(gpu_id: int) -> bool:
    """Check if a specific GPU is available (low utilization)"""
    utilization = get_gpu_utilization()
    return utilization.get(gpu_id, 100.0) < 10.0  # Consider available if < 10% utilization


def get_available_gpus(threshold: float = 10.0) -> List[int]:
    """Get list of available GPUs based on utilization threshold"""
    utilization = get_gpu_utilization()
    return [gpu_id for gpu_id, util in utilization.items() if util < threshold]


def get_all_gpu_ids() -> List[int]:
    """Get list of all GPU IDs on this system (alternative to get_available_gpus)"""
    try:
        # Try using nvidia-smi to detect GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            gpu_ids = []
            for line in result.stdout.strip().split('\n'):
                if line.strip().isdigit():
                    gpu_ids.append(int(line.strip()))
            return gpu_ids
        else:
            logger.warning("nvidia-smi failed, assuming no GPUs available")
            return []
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not run nvidia-smi, assuming no GPUs available")
        return []
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return []


def set_gpu_environment(gpu_ids: List[int]) -> Dict[str, str]:
    """Set environment variables for GPU usage"""
    if gpu_ids:
        gpu_str = ','.join(map(str, gpu_ids))
        return {
            'CUDA_VISIBLE_DEVICES': gpu_str,
            'NVIDIA_VISIBLE_DEVICES': gpu_str
        }
    else:
        return {
            'CUDA_VISIBLE_DEVICES': '',
            'NVIDIA_VISIBLE_DEVICES': ''
        }


def setup_gpu_environment(env: Dict[str, str], gpu_ids: List[int]) -> Dict[str, str]:
    """
    Setup environment variables for GPU usage
    
    Args:
        env: Base environment dictionary to modify
        gpu_ids: List of GPU IDs to make available
        
    Returns:
        Modified environment dictionary
    """
    # Create a copy of the environment
    new_env = env.copy()
    
    if gpu_ids:
        # Set CUDA_VISIBLE_DEVICES to restrict GPU access
        gpu_list = ','.join(map(str, gpu_ids))
        new_env['CUDA_VISIBLE_DEVICES'] = gpu_list
        new_env['NVIDIA_VISIBLE_DEVICES'] = gpu_list
        logger.debug(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")
    else:
        # No GPUs available - hide all GPUs
        new_env['CUDA_VISIBLE_DEVICES'] = ""
        new_env['NVIDIA_VISIBLE_DEVICES'] = ""
        logger.debug("No GPUs allocated - hiding all GPUs")
    
    return new_env
