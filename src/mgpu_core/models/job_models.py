"""
Core data models for Multi-GPU Scheduler
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import time
import subprocess


@dataclass
class SimpleJob:
    """Job representation with all necessary attributes"""
    id: str
    user: str
    cmd: str
    gpus_needed: int
    node_gpu_ids: Optional[Dict[str, List[int]]] = None
    priority: int = 0
    status: str = "queued"  # queued, running, completed, failed
    interactive: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    assigned_node: Optional[str] = None
    assigned_gpus: Optional[List[int]] = None
    retry_count: int = 0

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user': self.user,
            'cmd': self.cmd,
            'gpus_needed': self.gpus_needed,
            'node_gpu_ids': self.node_gpu_ids,
            'priority': self.priority,
            'status': self.status,
            'interactive': self.interactive,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'exit_code': self.exit_code,
            'assigned_node': self.assigned_node,
            'assigned_gpus': self.assigned_gpus,
            'retry_count': self.retry_count
        }


class NodeInfo:
    """Node information container"""
    def __init__(self, node_id: str, host: str, port: int, gpu_count: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.gpu_count = gpu_count
        self.available_gpus = list(range(gpu_count))
        self.running_jobs = []
        self.last_heartbeat = time.time()
        self.failure_count = 0


class JobProcess:
    """Running job process management"""
    def __init__(self, job_id: str, process: subprocess.Popen, gpus: List[int], interactive: bool = False):
        self.job_id = job_id
        self.process = process
        self.gpus = gpus
        self.interactive = interactive
        self.start_time = time.time()
        self.interactive_clients = []


class MessageType:
    """Message type constants"""
    SUBMIT = 'submit'
    QUEUE = 'queue'
    CANCEL = 'cancel'
    GET_JOB_OUTPUT = 'get_job_output'
    NODE_REGISTER = 'node_register'
    NODE_STATUS = 'node_status'
    JOB_COMPLETE = 'job_complete'
    INTERACTIVE_COMPLETE = 'interactive_complete'
    JOB_OUTPUT = 'job_output'
    RUN = 'run'
    INTERACTIVE = 'interactive'
    STATUS = 'status'
