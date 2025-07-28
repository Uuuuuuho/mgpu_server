"""
Core data models for Multi-GPU Scheduler
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import subprocess


@dataclass
class SimpleJob:
    """Represents a job to be executed"""
    id: str
    user: str
    command: str
    gpus: int
    node_gpu_ids: Optional[Dict[str, List[int]]] = None
    priority: int = 0
    status: str = "queued"  # queued, running, completed, failed
    interactive: bool = False
    submit_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    assigned_node: Optional[str] = None
    assigned_gpus: Optional[List[int]] = None

    def __post_init__(self):
        if self.submit_time is None:
            self.submit_time = time.time()

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user': self.user,
            'command': self.command,
            'gpus': self.gpus,
            'node_gpu_ids': self.node_gpu_ids,
            'priority': self.priority,
            'status': self.status,
            'interactive': self.interactive,
            'submit_time': self.submit_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'exit_code': self.exit_code,
            'assigned_node': self.assigned_node,
            'assigned_gpus': self.assigned_gpus
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleJob':
        """Create job from dictionary"""
        return cls(**data)


class JobProcess:
    """Running job process management"""
    def __init__(self, job_id: str, process: subprocess.Popen, gpus: List[int], interactive: bool = False):
        self.job_id = job_id
        self.process = process
        self.gpus = gpus
        self.interactive = interactive
        self.start_time = time.time()
        self.interactive_clients = []  # List of client sockets for interactive output


@dataclass
class NodeInfo:
    """Node information representation"""
    id: str
    host: str
    port: int
    gpu_count: int
    available_gpus: List[int]
    load_avg: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: Optional[float] = None
    status: str = "online"  # online, offline, maintenance

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'host': self.host,
            'port': self.port,
            'gpu_count': self.gpu_count,
            'available_gpus': self.available_gpus,
            'load_avg': self.load_avg,
            'memory_usage': self.memory_usage,
            'last_heartbeat': self.last_heartbeat,
            'status': self.status
        }


# Job status constants
class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Node status constants
class NodeStatus:
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


# Communication message types
class MessageType:
    # Job management
    SUBMIT = "submit"
    CANCEL = "cancel"
    QUEUE = "queue"
    
    # Job execution
    RUN = "run"
    RUN_JOB = "run_job"
    CANCEL_JOB = "cancel_job"
    KILL = "kill"
    STATUS = "status"
    
    # Interactive sessions
    INTERACTIVE_START = "interactive_start"
    INTERACTIVE_STOP = "interactive_stop"
    
    # Output streaming
    OUTPUT = "output"
    COMPLETION = "completion"
    ERROR = "error"
    
    # Node management
    REGISTER = "register"
    NODE_REGISTER = "node_register"
    NODE_HEARTBEAT = "node_heartbeat"
    HEARTBEAT = "heartbeat"
    NODE_STATUS = "node_status"
    JOB_UPDATE = "job_update"
    
    # Job output requests
    GET_JOB_OUTPUT = "get_job_output"
    JOB_OUTPUT = "job_output"
    JOB_COMPLETE = "job_complete"
