"""
Multi-GPU Scheduler Package

A simplified, reliable multi-GPU job scheduling system for distributed computing environments.
"""

__version__ = "2.0.0"
__author__ = "Multi-GPU Scheduler Development Team"

# Import main components for easy access
from .client.client import SimpleClient
from .server.master import SimpleMaster
from .node.agent import SimpleNode, NodeManager
from .core.models import SimpleJob, JobProcess, NodeInfo, MessageType
from .core.config import Config

__all__ = [
    'SimpleClient',
    'SimpleMaster', 
    'SimpleNode',
    'NodeManager',
    'SimpleJob',
    'JobProcess', 
    'NodeInfo',
    'MessageType',
    'Config'
]
