"""
Configuration settings for Multi-GPU Scheduler
"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for Multi-GPU Scheduler"""
    
    # Default server settings
    DEFAULT_MASTER_HOST = '127.0.0.1'
    DEFAULT_MASTER_PORT = 8080
    DEFAULT_NODE_PORT = 8081
    
    # Default timeout settings
    DEFAULT_SESSION_TIMEOUT = 7200  # 2 hours
    DEFAULT_CONNECTION_TIMEOUT = 30  # 30 seconds
    DEFAULT_MAX_WAIT_TIME = 300  # 5 minutes
    DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 30
    
    # Job management settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds
    DEFAULT_NODE_TIMEOUT = 60  # seconds
    
    # Output management
    DEFAULT_OUTPUT_BUFFER_SIZE = 8192
    DEFAULT_MAX_OUTPUT_LINES = 10000
    
    # Resource limits
    DEFAULT_MAX_JOBS_PER_NODE = 100
    DEFAULT_MAX_GPUS_PER_JOB = 8
    
    @classmethod
    def get_default_timeout_config(cls) -> Dict[str, Any]:
        """Get default timeout configuration"""
        return {
            'session_timeout': cls.DEFAULT_SESSION_TIMEOUT,
            'connection_timeout': cls.DEFAULT_CONNECTION_TIMEOUT,
            'max_wait_time': cls.DEFAULT_MAX_WAIT_TIME,
            'max_consecutive_timeouts': cls.DEFAULT_MAX_CONSECUTIVE_TIMEOUTS
        }
    
    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'master_host': os.getenv('MGPU_MASTER_HOST', cls.DEFAULT_MASTER_HOST),
            'master_port': int(os.getenv('MGPU_MASTER_PORT', cls.DEFAULT_MASTER_PORT)),
            'node_port': int(os.getenv('MGPU_NODE_PORT', cls.DEFAULT_NODE_PORT)),
            'session_timeout': int(os.getenv('MGPU_SESSION_TIMEOUT', cls.DEFAULT_SESSION_TIMEOUT)),
            'connection_timeout': int(os.getenv('MGPU_CONNECTION_TIMEOUT', cls.DEFAULT_CONNECTION_TIMEOUT)),
            'max_wait_time': int(os.getenv('MGPU_MAX_WAIT_TIME', cls.DEFAULT_MAX_WAIT_TIME)),
            'heartbeat_interval': int(os.getenv('MGPU_HEARTBEAT_INTERVAL', cls.DEFAULT_HEARTBEAT_INTERVAL)),
            'node_timeout': int(os.getenv('MGPU_NODE_TIMEOUT', cls.DEFAULT_NODE_TIMEOUT))
        }
