#!/usr/bin/env python3
"""
Multi-GPU Scheduler Node Agent Entry Point
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from mgpu_node.node_agent import NodeAgent
from mgpu_core.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Node Agent')
    parser.add_argument('--node-id', required=True, help='Unique node identifier')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to bind to')
    parser.add_argument('--master-host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--master-port', type=int, default=8080, help='Master server port')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs on this node')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    agent = NodeAgent(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        master_host=args.master_host,
        master_port=args.master_port,
        gpu_count=args.gpu_count
    )
    
    try:
        agent.start_agent()
    except KeyboardInterrupt:
        logger.info("Node agent stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
