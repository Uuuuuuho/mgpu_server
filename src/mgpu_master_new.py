#!/usr/bin/env python3
"""
Multi-GPU Scheduler Master Server Entry Point
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from mgpu_server.master_server import MasterServer
from mgpu_core.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Master Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = MasterServer(args.host, args.port)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        logger.info("Master server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
