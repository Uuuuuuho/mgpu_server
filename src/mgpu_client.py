#!/usr/bin/env python3
"""
Multi-GPU Scheduler Client Entry Point
"""

import sys
import os
import argparse
from typing import Dict, List

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from mgpu_client.job_client import JobClient
from mgpu_core.utils.logging_utils import setup_logger
from mgpu_core.utils.system_utils import TimeoutConfig

logger = setup_logger(__name__)


def parse_node_gpu_ids(node_gpu_str: str) -> Dict[str, List[int]]:
    """Parse node-gpu mapping string like 'node1:0,1;node2:2,3'"""
    if not node_gpu_str:
        return {}
    
    result = {}
    for mapping in node_gpu_str.split(';'):
        if ':' in mapping:
            node, gpus_str = mapping.split(':', 1)
            gpus = [int(g.strip()) for g in gpus_str.split(',')]
            result[node] = gpus
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Client')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a job')
    submit_parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs needed')
    submit_parser.add_argument('--interactive', action='store_true', help='Interactive session')
    submit_parser.add_argument('--node-gpu-ids', help='Specific node-GPU mapping (e.g., node1:0,1;node2:2)')
    submit_parser.add_argument('--session-timeout', type=int, help='Session timeout in seconds (no limit if not specified)')
    submit_parser.add_argument('--connection-timeout', type=int, help='Connection timeout in seconds (no limit if not specified)')
    submit_parser.add_argument('--max-wait-time', type=int, help='Maximum wait time for job output (no limit if not specified)')
    submit_parser.add_argument('--max-consecutive-timeouts', type=int, help='Maximum consecutive timeouts (no limit if not specified)')
    submit_parser.add_argument('cmd', help='Command to execute')
    
    # Queue command
    queue_parser = subparsers.add_parser('queue', help='Show queue status')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='Job ID to cancel')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor job output')
    monitor_parser.add_argument('job_id', help='Job ID to monitor')
    monitor_parser.add_argument('--max-wait-time', type=int, help='Maximum wait time (no limit if not specified)')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    client = JobClient(args.host, args.port)
    
    try:
        if args.command == 'submit':
            node_gpu_ids = parse_node_gpu_ids(args.node_gpu_ids) if args.node_gpu_ids else None
            
            # Create timeout configuration only if any timeout is specified
            timeout_config = None
            if any([args.session_timeout, args.connection_timeout, args.max_wait_time, args.max_consecutive_timeouts]):
                timeout_config = {}
                if args.session_timeout is not None:
                    timeout_config['session_timeout'] = args.session_timeout
                if args.connection_timeout is not None:
                    timeout_config['connection_timeout'] = args.connection_timeout
                if args.max_wait_time is not None:
                    timeout_config['max_wait_time'] = args.max_wait_time
                if args.max_consecutive_timeouts is not None:
                    timeout_config['max_consecutive_timeouts'] = args.max_consecutive_timeouts
            
            success = client.submit_job(
                gpus=args.gpus,
                cmd=args.cmd,
                interactive=args.interactive,
                node_gpu_ids=node_gpu_ids,
                timeout_config=timeout_config
            )
            
        elif args.command == 'queue':
            success = client.get_queue_status()
            
        elif args.command == 'cancel':
            success = client.cancel_job(args.job_id)
            
        elif args.command == 'monitor':
            timeout_config = None
            if args.max_wait_time is not None:
                timeout_config = {'max_wait_time': args.max_wait_time}
            success = client.monitor_job_output(args.job_id, timeout_config)
        
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Client error: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
