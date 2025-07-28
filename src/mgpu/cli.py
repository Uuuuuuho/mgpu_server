"""
Command-line entry points for Multi-GPU Scheduler
"""

import argparse
import sys
import time
import signal
from typing import Dict, Any, Optional

from .client.client import SimpleClient
from .server.master import SimpleMaster
from .node.agent import SimpleNode, NodeManager
from .core.config import Config
from .utils.logging import setup_logger

logger = setup_logger(__name__)


def main_client():
    """Entry point for mgpu_client command"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Client')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs required')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--timeout-session', type=int, help='Session timeout in seconds')
    parser.add_argument('--timeout-connection', type=int, help='Connection timeout in seconds')
    parser.add_argument('--timeout-monitoring', type=int, help='Monitoring timeout in seconds')
    parser.add_argument('--queue', action='store_true', help='Show queue status')
    parser.add_argument('--cancel', help='Cancel job by ID')
    parser.add_argument('--monitor', help='Monitor job output by ID')
    parser.add_argument('command', nargs='?', help='Command to execute')
    
    args = parser.parse_args()
    
    # Setup timeout configuration
    timeout_config = Config.get_default_timeout_config()
    if args.timeout_session:
        timeout_config['session_timeout'] = args.timeout_session
    if args.timeout_connection:
        timeout_config['connection_timeout'] = args.timeout_connection
    if args.timeout_monitoring:
        timeout_config['max_wait_time'] = args.timeout_monitoring
    
    client = SimpleClient(args.host, args.port)
    
    try:
        if args.queue:
            success = client.get_queue_status()
        elif args.cancel:
            success = client.cancel_job(args.cancel)
        elif args.monitor:
            success = client.monitor_job_output(args.monitor, timeout_config)
        elif args.command:
            success = client.submit_job(
                gpus=args.gpus,
                cmd=args.command,
                interactive=args.interactive,
                timeout_config=timeout_config
            )
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main_server():
    """Entry point for mgpu_server command"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Master Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server bind host')
    parser.add_argument('--port', type=int, default=8080, help='Server bind port')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = SimpleMaster(args.host, args.port)
    
    def signal_handler(signum, frame):
        print("\nShutting down server...")
        server.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start_server()
    except Exception as e:
        print(f"Server error: {e}")
        return 1
    
    return 0


def main_node():
    """Entry point for mgpu_node command"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Node Agent')
    parser.add_argument('--node-id', required=True, help='Unique node identifier')
    parser.add_argument('--master-host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--master-port', type=int, default=8080, help='Master server port')
    parser.add_argument('--node-port', type=int, default=8081, help='Node agent port')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    node = SimpleNode(
        node_id=args.node_id,
        master_host=args.master_host,
        master_port=args.master_port,
        node_port=args.node_port
    )
    
    def signal_handler(signum, frame):
        print("\nShutting down node agent...")
        node.stop_agent()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        node.start_agent()
    except Exception as e:
        print(f"Node agent error: {e}")
        return 1
    
    return 0


def main_queue():
    """Entry point for mgpu_queue command (alias for client --queue)"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Queue Status')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    
    args = parser.parse_args()
    
    client = SimpleClient(args.host, args.port)
    
    try:
        success = client.get_queue_status()
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main_cancel():
    """Entry point for mgpu_cancel command"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Job Cancellation')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    parser.add_argument('job_id', help='Job ID to cancel')
    
    args = parser.parse_args()
    
    client = SimpleClient(args.host, args.port)
    
    try:
        success = client.cancel_job(args.job_id)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main_srun():
    """Entry point for mgpu_srun command (simplified SLURM-like interface)"""
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Job Submission (SLURM-like)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs required')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    parser.add_argument('--timeout', type=int, help='Job timeout in seconds')
    parser.add_argument('command', nargs='+', help='Command and arguments to execute')
    
    args = parser.parse_args()
    
    # Join command arguments
    command = ' '.join(args.command)
    
    # Setup timeout configuration
    timeout_config = Config.get_default_timeout_config()
    if args.timeout:
        timeout_config['session_timeout'] = args.timeout
        timeout_config['max_wait_time'] = args.timeout
    
    client = SimpleClient(args.host, args.port)
    
    try:
        success = client.submit_job(
            gpus=args.gpus,
            cmd=command,
            interactive=args.interactive,
            timeout_config=timeout_config
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    # Allow running individual components directly
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'client':
            sys.argv.pop(1)  # Remove 'client' from args
            sys.exit(main_client())
        elif sys.argv[1] == 'server':
            sys.argv.pop(1)  # Remove 'server' from args
            sys.exit(main_server())
        elif sys.argv[1] == 'node':
            sys.argv.pop(1)  # Remove 'node' from args
            sys.exit(main_node())
        else:
            print("Usage: python -m mgpu.cli [client|server|node] [args...]")
            sys.exit(1)
    else:
        print("Usage: python -m mgpu.cli [client|server|node] [args...]")
        sys.exit(1)
