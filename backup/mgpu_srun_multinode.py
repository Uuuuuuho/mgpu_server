#!/usr/bin/env python3
"""
mgpu_srun_multinode.py

Unified srun client for single-node and multi-node GPU scheduling.
Mirrors mgpu_srun options plus multi-node flags.
"""
import sys
import os
import socket
import json
import argparse
import getpass
import random
import string

def generate_job_id():
    """Generate unique 8-character job ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a GPU job to mgpu_scheduler (single- or multi-node)."
    )
    # Single-node options
    parser.add_argument('--gpu-ids', type=str,
                        help='Comma-separated GPU IDs for single-node')
    parser.add_argument('--mem', type=int,
                        help='Memory requirement per GPU (MB)')
    parser.add_argument('--time-limit', type=int,
                        help='Job time limit (seconds)')
    parser.add_argument('--priority', type=int, default=0,
                        help='Job priority (higher = sooner)')
    parser.add_argument('--env-setup-cmd', type=str,
                        help='Environment setup command')
    parser.add_argument('--interactive', action='store_true',
                        help='Run job interactively (stream output)')
    parser.add_argument('--background', action='store_true',
                        help='Submit job and exit immediately')
    # Multi-node options
    parser.add_argument('--nodes', type=int,
                        help='Number of nodes to allocate')
    parser.add_argument('--gpus-per-node', type=int, default=1,
                        help='GPUs per node when using --nodes')
    parser.add_argument('--nodelist', type=str,
                        help='Comma-separated list of specific node IDs')
    parser.add_argument('--exclude', type=str,
                        help='Comma-separated list of node IDs to exclude')
    parser.add_argument('--mpi', action='store_true',
                        help='Use MPI for distributed execution')
    parser.add_argument('--distributed', action='store_true',
                        help='Use PyTorch distributed execution')
    # Master server connection
    parser.add_argument('--master-host',
                        default=os.getenv('MGPU_MASTER_HOST', 'localhost'),
                        help='Master server hostname or IP')
    parser.add_argument('--master-port',
                        type=int,
                        default=int(os.getenv('MGPU_MASTER_PORT', '8080')),
                        help='Master server port')
    # GPUs per specific node mapping: node:gpu_ids pairs
    parser.add_argument('--node-gpu-ids', type=str,
                        help='Semicolon-separated list of node:gpu_ids pairs, e.g. node1:0,1;node2:2,3')
    # Command to execute
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to run under scheduler')
    return parser.parse_args()

def build_node_requirements(args):
    req = {}
    # Legacy single-node
    if args.gpu_ids:
        req['gpu_ids'] = [int(x) for x in args.gpu_ids.split(',')]
    # Multi-node
    if args.nodes:
        req['nodes'] = args.nodes
        req['gpus_per_node'] = args.gpus_per_node
    if args.nodelist:
        req['nodelist'] = args.nodelist.split(',')
    if args.exclude:
        req['exclude'] = args.exclude.split(',')
    # Specific GPUs per node
    if getattr(args, 'node_gpu_ids', None):
        mapping = {}
        for pair in args.node_gpu_ids.split(';'):
            if ':' in pair:
                node, ids = pair.split(':', 1)
                mapping[node] = [int(x) for x in ids.split(',') if x]
        if mapping:
            req['node_gpu_ids'] = mapping
    return req

def main():
    args = parse_args()
    if not args.command:
        print("Error: no command specified", file=sys.stderr)
        sys.exit(1)

    user = getpass.getuser()
    job_id = generate_job_id()
    
    # Remove leading '--' from command if present (argparse artifact)
    command_parts = args.command
    if command_parts and command_parts[0] == '--':
        command_parts = command_parts[1:]
    
    if not command_parts:
        print("Error: no command specified after --", file=sys.stderr)
        sys.exit(1)
    
    cmdline = ' '.join(command_parts)

    node_req = build_node_requirements(args)
    # Determine total_gpus, prioritizing specific per-node GPU mapping
    if 'node_gpu_ids' in node_req:
        total_gpus = sum(len(lst) for lst in node_req['node_gpu_ids'].values())
    elif 'nodes' in node_req and 'gpus_per_node' in node_req:
        total_gpus = node_req['nodes'] * node_req['gpus_per_node']
    elif 'gpu_ids' in node_req:
        total_gpus = len(node_req['gpu_ids'])
    else:
        total_gpus = 1

    # Determine distributed type
    if args.mpi:
        dist_type = 'mpi'
    elif args.distributed:
        dist_type = 'pytorch'
    else:
        dist_type = 'single'

    # Build request
    request = {
        'cmd': 'submit',
        'job_id': job_id,
        'user': user,
        'cmdline': cmdline,
        'node_requirements': node_req,
        'total_gpus': total_gpus,
        'priority': args.priority,
        'distributed_type': dist_type,
        'interactive': args.interactive
    }
    if args.mem:
        request['mem'] = args.mem
    if args.time_limit:
        request['time_limit'] = args.time_limit
    if args.env_setup_cmd:
        request['env_setup_cmd'] = args.env_setup_cmd

    # Connect to master server and send request
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.master_host, args.master_port))
        sock.send(json.dumps(request).encode())

        if args.interactive:
            # First receive the submission response
            response = sock.recv(8192).decode()
            result = json.loads(response)
            
            if result.get('status') != 'ok':
                print(f"Error: {result.get('message')}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Job {result.get('job_id')} submitted")
            print("Starting interactive session...")
            print("=" * 50)
            
            # Stream JSON messages until completion
            buffer = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                
                # Parse JSON messages
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if line.strip():
                        try:
                            msg = json.loads(line.decode())
                            if msg.get('type') == 'output':
                                print(msg.get('data', '').rstrip())
                            elif msg.get('type') == 'completion':
                                print("=" * 50)
                                print(f"Job completed with exit code: {msg.get('exit_code')}")
                                return
                            elif msg.get('type') == 'error':
                                print(f"ERROR: {msg.get('message')}")
                                return
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line}")
        else:
            resp = sock.recv(4096).decode()
            try:
                info = json.loads(resp)
                if info.get('status') == 'ok':
                    print(f"Job {info.get('job_id')} submitted")
                    if args.background:
                        print("Background mode, exiting.")
                    else:
                        print("Use mgpu_queue_multinode to check status.")
                else:
                    print(f"Error: {info.get('message')}", file=sys.stderr)
                    sys.exit(1)
            except Exception:
                print(f"Invalid response: {resp}", file=sys.stderr)
                sys.exit(1)
    except ConnectionRefusedError:
        print(f"Error: Cannot connect to master server at {args.master_host}:{args.master_port}", file=sys.stderr)
        sys.exit(1)
    finally:
        sock.close()

if __name__ == '__main__':
    main()
