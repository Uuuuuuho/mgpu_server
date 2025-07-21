#!/usr/bin/env python3
"""
Multi-Node GPU Scheduler - Queue Status Tool
Query job queue and running jobs from multi-node master server
"""
import os
import socket
import json

def main():
    # Master server connection information
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    # Query the master server for the current queue and running jobs
    req = {'cmd': 'queue'}
    
    try:
        # TCP connection to multi-node master server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((master_host, master_port))
        s.send(json.dumps(req).encode())
        resp = json.loads(s.recv(4096).decode())
        
        if resp['status'] == 'ok':
            print('=== Multi-Node Cluster Status ===')
            
            # Display node status
            nodes = resp.get('nodes', {})
            if nodes:
                print('\n--- Cluster Nodes ---')
                for node_id, status in nodes.items():
                    print(f"Node {node_id}: {status}")
            
            print('\n--- Running Jobs ---')
            running_jobs = resp.get('running', [])
            if running_jobs:
                for job in running_jobs:
                    # Multi-node job information display
                    job_id = job.get('id', 'N/A')
                    user = job.get('user', 'N/A')
                    cmd = job.get('cmd', 'N/A')
                    total_gpus = job.get('total_gpus', 'N/A')
                    distributed_type = job.get('distributed_type', 'single')
                    assigned_nodes = job.get('assigned_nodes', [])
                    priority = job.get('priority', 0)
                    interactive = job.get('interactive', False)
                    
                    print(f"ID: {job_id} | User: {user} | GPUs: {total_gpus} | Type: {distributed_type}")
                    print(f"  Priority: {priority} | Interactive: {interactive}")
                    if assigned_nodes:
                        print(f"  Assigned Nodes: {', '.join(assigned_nodes)}")
                    print(f"  Command: {cmd}")
                    print(f"  Status: running")
                    print()
            else:
                print("No running jobs")
            
            print('--- Job Queue ---')
            queued_jobs = resp.get('queue', [])
            if queued_jobs:
                for job in queued_jobs:
                    job_id = job.get('id', 'N/A')
                    user = job.get('user', 'N/A')
                    cmd = job.get('cmd', 'N/A')
                    total_gpus = job.get('total_gpus', 'N/A')
                    distributed_type = job.get('distributed_type', 'single')
                    node_requirements = job.get('node_requirements', {})
                    priority = job.get('priority', 0)
                    interactive = job.get('interactive', False)
                    
                    print(f"ID: {job_id} | User: {user} | GPUs: {total_gpus} | Type: {distributed_type}")
                    print(f"  Priority: {priority} | Interactive: {interactive}")
                    
                    # Display node requirements
                    if 'nodes' in node_requirements:
                        print(f"  Nodes needed: {node_requirements['nodes']}")
                    if 'gpus_per_node' in node_requirements:
                        print(f"  GPUs per node: {node_requirements['gpus_per_node']}")
                    if 'nodelist' in node_requirements:
                        print(f"  Specific nodes: {', '.join(node_requirements['nodelist'])}")
                    if 'gpu_ids' in node_requirements:
                        print(f"  Specific GPUs: {node_requirements['gpu_ids']}")
                        
                    print(f"  Command: {cmd}")
                    print(f"  Status: queued")
                    print()
            else:
                print("No queued jobs")
                
        else:
            print(f"Error: {resp.get('message', 'Unknown error')}")
        
        s.close()
    
    except ConnectionRefusedError:
        print(f"Error: Cannot connect to master server at {master_host}:{master_port}")
        print("Make sure mgpu_master_server is running.")
        print("\nConnection settings:")
        print(f"  export MGPU_MASTER_HOST={master_host}")
        print(f"  export MGPU_MASTER_PORT={master_port}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
