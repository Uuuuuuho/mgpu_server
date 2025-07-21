#!/usr/bin/env python3
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
        # TCP connection (multi-node master server)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((master_host, master_port))
        s.send(json.dumps(req).encode())
        resp = json.loads(s.recv(4096).decode())
        
        if resp['status'] == 'ok':
            print('--- Running Jobs ---')
            running_jobs = resp.get('running', [])
            if running_jobs:
                for job in running_jobs:
                    # Print multi-node job information
                    job_id = job.get('id', 'N/A')
                    user = job.get('user', 'N/A')
                    cmd = job.get('cmd', job.get('cmdline', 'N/A'))
                    total_gpus = job.get('total_gpus', job.get('gpus', 'N/A'))
                    distributed_type = job.get('distributed_type', 'single')
                    assigned_nodes = job.get('assigned_nodes', [])
                    
                    print(f"ID: {job_id} | User: {user} | GPUs: {total_gpus} | Type: {distributed_type}")
                    if assigned_nodes:
                        print(f"  Nodes: {', '.join(assigned_nodes)}")
                    print(f"  CMD: {cmd}")
                    print(f"  Status: running")
                    print()
            else:
                print("No running jobs")
            
            print('--- Queue ---')
            queued_jobs = resp.get('queue', [])
            if queued_jobs:
                for job in queued_jobs:
                    job_id = job.get('id', 'N/A')
                    user = job.get('user', 'N/A')
                    cmd = job.get('cmd', job.get('cmdline', 'N/A'))
                    total_gpus = job.get('total_gpus', job.get('gpus', 'N/A'))
                    distributed_type = job.get('distributed_type', 'single')
                    priority = job.get('priority', 0)
                    
                    print(f"ID: {job_id} | User: {user} | GPUs: {total_gpus} | Type: {distributed_type} | Priority: {priority}")
                    print(f"  CMD: {cmd}")
                    print(f"  Status: queued")
                    print()
            else:
                print("No queued jobs")
            
            print('--- Cluster Status ---')
            nodes = resp.get('nodes', {})
            if nodes:
                for node_id, status in nodes.items():
                    print(f"Node {node_id}: {status}")
            else:
                print("No node information available")
                
        else:
            print('Query failed:', resp.get('msg', resp.get('message', '')))
            
        s.close()
        
    except ConnectionRefusedError:
        print(f"Error: Cannot connect to master server at {master_host}:{master_port}")
        print("Make sure mgpu_master_server is running.")
        print("\nAlternatively, you can set environment variables:")
        print("  export MGPU_MASTER_HOST=<master_ip>")
        print("  export MGPU_MASTER_PORT=<master_port>")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
