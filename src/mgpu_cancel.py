#!/usr/bin/env python3
import sys
import os
import socket
import json

def main():
    # Cancel a job by job ID
    if len(sys.argv) < 2:
        print('Usage: mgpu_cancel <job_id>')
        sys.exit(1)
    
    job_id = sys.argv[1]
    
    # Master server connection information
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    req = {'cmd': 'cancel', 'job_id': job_id}
    
    try:
        # TCP connection (multi-node master server)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((master_host, master_port))
        s.send(json.dumps(req).encode())
        resp = json.loads(s.recv(4096).decode())
        
        if resp['status'] == 'ok':
            print(f'Job {job_id} cancelled successfully.')
        else:
            print(f'Cancel failed: {resp.get("message", "Unknown error")}')
            
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
