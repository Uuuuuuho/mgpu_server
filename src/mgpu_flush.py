#!/usr/bin/env python3
import os
import socket
import json

def main():
    """Flush (cancel) all jobs in the queue and running"""
    # Master server connection info
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    req = {'cmd': 'flush'}
    
    try:
        # TCP connection (multi-node master server)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((master_host, master_port))
        s.send(json.dumps(req).encode())
        resp = json.loads(s.recv(4096).decode())
        
        if resp['status'] == 'ok':
            print(resp['message'])
        else:
            print(f'Flush failed: {resp.get("message", "Unknown error")}')
            
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
