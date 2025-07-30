#!/usr/bin/env python3
"""
Interactive job monitoring tool
"""
import socket
import json
import time
import threading

def monitor_queue():
    """Monitor queue status continuously"""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(('localhost', 8080))
            
            request = {'cmd': 'queue'}
            sock.send(json.dumps(request).encode())
            
            response = sock.recv(8192).decode()
            data = json.loads(response)
            
            print(f"\n=== Queue Status ({time.strftime('%H:%M:%S')}) ===")
            if data.get('status') == 'ok':
                print(f"Queued jobs: {len(data.get('queue', []))}")
                print(f"Running jobs: {len(data.get('running', []))}")
                
                for job in data.get('running', []):
                    print(f"  Running: {job.get('id')} - {job.get('cmd')[:50]}...")
                    print(f"    Status: {job.get('status')}, Interactive: {job.get('interactive')}")
                    
                for job in data.get('queue', []):
                    print(f"  Queued: {job.get('id')} - {job.get('cmd')[:50]}...")
            else:
                print(f"Error: {data.get('message')}")
            
            sock.close()
            
        except Exception as e:
            print(f"Monitor error: {e}")
            
        time.sleep(3)

def test_simple_interactive():
    """Test simple interactive job"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8080))
        
        request = {
            'cmd': 'submit',
            'job_id': 'TEST_SIMPLE',
            'user': 'uho',
            'cmdline': 'echo "Hello Interactive"; sleep 5; echo "Goodbye"',
            'node_requirements': {'node_gpu_ids': {'node001': [0]}},
            'total_gpus': 1,
            'priority': 0,
            'distributed_type': 'single',
            'interactive': True
        }
        
        sock.send(json.dumps(request).encode())
        
        print("Waiting for interactive output...")
        
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
                            print(f"OUTPUT: {msg.get('data', '').strip()}")
                        elif msg.get('type') == 'completion':
                            print(f"COMPLETED: Job {msg.get('job_id')} with exit code {msg.get('exit_code')}")
                            return
                        else:
                            print(f"MSG: {msg}")
                    except json.JSONDecodeError as e:
                        print(f"JSON Error: {e}, data: {line}")
        
        sock.close()
        
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_queue()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_simple_interactive()
    else:
        print("Usage: python monitor_interactive.py [monitor|test]")
