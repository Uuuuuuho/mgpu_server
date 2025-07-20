#!/usr/bin/env python3
"""
Simple connection test between master and agent
"""
import socket
import json
import time

def test_node_agent_connection(host='172.24.27.42', port=8081):
    """Test connection to node agent"""
    print(f"Testing connection to {host}:{port}")
    
    sock = None
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        
        # Connect
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))
        print("Connection established!")
        
        # Send get_resources request
        request = {"cmd": "get_resources"}
        request_data = json.dumps(request).encode()
        print(f"Sending request: {request}")
        sock.send(request_data)
        
        # Receive response
        print("Waiting for response...")
        response_data = sock.recv(4096).decode()
        print(f"Received {len(response_data)} bytes")
        
        if response_data:
            response = json.loads(response_data)
            print(f"Response: {response}")
            
            if response.get('status') == 'ok':
                print("✓ SUCCESS: Resource query successful")
                if 'resources' in response:
                    resources = response['resources']
                    print(f"  Node ID: {resources.get('node_id')}")
                    print(f"  GPU Count: {resources.get('gpu_count')}")
                    print(f"  Available GPUs: {resources.get('available_gpus')}")
                return True
            else:
                print(f"✗ FAILED: {response.get('message', 'Unknown error')}")
                return False
        else:
            print("✗ FAILED: Empty response")
            return False
            
    except ConnectionRefusedError:
        print("✗ FAILED: Connection refused - node agent may not be running")
        return False
    except socket.timeout:
        print("✗ FAILED: Connection timeout")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass

def test_heartbeat(host='172.24.27.42', port=8080):
    """Test heartbeat to master server"""
    print(f"\nTesting heartbeat to master server {host}:{port}")
    
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        
        print(f"Connecting to master at {host}:{port}...")
        sock.connect((host, port))
        print("Connected to master!")
        
        heartbeat = {
            'cmd': 'heartbeat',
            'node_id': 'node001',
            'timestamp': time.time()
        }
        
        heartbeat_data = json.dumps(heartbeat).encode()
        print(f"Sending heartbeat: {heartbeat}")
        sock.send(heartbeat_data)
        
        response_data = sock.recv(4096).decode()
        if response_data:
            response = json.loads(response_data)
            print(f"Master response: {response}")
            if response.get('status') == 'ok':
                print("✓ SUCCESS: Heartbeat acknowledged")
                return True
            else:
                print(f"✗ FAILED: {response}")
                return False
        else:
            print("✗ FAILED: Empty response from master")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass

if __name__ == "__main__":
    print("=== Connection Test ===")
    
    # Test node agent connection
    agent_success = test_node_agent_connection()
    
    # Test master server heartbeat
    master_success = test_heartbeat()
    
    print(f"\n=== Summary ===")
    print(f"Node Agent Connection: {'✓ OK' if agent_success else '✗ FAILED'}")
    print(f"Master Server Heartbeat: {'✓ OK' if master_success else '✗ FAILED'}")
    
    if agent_success and master_success:
        print("\n✓ All tests passed! The connection issue should be resolved.")
    else:
        print("\n✗ Some tests failed. Check if both master and agent are running.")
