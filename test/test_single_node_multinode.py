#!/usr/bin/env python3
"""
Single Node에서 Multi-Node 서버 테스트
"""
import os
import sys
import time
import socket
import json
import subprocess
import threading

def test_master_server_standalone():
    """마스터 서버 단독 실행 테스트"""
    print("=== Testing Master Server in Standalone Mode ===")
    
    # 현재 디렉토리에서 마스터 서버 실행
    master_script = os.path.join('src', 'mgpu_master_server.py')
    config_file = 'cluster_config_localhost.yaml'
    
    if not os.path.exists(master_script):
        print(f"✗ Master server script not found: {master_script}")
        return False
    
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return False
    
    try:
        # 마스터 서버를 백그라운드에서 실행
        print(f"Starting master server with config: {config_file}")
        proc = subprocess.Popen([
            sys.executable, master_script, 
            '--config', config_file,
            '--port', '8080'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 서버 시작 대기
        print("Waiting for server to start...")
        time.sleep(3)
        
        # 서버 연결 테스트
        success = test_master_server_connection()
        
        # 서버 종료
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            print("Server output:")
            for line in stdout.split('\n')[:10]:  # 첫 10줄만 표시
                if line.strip():
                    print(f"  {line}")
        except subprocess.TimeoutExpired:
            proc.kill()
        
        return success
        
    except Exception as e:
        print(f"✗ Failed to start master server: {e}")
        return False

def test_master_server_connection():
    """마스터 서버 연결 및 기본 기능 테스트"""
    print("\n--- Testing Master Server Connection ---")
    
    try:
        # 마스터 서버에 연결
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('localhost', 8080))
        
        # 큐 상태 조회
        request = {'cmd': 'queue'}
        sock.send(json.dumps(request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            print("✓ Queue status query successful")
            print(f"  Queue: {len(response.get('queue', []))} jobs")
            print(f"  Running: {len(response.get('running', []))} jobs")
            
            nodes = response.get('nodes', {})
            print(f"  Nodes: {len(nodes)} registered")
            for node_id, status in nodes.items():
                print(f"    {node_id}: {status}")
        else:
            print(f"✗ Queue query failed: {response.get('message', 'Unknown error')}")
            sock.close()
            return False
        
        sock.close()
        
        # 작업 제출 테스트
        return test_job_submission()
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def test_job_submission():
    """작업 제출 테스트"""
    print("\n--- Testing Job Submission ---")
    
    try:
        # 단일 GPU 작업 제출
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8080))
        
        job_request = {
            'cmd': 'submit',
            'job_id': 'TEST_SINGLE_001',
            'user': 'testuser',
            'cmdline': 'echo "Hello from single GPU job"',
            'node_requirements': {},  # 기본 단일 노드
            'total_gpus': 1,
            'distributed_type': 'single',
            'priority': 1
        }
        
        sock.send(json.dumps(job_request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            print(f"✓ Single GPU job submitted: {response['job_id']}")
        else:
            print(f"✗ Job submission failed: {response.get('message', '')}")
            sock.close()
            return False
        
        sock.close()
        
        # 잠시 후 큐 상태 다시 확인
        time.sleep(1)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8080))
        
        request = {'cmd': 'queue'}
        sock.send(json.dumps(request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            queue = response.get('queue', [])
            running = response.get('running', [])
            total_jobs = len(queue) + len(running)
            
            print(f"✓ Queue updated - Total jobs: {total_jobs}")
            if total_jobs > 0:
                print("  Recent job:")
                recent_job = (running + queue)[-1] if (running + queue) else None
                if recent_job:
                    print(f"    ID: {recent_job.get('id', 'N/A')}")
                    print(f"    Status: {recent_job.get('status', 'N/A')}")
                    print(f"    Command: {recent_job.get('cmd', 'N/A')}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"✗ Job submission test failed: {e}")
        return False

def main():
    """메인 함수"""
    print("Single Node Multi-Node Server Test")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # 현재 디렉토리 확인
    print(f"Working directory: {os.getcwd()}")
    
    # 필요한 파일들 확인
    required_files = [
        'src/mgpu_master_server.py',
        'cluster_config_localhost.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    # 마스터 서버 단독 실행 테스트
    success = test_master_server_standalone()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Single Node Multi-Node Server Test PASSED!")
        print("\nThe multi-node server can run successfully in single-node mode.")
        print("It creates a virtual 'localhost' node when no real nodes are available.")
        return True
    else:
        print("❌ Single Node Multi-Node Server Test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
