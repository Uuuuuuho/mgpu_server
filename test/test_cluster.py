#!/usr/bin/env python3
"""
Multi-node cluster functionality test
Tests cluster setup, node communication, and job scheduling
"""
import time
import sys
import socket
import json
import os
import subprocess

def test_cluster_connectivity():
    """클러스터 연결성 테스트"""
    print("=== Testing Cluster Connectivity ===")
    
    # 마스터 서버 연결 테스트
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((master_host, master_port))
        
        # 클러스터 상태 조회
        request = {'cmd': 'queue'}
        sock.send(json.dumps(request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            print(f"✓ Master server connection successful")
            print(f"  Queue: {len(response.get('queue', []))} jobs")
            print(f"  Running: {len(response.get('running', []))} jobs")
            
            nodes = response.get('nodes', {})
            print(f"  Nodes: {len(nodes)} total")
            for node_id, status in nodes.items():
                print(f"    {node_id}: {status}")
        else:
            print(f"✗ Master server error: {response.get('message', 'Unknown error')}")
            return False
            
        sock.close()
        return True
        
    except Exception as e:
        print(f"✗ Master server connection failed: {e}")
        return False

def test_job_submission():
    """작업 제출 테스트"""
    print("\n=== Testing Job Submission ===")
    
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    try:
        # 단일 노드 작업 제출
        print("Testing single-node job submission...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((master_host, master_port))
        
        job_request = {
            'cmd': 'submit',
            'job_id': 'TEST001',
            'user': 'testuser',
            'cmdline': 'echo "Hello from single node"',
            'node_requirements': {'gpu_ids': [0], 'total_gpus': 1},
            'total_gpus': 1,
            'distributed_type': 'single',
            'priority': 1,
            'interactive': False
        }
        
        sock.send(json.dumps(job_request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            print(f"✓ Single-node job submitted: {response['job_id']}")
        else:
            print(f"✗ Single-node job submission failed: {response.get('message', '')}")
        
        sock.close()
        
        # 멀티노드 작업 제출 테스트
        print("Testing multi-node job submission...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((master_host, master_port))
        
        multinode_request = {
            'cmd': 'submit',
            'job_id': 'TEST002',
            'user': 'testuser',
            'cmdline': 'echo "Hello from multi-node"',
            'node_requirements': {'nodes': 2, 'gpus_per_node': 1, 'total_gpus': 2},
            'total_gpus': 2,
            'distributed_type': 'pytorch',
            'priority': 2,
            'interactive': False
        }
        
        sock.send(json.dumps(multinode_request).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response['status'] == 'ok':
            print(f"✓ Multi-node job submitted: {response['job_id']}")
        else:
            print(f"✗ Multi-node job submission failed: {response.get('message', '')}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"✗ Job submission test failed: {e}")
        return False

def test_resource_monitoring():
    """리소스 모니터링 테스트"""
    print("\n=== Testing Resource Monitoring ===")
    
    # nvidia-smi 실행 가능 여부 확인
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ nvidia-smi accessible")
            gpu_info = result.stdout.strip().split('\n')
            print(f"  Detected {len(gpu_info)} GPUs:")
            for i, gpu in enumerate(gpu_info):
                parts = gpu.split(', ')
                if len(parts) >= 4:
                    index, name, total_mem, free_mem = parts[:4]
                    print(f"    GPU {index}: {name} ({free_mem}MB/{total_mem}MB free)")
        else:
            print("✗ nvidia-smi not accessible")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi timeout")
        return False
    except Exception as e:
        print(f"✗ nvidia-smi error: {e}")
        return False
    
    return True

def test_environment_variables():
    """환경 변수 테스트"""
    print("\n=== Testing Environment Variables ===")
    
    required_vars = {
        'MGPU_MASTER_HOST': 'Master server hostname',
        'MGPU_MASTER_PORT': 'Master server port'
    }
    
    optional_vars = {
        'CUDA_VISIBLE_DEVICES': 'Visible CUDA devices',
        'PYTHONUNBUFFERED': 'Python output buffering'
    }
    
    all_good = True
    
    for var, desc in required_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value} ({desc})")
        else:
            print(f"✗ {var}: Not set ({desc})")
            all_good = False
    
    for var, desc in optional_vars.items():
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value} ({desc})")
    
    return all_good

def run_integration_test():
    """통합 테스트 실행"""
    print("=== Multi-Node GPU Scheduler Integration Test ===")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hostname: {socket.gethostname()}")
    print("-" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Resource Monitoring", test_resource_monitoring),
        ("Cluster Connectivity", test_cluster_connectivity),
        ("Job Submission", test_job_submission)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] Running {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Cluster is ready.")
        return True
    else:
        print("❌ Some tests FAILED. Check configuration.")
        return False

def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == '--integration':
        success = run_integration_test()
        sys.exit(0 if success else 1)
    else:
        print("Multi-Node Cluster Test Script")
        print("Usage:")
        print("  python test_cluster.py --integration    # Run full integration test")
        print("\nEnvironment variables:")
        print("  MGPU_MASTER_HOST  - Master server hostname (default: localhost)")
        print("  MGPU_MASTER_PORT  - Master server port (default: 8080)")

if __name__ == "__main__":
    main()
