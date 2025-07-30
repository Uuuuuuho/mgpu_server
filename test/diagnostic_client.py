#!/usr/bin/env python3
"""
Enhanced Simple Client with Diagnostic Features
특정 노드 할당 문제 디버깅을 위한 진단 기능이 포함된 클라이언트
"""

import socket
import json
import argparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticClient:
    """진단 기능이 향상된 클라이언트"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
    
    def send_request(self, request):
        """서버에 요청 전송"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((self.host, self.port))
            sock.send(json.dumps(request).encode())
            
            response = sock.recv(8192).decode()
            sock.close()
            
            return json.loads(response)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_node_assignment(self, node_id, gpu_id=0):
        """특정 노드 할당 테스트"""
        print(f"\n=== Testing Node Assignment: {node_id} ===")
        
        # 1. 현재 클러스터 상태 확인
        queue_status = self.send_request({'cmd': 'queue'})
        print(f"Cluster Status: {queue_status.get('status')}")
        
        if 'nodes' in queue_status and isinstance(queue_status['nodes'], dict):
            print("Available Nodes:")
            for nid, info in queue_status['nodes'].items():
                print(f"  {nid}: GPUs={info.get('available_gpus', [])}")
        
        # 2. 테스트 작업 제출
        test_cmd = f"""
echo "=== NODE ASSIGNMENT TEST ==="
echo "Expected Node: {node_id}"
echo "Actual Hostname: $(hostname)"
echo "Actual IP: $(hostname -I | cut -d' ' -f1)"
echo "Running Processes: $(ps aux | grep mgpu_simple | head -2)"
echo "Network Interfaces: $(ip addr show | grep 'inet ' | grep -v '127.0.0.1')"
echo "==========================="
sleep 2
        """.strip()
        
        submit_request = {
            'cmd': 'submit',
            'user': 'diagnostic_test',
            'command': test_cmd,
            'gpus': 1,
            'node_gpu_ids': {node_id: [gpu_id]},
            'interactive': False
        }
        
        print(f"Submitting test job to {node_id}:GPU{gpu_id}")
        result = self.send_request(submit_request)
        
        if result.get('status') == 'ok':
            job_id = result['job_id']
            print(f"Job submitted: {job_id}")
            
            # 3. 작업 완료까지 대기 및 출력 수집
            self.wait_and_collect_output(job_id)
        else:
            print(f"Job submission failed: {result}")
    
    def wait_and_collect_output(self, job_id):
        """작업 완료까지 대기하고 출력 수집"""
        print(f"Waiting for job {job_id} to complete...")
        
        for i in range(30):  # 30초 대기
            time.sleep(1)
            
            # 출력 요청
            output_request = {
                'cmd': 'get_job_output',
                'job_id': job_id
            }
            
            result = self.send_request(output_request)
            
            if result.get('status') == 'ok':
                job_status = result.get('job_status')
                output_lines = result.get('output', [])
                
                if job_status in ['completed', 'failed']:
                    print(f"Job {job_status}!")
                    print("Output:")
                    for line in output_lines:
                        print(f"  {line}")
                    
                    # 결과 분석
                    self.analyze_execution_location(output_lines, job_id)
                    break
                elif job_status == 'running':
                    print(f"  Job still running... ({i+1}/30)")
                else:
                    print(f"  Job status: {job_status}")
            
            if i == 29:
                print("Timeout waiting for job completion")
    
    def analyze_execution_location(self, output_lines, job_id):
        """실행 위치 분석"""
        print(f"\n=== Execution Location Analysis for {job_id} ===")
        
        analysis = {}
        for line in output_lines:
            if "Expected Node:" in line:
                analysis['expected'] = line.split(":")[1].strip()
            elif "Actual Hostname:" in line:
                analysis['hostname'] = line.split(":")[1].strip()
            elif "Actual IP:" in line:
                analysis['ip'] = line.split(":")[1].strip()
        
        print(f"Expected Node: {analysis.get('expected', 'Unknown')}")
        print(f"Actual Hostname: {analysis.get('hostname', 'Unknown')}")
        print(f"Actual IP: {analysis.get('ip', 'Unknown')}")
        
        # 일치성 판단
        if analysis.get('expected') == analysis.get('hostname'):
            print("✅ MATCH: Job executed on expected node")
        else:
            print("❌ MISMATCH: Job executed on different node!")
            print("This indicates a potential issue with node assignment")
    
    def cluster_health_check(self):
        """클러스터 전체 상태 확인"""
        print("\n=== Cluster Health Check ===")
        
        result = self.send_request({'cmd': 'queue'})
        
        if result.get('status') == 'ok':
            nodes = result.get('nodes', {})
            if isinstance(nodes, dict):
                print(f"Total nodes: {len(nodes)}")
                
                for node_id, info in nodes.items():
                    available_gpus = info.get('available_gpus', [])
                    last_heartbeat = info.get('last_heartbeat', 0)
                    time_since = time.time() - last_heartbeat
                    
                    status = "🟢 Healthy" if time_since < 60 else "🔴 Stale"
                    print(f"  {node_id}: {status}")
                    print(f"    Available GPUs: {available_gpus}")
                    print(f"    Last seen: {time_since:.1f}s ago")
            else:
                print("No valid node information received")
        else:
            print(f"Failed to get cluster status: {result}")

def main():
    parser = argparse.ArgumentParser(description='Diagnostic Client for Multi-GPU Scheduler')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Test node assignment
    test_parser = subparsers.add_parser('test-node', help='Test specific node assignment')
    test_parser.add_argument('node_id', help='Node ID to test')
    test_parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    # Health check
    health_parser = subparsers.add_parser('health', help='Check cluster health')
    
    args = parser.parse_args()
    
    client = DiagnosticClient(args.host, args.port)
    
    if args.command == 'test-node':
        client.test_node_assignment(args.node_id, args.gpu)
    elif args.command == 'health':
        client.cluster_health_check()
    else:
        print("Use 'test-node' or 'health' command")
        parser.print_help()

if __name__ == "__main__":
    main()
