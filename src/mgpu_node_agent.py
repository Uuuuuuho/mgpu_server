#!/usr/bin/env python3
"""
Multi-Node GPU Scheduler - Node Agent
Runs on each compute node to manage local GPU resources and execute jobs
"""
import os
import socket
import threading
import subprocess
import json
import time
import psutil
import argparse
from typing import Dict, List

def get_available_gpus():
    """로컬 GPU 리소스 조회"""
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'])
        gpus = []
        for line in out.decode().strip().split('\n'):
            index, mem_free, mem_total = line.split(', ')
            gpus.append({
                'index': int(index),
                'memory_free': int(mem_free),
                'memory_total': int(mem_total),
                'utilization': (int(mem_total) - int(mem_free)) / int(mem_total) * 100
            })
        return gpus
    except Exception as e:
        print(f"[ERROR] Failed to get GPU info: {e}")
        return []

class NodeAgent:
    """노드 에이전트 - 로컬 리소스 관리 및 작업 실행"""
    
    def __init__(self, node_id: str, master_host: str, master_port: int, agent_port: int):
        self.node_id = node_id
        self.master_host = master_host
        self.master_port = master_port
        self.agent_port = agent_port
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.allocated_gpus: List[int] = []  # 현재 할당된 GPU 목록
        self.lock = threading.Lock()
        
    def get_node_resources(self) -> Dict:
        """노드 리소스 정보 반환"""
        gpus = get_available_gpus()
        available_gpu_indices = []
        
        for gpu in gpus:
            if gpu['index'] not in self.allocated_gpus:
                # GPU 메모리 사용률이 10% 미만이면 사용 가능한 것으로 간주
                if gpu['utilization'] < 10:
                    available_gpu_indices.append(gpu['index'])
        
        return {
            'node_id': self.node_id,
            'gpu_count': len(gpus),
            'available_gpus': available_gpu_indices,
            'gpu_details': gpus,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available // (1024**3),  # GB
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def start_job(self, job_info: Dict) -> bool:
        """단일 노드 작업 실행"""
        job_id = job_info['job_id']
        user = job_info['user']
        command = job_info['command']
        gpu_ids = job_info['gpu_ids']
        
        try:
            with self.lock:
                # GPU 할당
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        raise Exception(f"GPU {gpu_id} already allocated")
                    self.allocated_gpus.append(gpu_id)
            
            # CUDA_VISIBLE_DEVICES 설정
            cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"
            home_dir = os.path.expanduser(f'~{user}')
            
            full_command = f"cd {home_dir} && PYTHONUNBUFFERED=1 {cuda_env} {command}"
            
            # 작업 실행
            proc = subprocess.Popen([
                'sudo', '-u', user, 'bash', '-lc', full_command
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            universal_newlines=True, preexec_fn=os.setsid)
            
            with self.lock:
                self.running_jobs[job_id] = proc
            
            print(f"[INFO] Started job {job_id} on GPUs {gpu_ids}")
            
            # 작업 완료 모니터링
            def monitor_job():
                proc.wait()
                with self.lock:
                    # GPU 해제
                    for gpu_id in gpu_ids:
                        if gpu_id in self.allocated_gpus:
                            self.allocated_gpus.remove(gpu_id)
                    # 실행 중 작업 목록에서 제거
                    if job_id in self.running_jobs:
                        del self.running_jobs[job_id]
                print(f"[INFO] Job {job_id} completed with exit code {proc.returncode}")
            
            threading.Thread(target=monitor_job, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start job {job_id}: {e}")
            # 할당된 GPU 해제
            with self.lock:
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        self.allocated_gpus.remove(gpu_id)
            return False
    
    def start_distributed_job(self, job_info: Dict) -> bool:
        """분산 작업 실행"""
        job_id = job_info['job_id']
        user = job_info['user']
        command = job_info['command']
        gpu_ids = job_info['gpu_ids']
        distributed_type = job_info.get('distributed_type', 'pytorch')
        rank = job_info['rank']
        world_size = job_info['world_size']
        master_node = job_info['master_node']
        
        try:
            with self.lock:
                # GPU 할당
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        raise Exception(f"GPU {gpu_id} already allocated")
                    self.allocated_gpus.append(gpu_id)
            
            # 분산 실행 환경 설정
            env_vars = {
                'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpu_ids)),
                'PYTHONUNBUFFERED': '1'
            }
            
            if distributed_type == 'pytorch':
                env_vars.update({
                    'RANK': str(rank),
                    'WORLD_SIZE': str(world_size),
                    'MASTER_ADDR': master_node,
                    'MASTER_PORT': '29500'  # PyTorch 기본 포트
                })
            elif distributed_type == 'mpi':
                # MPI 환경 설정은 mpirun에서 처리
                pass
            
            # 환경 변수 문자열 생성
            env_str = ' '.join([f"{k}={v}" for k, v in env_vars.items()])
            
            home_dir = os.path.expanduser(f'~{user}')
            full_command = f"cd {home_dir} && {env_str} {command}"
            
            # 분산 작업 실행
            proc = subprocess.Popen([
                'sudo', '-u', user, 'bash', '-lc', full_command
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, preexec_fn=os.setsid)
            
            with self.lock:
                self.running_jobs[job_id] = proc
            
            print(f"[INFO] Started distributed job {job_id} rank {rank} on GPUs {gpu_ids}")
            
            # 작업 완료 모니터링
            def monitor_distributed_job():
                proc.wait()
                with self.lock:
                    # GPU 해제
                    for gpu_id in gpu_ids:
                        if gpu_id in self.allocated_gpus:
                            self.allocated_gpus.remove(gpu_id)
                    # 실행 중 작업 목록에서 제거
                    if job_id in self.running_jobs:
                        del self.running_jobs[job_id]
                print(f"[INFO] Distributed job {job_id} rank {rank} completed with exit code {proc.returncode}")
            
            threading.Thread(target=monitor_distributed_job, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start distributed job {job_id}: {e}")
            # 할당된 GPU 해제
            with self.lock:
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        self.allocated_gpus.remove(gpu_id)
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        with self.lock:
            if job_id in self.running_jobs:
                proc = self.running_jobs[job_id]
                try:
                    # 프로세스 트리 전체 종료
                    parent = psutil.Process(proc.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except:
                            pass
                    parent.kill()
                    del self.running_jobs[job_id]
                    print(f"[INFO] Canceled job {job_id}")
                    return True
                except Exception as e:
                    print(f"[ERROR] Failed to cancel job {job_id}: {e}")
                    return False
            else:
                print(f"[WARNING] Job {job_id} not found")
                return False
    
    def handle_request(self, conn, addr):
        """마스터 서버로부터의 요청 처리"""
        print(f"[DEBUG] Received connection from {addr}")
        
        try:
            # 소켓 타임아웃 설정
            conn.settimeout(10.0)
            
            data = conn.recv(4096)
            print(f"[DEBUG] Received data length: {len(data)} bytes")
            
            if not data:
                print(f"[WARNING] Empty data received from {addr}")
                return
            
            print(f"[DEBUG] Raw data: {data[:200]}...")  # 처음 200바이트만 출력
            
            try:
                request = json.loads(data.decode())
                print(f"[DEBUG] Parsed request: {request}")
            except json.JSONDecodeError as je:
                print(f"[ERROR] JSON decode error: {je}")
                print(f"[ERROR] Raw data was: {data}")
                error_response = {'status': 'error', 'message': f'Invalid JSON: {str(je)}'}
                conn.send(json.dumps(error_response).encode())
                return
            
            cmd = request.get('cmd')
            print(f"[DEBUG] Processing command: {cmd}")
            
            response = {'status': 'error', 'message': 'Unknown command'}
            
            if cmd == 'get_resources':
                try:
                    resources = self.get_node_resources()
                    response = {'status': 'ok', 'resources': resources}
                    print(f"[DEBUG] Sending resources: {resources}")
                except Exception as e:
                    print(f"[ERROR] Failed to get resources: {e}")
                    response = {'status': 'error', 'message': f'Failed to get resources: {str(e)}'}
            
            elif cmd == 'start_job':
                success = self.start_job(request)
                response = {'status': 'ok' if success else 'error'}
            
            elif cmd == 'start_distributed_job':
                success = self.start_distributed_job(request)
                response = {'status': 'ok' if success else 'error'}
            
            elif cmd == 'cancel_job':
                success = self.cancel_job(request['job_id'])
                response = {'status': 'ok' if success else 'error'}
            
            elif cmd == 'ping':
                response = {'status': 'ok', 'timestamp': time.time()}
            
            elif cmd == 'heartbeat':
                # 하트비트는 단순히 OK 응답만 보냄
                response = {'status': 'ok', 'message': 'heartbeat received'}
                print(f"[DEBUG] Heartbeat received from {addr}")
            
            # 응답 전송
            response_data = json.dumps(response).encode()
            print(f"[DEBUG] Sending response: {len(response_data)} bytes")
            conn.send(response_data)
            print(f"[DEBUG] Response sent successfully to {addr}")
            
        except socket.timeout:
            print(f"[ERROR] Socket timeout with {addr}")
        except ConnectionResetError:
            print(f"[WARNING] Connection reset by {addr}")
        except Exception as e:
            print(f"[ERROR] Error handling request from {addr}: {e}")
            error_response = {'status': 'error', 'message': str(e)}
            try:
                conn.send(json.dumps(error_response).encode())
                print(f"[DEBUG] Error response sent to {addr}")
            except Exception as send_error:
                print(f"[ERROR] Failed to send error response: {send_error}")
        finally:
            try:
                conn.close()
                print(f"[DEBUG] Connection closed with {addr}")
            except:
                pass
    
    def start_agent_server(self):
        """에이전트 서버 시작"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.agent_port))
        server.listen(10)
        
        print(f"[INFO] Node agent {self.node_id} started on port {self.agent_port}")
        
        while True:
            conn, addr = server.accept()
            threading.Thread(target=self.handle_request, args=(conn, addr), daemon=True).start()
    
    def send_heartbeat(self):
        """마스터 서버에 하트비트 전송"""
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 5초 타임아웃
                
                print(f"[DEBUG] Attempting heartbeat to {self.master_host}:{self.master_port}")
                sock.connect((self.master_host, self.master_port))
                
                heartbeat = {
                    'cmd': 'heartbeat',
                    'node_id': self.node_id,
                    'timestamp': time.time(),
                    'resources': self.get_node_resources()
                }
                
                heartbeat_data = json.dumps(heartbeat).encode()
                sock.send(heartbeat_data)
                
                # 응답 받기
                response_data = sock.recv(4096)
                if response_data:
                    response = json.loads(response_data.decode())
                    if response.get('status') == 'ok':
                        print(f"[DEBUG] Heartbeat acknowledged by master")
                    else:
                        print(f"[WARNING] Heartbeat response: {response}")
                else:
                    print(f"[WARNING] Empty response to heartbeat")
                
                sock.close()
                
            except socket.timeout:
                print(f"[WARNING] Heartbeat timeout to {self.master_host}:{self.master_port}")
            except ConnectionRefusedError:
                print(f"[WARNING] Heartbeat connection refused to {self.master_host}:{self.master_port} - master server may not be running")
            except Exception as e:
                print(f"[WARNING] Failed to send heartbeat: {e}")
            
            time.sleep(30)  # 30초마다 하트비트

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', required=True, help='Node identifier')
    parser.add_argument('--master-host', default='localhost', help='Master server hostname')
    parser.add_argument('--master-port', type=int, default=8080, help='Master server port')
    parser.add_argument('--agent-port', type=int, default=8081, help='Agent server port')
    args = parser.parse_args()
    
    agent = NodeAgent(args.node_id, args.master_host, args.master_port, args.agent_port)
    
    # 하트비트 스레드 시작
    threading.Thread(target=agent.send_heartbeat, daemon=True).start()
    
    # 에이전트 서버 시작
    agent.start_agent_server()

if __name__ == "__main__":
    main()
