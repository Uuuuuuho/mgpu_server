#!/usr/bin/env python3
"""
Multi-Node GPU Scheduler - Master Server
Manages cluster-wide resource allocation and job scheduling
"""
import os
import socket
import threading
import subprocess
import json
import time
import yaml
from collections import deque
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class Node:
    node_id: str
    hostname: str
    ip: str
    port: int
    gpu_count: int
    gpu_type: str = "unknown"
    status: str = "online"  # online, offline, maintenance
    last_heartbeat: float = 0
    available_gpus: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.available_gpus is None:
            self.available_gpus = list(range(self.gpu_count))
        self.last_heartbeat = time.time()

@dataclass  
class DistributedJob:
    id: str
    user: str
    cmd: str
    node_requirements: Dict  # {"nodes": 2, "gpus_per_node": 4, "nodelist": ["node001"]}
    total_gpus: int
    assigned_nodes: Optional[List[str]] = None
    assigned_gpus: Optional[Dict[str, List[int]]] = None  # {"node001": [0,1], "node002": [2,3]}
    status: str = "queued"  # queued, running, completed, failed
    priority: int = 0
    distributed_type: str = "single"  # single, mpi, pytorch, custom
    master_node: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class ClusterResourceManager:
    """클러스터 리소스 관리"""
    
    def __init__(self, config_path: str):
        self.nodes: Dict[str, Node] = {}
        self.load_config(config_path)
        
    def load_config(self, config_path: str):
        """클러스터 설정 로드"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for node_config in config['nodes']:
            node = Node(**node_config)
            self.nodes[node.node_id] = node
    
    def connect_to_nodes(self):
        """모든 노드의 연결 상태 확인"""
        available_count = 0
        for node_id, node in self.nodes.items():
            try:
                # 연결 테스트용 임시 소켓 생성
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)  # 5초 타임아웃
                sock.connect((node.ip, node.port))
                sock.close()  # 즉시 연결 해제
                
                node.status = "online"
                available_count += 1
                print(f"[INFO] Node {node_id} ({node.hostname}) is available")
            except Exception as e:
                print(f"[WARNING] Node {node_id} is not available: {e}")
                print(f"[INFO] Node {node_id} will be managed locally (single-node mode)")
                node.status = "offline"
        
        if available_count == 0:
            print(f"[WARNING] No nodes available. Master server will run in standalone mode.")
            # standalone 모드에서는 localhost 노드를 생성
            localhost_node = Node(
                node_id="localhost",
                hostname="localhost", 
                ip="127.0.0.1",
                port=0,  # 실제 연결은 하지 않음
                gpu_count=1,
                gpu_type="virtual"
            )
            localhost_node.status = "online"
            self.nodes["localhost"] = localhost_node
            print(f"[INFO] Created virtual localhost node for standalone mode")
        else:
            print(f"[INFO] Found {available_count}/{len(self.nodes)} available nodes")
    
    def get_cluster_resources(self) -> Dict:
        """클러스터 전체 리소스 조회"""
        cluster_resources = {}
        for node_id, node in self.nodes.items():
            if node.status == "online":
                try:
                    # localhost 노드는 실제 쿼리하지 않고 가상 리소스 반환
                    if node_id == "localhost":
                        cluster_resources[node_id] = {
                            "available_gpus": list(range(node.gpu_count)),
                            "total_gpus": node.gpu_count,
                            "gpu_type": node.gpu_type
                        }
                    else:
                        resources = self.query_node_resources(node_id)
                        cluster_resources[node_id] = resources
                except Exception as e:
                    print(f"[WARNING] Failed to get resources from {node_id}: {e}")
                    print(f"[DEBUG] Node {node_id} config: {node.ip}:{node.port}")
                    print(f"[DEBUG] Node communication failed, checking if node is offline")
                    node.status = "offline"
            else:
                # 오프라인 노드도 기본 리소스 정보 제공
                print(f"[INFO] Node {node_id} is offline, using default resource info")
                cluster_resources[node_id] = {
                    "available_gpus": list(range(node.gpu_count)),
                    "total_gpus": node.gpu_count,
                    "gpu_type": node.gpu_type,
                    "status": "offline"
                }
        return cluster_resources
    
    def query_node_resources(self, node_id: str) -> Dict:
        """특정 노드의 리소스 조회"""
        if node_id not in self.nodes:
            raise Exception(f"Unknown node {node_id}")
        
        node = self.nodes[node_id]
        response_data = ""
        sock = None
        
        try:
            # 새로운 연결 생성 (heartbeat 방식과 동일)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((node.ip, node.port))
            
            request = {"cmd": "get_resources"}
            sock.send(json.dumps(request).encode())
            
            response_data = sock.recv(4096).decode()
            if not response_data.strip():
                raise Exception("Empty response from node agent")
            
            response = json.loads(response_data)
            
            # 응답에서 실제 리소스 정보 추출
            if response.get('status') == 'ok' and 'resources' in response:
                return response['resources']
            elif response.get('status') == 'error':
                raise Exception(f"Node agent returned error: {response.get('message', 'Unknown error')}")
            else:
                # 노드 에이전트가 직접 리소스를 반환하는 경우 (이전 버전 호환성)
                return response
            
        except socket.timeout:
            raise Exception(f"Timeout waiting for response from node agent")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from node agent: {response_data[:100]}... - {e}")
        except Exception as e:
            raise Exception(f"Communication error with node agent: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass

class MultiNodeScheduler:
    """멀티노드 스케줄러"""
    
    def __init__(self, resource_manager: ClusterResourceManager):
        self.resource_manager = resource_manager
        self.job_queue = deque()
        self.running_jobs: Dict[str, DistributedJob] = {}
        self.lock = threading.Lock()
    
    def submit_job(self, job: DistributedJob) -> str:
        """작업 제출"""
        with self.lock:
            self.job_queue.append(job)
            return job.id
    
    def try_schedule_jobs(self):
        """작업 스케줄링 시도"""
        with self.lock:
            cluster_resources = self.resource_manager.get_cluster_resources()
            
            for job in list(self.job_queue):
                assignment = self.find_node_assignment(job, cluster_resources)
                if assignment:
                    self.start_distributed_job(job, assignment)
                    self.job_queue.remove(job)
                    self.running_jobs[job.id] = job
    
    def find_node_assignment(self, job: DistributedJob, cluster_resources: Dict) -> Optional[Dict]:
        """작업에 적합한 노드 조합 찾기"""
        requirements = job.node_requirements
        
        if "nodelist" in requirements:
            # 특정 노드 지정된 경우
            return self.assign_specific_nodes(job, requirements["nodelist"], cluster_resources)
        elif "nodes" in requirements:
            # 노드 수만 지정된 경우
            return self.assign_best_nodes(job, requirements["nodes"], requirements.get("gpus_per_node", 1), cluster_resources)
        else:
            # 단일 노드에서 실행
            return self.assign_single_node(job, cluster_resources)
    
    def assign_specific_nodes(self, job: DistributedJob, nodelist: List[str], cluster_resources: Dict) -> Optional[Dict]:
        """지정된 노드들에 할당 시도"""
        assignment = {}
        required_gpus_per_node = job.node_requirements.get("gpus_per_node", 1)
        
        for node_id in nodelist:
            if node_id not in cluster_resources:
                return None  # 노드가 오프라인이거나 없음
            
            available = cluster_resources[node_id]["available_gpus"]
            if len(available) < required_gpus_per_node:
                return None  # GPU 부족
            
            assignment[node_id] = available[:required_gpus_per_node]
        
        return assignment
    
    def assign_best_nodes(self, job: DistributedJob, node_count: int, gpus_per_node: int, cluster_resources: Dict) -> Optional[Dict]:
        """최적의 노드 조합 찾기"""
        # 사용 가능한 노드들을 GPU 수 기준으로 정렬
        available_nodes = []
        for node_id, resources in cluster_resources.items():
            available_gpus = len(resources["available_gpus"])
            if available_gpus >= gpus_per_node:
                available_nodes.append((node_id, available_gpus))
        
        # GPU가 많은 노드부터 우선 선택 (fill-first 정책)
        available_nodes.sort(key=lambda x: x[1], reverse=True)
        
        if len(available_nodes) < node_count:
            return None  # 사용 가능한 노드 부족
        
        assignment = {}
        for i in range(node_count):
            node_id = available_nodes[i][0]
            available = cluster_resources[node_id]["available_gpus"]
            assignment[node_id] = available[:gpus_per_node]
        
        return assignment
    
    def assign_single_node(self, job: DistributedJob, cluster_resources: Dict) -> Optional[Dict]:
        """단일 노드에 할당"""
        required_gpus = job.total_gpus
        
        # 클러스터에 노드가 없는 경우 (테스트 환경 등)
        if not cluster_resources:
            print(f"Warning: No nodes available in cluster, creating mock assignment for job {job.id}")
            return {"localhost": list(range(required_gpus))}
        
        for node_id, resources in cluster_resources.items():
            available = resources["available_gpus"]
            if len(available) >= required_gpus:
                return {node_id: available[:required_gpus]}
        
        return None
    
    def start_distributed_job(self, job: DistributedJob, assignment: Dict):
        """분산 작업 실행"""
        job.assigned_nodes = list(assignment.keys())
        job.assigned_gpus = assignment
        job.status = "running"
        
        if len(assignment) == 1:
            # 단일 노드 실행
            node_id = list(assignment.keys())[0]
            self.start_single_node_job(job, node_id, assignment[node_id])
        else:
            # 멀티 노드 실행
            job.master_node = list(assignment.keys())[0]  # 첫 번째 노드를 마스터로
            self.start_multi_node_job(job, assignment)
    
    def start_single_node_job(self, job: DistributedJob, node_id: str, gpu_ids: List[int]):
        """단일 노드에서 작업 실행"""
        try:
            # localhost 노드인 경우 실제 작업 실행 없이 로그만 출력
            if node_id == "localhost":
                print(f"[INFO] Started local job {job.id} on localhost with GPUs {gpu_ids}")
                print(f"[INFO] Command: {job.cmd}")
                # 실제 환경에서는 subprocess로 실행하거나 단일 노드 스케줄러로 전달
                return
                
            # 원격 노드인 경우 네트워크로 전송
            node = self.resource_manager.nodes.get(node_id)
            if not node or node.status != "online":
                print(f"[WARNING] Node {node_id} is not available")
                job.status = "failed"
                return
                
            # 개별 연결 생성
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((node.ip, node.port))
            
            request = {
                "cmd": "start_job",
                "job_id": job.id,
                "user": job.user,
                "command": job.cmd,
                "gpu_ids": gpu_ids,
                "distributed": False
            }
            sock.send(json.dumps(request).encode())
            
            # 응답 받기
            response_data = sock.recv(4096).decode()
            response = json.loads(response_data)
            
            if response.get('status') == 'ok':
                print(f"[INFO] Started job {job.id} on node {node_id}")
            else:
                print(f"[ERROR] Failed to start job {job.id} on node {node_id}: {response.get('message', 'Unknown error')}")
                job.status = "failed"
                
            sock.close()
            
        except Exception as e:
            print(f"[ERROR] Failed to start job {job.id} on node {node_id}: {e}")
            job.status = "failed"
    
    def start_multi_node_job(self, job: DistributedJob, assignment: Dict):
        """멀티 노드에서 분산 작업 실행"""
        # 각 노드에 분산 실행 정보 전송
        for rank, (node_id, gpu_ids) in enumerate(assignment.items()):
            try:
                # localhost 노드인 경우
                if node_id == "localhost":
                    print(f"[INFO] Started distributed job {job.id} rank {rank} on localhost with GPUs {gpu_ids}")
                    continue
                
                # 원격 노드인 경우
                node = self.resource_manager.nodes.get(node_id)
                if not node or node.status != "online":
                    print(f"[WARNING] Node {node_id} is not available, skipping rank {rank}")
                    continue
                    
                # 개별 연결 생성
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((node.ip, node.port))
                
                request = {
                    "cmd": "start_distributed_job",
                    "job_id": job.id,
                    "user": job.user,
                    "command": job.cmd,
                    "gpu_ids": gpu_ids,
                    "distributed": True,
                    "distributed_type": job.distributed_type,
                    "rank": rank,
                    "world_size": len(assignment),
                    "master_node": job.master_node,
                    "node_list": list(assignment.keys())
                }
                sock.send(json.dumps(request).encode())
                
                # 응답 받기
                response_data = sock.recv(4096).decode()
                response = json.loads(response_data)
                
                if response.get('status') == 'ok':
                    print(f"[INFO] Started distributed job {job.id} rank {rank} on node {node_id}")
                else:
                    print(f"[ERROR] Failed to start distributed job {job.id} on node {node_id}: {response.get('message', 'Unknown error')}")
                    job.status = "failed"
                    
                sock.close()
                
            except Exception as e:
                print(f"[ERROR] Failed to start distributed job {job.id} on node {node_id}: {e}")
                job.status = "failed"
                break
    
    def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        with self.lock:
            # 큐에서 찾기
            for job in list(self.job_queue):
                if job.id == job_id:
                    self.job_queue.remove(job)
                    print(f"[INFO] Cancelled queued job {job_id}")
                    return True
            
            # 실행 중인 작업에서 찾기
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                try:
                    # 각 노드에 취소 요청 전송
                    if job.assigned_nodes:
                        for node_id in job.assigned_nodes:
                            if node_id == "localhost":
                                continue  # localhost는 실제 취소 없이 로그만
                            
                            node = self.resource_manager.nodes.get(node_id)
                            if node and node.status == "online":
                                try:
                                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    sock.settimeout(5.0)
                                    sock.connect((node.ip, node.port))
                                    
                                    cancel_request = {
                                        "cmd": "cancel_job",
                                        "job_id": job_id
                                    }
                                    sock.send(json.dumps(cancel_request).encode())
                                    
                                    response_data = sock.recv(4096).decode()
                                    response = json.loads(response_data)
                                    
                                    if response.get('status') == 'ok':
                                        print(f"[INFO] Cancelled job {job_id} on node {node_id}")
                                    else:
                                        print(f"[WARNING] Failed to cancel job {job_id} on node {node_id}")
                                    
                                    sock.close()
                                except Exception as e:
                                    print(f"[ERROR] Error cancelling job {job_id} on node {node_id}: {e}")
                    
                    # 실행 중 작업 목록에서 제거
                    del self.running_jobs[job_id]
                    job.status = "cancelled"
                    print(f"[INFO] Cancelled running job {job_id}")
                    return True
                    
                except Exception as e:
                    print(f"[ERROR] Error cancelling job {job_id}: {e}")
                    return False
            
            print(f"[WARNING] Job {job_id} not found")
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cluster_config.yaml', help='Cluster configuration file')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    args = parser.parse_args()
    
    # 리소스 매니저 초기화
    resource_manager = ClusterResourceManager(args.config)
    resource_manager.connect_to_nodes()
    
    # 스케줄러 초기화
    scheduler = MultiNodeScheduler(resource_manager)
    
    # 백그라운드 스케줄링 스레드
    def scheduling_loop():
        while True:
            scheduler.try_schedule_jobs()
            time.sleep(2)
    
    threading.Thread(target=scheduling_loop, daemon=True).start()
    
    # 클라이언트 요청 처리 서버
    def handle_client(conn, addr):
        try:
            print(f"[DEBUG] Client connected from {addr}")
            data = conn.recv(4096)
            
            if not data:
                print(f"[WARNING] Empty data from {addr}")
                return
                
            print(f"[DEBUG] Received {len(data)} bytes from {addr}")
            request = json.loads(data.decode())
            print(f"[DEBUG] Request: {request.get('cmd', 'unknown')}")
            
            if request['cmd'] == 'submit':
                # 분산 작업 제출 처리
                job = DistributedJob(
                    id=request['job_id'],
                    user=request['user'],
                    cmd=request['cmdline'],
                    node_requirements=request.get('node_requirements', {}),
                    total_gpus=request.get('total_gpus', 1),
                    priority=request.get('priority', 0),
                    distributed_type=request.get('distributed_type', 'single')
                )
                
                job_id = scheduler.submit_job(job)
                response = {'status': 'ok', 'job_id': job_id}
                conn.send(json.dumps(response).encode())
                print(f"[DEBUG] Job submitted: {job_id}")
            
            elif request['cmd'] == 'queue':
                # 큐 상태 조회
                queue_info = {
                    'status': 'ok',
                    'queue': [job.to_dict() for job in scheduler.job_queue],
                    'running': [job.to_dict() for job in scheduler.running_jobs.values()],
                    'nodes': {node_id: node.status for node_id, node in resource_manager.nodes.items()}
                }
                conn.send(json.dumps(queue_info).encode())
                print(f"[DEBUG] Queue status sent to {addr}")
            
            elif request['cmd'] == 'cancel':
                # 작업 취소 처리
                job_id = request.get('job_id')
                if job_id:
                    success = scheduler.cancel_job(job_id)
                    if success:
                        response = {'status': 'ok', 'message': f'Job {job_id} cancelled'}
                    else:
                        response = {'status': 'error', 'message': f'Failed to cancel job {job_id}'}
                    print(f"[DEBUG] Cancel request for job {job_id}: {'success' if success else 'failed'}")
                else:
                    response = {'status': 'error', 'message': 'No job_id provided'}
                    print(f"[DEBUG] Cancel request failed: No job_id provided")
                conn.send(json.dumps(response).encode())
            
            elif request['cmd'] == 'heartbeat':
                # 노드 에이전트로부터의 하트비트 처리
                node_id = request.get('node_id')
                if node_id and node_id in resource_manager.nodes:
                    resource_manager.nodes[node_id].last_heartbeat = time.time()
                    resource_manager.nodes[node_id].status = "online"
                    print(f"[INFO] Heartbeat received from {node_id}")
                else:
                    print(f"[WARNING] Heartbeat from unknown node: {node_id}")
                
                response = {'status': 'ok', 'message': 'heartbeat acknowledged'}
                conn.send(json.dumps(response).encode())
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error from {addr}: {e}")
            error_response = {'status': 'error', 'message': 'Invalid JSON'}
            try:
                conn.send(json.dumps(error_response).encode())
            except:
                pass
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            conn.send(json.dumps(error_response).encode())
        finally:
            conn.close()
    
    # 마스터 서버 시작
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(10)
    
    print(f"[INFO] Multi-node GPU scheduler master started on port {args.port}")
    
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
