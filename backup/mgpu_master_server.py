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
    interactive: bool = False
    client_conn: Optional[socket.socket] = None
    
    def to_dict(self):
        result = asdict(self)
        # Remove non-serializable socket object and other problematic fields
        result.pop('client_conn', None)
        # Ensure all values are JSON serializable
        for key, value in result.items():
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                result[key] = str(value)
        return result

class ClusterResourceManager:
    """Cluster resource management"""
    
    def __init__(self, config_path: str):
        self.nodes: Dict[str, Node] = {}
        self.load_config(config_path)
        
    def load_config(self, config_path: str):
        """Load cluster configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for node_config in config['nodes']:
            node = Node(**node_config)
            self.nodes[node.node_id] = node
    
    def connect_to_nodes(self):
        """Check connection status of all nodes"""
        available_count = 0
        for node_id, node in self.nodes.items():
            try:
                # Create temporary socket for connection test
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)  # 5 second timeout
                sock.connect((node.ip, node.port))
                sock.close()  # Close connection immediately
                
                node.status = "online"
                available_count += 1
                print(f"[INFO] Node {node_id} ({node.hostname}) is available")
            except Exception as e:
                print(f"[WARNING] Node {node_id} is not available: {e}")
                print(f"[INFO] Node {node_id} will be managed locally (single-node mode)")
                node.status = "offline"
        
        if available_count == 0:
            print(f"[WARNING] No nodes available. Master server will run in standalone mode.")
            # Create localhost node for standalone mode with multiple virtual GPUs
            localhost_node = Node(
                node_id="localhost",
                hostname="localhost", 
                ip="127.0.0.1",
                port=0,  # No actual connection
                gpu_count=4,  # Increased to 4 GPUs for testing
                gpu_type="virtual"
            )
            localhost_node.status = "online"
            self.nodes["localhost"] = localhost_node
            print(f"[INFO] Created virtual localhost node with {localhost_node.gpu_count} GPUs for standalone mode")
        else:
            print(f"[INFO] Found {available_count}/{len(self.nodes)} available nodes")
    
    def get_cluster_resources(self) -> Dict:
        """Query cluster-wide resources"""
        cluster_resources = {}
        for node_id, node in self.nodes.items():
            if node.status == "online":
                try:
                    # Return virtual resources for localhost node without actual query
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
                # Provide default resource info for offline nodes
                print(f"[INFO] Node {node_id} is offline, using default resource info")
                cluster_resources[node_id] = {
                    "available_gpus": list(range(node.gpu_count)),
                    "total_gpus": node.gpu_count,
                    "gpu_type": node.gpu_type,
                    "status": "offline"
                }
        return cluster_resources
    
    def query_node_resources(self, node_id: str) -> Dict:
        """Query resources from specific node"""
        if node_id not in self.nodes:
            raise Exception(f"Unknown node {node_id}")
        
        node = self.nodes[node_id]
        response_data = ""
        sock = None
        
        try:
            # Create new connection (same as heartbeat method)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((node.ip, node.port))
            
            request = {"cmd": "get_resources"}
            sock.send(json.dumps(request).encode())
            
            response_data = sock.recv(4096).decode()
            if not response_data.strip():
                raise Exception("Empty response from node agent")
            
            response = json.loads(response_data)
            
            # Extract actual resource information from response
            if response.get('status') == 'ok' and 'resources' in response:
                return response['resources']
            elif response.get('status') == 'error':
                raise Exception(f"Node agent returned error: {response.get('message', 'Unknown error')}")
            else:
                # For backward compatibility when node agent returns resources directly
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
    """Multi-node scheduler"""
    
    def __init__(self, resource_manager: ClusterResourceManager):
        self.resource_manager = resource_manager
        self.job_queue = deque()
        self.running_jobs: Dict[str, DistributedJob] = {}
        self.lock = threading.Lock()
        self.interactive_clients = {}  # job_id -> list of client sockets
    
    def submit_job(self, job: DistributedJob) -> str:
        """Submit job"""
        with self.lock:
            self.job_queue.append(job)
            return job.id
    
    def get_queue_status(self) -> Dict:
        """Get queue status without blocking - thread-safe snapshot"""
        # Take a quick snapshot to avoid blocking
        with self.lock:
            queued_jobs = [job.to_dict() for job in list(self.job_queue)]
            running_jobs = [job.to_dict() for job in list(self.running_jobs.values())]
            node_status = {node_id: node.status for node_id, node in self.resource_manager.nodes.items()}
        
        return {
            'status': 'ok',
            'queue': queued_jobs,
            'running': running_jobs,
            'nodes': node_status
        }
    
    def try_schedule_jobs(self):
        """Try to schedule jobs"""
        with self.lock:
            if not self.job_queue:
                return  # No jobs to schedule
            
            print(f"[DEBUG] Attempting to schedule {len(self.job_queue)} jobs")
            cluster_resources = self.resource_manager.get_cluster_resources()
            print(f"[DEBUG] Available cluster resources: {cluster_resources}")
            
            for job in list(self.job_queue):
                print(f"[DEBUG] Trying to schedule job {job.id} with requirements: {job.node_requirements}")
                assignment = self.find_node_assignment(job, cluster_resources)
                
                if assignment:
                    print(f"[DEBUG] Found assignment for job {job.id}: {assignment}")
                    self.start_distributed_job(job, assignment)
                    self.job_queue.remove(job)
                    self.running_jobs[job.id] = job
                    
                    # Update cluster resources after assignment
                    for node_id, gpu_ids in assignment.items():
                        if node_id in cluster_resources:
                            available_gpus = cluster_resources[node_id]["available_gpus"]
                            for gpu_id in gpu_ids:
                                if gpu_id in available_gpus:
                                    available_gpus.remove(gpu_id)
                    print(f"[INFO] Successfully scheduled job {job.id}")
                else:
                    print(f"[DEBUG] No suitable assignment found for job {job.id}")
                    break  # Wait for resources to become available
    
    def find_node_assignment(self, job: DistributedJob, cluster_resources: Dict) -> Optional[Dict]:
        """Find suitable node combination for job"""
        requirements = job.node_requirements
        # If user specified exact GPUs per node, honor that mapping
        if 'node_gpu_ids' in requirements:
            mapping = requirements['node_gpu_ids']
            print(f"[DEBUG] Processing node_gpu_ids mapping: {mapping}")
            
            # Verify nodes and GPU availability
            for node_id, gpu_ids in mapping.items():
                print(f"[DEBUG] Checking node {node_id} with requested GPUs {gpu_ids}")
                
                if node_id not in cluster_resources:
                    print(f"[ERROR] Node {node_id} not found in cluster resources. Available nodes: {list(cluster_resources.keys())}")
                    return None
                
                available = cluster_resources[node_id]['available_gpus']
                print(f"[DEBUG] Available GPUs on {node_id}: {available}")
                
                for g in gpu_ids:
                    if g not in available:
                        print(f"[ERROR] GPU {g} not available on node {node_id}. Available GPUs: {available}")
                        return None
                
                print(f"[DEBUG] Node {node_id} has all requested GPUs available")
            
            print(f"[INFO] All nodes and GPUs are available for job {job.id}, returning mapping: {mapping}")
            return mapping
        if "nodelist" in requirements:
            # When specific nodes are specified
            return self.assign_specific_nodes(job, requirements["nodelist"], cluster_resources)
        elif "nodes" in requirements:
            # When only node count is specified
            return self.assign_best_nodes(job, requirements["nodes"], requirements.get("gpus_per_node", 1), cluster_resources)
        else:
            # Execute on single node
            return self.assign_single_node(job, cluster_resources)
    
    def assign_specific_nodes(self, job: DistributedJob, nodelist: List[str], cluster_resources: Dict) -> Optional[Dict]:
        """Try to assign to specified nodes"""
        assignment = {}
        required_gpus_per_node = job.node_requirements.get("gpus_per_node", 1)
        
        for node_id in nodelist:
            if node_id not in cluster_resources:
                return None  # Node is offline or doesn't exist
            
            available = cluster_resources[node_id]["available_gpus"]
            if len(available) < required_gpus_per_node:
                return None  # Insufficient GPUs
            
            assignment[node_id] = available[:required_gpus_per_node]
        
        return assignment
    
    def assign_best_nodes(self, job: DistributedJob, node_count: int, gpus_per_node: int, cluster_resources: Dict) -> Optional[Dict]:
        """Find optimal node combination"""
        # Sort available nodes by GPU count, prioritizing online nodes
        available_nodes = []
        for node_id, resources in cluster_resources.items():
            available_gpus = len(resources["available_gpus"])
            if available_gpus >= gpus_per_node:
                # Prioritize online nodes by giving them higher priority
                priority = 1000 if resources.get("status") != "offline" else 0
                available_nodes.append((node_id, available_gpus, priority))
        
        # Sort by priority first (online vs offline), then by GPU count (fill-first policy)
        available_nodes.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        if len(available_nodes) < node_count:
            return None  # Insufficient available nodes
        
        assignment = {}
        for i in range(node_count):
            node_id = available_nodes[i][0]
            available = cluster_resources[node_id]["available_gpus"]
            assignment[node_id] = available[:gpus_per_node]
        
        return assignment
    
    def assign_single_node(self, job: DistributedJob, cluster_resources: Dict) -> Optional[Dict]:
        """Assign to single node"""
        required_gpus = job.total_gpus
        
        # For cases when no nodes are available in cluster (test environment, etc.)
        if not cluster_resources:
            print(f"Warning: No nodes available in cluster, creating mock assignment for job {job.id}")
            return {"localhost": list(range(required_gpus))}
        
        # Prioritize online nodes first, then offline nodes
        online_nodes = []
        offline_nodes = []
        
        for node_id, resources in cluster_resources.items():
            available = resources["available_gpus"]
            if len(available) >= required_gpus:
                if resources.get("status") == "offline":
                    offline_nodes.append((node_id, available))
                else:
                    online_nodes.append((node_id, available))
        
        # Try online nodes first
        for node_id, available in online_nodes:
            return {node_id: available[:required_gpus]}
        
        # If no online nodes available, try offline nodes (but this shouldn't happen for localhost)
        for node_id, available in offline_nodes:
            return {node_id: available[:required_gpus]}
        
        return None
    
    def start_distributed_job(self, job: DistributedJob, assignment: Dict):
        """Execute distributed job"""
        job.assigned_nodes = list(assignment.keys())
        job.assigned_gpus = assignment
        job.status = "running"
        
        if len(assignment) == 1:
            # Single node execution
            node_id = list(assignment.keys())[0]
            self.start_single_node_job(job, node_id, assignment[node_id])
        else:
            # Multi-node execution
            job.master_node = list(assignment.keys())[0]  # First node becomes master
            self.start_multi_node_job(job, assignment)
    
    def start_single_node_job(self, job: DistributedJob, node_id: str, gpu_ids: List[int]):
        """Execute job on single node"""
        try:
            # For localhost node, execute directly on local GPUs
            if node_id == "localhost":
                print(f"[INFO] Starting local job {job.id} on localhost with GPUs {gpu_ids}")
                print(f"[INFO] Command: {job.cmd}")
                
                # Set CUDA_VISIBLE_DEVICES for GPU assignment
                cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"
                home_dir = os.path.expanduser(f'~{job.user}')
                
                # Use the user's shell with proper environment
                full_command = f"cd {home_dir} && export {cuda_env} && export PYTHONUNBUFFERED=1 && {job.cmd}"
                
                # Execute job locally
                def execute_local_job():
                    try:
                        proc = subprocess.Popen([
                            'sudo', '-u', job.user, 'bash', '-lc', full_command
                        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                        universal_newlines=True, preexec_fn=os.setsid)
                        
                        print(f"[INFO] Started local job {job.id} with PID {proc.pid}")
                        
                        # Handle interactive output streaming
                        if job.interactive and job.client_conn and proc.stdout:
                            try:
                                # Stream output to client
                                while True:
                                    output = proc.stdout.readline()
                                    if output == '' and proc.poll() is not None:
                                        break
                                    if output:
                                        output_msg = {
                                            'type': 'output',
                                            'data': output
                                        }
                                        try:
                                            job.client_conn.send((json.dumps(output_msg) + '\n').encode())
                                        except (BrokenPipeError, ConnectionResetError):
                                            print(f"[INFO] Client disconnected from job {job.id}")
                                            break
                                
                                # Wait for completion and send completion message
                                proc.wait()
                                completion_msg = {
                                    'type': 'completion',
                                    'job_id': job.id,
                                    'exit_code': proc.returncode
                                }
                                try:
                                    job.client_conn.send((json.dumps(completion_msg) + '\n').encode())
                                    job.client_conn.close()
                                except (BrokenPipeError, ConnectionResetError):
                                    pass
                                    
                            except Exception as e:
                                print(f"[ERROR] Error streaming output for job {job.id}: {e}")
                        else:
                            # Non-interactive mode, just wait for completion
                            proc.wait()
                        
                        print(f"[INFO] Local job {job.id} completed with exit code {proc.returncode}")
                        
                        # Remove from running jobs
                        with self.lock:
                            if job.id in self.running_jobs:
                                del self.running_jobs[job.id]
                                
                    except Exception as e:
                        print(f"[ERROR] Error executing local job {job.id}: {e}")
                        job.status = "failed"
                
                # Start job in background thread
                threading.Thread(target=execute_local_job, daemon=True).start()
                return
                
            # For remote nodes, send via network
            node = self.resource_manager.nodes.get(node_id)
            if not node or node.status != "online":
                print(f"[WARNING] Node {node_id} is not available")
                job.status = "failed"
                return
                
            # Create individual connection
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
            
            # Receive response
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
        """Execute distributed job on multiple nodes"""
        # Send distributed execution information to each node
        for rank, (node_id, gpu_ids) in enumerate(assignment.items()):
            try:
                # For localhost node
                if node_id == "localhost":
                    print(f"[INFO] Started distributed job {job.id} rank {rank} on localhost with GPUs {gpu_ids}")
                    continue
                
                # For remote nodes
                node = self.resource_manager.nodes.get(node_id)
                if not node or node.status != "online":
                    print(f"[WARNING] Node {node_id} is not available, skipping rank {rank}")
                    continue
                    
                # Create individual connection
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
                
                # Receive response
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
        """Cancel job"""
        with self.lock:
            # Look for job in queue
            for job in list(self.job_queue):
                if job.id == job_id:
                    self.job_queue.remove(job)
                    print(f"[INFO] Cancelled queued job {job_id}")
                    return True
            
            # Look for job in running jobs
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                try:
                    # Send cancel request to each node
                    if job.assigned_nodes:
                        for node_id in job.assigned_nodes:
                            if node_id == "localhost":
                                continue  # For localhost, only log without actual cancellation
                            
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
                    
                    # Remove from running jobs list
                    del self.running_jobs[job_id]
                    job.status = "cancelled"
                    print(f"[INFO] Cancelled running job {job_id}")
                    return True
                    
                except Exception as e:
                    print(f"[ERROR] Error cancelling job {job_id}: {e}")
                    return False
            
            print(f"[WARNING] Job {job_id} not found")
            return False
    
    def flush_all_jobs(self) -> int:
        """Cancel all queued and running jobs"""
        with self.lock:
            cancelled_count = 0
            
            # Cancel all queued jobs
            queued_count = len(self.job_queue)
            for job in list(self.job_queue):
                print(f"[INFO] Cancelled queued job {job.id}")
                cancelled_count += 1
            self.job_queue.clear()
            
            # Cancel all running jobs
            running_jobs = list(self.running_jobs.values())
            for job in running_jobs:
                if self.cancel_job(job.id):
                    cancelled_count += 1
            
            print(f"[INFO] Flushed {cancelled_count} jobs ({queued_count} queued, {len(running_jobs)} running)")
            return cancelled_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cluster_config.yaml', help='Cluster configuration file')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    args = parser.parse_args()
    
    # Initialize resource manager
    resource_manager = ClusterResourceManager(args.config)
    resource_manager.connect_to_nodes()
    
    # Initialize scheduler
    scheduler = MultiNodeScheduler(resource_manager)
    
    # Background scheduling thread
    def scheduling_loop():
        while True:
            scheduler.try_schedule_jobs()
            time.sleep(2)
    
    threading.Thread(target=scheduling_loop, daemon=True).start()
    
    # Client request handler server
    def handle_client(conn, addr):
        close_connection = True  # Default behavior
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
                # Handle distributed job submission
                interactive = request.get('interactive', False)
                
                job = DistributedJob(
                    id=request['job_id'],
                    user=request['user'],
                    cmd=request['cmdline'],
                    node_requirements=request.get('node_requirements', {}),
                    total_gpus=request.get('total_gpus', 1),
                    priority=request.get('priority', 0),
                    distributed_type=request.get('distributed_type', 'single'),
                    interactive=interactive,
                    client_conn=conn if interactive else None
                )
                
                job_id = scheduler.submit_job(job)
                
                if interactive:
                    # For interactive jobs, add client to interactive clients list
                    if job_id not in scheduler.interactive_clients:
                        scheduler.interactive_clients[job_id] = []
                    scheduler.interactive_clients[job_id].append(conn)
                    
                    # Send initial response but keep connection open
                    response = {'status': 'ok', 'job_id': job_id, 'interactive': True}
                    conn.send(json.dumps(response).encode())
                    print(f"[DEBUG] Interactive job submitted: {job_id}")
                    close_connection = False  # Don't close connection for interactive jobs
                    return  # Don't close connection
                else:
                    # For non-interactive jobs, send response and close connection
                    response = {'status': 'ok', 'job_id': job_id}
                    conn.send(json.dumps(response).encode())
                    print(f"[DEBUG] Job submitted: {job_id}")
            
            elif request['cmd'] == 'queue':
                # Query queue status using thread-safe method
                queue_info = scheduler.get_queue_status()
                conn.send(json.dumps(queue_info).encode())
                print(f"[DEBUG] Queue status sent to {addr}")
            
            elif request['cmd'] == 'cancel':
                # Handle job cancellation
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
            
            elif request['cmd'] == 'flush':
                # Flush all jobs
                cancelled_count = scheduler.flush_all_jobs()
                response = {'status': 'ok', 'message': f'Flushed {cancelled_count} jobs'}
                conn.send(json.dumps(response).encode())
                print(f"[DEBUG] Flush request: cancelled {cancelled_count} jobs")
            
            elif request['cmd'] == 'heartbeat':
                # Handle heartbeat from node agents
                node_id = request.get('node_id')
                if node_id and node_id in resource_manager.nodes:
                    resource_manager.nodes[node_id].last_heartbeat = time.time()
                    resource_manager.nodes[node_id].status = "online"
                    print(f"[INFO] Heartbeat received from {node_id}")
                else:
                    print(f"[WARNING] Heartbeat from unknown node: {node_id}")
                
                response = {'status': 'ok', 'message': 'heartbeat acknowledged'}
                conn.send(json.dumps(response).encode())
            
            elif request['cmd'] == 'interactive_output':
                # Handle interactive output from node
                job_id = request.get('job_id')
                data = request.get('data', '')
                
                if job_id in scheduler.interactive_clients:
                    # Send output to all connected interactive clients
                    dead_clients = []
                    for client_socket in scheduler.interactive_clients[job_id]:
                        try:
                            output_msg = {
                                'type': 'output',
                                'job_id': job_id,
                                'data': data
                            }
                            client_socket.send((json.dumps(output_msg) + '\n').encode())
                        except:
                            dead_clients.append(client_socket)
                    
                    # Remove dead clients
                    for client in dead_clients:
                        scheduler.interactive_clients[job_id].remove(client)
                
                response = {'status': 'ok', 'message': 'Output forwarded'}
                conn.send(json.dumps(response).encode())
            
            elif request['cmd'] == 'interactive_complete':
                # Handle interactive job completion
                job_id = request.get('job_id')
                exit_code = request.get('exit_code', 0)
                
                # Send completion to interactive clients
                if job_id in scheduler.interactive_clients:
                    dead_clients = []
                    for client_socket in scheduler.interactive_clients[job_id]:
                        try:
                            completion_msg = {
                                'type': 'completion',
                                'job_id': job_id,
                                'exit_code': exit_code
                            }
                            client_socket.send((json.dumps(completion_msg) + '\n').encode())
                        except:
                            dead_clients.append(client_socket)
                    
                    # Clean up client list
                    for client in dead_clients:
                        try:
                            client.close()
                        except:
                            pass
                    
                    del scheduler.interactive_clients[job_id]
                
                response = {'status': 'ok', 'message': 'Interactive completion processed'}
                conn.send(json.dumps(response).encode())
            
            elif request['cmd'] == 'get_cluster_resources':
                # Get cluster-wide resource information
                try:
                    cluster_resources = resource_manager.get_cluster_resources()
                    response = {
                        'status': 'ok', 
                        'resources': cluster_resources,
                        'nodes': {node_id: {
                            'hostname': node.hostname,
                            'status': node.status,
                            'gpu_count': node.gpu_count,
                            'gpu_type': node.gpu_type,
                            'last_heartbeat': node.last_heartbeat
                        } for node_id, node in resource_manager.nodes.items()}
                    }
                    conn.send(json.dumps(response).encode())
                    print(f"[DEBUG] Cluster resources sent to {addr}")
                except Exception as e:
                    response = {'status': 'error', 'message': f'Failed to get cluster resources: {str(e)}'}
                    conn.send(json.dumps(response).encode())
                    print(f"[ERROR] Failed to get cluster resources: {e}")
            
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
            if close_connection:
                conn.close()
    
    # Start master server
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
