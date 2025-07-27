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
    """Query local GPU resources"""
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
    """Node agent - manages local resources and executes jobs"""
    
    def __init__(self, node_id: str, master_host: str, master_port: int, agent_port: int):
        self.node_id = node_id
        self.master_host = master_host
        self.master_port = master_port
        self.agent_port = agent_port
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.allocated_gpus: List[int] = []  # Currently allocated GPU list
        self.lock = threading.Lock()
        
    def get_node_resources(self) -> Dict:
        """Return node resource information"""
        gpus = get_available_gpus()
        available_gpu_indices = []
        
        for gpu in gpus:
            if gpu['index'] not in self.allocated_gpus:
                # Consider GPU available if memory utilization is less than 80% (increased for testing)
                if gpu['utilization'] < 80:
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
        """Execute single node job"""
        job_id = job_info['job_id']
        user = job_info['user']
        command = job_info['command']
        gpu_ids = job_info['gpu_ids']
        interactive = job_info.get('interactive', False)
        
        try:
            with self.lock:
                # Allocate GPUs
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        raise Exception(f"GPU {gpu_id} already allocated")
                    self.allocated_gpus.append(gpu_id)
            
            # Set CUDA_VISIBLE_DEVICES
            cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_ids))}"
            home_dir = os.path.expanduser(f'~{user}')
            
            full_command = f"cd {home_dir} && PYTHONUNBUFFERED=1 {cuda_env} {command}"
            
            # Execute job (different handling for interactive)
            if interactive:
                proc = subprocess.Popen([
                    'sudo', '-u', user, 'bash', '-lc', full_command
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, bufsize=1, universal_newlines=True, text=True,
                preexec_fn=os.setsid)
            else:
                proc = subprocess.Popen([
                    'sudo', '-u', user, 'bash', '-lc', full_command
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                universal_newlines=True, preexec_fn=os.setsid)
            
            with self.lock:
                self.running_jobs[job_id] = proc
            
            print(f"[INFO] Started {'interactive' if interactive else 'regular'} job {job_id} on GPUs {gpu_ids}")
            
            # Monitor job completion (different for interactive)
            if interactive:
                def monitor_interactive_job():
                    # Stream output in real-time for interactive jobs
                    while proc.poll() is None:
                        if proc.stdout:
                            line = proc.stdout.readline()
                            if line:
                                # Send output to master server
                                try:
                                    master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                    master_sock.settimeout(2.0)
                                    master_sock.connect((self.master_host, self.master_port))
                                    
                                    output_msg = {
                                        'cmd': 'interactive_output',
                                        'job_id': job_id,
                                        'data': line,
                                        'node_id': self.node_id
                                    }
                                    
                                    master_sock.send(json.dumps(output_msg).encode())
                                    master_sock.recv(1024)  # Receive response
                                    master_sock.close()
                                    
                                except Exception as e:
                                    print(f"[DEBUG] Failed to send interactive output: {e}")
                                    
                        time.sleep(0.1)
                    
                    # Get final exit code
                    exit_code = proc.wait()
                    
                    # Send completion notification
                    try:
                        master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        master_sock.settimeout(10.0)
                        master_sock.connect((self.master_host, self.master_port))
                        
                        completion_msg = {
                            'cmd': 'interactive_complete',
                            'job_id': job_id,
                            'exit_code': exit_code,
                            'node_id': self.node_id
                        }
                        
                        master_sock.send(json.dumps(completion_msg).encode())
                        master_sock.recv(1024)
                        master_sock.close()
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to notify interactive completion: {e}")
                    
                    # Cleanup
                    with self.lock:
                        for gpu_id in gpu_ids:
                            if gpu_id in self.allocated_gpus:
                                self.allocated_gpus.remove(gpu_id)
                        if job_id in self.running_jobs:
                            del self.running_jobs[job_id]
                    
                    print(f"[INFO] Interactive job {job_id} completed with exit code {exit_code}")
                
                threading.Thread(target=monitor_interactive_job, daemon=True).start()
            else:
                def monitor_job():
                    proc.wait()
                    with self.lock:
                        # Release GPUs
                        for gpu_id in gpu_ids:
                            if gpu_id in self.allocated_gpus:
                                self.allocated_gpus.remove(gpu_id)
                        # Remove from running jobs list
                        if job_id in self.running_jobs:
                            del self.running_jobs[job_id]
                    print(f"[INFO] Job {job_id} completed with exit code {proc.returncode}")
                
                threading.Thread(target=monitor_job, daemon=True).start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start job {job_id}: {e}")
            # Release allocated GPUs
            with self.lock:
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        self.allocated_gpus.remove(gpu_id)
            return False
    
    def start_distributed_job(self, job_info: Dict) -> bool:
        """Execute distributed job"""
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
                # Allocate GPUs
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        raise Exception(f"GPU {gpu_id} already allocated")
                    self.allocated_gpus.append(gpu_id)
            
            # Setup distributed execution environment
            env_vars = {
                'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpu_ids)),
                'PYTHONUNBUFFERED': '1'
            }
            
            if distributed_type == 'pytorch':
                env_vars.update({
                    'RANK': str(rank),
                    'WORLD_SIZE': str(world_size),
                    'MASTER_ADDR': master_node,
                    'MASTER_PORT': '29500'  # PyTorch default port
                })
            elif distributed_type == 'mpi':
                # MPI environment setup is handled by mpirun
                pass
            
            # Generate environment variables string
            env_str = ' '.join([f"{k}={v}" for k, v in env_vars.items()])
            
            home_dir = os.path.expanduser(f'~{user}')
            full_command = f"cd {home_dir} && {env_str} {command}"
            
            # Execute distributed job
            proc = subprocess.Popen([
                'sudo', '-u', user, 'bash', '-lc', full_command
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, preexec_fn=os.setsid)
            
            with self.lock:
                self.running_jobs[job_id] = proc
            
            print(f"[INFO] Started distributed job {job_id} rank {rank} on GPUs {gpu_ids}")
            
            # Monitor job completion
            def monitor_distributed_job():
                proc.wait()
                with self.lock:
                    # Release GPUs
                    for gpu_id in gpu_ids:
                        if gpu_id in self.allocated_gpus:
                            self.allocated_gpus.remove(gpu_id)
                    # Remove from running jobs list
                    if job_id in self.running_jobs:
                        del self.running_jobs[job_id]
                print(f"[INFO] Distributed job {job_id} rank {rank} completed with exit code {proc.returncode}")
            
            threading.Thread(target=monitor_distributed_job, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start distributed job {job_id}: {e}")
            # Release allocated GPUs
            with self.lock:
                for gpu_id in gpu_ids:
                    if gpu_id in self.allocated_gpus:
                        self.allocated_gpus.remove(gpu_id)
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job"""
        with self.lock:
            if job_id in self.running_jobs:
                proc = self.running_jobs[job_id]
                try:
                    # Terminate entire process tree
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
        """Handle requests from master server"""
        print(f"[DEBUG] Received connection from {addr}")
        
        try:
            # Set socket timeout
            conn.settimeout(10.0)
            
            data = conn.recv(4096)
            print(f"[DEBUG] Received data length: {len(data)} bytes")
            
            if not data:
                print(f"[WARNING] Empty data received from {addr}")
                return
            
            print(f"[DEBUG] Raw data: {data[:200]}...")  # Print only first 200 bytes
            
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
                # Heartbeat only sends OK response
                response = {'status': 'ok', 'message': 'heartbeat received'}
                print(f"[DEBUG] Heartbeat received from {addr}")
            
            # Send response
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
        """Start agent server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.agent_port))
        server.listen(10)
        
        print(f"[INFO] Node agent {self.node_id} started on port {self.agent_port}")
        
        while True:
            conn, addr = server.accept()
            threading.Thread(target=self.handle_request, args=(conn, addr), daemon=True).start()
    
    def send_heartbeat(self):
        """Send heartbeat to master server"""
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 5 second timeout
                
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
                
                # Receive response
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
            
            time.sleep(30)  # Heartbeat every 30 seconds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', required=True, help='Node identifier')
    parser.add_argument('--master-host', default='localhost', help='Master server hostname')
    parser.add_argument('--master-port', type=int, default=8080, help='Master server port')
    parser.add_argument('--agent-port', type=int, default=8081, help='Agent server port')
    args = parser.parse_args()
    
    agent = NodeAgent(args.node_id, args.master_host, args.master_port, args.agent_port)
    
    # Start heartbeat thread
    threading.Thread(target=agent.send_heartbeat, daemon=True).start()
    
    # Start agent server
    agent.start_agent_server()

if __name__ == "__main__":
    main()
