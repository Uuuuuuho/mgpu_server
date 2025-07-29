#!/usr/bin/env python3
"""
Simplified Master Server for Multi-GPU Scheduler
Minimal command set: submit, queue, cancel, node_status
"""

import socket
import json
import threading
import time
import uuid
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimpleJob:
    """Simplified job representation"""
    id: str
    user: str
    cmd: str
    gpus_needed: int
    node_gpu_ids: Optional[Dict[str, List[int]]] = None
    priority: int = 0
    status: str = "queued"  # queued, running, completed, failed
    interactive: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    assigned_node: Optional[str] = None
    assigned_gpus: Optional[List[int]] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user': self.user,
            'cmd': self.cmd,
            'gpus_needed': self.gpus_needed,
            'node_gpu_ids': self.node_gpu_ids,
            'priority': self.priority,
            'status': self.status,
            'interactive': self.interactive,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'exit_code': self.exit_code,
            'assigned_node': self.assigned_node,
            'assigned_gpus': self.assigned_gpus
        }

class NodeInfo:
    """Node information"""
    def __init__(self, node_id: str, host: str, port: int, gpu_count: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.gpu_count = gpu_count
        self.available_gpus = list(range(gpu_count))  # All GPUs available initially
        self.running_jobs = []
        self.last_heartbeat = time.time()

class SimpleMaster:
    """Simplified Master Server"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.nodes = {}  # node_id -> NodeInfo
        self.job_queue = queue.Queue()
        self.running_jobs = {}  # job_id -> SimpleJob
        self.completed_jobs = {}  # job_id -> SimpleJob
        self.job_outputs = {}  # job_id -> List[str] - store output for non-interactive jobs
        self.interactive_clients = {}  # job_id -> List[socket] - interactive client connections
        self.lock = threading.RLock()
        self.running = False
        
    def add_node(self, node_id: str, host: str, port: int, gpu_count: int):
        """Add a node to the cluster"""
        with self.lock:
            self.nodes[node_id] = NodeInfo(node_id, host, port, gpu_count)
            # Note: Detailed logging is now handled in handle_node_register
            logger.info(f"Node {node_id} added to cluster (total nodes: {len(self.nodes)})")
    
    def handle_submit(self, request: Dict) -> Dict:
        """Handle job submission"""
        try:
            # Create job
            job = SimpleJob(
                id=request.get('job_id', str(uuid.uuid4())[:8].upper()),
                user=request.get('user', 'unknown'),
                cmd=request.get('command', ''),  # Use command field from new client
                gpus_needed=request.get('gpus', 1),  # Use gpus field from new client
                node_gpu_ids=request.get('node_gpu_ids'),
                priority=request.get('priority', 0),
                interactive=request.get('interactive', False)
            )
            
            # Add to queue
            self.job_queue.put(job)
            logger.info(f"Job {job.id} submitted: {job.cmd[:50]}...")
            
            return {'status': 'ok', 'job_id': job.id, 'message': 'Job submitted'}
            
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_queue_status(self, request: Dict) -> Dict:
        """Handle queue status request"""
        try:
            with self.lock:
                # Get queued jobs
                queued = []
                temp_queue = queue.Queue()
                
                while not self.job_queue.empty():
                    job = self.job_queue.get()
                    queued.append(job.to_dict())
                    temp_queue.put(job)
                
                # Restore queue
                while not temp_queue.empty():
                    self.job_queue.put(temp_queue.get())
                
                # Get running jobs
                running = [job.to_dict() for job in self.running_jobs.values()]
                
                return {
                    'status': 'ok',
                    'queue': queued,
                    'running': running,
                    'nodes': {nid: {
                        'available_gpus': node.available_gpus,
                        'running_jobs': node.running_jobs,
                        'last_heartbeat': node.last_heartbeat
                    } for nid, node in self.nodes.items()}
                }
                
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_cancel(self, request: Dict) -> Dict:
        """Handle job cancellation"""
        job_id = request.get('job_id')
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            with self.lock:
                # Check if job is running
                if job_id in self.running_jobs:
                    job = self.running_jobs[job_id]
                    # Send cancel to node
                    self.send_to_node(job.assigned_node, {
                        'cmd': 'cancel',
                        'job_id': job_id
                    })
                    job.status = 'cancelled'
                    self.completed_jobs[job_id] = job
                    del self.running_jobs[job_id]
                    return {'status': 'ok', 'message': f'Job {job_id} cancelled'}
                
                # Check if job is queued
                temp_queue = queue.Queue()
                found = False
                
                while not self.job_queue.empty():
                    job = self.job_queue.get()
                    if job.id == job_id:
                        job.status = 'cancelled'
                        self.completed_jobs[job_id] = job
                        found = True
                    else:
                        temp_queue.put(job)
                
                # Restore queue
                while not temp_queue.empty():
                    self.job_queue.put(temp_queue.get())
                
                if found:
                    return {'status': 'ok', 'message': f'Job {job_id} cancelled'}
                else:
                    return {'status': 'error', 'message': f'Job {job_id} not found'}
                    
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_node_status(self, request: Dict) -> Dict:
        """Handle node status update"""
        node_id = request.get('node_id')
        if not node_id or node_id not in self.nodes:
            return {'status': 'error', 'message': 'Invalid node_id'}
        
        try:
            with self.lock:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                # Update GPU availability
                if 'available_gpus' in request:
                    node.available_gpus = request['available_gpus']
                
                # Update running jobs
                if 'running_jobs' in request:
                    node.running_jobs = request['running_jobs']
                
                return {'status': 'ok', 'message': 'Node status updated'}
                
        except Exception as e:
            logger.error(f"Node status error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_node_register(self, request: Dict) -> Dict:
        """Handle node registration"""
        try:
            node_id = request.get('node_id')
            host = request.get('host', '127.0.0.1')
            port = request.get('port', 8081)
            gpu_count = request.get('gpu_count', 1)
            gpu_info = request.get('gpu_info', [])  # Optional detailed GPU info
            
            if not node_id:
                return {'status': 'error', 'message': 'node_id required'}
            
            # Add or update node
            self.add_node(node_id, host, port, gpu_count)
            
            # Enhanced logging with GPU information
            if gpu_info:
                logger.info(f"🔗 Node {node_id} connected from {host}:{port}")
                logger.info(f"   └─ 🎮 {gpu_count} GPU(s) detected:")
                for i, gpu in enumerate(gpu_info):
                    gpu_name = gpu.get('name', 'Unknown GPU')
                    gpu_memory = gpu.get('memory', 'Unknown')
                    logger.info(f"      ├─ GPU {i}: {gpu_name} ({gpu_memory})")
            else:
                logger.info(f"🔗 Node {node_id} connected from {host}:{port} with {gpu_count} GPU(s)")
            
            return {'status': 'ok', 'message': f'Node {node_id} registered successfully'}
            
        except Exception as e:
            logger.error(f"Node registration error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def send_to_node(self, node_id: str, message: Dict) -> Optional[Dict]:
        """Send message to node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((node.host, node.port))
            sock.send(json.dumps(message).encode())
            
            response = sock.recv(8192).decode()
            sock.close()
            
            return json.loads(response) if response else None
            
        except Exception as e:
            logger.error(f"Failed to send to node {node_id}: {e}")
            return None
    
    def find_available_node(self, job: SimpleJob) -> Optional[str]:
        """Find available node for job"""
        with self.lock:
            # If specific node-gpu mapping is requested
            if job.node_gpu_ids:
                for node_id, gpu_list in job.node_gpu_ids.items():
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        if all(gpu in node.available_gpus for gpu in gpu_list):
                            return node_id
                return None
            
            # Find any node with enough GPUs
            for node_id, node in self.nodes.items():
                if len(node.available_gpus) >= job.gpus_needed:
                    return node_id
            
            return None
    
    def schedule_jobs(self):
        """Job scheduler thread"""
        while self.running:
            try:
                if not self.job_queue.empty():
                    job = self.job_queue.get(timeout=1.0)
                    
                    # Find available node
                    node_id = self.find_available_node(job)
                    if node_id:
                        # Assign GPUs
                        node = self.nodes[node_id]
                        if job.node_gpu_ids and node_id in job.node_gpu_ids:
                            assigned_gpus = job.node_gpu_ids[node_id]
                        else:
                            assigned_gpus = node.available_gpus[:job.gpus_needed]
                        
                        # Update node state
                        for gpu in assigned_gpus:
                            if gpu in node.available_gpus:
                                node.available_gpus.remove(gpu)
                        
                        # Update job
                        job.status = 'running'
                        job.assigned_node = node_id
                        job.assigned_gpus = assigned_gpus
                        job.start_time = time.time()
                        
                        # Send to node
                        node_message = {
                            'cmd': 'run',
                            'job_id': job.id,
                            'command': job.cmd,
                            'gpus': assigned_gpus,
                            'interactive': job.interactive
                        }
                        
                        response = self.send_to_node(node_id, node_message)
                        if response and response.get('status') == 'ok':
                            self.running_jobs[job.id] = job
                            logger.info(f"Job {job.id} started on {node_id} with GPUs {assigned_gpus}")
                        else:
                            # Restore GPUs and requeue
                            for gpu in assigned_gpus:
                                node.available_gpus.append(gpu)
                            job.status = 'queued'
                            self.job_queue.put(job)
                            logger.error(f"Failed to start job {job.id} on {node_id}")
                    else:
                        # No available resources, put back in queue
                        self.job_queue.put(job)
                        time.sleep(1.0)
                else:
                    time.sleep(0.5)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def handle_job_completion(self, request: Dict) -> Dict:
        """Handle job completion from node"""
        job_id = request.get('job_id')
        exit_code = request.get('exit_code', 0)
        
        if job_id not in self.running_jobs:
            return {'status': 'error', 'message': 'Job not found'}
        
        try:
            with self.lock:
                job = self.running_jobs[job_id]
                job.status = 'completed' if exit_code == 0 else 'failed'
                job.exit_code = exit_code
                job.end_time = time.time()
                
                # Restore GPU availability
                if job.assigned_node and job.assigned_gpus:
                    node = self.nodes[job.assigned_node]
                    for gpu in job.assigned_gpus:
                        if gpu not in node.available_gpus:
                            node.available_gpus.append(gpu)
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.running_jobs[job_id]
                
                logger.info(f"Job {job_id} completed with exit code {exit_code}")
                return {'status': 'ok', 'message': 'Job completion recorded'}
                
        except Exception as e:
            logger.error(f"Job completion error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    # Interactive job handlers
    interactive_clients = {}  # job_id -> list of client sockets
    
    def handle_get_job_output(self, request: Dict) -> Dict:
        """Handle request to get job output for non-interactive jobs"""
        job_id = request.get('job_id')
        from_line = request.get('from_line', 0)
        
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            with self.lock:
                # Get job status
                job_status = 'unknown'
                exit_code = None
                
                if job_id in self.running_jobs:
                    job_status = 'running'
                elif job_id in self.completed_jobs:
                    job = self.completed_jobs[job_id]
                    job_status = job.status
                    exit_code = job.exit_code
                
                # Get output lines
                output_lines = self.job_outputs.get(job_id, [])
                
                return {
                    'status': 'ok',
                    'job_status': job_status,
                    'exit_code': exit_code,
                    'output': output_lines,
                    'total_lines': len(output_lines)
                }
                
        except Exception as e:
            logger.error(f"Get job output error: {e}")
            return {'status': 'error', 'message': str(e)}

    def handle_job_output(self, request: Dict) -> Dict:
        """Handle job output from nodes and forward to clients"""
        job_id = request.get('job_id')
        data = request.get('data', '')
        interactive = request.get('interactive', False)
        
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            # For non-interactive jobs, store output for later retrieval
            if not interactive:
                if job_id not in self.job_outputs:
                    self.job_outputs[job_id] = []
                self.job_outputs[job_id].append(data)
            
            # For interactive jobs, forward to connected clients
            if interactive and job_id in self.interactive_clients:
                output_msg = {
                    'type': 'output',
                    'job_id': job_id,
                    'data': data
                }
                
                # Send to all connected interactive clients
                dead_clients = []
                for client in self.interactive_clients[job_id]:
                    try:
                        client.send((json.dumps(output_msg) + '\n').encode())
                    except:
                        dead_clients.append(client)
                
                # Remove dead clients
                for client in dead_clients:
                    try:
                        client.close()
                    except:
                        pass
                    self.interactive_clients[job_id].remove(client)
            
            return {'status': 'ok', 'message': 'Output received'}
            
        except Exception as e:
            logger.error(f"Job output error: {e}")
            return {'status': 'error', 'message': str(e)}

    def handle_interactive_output(self, request: Dict) -> Dict:
        """Handle interactive output from node"""
        job_id = request.get('job_id')
        data = request.get('data', '')
        
        if job_id in self.interactive_clients:
            # Send output to all connected interactive clients
            dead_clients = []
            for client_socket in self.interactive_clients[job_id]:
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
                self.interactive_clients[job_id].remove(client)
        
        return {'status': 'ok', 'message': 'Output forwarded'}
    
    def handle_interactive_completion(self, request: Dict) -> Dict:
        """Handle interactive job completion with improved cleanup"""
        job_id = request.get('job_id')
        exit_code = request.get('exit_code', 0)
        
        logger.info(f"Interactive job {job_id} completion requested with exit code {exit_code}")
        
        # Send completion to interactive clients
        if job_id in self.interactive_clients:
            completion_msg = {
                'type': 'completion',
                'job_id': job_id,
                'exit_code': exit_code
            }
            
            dead_clients = []
            for client_socket in self.interactive_clients[job_id]:
                try:
                    client_socket.send((json.dumps(completion_msg) + '\n').encode())
                    # Give client time to receive the message
                    time.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Failed to send completion to client: {e}")
                    dead_clients.append(client_socket)
            
            # Clean up all clients for this job
            for client in self.interactive_clients[job_id]:
                try:
                    client.close()
                except:
                    pass
            
            # Remove the job from interactive clients
            del self.interactive_clients[job_id]
            logger.info(f"Cleaned up interactive clients for job {job_id}")
        
        # Also handle regular job completion
        return self.handle_job_completion(request)
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            data = client_socket.recv(8192).decode()
            if not data:
                return
            
            request = json.loads(data)
            cmd = request.get('cmd')
            
            if cmd == 'submit':
                response = self.handle_submit(request)
                
                # For interactive jobs, keep connection open
                if request.get('interactive') and response.get('status') == 'ok':
                    job_id = response['job_id']
                    # Send immediate response first
                    client_socket.send(json.dumps(response).encode())
                    
                    # Then handle interactive session
                    self.handle_interactive_client(client_socket, job_id)
                    return  # Don't close socket here
                else:
                    # Non-interactive job, send response and close
                    client_socket.send(json.dumps(response).encode())
                    
            elif cmd == 'queue':
                response = self.handle_queue_status(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'cancel':
                response = self.handle_cancel(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'node_status':
                response = self.handle_node_status(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'node_register':
                response = self.handle_node_register(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'job_complete':
                response = self.handle_job_completion(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'get_job_output':
                response = self.handle_get_job_output(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'job_output':
                response = self.handle_job_output(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'interactive_output':
                response = self.handle_interactive_output(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'interactive_complete':
                response = self.handle_interactive_completion(request)
                client_socket.send(json.dumps(response).encode())
                
            else:
                response = {'status': 'error', 'message': f'Unknown command: {cmd}'}
                client_socket.send(json.dumps(response).encode())
                
        except Exception as e:
            logger.error(f"Client handler error: {e}")
            try:
                error_response = {'status': 'error', 'message': str(e)}
                client_socket.send(json.dumps(error_response).encode())
            except:
                pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def handle_interactive_client(self, client_socket: socket.socket, job_id: str):
        """Handle interactive client connection with proper timeout and cleanup"""
        try:
            # Add client to interactive clients list
            if job_id not in self.interactive_clients:
                self.interactive_clients[job_id] = []
            self.interactive_clients[job_id].append(client_socket)
            
            logger.info(f"Interactive client connected for job {job_id}")
            
            # Keep connection alive until job completes with proper timeout
            start_time = time.time()
            max_session_time = 7200  # 2 hours maximum session time (increased for GPU tests)
            
            while True:
                try:
                    # Check various termination conditions
                    current_time = time.time()
                    
                    # 1. Check if session has timed out
                    if current_time - start_time > max_session_time:
                        logger.info(f"Interactive session {job_id} timed out after {max_session_time} seconds")
                        break
                    
                    # 2. Check if job is no longer running (but allow time for job to start)
                    with self.lock:
                        # Give more time for job to be scheduled and start (increased to 90 seconds for GPU initialization)
                        if current_time - start_time > 90.0 and job_id not in self.running_jobs:
                            # Also check if job completed recently
                            if job_id not in self.completed_jobs:
                                logger.info(f"Job {job_id} is no longer running and not completed, ending interactive session")
                                break
                    
                    # 3. Check if job completed (more comprehensive check)
                    job_completed = False
                    with self.lock:
                        if job_id in self.completed_jobs:
                            job_completed = True
                    
                    if job_completed:
                        logger.info(f"Job {job_id} completed, waiting for final output before ending session")
                        # Wait longer to ensure all output is sent (increased from 3.0 to 10.0)
                        time.sleep(10.0)
                        break
                    
                    # Try to read from client (with short timeout)
                    client_socket.settimeout(5.0)  # Increased from 2.0 to 5.0 seconds
                    try:
                        data = client_socket.recv(1024)
                        if not data:
                            logger.info(f"Client disconnected for job {job_id}")
                            break
                    except socket.timeout:
                        # Timeout is normal, just continue
                        continue
                    except (ConnectionResetError, BrokenPipeError):
                        logger.info(f"Client connection lost for job {job_id}")
                        break
                        
                except Exception as e:
                    logger.debug(f"Interactive client loop error: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Interactive client handler error: {e}")
        finally:
            # Clean up client
            try:
                if job_id in self.interactive_clients and client_socket in self.interactive_clients[job_id]:
                    self.interactive_clients[job_id].remove(client_socket)
                    logger.info(f"Removed interactive client for job {job_id}")
                
                # If no more clients for this job, clean up the entry
                if job_id in self.interactive_clients and len(self.interactive_clients[job_id]) == 0:
                    del self.interactive_clients[job_id]
                    
            except Exception as e:
                logger.debug(f"Error during interactive client cleanup: {e}")
            
            try:
                client_socket.close()
            except:
                pass
    
    def forward_interactive_data(self, client_sock: socket.socket, node_sock: socket.socket):
        """Forward data between client and node for interactive sessions"""
        def forward(src, dst, name):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.send(data)
            except Exception as e:
                logger.debug(f"Forward {name} ended: {e}")
        
        # Start forwarding threads
        client_to_node = threading.Thread(target=forward, args=(client_sock, node_sock, "client->node"))
        node_to_client = threading.Thread(target=forward, args=(node_sock, client_sock, "node->client"))
        
        client_to_node.daemon = True
        node_to_client.daemon = True
        
        client_to_node.start()
        node_to_client.start()
        
        # Wait for either thread to finish
        client_to_node.join(timeout=0.1)
        node_to_client.join(timeout=0.1)
        
        # Close connections
        try:
            node_sock.close()
        except:
            pass
    
    def start(self):
        """Start the master server"""
        self.running = True
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self.schedule_jobs)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Start server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(10)
        
        logger.info(f"Simple Master Server started on {self.host}:{self.port}")
        
        try:
            while self.running:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            self.running = False
            server_socket.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple Master Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--config', help='Cluster configuration file')
    
    args = parser.parse_args()
    
    master = SimpleMaster(args.host, args.port)
    
    # Add default node for testing
    master.add_node('node001', '127.0.0.1', 8081, 1)
    
    # Load config if provided
    if args.config:
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Clear default nodes and add from config
            master.nodes.clear()
            for node in config.get('nodes', []):
                master.add_node(
                    node['id'],
                    node['host'],
                    node['port'],
                    len(node.get('gpu_ids', [0]))
                )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    try:
        master.start()
    except KeyboardInterrupt:
        logger.info("Master server stopped")

if __name__ == "__main__":
    main()
