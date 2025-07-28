"""
Master server implementation for Multi-GPU Scheduler
"""

import socket
import threading
import json
import time
import uuid
import copy
from typing import Dict, List, Optional, Set, Any

from ..core.models import MessageType, SimpleJob, JobProcess, NodeInfo
from ..core.config import Config
from ..utils.logging import setup_logger
from ..utils.network import send_json_message, receive_json_message

logger = setup_logger(__name__)


class SimpleMaster:
    """Simplified Master Server for Multi-GPU Job Scheduling"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
        # Job management
        self.job_queue = []
        self.running_jobs = {}
        self.completed_jobs = {}
        self.interactive_jobs = {}
        
        # Node management
        self.nodes = {}
        self.lock = threading.Lock()
        
        # Job ID counter
        self.job_counter = 0
    
    def start_server(self):
        """Start the master server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(10)
            self.running = True
            
            logger.info(f"Master server started on {self.host}:{self.port}")
            print(f"Master server started on {self.host}:{self.port}")
            
            # Start scheduler thread
            scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            scheduler_thread.start()
            
            # Start node monitor thread
            monitor_thread = threading.Thread(target=self._node_monitor_loop, daemon=True)
            monitor_thread.start()
            
            # Accept connections
            while self.running:
                try:
                    client_socket, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Accept error: {e}")
                        
        except Exception as e:
            logger.error(f"Server start error: {e}")
            print(f"Failed to start server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the master server"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("Master server stopped")
    
    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle client connection"""
        try:
            logger.debug(f"Client connected: {addr}")
            
            # Receive request
            request = receive_json_message(client_socket, 30.0)
            if not request:
                return
            
            cmd = request.get('cmd')
            if cmd == MessageType.SUBMIT:
                self._handle_job_submit(client_socket, request)
            elif cmd == MessageType.QUEUE:
                self._handle_queue_request(client_socket)
            elif cmd == MessageType.CANCEL:
                self._handle_cancel_request(client_socket, request)
            elif cmd == MessageType.GET_JOB_OUTPUT:
                self._handle_output_request(client_socket, request)
            elif cmd == MessageType.NODE_REGISTER:
                self._handle_node_register(client_socket, request)
            elif cmd == MessageType.NODE_HEARTBEAT:
                self._handle_node_heartbeat(client_socket, request)
            elif cmd == MessageType.JOB_UPDATE:
                self._handle_job_update(client_socket, request)
            else:
                response = {'status': 'error', 'message': f'Unknown command: {cmd}'}
                send_json_message(client_socket, response, 10.0)
                
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_job_submit(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job submission"""
        try:
            with self.lock:
                self.job_counter += 1
                job_id = f"job_{self.job_counter:06d}"
                
                job = SimpleJob(
                    id=job_id,
                    user=request.get('user', 'unknown'),
                    command=request['command'],
                    gpus=request['gpus'],
                    interactive=request.get('interactive', False),
                    node_gpu_ids=request.get('node_gpu_ids')
                )
                
                self.job_queue.append(job)
                logger.info(f"Job submitted: {job_id} - {job.command[:50]}...")
            
            if job.interactive:
                # Store socket for interactive session
                self.interactive_jobs[job_id] = {
                    'socket': client_socket,
                    'job': job,
                    'start_time': time.time()
                }
                
                response = {'status': 'ok', 'job_id': job_id, 'message': 'Interactive job submitted'}
                send_json_message(client_socket, response, 10.0)
                # Don't close socket for interactive jobs
            else:
                response = {'status': 'ok', 'job_id': job_id, 'message': 'Job submitted'}
                send_json_message(client_socket, response, 10.0)
                client_socket.close()
                
        except Exception as e:
            logger.error(f"Job submit error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
            client_socket.close()
    
    def _handle_queue_request(self, client_socket: socket.socket):
        """Handle queue status request"""
        try:
            with self.lock:
                queue_data = []
                for job in self.job_queue:
                    queue_data.append({
                        'id': job.id,
                        'user': job.user,
                        'cmd': job.command,
                        'gpus': job.gpus,
                        'submit_time': job.submit_time
                    })
                
                running_data = []
                for job_id, job_process in self.running_jobs.items():
                    running_data.append({
                        'id': job_id,
                        'user': job_process.job.user,
                        'cmd': job_process.job.command,
                        'gpus': job_process.job.gpus,
                        'assigned_node': job_process.assigned_node,
                        'start_time': job_process.start_time
                    })
                
                nodes_data = {}
                for node_id, node_info in self.nodes.items():
                    nodes_data[node_id] = {
                        'available_gpus': node_info.available_gpus,
                        'total_gpus': node_info.total_gpus,
                        'last_heartbeat': node_info.last_heartbeat
                    }
            
            response = {
                'status': 'ok',
                'queue': queue_data,
                'running': running_data,
                'nodes': nodes_data
            }
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Queue request error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _handle_cancel_request(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job cancellation request"""
        try:
            job_id = request.get('job_id')
            if not job_id:
                response = {'status': 'error', 'message': 'job_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            success = False
            message = ""
            
            with self.lock:
                # Check if job is in queue
                for i, job in enumerate(self.job_queue):
                    if job.id == job_id:
                        self.job_queue.pop(i)
                        success = True
                        message = f"Job {job_id} removed from queue"
                        break
                
                # Check if job is running
                if not success and job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    # Send cancel request to node
                    self._send_cancel_to_node(job_process.assigned_node, job_id)
                    success = True
                    message = f"Cancel request sent for job {job_id}"
                
                # Check if job is interactive
                if not success and job_id in self.interactive_jobs:
                    interactive_info = self.interactive_jobs[job_id]
                    try:
                        interactive_info['socket'].close()
                    except:
                        pass
                    del self.interactive_jobs[job_id]
                    success = True
                    message = f"Interactive job {job_id} cancelled"
            
            if not success:
                message = f"Job {job_id} not found"
            
            response = {'status': 'ok' if success else 'error', 'message': message}
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Cancel request error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _handle_output_request(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job output request"""
        try:
            job_id = request.get('job_id')
            from_line = request.get('from_line', 0)
            
            if not job_id:
                response = {'status': 'error', 'message': 'job_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            job_status = 'unknown'
            output_lines = []
            exit_code = None
            
            with self.lock:
                # Check running jobs
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    job_status = 'running'
                    output_lines = job_process.output_lines[from_line:]
                
                # Check completed jobs
                elif job_id in self.completed_jobs:
                    job_process = self.completed_jobs[job_id]
                    job_status = job_process.status
                    output_lines = job_process.output_lines[from_line:]
                    exit_code = job_process.exit_code
                
                # Check queued jobs
                else:
                    for job in self.job_queue:
                        if job.id == job_id:
                            job_status = 'queued'
                            break
            
            response = {
                'status': 'ok',
                'job_status': job_status,
                'output': output_lines,
                'exit_code': exit_code
            }
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Output request error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _handle_node_register(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle node registration"""
        try:
            node_id = request.get('node_id')
            total_gpus = request.get('total_gpus', [])
            
            if not node_id:
                response = {'status': 'error', 'message': 'node_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            with self.lock:
                self.nodes[node_id] = NodeInfo(
                    id=node_id,
                    host=request.get('host', 'unknown'),
                    port=request.get('port', 0),
                    total_gpus=total_gpus,
                    available_gpus=total_gpus.copy(),
                    last_heartbeat=time.time()
                )
            
            logger.info(f"Node registered: {node_id} with {len(total_gpus)} GPUs")
            response = {'status': 'ok', 'message': 'Node registered'}
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Node register error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _handle_node_heartbeat(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle node heartbeat"""
        try:
            node_id = request.get('node_id')
            available_gpus = request.get('available_gpus', [])
            
            if not node_id:
                response = {'status': 'error', 'message': 'node_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            with self.lock:
                if node_id in self.nodes:
                    self.nodes[node_id].available_gpus = available_gpus
                    self.nodes[node_id].last_heartbeat = time.time()
                    logger.debug(f"Heartbeat from {node_id}: {len(available_gpus)} GPUs available")
                else:
                    logger.warning(f"Heartbeat from unregistered node: {node_id}")
            
            response = {'status': 'ok'}
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Node heartbeat error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _handle_job_update(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job status update from node"""
        try:
            job_id = request.get('job_id')
            status = request.get('status')
            
            if not job_id:
                response = {'status': 'error', 'message': 'job_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            with self.lock:
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    
                    if status == 'completed' or status == 'failed':
                        # Move to completed jobs
                        job_process.status = status
                        job_process.exit_code = request.get('exit_code')
                        job_process.end_time = time.time()
                        
                        # Add any final output
                        if 'output' in request:
                            job_process.output_lines.extend(request['output'])
                        
                        self.completed_jobs[job_id] = job_process
                        del self.running_jobs[job_id]
                        
                        logger.info(f"Job {job_id} {status} with exit code {job_process.exit_code}")
                        
                        # Handle interactive job completion
                        if job_id in self.interactive_jobs:
                            self._handle_interactive_completion(job_id, job_process)
                    
                    elif status == 'running':
                        # Update output
                        if 'output' in request:
                            new_lines = request['output']
                            job_process.output_lines.extend(new_lines)
                            
                            # Handle interactive job output
                            if job_id in self.interactive_jobs:
                                self._send_interactive_output(job_id, new_lines)
            
            response = {'status': 'ok'}
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Job update error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
        finally:
            client_socket.close()
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._schedule_jobs()
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5.0)
    
    def _schedule_jobs(self):
        """Schedule jobs to available nodes"""
        with self.lock:
            if not self.job_queue or not self.nodes:
                return
            
            # Try to schedule each job in queue
            scheduled_jobs = []
            
            for job in self.job_queue:
                best_node = self._find_best_node(job)
                if best_node:
                    # Assign job to node
                    assigned_gpus = self._allocate_gpus(best_node, job)
                    if assigned_gpus:
                        job_process = JobProcess(
                            job=job,
                            assigned_node=best_node,
                            assigned_gpus=assigned_gpus,
                            start_time=time.time()
                        )
                        
                        # Send job to node
                        if self._send_job_to_node(best_node, job, assigned_gpus):
                            self.running_jobs[job.id] = job_process
                            scheduled_jobs.append(job)
                            logger.info(f"Job {job.id} scheduled to node {best_node} with GPUs {assigned_gpus}")
            
            # Remove scheduled jobs from queue
            for job in scheduled_jobs:
                self.job_queue.remove(job)
    
    def _find_best_node(self, job: SimpleJob) -> Optional[str]:
        """Find the best node for a job"""
        # Check if specific node GPUs are requested
        if job.node_gpu_ids:
            for node_id, requested_gpus in job.node_gpu_ids.items():
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    available_set = set(node.available_gpus)
                    requested_set = set(requested_gpus)
                    if requested_set.issubset(available_set):
                        return node_id
            return None
        
        # Find node with enough available GPUs
        best_node = None
        min_waste = float('inf')
        
        for node_id, node in self.nodes.items():
            if len(node.available_gpus) >= job.gpus:
                waste = len(node.available_gpus) - job.gpus
                if waste < min_waste:
                    min_waste = waste
                    best_node = node_id
        
        return best_node
    
    def _allocate_gpus(self, node_id: str, job: SimpleJob) -> List[int]:
        """Allocate GPUs for a job on a node"""
        node = self.nodes[node_id]
        
        if job.node_gpu_ids and node_id in job.node_gpu_ids:
            # Use specific requested GPUs
            requested_gpus = job.node_gpu_ids[node_id]
            available_set = set(node.available_gpus)
            requested_set = set(requested_gpus)
            
            if requested_set.issubset(available_set):
                # Remove allocated GPUs from available list
                for gpu_id in requested_gpus:
                    node.available_gpus.remove(gpu_id)
                return requested_gpus
        else:
            # Allocate first available GPUs
            if len(node.available_gpus) >= job.gpus:
                allocated = node.available_gpus[:job.gpus]
                node.available_gpus = node.available_gpus[job.gpus:]
                return allocated
        
        return []
    
    def _send_job_to_node(self, node_id: str, job: SimpleJob, assigned_gpus: List[int]) -> bool:
        """Send job to node for execution"""
        try:
            node = self.nodes[node_id]
            
            # Connect to node
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((node.host, node.port))
            
            # Send job assignment
            request = {
                'cmd': MessageType.RUN_JOB,
                'job_id': job.id,
                'command': job.command,
                'gpus': assigned_gpus,
                'interactive': job.interactive
            }
            
            success = send_json_message(sock, request, 10.0)
            sock.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send job to node {node_id}: {e}")
            return False
    
    def _send_cancel_to_node(self, node_id: str, job_id: str):
        """Send cancel request to node"""
        try:
            node = self.nodes[node_id]
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((node.host, node.port))
            
            request = {
                'cmd': MessageType.CANCEL_JOB,
                'job_id': job_id
            }
            
            send_json_message(sock, request, 10.0)
            sock.close()
            
        except Exception as e:
            logger.error(f"Failed to send cancel to node {node_id}: {e}")
    
    def _node_monitor_loop(self):
        """Monitor node health"""
        while self.running:
            try:
                current_time = time.time()
                dead_nodes = []
                
                with self.lock:
                    for node_id, node in self.nodes.items():
                        if current_time - node.last_heartbeat > 60.0:  # 60 seconds timeout
                            dead_nodes.append(node_id)
                
                # Remove dead nodes
                with self.lock:
                    for node_id in dead_nodes:
                        logger.warning(f"Node {node_id} appears to be dead, removing")
                        del self.nodes[node_id]
                        
                        # Cancel jobs running on dead node
                        cancelled_jobs = []
                        for job_id, job_process in self.running_jobs.items():
                            if job_process.assigned_node == node_id:
                                cancelled_jobs.append(job_id)
                        
                        for job_id in cancelled_jobs:
                            job_process = self.running_jobs[job_id]
                            job_process.status = 'failed'
                            job_process.exit_code = -1
                            job_process.end_time = current_time
                            job_process.output_lines.append(f"Job cancelled due to node failure")
                            
                            self.completed_jobs[job_id] = job_process
                            del self.running_jobs[job_id]
                            
                            logger.warning(f"Job {job_id} cancelled due to node {node_id} failure")
                
                time.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Node monitor error: {e}")
                time.sleep(30.0)
    
    def _handle_interactive_completion(self, job_id: str, job_process: JobProcess):
        """Handle completion of interactive job"""
        if job_id in self.interactive_jobs:
            interactive_info = self.interactive_jobs[job_id]
            try:
                completion_msg = {
                    'type': 'completion',
                    'exit_code': job_process.exit_code,
                    'status': job_process.status
                }
                message = json.dumps(completion_msg) + '\n'
                interactive_info['socket'].send(message.encode())
            except Exception as e:
                logger.debug(f"Error sending completion to interactive job {job_id}: {e}")
            finally:
                try:
                    interactive_info['socket'].close()
                except:
                    pass
                del self.interactive_jobs[job_id]
    
    def _send_interactive_output(self, job_id: str, output_lines: List[str]):
        """Send output to interactive job client"""
        if job_id in self.interactive_jobs:
            interactive_info = self.interactive_jobs[job_id]
            try:
                for line in output_lines:
                    output_msg = {
                        'type': 'output',
                        'data': line
                    }
                    message = json.dumps(output_msg) + '\n'
                    interactive_info['socket'].send(message.encode())
            except Exception as e:
                logger.debug(f"Error sending output to interactive job {job_id}: {e}")
                # Clean up broken connection
                try:
                    interactive_info['socket'].close()
                except:
                    pass
                del self.interactive_jobs[job_id]
