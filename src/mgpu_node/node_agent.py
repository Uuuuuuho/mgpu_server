"""
Node Agent for Multi-GPU Scheduler
"""

import socket
import subprocess
import threading
import json
import time
import os
import sys
from typing import Dict, List, Optional, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu_core.models.job_models import JobProcess, MessageType
from mgpu_core.network.network_manager import NetworkManager
from mgpu_core.utils.logging_utils import setup_logger
from mgpu_core.utils.system_utils import GPUManager, IPManager


logger = setup_logger(__name__)


class NodeAgent:
    """Node agent for job execution"""
    
    def __init__(self, node_id: str, host: str = '0.0.0.0', port: int = 8081, 
                 master_host: str = '127.0.0.1', master_port: int = 8080, gpu_count: int = 1):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.master_host = master_host
        self.master_port = master_port
        self.gpu_count = gpu_count
        
        self.running_jobs = {}  # job_id -> JobProcess
        self.available_gpus = list(range(gpu_count))
        self.lock = threading.RLock()
        self.running = False
        self.server_socket = None
    
    def get_actual_ip_address(self) -> str:
        """Get actual IP address using multiple detection methods"""
        return IPManager.get_actual_ip_address(self.master_host, self.master_port)
    
    def register_with_master(self) -> bool:
        """Register this node with the master server"""
        try:
            # Get actual IP address
            actual_ip = self.get_actual_ip_address()
            
            # Get GPU information
            gpu_info = GPUManager.get_gpu_info(self.gpu_count)
            
            register_request = {
                'cmd': MessageType.NODE_REGISTER,
                'node_id': self.node_id,
                'host': actual_ip,  # Use actual IP instead of configured host
                'port': self.port,
                'gpu_count': self.gpu_count,
                'gpu_info': gpu_info
            }
            
            logger.info(f"Registering with master using IP: {actual_ip}")
            
            sock = NetworkManager.connect_to_server(self.master_host, self.master_port, 10.0)
            if not sock:
                logger.error(f"Cannot connect to master at {self.master_host}:{self.master_port}")
                return False
            
            if not NetworkManager.send_json_message(sock, register_request, 10.0):
                logger.error("Failed to send registration request")
                sock.close()
                return False
            
            response = NetworkManager.receive_json_message(sock, 10.0)
            sock.close()
            
            if response and response.get('status') == 'ok':
                logger.info(f"Successfully registered with master server")
                return True
            else:
                logger.error(f"Registration failed: {response.get('message', 'Unknown error') if response else 'No response'}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register with master: {e}")
            return False
    
    def handle_run_job(self, request: Dict) -> Dict:
        """Handle job execution request"""
        job_id = request.get('job_id')
        command = request.get('command')
        gpus = request.get('gpus', [])
        interactive = request.get('interactive', False)
        
        if not job_id or not command:
            return {'status': 'error', 'message': 'job_id and command required'}
        
        try:
            with self.lock:
                # Check if job already running
                if job_id in self.running_jobs:
                    return {'status': 'error', 'message': f'Job {job_id} already running'}
                
                # Check GPU availability
                for gpu in gpus:
                    if gpu not in self.available_gpus:
                        return {'status': 'error', 'message': f'GPU {gpu} not available'}
                
                # Reserve GPUs
                for gpu in gpus:
                    self.available_gpus.remove(gpu)
                
                # Set environment
                env = os.environ.copy()
                if gpus:
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
                else:
                    env['CUDA_VISIBLE_DEVICES'] = ''
                
                # Start process with output capture for all jobs
                process = subprocess.Popen(
                    command,
                    shell=True,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE if interactive else None,
                    bufsize=1,
                    universal_newlines=True,
                    text=True
                )
                
                # Store job
                job_process = JobProcess(job_id, process, gpus, interactive)
                self.running_jobs[job_id] = job_process
                
                # Start monitoring thread - all jobs now stream output
                monitor_thread = threading.Thread(target=self.monitor_job_with_output, args=(job_id,))
                monitor_thread.daemon = True
                monitor_thread.start()
                
                logger.info(f"Started job {job_id} on GPUs {gpus}: {command[:50]}...")
                return {'status': 'ok', 'message': f'Job {job_id} started'}
                
        except Exception as e:
            # Restore GPUs on error
            with self.lock:
                for gpu in gpus:
                    if gpu not in self.available_gpus:
                        self.available_gpus.append(gpu)
            
            logger.error(f"Failed to start job {job_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_cancel_job(self, request: Dict) -> Dict:
        """Handle job cancellation"""
        job_id = request.get('job_id')
        
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            with self.lock:
                if job_id not in self.running_jobs:
                    return {'status': 'error', 'message': f'Job {job_id} not found'}
                
                job_process = self.running_jobs[job_id]
                
                # Terminate process
                try:
                    job_process.process.terminate()
                    job_process.process.wait(timeout=10)
                except:
                    job_process.process.kill()
                
                # Restore GPUs
                for gpu in job_process.gpus:
                    if gpu not in self.available_gpus:
                        self.available_gpus.append(gpu)
                
                # Remove job
                del self.running_jobs[job_id]
                
                logger.info(f"Cancelled job {job_id}")
                return {'status': 'ok', 'message': f'Job {job_id} cancelled'}
                
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_status_request(self, request: Dict) -> Dict:
        """Handle status request"""
        try:
            with self.lock:
                status = {
                    'node_id': self.node_id,
                    'available_gpus': self.available_gpus.copy(),
                    'running_jobs': list(self.running_jobs.keys()),
                    'gpu_utilization': {
                        str(i): GPUManager.get_gpu_utilization(i, self.available_gpus) 
                        for i in range(self.gpu_count)
                    }
                }
                
                return {'status': 'ok', 'data': status}
                
        except Exception as e:
            logger.error(f"Status request error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def monitor_job_with_output(self, job_id: str):
        """Monitor job completion with output streaming to client"""
        try:
            if job_id not in self.running_jobs:
                return
            
            job_process = self.running_jobs[job_id]
            process = job_process.process
            
            # Stream output in real-time to master for forwarding to client
            while process.poll() is None:
                if process.stdout:
                    try:
                        line = process.stdout.readline()
                        if line:
                            # Send output to master
                            self.send_output_to_master(job_id, line, job_process.interactive)
                    except:
                        break
                        
                time.sleep(0.05)  # Small delay to prevent flooding
            
            # Get final exit code
            exit_code = process.wait()
            
            # Send completion notification
            self.send_completion_to_master(job_id, exit_code, job_process.interactive)
            
            # Clean up job
            with self.lock:
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    # Restore GPUs
                    for gpu in job_process.gpus:
                        if gpu not in self.available_gpus:
                            self.available_gpus.append(gpu)
                    del self.running_jobs[job_id]
            
            logger.info(f"Job {job_id} completed with exit code {exit_code}")
            
        except Exception as e:
            logger.error(f"Job monitoring error for {job_id}: {e}")
    
    def send_output_to_master(self, job_id: str, data: str, interactive: bool):
        """Send job output to master server"""
        try:
            sock = NetworkManager.connect_to_server(self.master_host, self.master_port, 5.0)
            if not sock:
                return
            
            output_msg = {
                'cmd': MessageType.JOB_OUTPUT,
                'job_id': job_id,
                'data': data,
                'interactive': interactive,
                'node_id': self.node_id
            }
            
            NetworkManager.send_json_message(sock, output_msg, 5.0)
            sock.close()
            
        except Exception as e:
            logger.debug(f"Failed to send output to master: {e}")
    
    def send_completion_to_master(self, job_id: str, exit_code: int, interactive: bool):
        """Send job completion notification to master"""
        try:
            sock = NetworkManager.connect_to_server(self.master_host, self.master_port, 10.0)
            if not sock:
                logger.error("Cannot connect to master for completion notification")
                return
            
            completion_msg = {
                'cmd': MessageType.INTERACTIVE_COMPLETE if interactive else MessageType.JOB_COMPLETE,
                'job_id': job_id,
                'exit_code': exit_code,
                'node_id': self.node_id
            }
            
            NetworkManager.send_json_message(sock, completion_msg, 10.0)
            response = NetworkManager.receive_json_message(sock, 10.0)
            sock.close()
            
            logger.info(f"Sent completion notification for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to notify job completion: {e}")
    
    def send_heartbeat(self):
        """Send periodic heartbeat to master"""
        while self.running:
            try:
                sock = NetworkManager.connect_to_server(self.master_host, self.master_port, 5.0)
                if sock:
                    heartbeat_msg = {
                        'cmd': MessageType.NODE_STATUS,
                        'node_id': self.node_id,
                        'available_gpus': self.available_gpus.copy(),
                        'running_jobs': list(self.running_jobs.keys())
                    }
                    
                    NetworkManager.send_json_message(sock, heartbeat_msg, 5.0)
                    sock.close()
                    
                time.sleep(30)  # Send heartbeat every 30 seconds
                    
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                time.sleep(30)
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            data = client_socket.recv(8192).decode()
            if not data:
                return
            
            request = json.loads(data)
            cmd = request.get('cmd')
            
            if cmd == MessageType.RUN:
                response = self.handle_run_job(request)
                
            elif cmd == MessageType.CANCEL:
                response = self.handle_cancel_job(request)
                
            elif cmd == MessageType.STATUS:
                response = self.handle_status_request(request)
                
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
    
    def start_agent(self):
        """Start the node agent"""
        self.running = True
        
        # Start server first
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        logger.info(f"Node Agent {self.node_id} started on {self.host}:{self.port}")
        logger.info(f"Master server: {self.master_host}:{self.master_port}")
        logger.info(f"Available GPUs: {self.available_gpus}")
        
        # Register with master server
        logger.info("Registering with master server...")
        if self.register_with_master():
            logger.info("Registration successful")
        else:
            logger.warning("Registration failed, but continuing to serve requests")
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        try:
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.debug(f"Connection from {address}")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Accept error: {e}")
                        
        except KeyboardInterrupt:
            logger.info("Node agent shutdown requested")
        finally:
            self.stop_agent()
    
    def stop_agent(self):
        """Stop the node agent"""
        logger.info("Stopping node agent...")
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("Node agent stopped")
