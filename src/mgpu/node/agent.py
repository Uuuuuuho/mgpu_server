"""
Node agent implementation for Multi-GPU Scheduler
"""

import socket
import threading
import json
import subprocess
import time
import os
import signal
import queue
import select
from typing import Dict, List, Optional, Any

from ..core.models import MessageType
from ..core.config import Config
from ..utils.logging import setup_logger
from ..utils.network import send_json_message, receive_json_message, connect_to_server
from ..utils.gpu import get_all_gpu_ids, setup_gpu_environment

logger = setup_logger(__name__)


class SimpleNode:
    """Simplified Node Agent for GPU Job Execution"""
    
    def __init__(self, node_id: str, master_host: str = '127.0.0.1', 
                 master_port: int = 8080, node_port: int = 8081):
        self.node_id = node_id
        self.master_host = master_host
        self.master_port = master_port
        self.node_port = node_port
        
        self.socket = None
        self.running = False
        
        # GPU management
        self.total_gpus = get_all_gpu_ids()
        self.available_gpus = self.total_gpus.copy()
        
        # Job management
        self.running_jobs = {}
        self.lock = threading.Lock()
        
        logger.info(f"Node {node_id} initialized with GPUs: {self.total_gpus}")
    
    def start_agent(self):
        """Start the node agent"""
        try:
            # Start server socket for receiving jobs from master
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.node_port))
            self.socket.listen(5)
            self.running = True
            
            logger.info(f"Node agent started on port {self.node_port}")
            print(f"Node agent '{self.node_id}' started on port {self.node_port}")
            
            # Register with master
            if not self._register_with_master():
                logger.error("Failed to register with master")
                return
            
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            
            # Accept connections from master
            while self.running:
                try:
                    client_socket, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_master_request,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Accept error: {e}")
                        
        except Exception as e:
            logger.error(f"Agent start error: {e}")
            print(f"Failed to start agent: {e}")
        finally:
            self.stop_agent()
    
    def stop_agent(self):
        """Stop the node agent"""
        self.running = False
        
        # Terminate all running jobs
        with self.lock:
            for job_id, job_info in self.running_jobs.items():
                try:
                    job_info['process'].terminate()
                except:
                    pass
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        logger.info("Node agent stopped")
    
    def _register_with_master(self) -> bool:
        """Register with master server"""
        try:
            sock = connect_to_server(self.master_host, self.master_port, 10.0)
            if not sock:
                return False
            
            request = {
                'cmd': MessageType.NODE_REGISTER,
                'node_id': self.node_id,
                'host': socket.gethostname(),
                'port': self.node_port,
                'total_gpus': self.total_gpus
            }
            
            if not send_json_message(sock, request, 10.0):
                sock.close()
                return False
            
            response = receive_json_message(sock, 10.0)
            sock.close()
            
            if response and response.get('status') == 'ok':
                logger.info("Successfully registered with master")
                return True
            else:
                logger.error(f"Registration failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to master"""
        while self.running:
            try:
                sock = connect_to_server(self.master_host, self.master_port, 5.0)
                if sock:
                    with self.lock:
                        available_gpus = self.available_gpus.copy()
                    
                    request = {
                        'cmd': MessageType.NODE_HEARTBEAT,
                        'node_id': self.node_id,
                        'available_gpus': available_gpus
                    }
                    
                    send_json_message(sock, request, 5.0)
                    receive_json_message(sock, 5.0)  # Read response
                    sock.close()
                
                time.sleep(10.0)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                time.sleep(30.0)  # Longer sleep on error
    
    def _handle_master_request(self, client_socket: socket.socket, addr):
        """Handle request from master"""
        try:
            request = receive_json_message(client_socket, 30.0)
            if not request:
                return
            
            cmd = request.get('cmd')
            if cmd == MessageType.RUN_JOB:
                self._handle_run_job(client_socket, request)
            elif cmd == MessageType.CANCEL_JOB:
                self._handle_cancel_job(client_socket, request)
            else:
                response = {'status': 'error', 'message': f'Unknown command: {cmd}'}
                send_json_message(client_socket, response, 10.0)
                
        except Exception as e:
            logger.error(f"Master request handling error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_run_job(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job execution request"""
        try:
            job_id = request.get('job_id')
            command = request.get('command')
            gpus = request.get('gpus', [])
            interactive = request.get('interactive', False)
            
            if not job_id or not command:
                response = {'status': 'error', 'message': 'job_id and command required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            # Allocate GPUs
            with self.lock:
                if not all(gpu in self.available_gpus for gpu in gpus):
                    response = {'status': 'error', 'message': 'Requested GPUs not available'}
                    send_json_message(client_socket, response, 10.0)
                    return
                
                # Remove allocated GPUs from available list
                for gpu in gpus:
                    self.available_gpus.remove(gpu)
            
            # Start job execution
            success = self._start_job(job_id, command, gpus, interactive)
            
            if success:
                response = {'status': 'ok', 'message': f'Job {job_id} started'}
                logger.info(f"Started job {job_id} with GPUs {gpus}")
            else:
                # Return GPUs if job start failed
                with self.lock:
                    self.available_gpus.extend(gpus)
                response = {'status': 'error', 'message': f'Failed to start job {job_id}'}
            
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Run job error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
    
    def _handle_cancel_job(self, client_socket: socket.socket, request: Dict[str, Any]):
        """Handle job cancellation request"""
        try:
            job_id = request.get('job_id')
            
            if not job_id:
                response = {'status': 'error', 'message': 'job_id required'}
                send_json_message(client_socket, response, 10.0)
                return
            
            success = False
            with self.lock:
                if job_id in self.running_jobs:
                    job_info = self.running_jobs[job_id]
                    try:
                        # Try graceful termination first
                        job_info['process'].terminate()
                        time.sleep(2.0)
                        
                        # Force kill if still running
                        if job_info['process'].poll() is None:
                            job_info['process'].kill()
                        
                        success = True
                        logger.info(f"Job {job_id} cancelled")
                    except Exception as e:
                        logger.error(f"Error cancelling job {job_id}: {e}")
            
            message = f"Job {job_id} {'cancelled' if success else 'not found or already completed'}"
            response = {'status': 'ok' if success else 'error', 'message': message}
            send_json_message(client_socket, response, 10.0)
            
        except Exception as e:
            logger.error(f"Cancel job error: {e}")
            response = {'status': 'error', 'message': str(e)}
            send_json_message(client_socket, response, 10.0)
    
    def _start_job(self, job_id: str, command: str, gpus: List[int], interactive: bool) -> bool:
        """Start executing a job"""
        try:
            # Setup environment
            env = os.environ.copy()
            env = setup_gpu_environment(env, gpus)
            
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=0,  # Unbuffered
                universal_newlines=True
            )
            
            # Store job info
            job_info = {
                'process': process,
                'gpus': gpus,
                'start_time': time.time(),
                'interactive': interactive,
                'output_lines': [],
                'last_output_sent': 0
            }
            
            with self.lock:
                self.running_jobs[job_id] = job_info
            
            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_job,
                args=(job_id, job_info),
                daemon=True
            )
            monitor_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {e}")
            return False
    
    def _monitor_job(self, job_id: str, job_info: Dict[str, Any]):
        """Monitor job execution and collect output"""
        process = job_info['process']
        output_lines = job_info['output_lines']
        
        try:
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    # Process has ended
                    break
                
                line = line.rstrip('\n\r')
                output_lines.append(line)
                
                # Send output to master for interactive jobs
                if job_info['interactive']:
                    self._send_output_to_master(job_id, [line])
                
                # For non-interactive jobs, batch output updates
                elif len(output_lines) - job_info['last_output_sent'] >= 10:
                    new_lines = output_lines[job_info['last_output_sent']:]
                    self._send_output_to_master(job_id, new_lines)
                    job_info['last_output_sent'] = len(output_lines)
            
            # Wait for process to complete
            exit_code = process.wait()
            
            # Send final status update
            final_output = output_lines[job_info['last_output_sent']:]
            self._send_job_completion(job_id, exit_code, final_output)
            
        except Exception as e:
            logger.error(f"Job monitoring error for {job_id}: {e}")
            self._send_job_completion(job_id, -1, [f"Monitoring error: {e}"])
        finally:
            # Clean up
            with self.lock:
                if job_id in self.running_jobs:
                    job_info = self.running_jobs[job_id]
                    # Return GPUs to available pool
                    self.available_gpus.extend(job_info['gpus'])
                    del self.running_jobs[job_id]
    
    def _send_output_to_master(self, job_id: str, output_lines: List[str]):
        """Send job output to master"""
        try:
            sock = connect_to_server(self.master_host, self.master_port, 5.0)
            if sock:
                request = {
                    'cmd': MessageType.JOB_UPDATE,
                    'job_id': job_id,
                    'status': 'running',
                    'output': output_lines
                }
                
                send_json_message(sock, request, 5.0)
                receive_json_message(sock, 5.0)  # Read response
                sock.close()
                
        except Exception as e:
            logger.debug(f"Failed to send output for job {job_id}: {e}")
    
    def _send_job_completion(self, job_id: str, exit_code: int, final_output: List[str]):
        """Send job completion status to master"""
        try:
            sock = connect_to_server(self.master_host, self.master_port, 10.0)
            if sock:
                status = 'completed' if exit_code == 0 else 'failed'
                
                request = {
                    'cmd': MessageType.JOB_UPDATE,
                    'job_id': job_id,
                    'status': status,
                    'exit_code': exit_code,
                    'output': final_output
                }
                
                send_json_message(sock, request, 10.0)
                receive_json_message(sock, 10.0)  # Read response
                sock.close()
                
                logger.info(f"Job {job_id} completed with exit code {exit_code}")
                
        except Exception as e:
            logger.error(f"Failed to send completion for job {job_id}: {e}")


class NodeManager:
    """Manager for running multiple node agents"""
    
    def __init__(self):
        self.nodes = {}
        self.running = False
    
    def add_node(self, node_id: str, master_host: str = '127.0.0.1', 
                 master_port: int = 8080, node_port: int = 8081):
        """Add a node agent"""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists")
            return False
        
        node = SimpleNode(node_id, master_host, master_port, node_port)
        self.nodes[node_id] = {
            'agent': node,
            'thread': None
        }
        
        logger.info(f"Added node {node_id}")
        return True
    
    def start_node(self, node_id: str) -> bool:
        """Start a specific node agent"""
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return False
        
        node_info = self.nodes[node_id]
        if node_info['thread'] and node_info['thread'].is_alive():
            logger.warning(f"Node {node_id} is already running")
            return False
        
        # Start node in separate thread
        thread = threading.Thread(
            target=node_info['agent'].start_agent,
            daemon=True
        )
        thread.start()
        node_info['thread'] = thread
        
        logger.info(f"Started node {node_id}")
        return True
    
    def start_all_nodes(self):
        """Start all node agents"""
        self.running = True
        for node_id in self.nodes:
            self.start_node(node_id)
    
    def stop_node(self, node_id: str):
        """Stop a specific node agent"""
        if node_id in self.nodes:
            self.nodes[node_id]['agent'].stop_agent()
            logger.info(f"Stopped node {node_id}")
    
    def stop_all_nodes(self):
        """Stop all node agents"""
        self.running = False
        for node_id in self.nodes:
            self.stop_node(node_id)
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all nodes"""
        status = {}
        for node_id, node_info in self.nodes.items():
            agent = node_info['agent']
            thread = node_info['thread']
            
            status[node_id] = {
                'running': thread and thread.is_alive(),
                'total_gpus': agent.total_gpus,
                'available_gpus': agent.available_gpus,
                'running_jobs': len(agent.running_jobs)
            }
        
        return status
