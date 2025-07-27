#!/usr/bin/env python3
"""
Simplified Node Agent for Multi-GPU Scheduler
Minimal command set: run, cancel, interactive, status
"""

import socket
import json
import threading
import time
import subprocess
import os
import signal
import select
from typing import Dict, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobProcess:
    """Running job process management"""
    def __init__(self, job_id: str, process: subprocess.Popen, gpus: List[int], interactive: bool = False):
        self.job_id = job_id
        self.process = process
        self.gpus = gpus
        self.interactive = interactive
        self.start_time = time.time()
        self.interactive_clients = []  # List of client sockets for interactive output

class SimpleNode:
    """Simplified Node Agent"""
    
    def __init__(self, node_id: str, host='0.0.0.0', port=8081, master_host='127.0.0.1', master_port=8080, gpu_count=1):
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
        
    def get_gpu_utilization(self, gpu_id: int) -> float:
        """Get GPU utilization (simplified - returns 0 if available)"""
        try:
            # For now, just check if GPU is in use
            return 0.0 if gpu_id in self.available_gpus else 100.0
        except:
            return 0.0
    
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
                # Always capture output to send to client via master
                process = subprocess.Popen(
                    command,
                    shell=True,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE if interactive else None,
                    bufsize=1,  # Line buffered
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
                    time.sleep(1.0)
                    if job_process.process.poll() is None:
                        job_process.process.kill()
                except:
                    pass
                
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
    
    def handle_interactive_request(self, request: Dict, client_socket: socket.socket) -> Dict:
        """Handle interactive session request"""
        job_id = request.get('job_id')
        
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        with self.lock:
            if job_id not in self.running_jobs:
                return {'status': 'error', 'message': f'Job {job_id} not found'}
            
            job_process = self.running_jobs[job_id]
            if not job_process.interactive:
                return {'status': 'error', 'message': f'Job {job_id} is not interactive'}
            
            # Add client to interactive clients list
            job_process.interactive_clients.append(client_socket)
            
            # Start interactive forwarding
            self.handle_interactive_forwarding(job_process, client_socket)
            
            return {'status': 'ok', 'message': 'Interactive session started'}
    
    def handle_interactive_forwarding(self, job_process: JobProcess, client_socket: socket.socket):
        """Handle interactive I/O forwarding"""
        try:
            process = job_process.process
            
            def forward_output():
                """Forward process output to client"""
                try:
                    while process.poll() is None:
                        # Check if stdout is available
                        if process.stdout is None:
                            time.sleep(0.1)
                            continue
                            
                        # Use select to check for available output
                        ready, _, _ = select.select([process.stdout], [], [], 0.1)
                        
                        if ready:
                            line = process.stdout.readline()
                            if line:
                                output_msg = {
                                    'type': 'output',
                                    'job_id': job_process.job_id,
                                    'data': line
                                }
                                try:
                                    client_socket.send((json.dumps(output_msg) + '\n').encode())
                                except:
                                    break
                    
                    # Send completion message
                    completion_msg = {
                        'type': 'completion',
                        'job_id': job_process.job_id,
                        'exit_code': process.returncode
                    }
                    try:
                        client_socket.send((json.dumps(completion_msg) + '\n').encode())
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Output forwarding error: {e}")
            
            def forward_input():
                """Forward client input to process"""
                try:
                    while process.poll() is None:
                        ready, _, _ = select.select([client_socket], [], [], 0.1)
                        if ready:
                            data = client_socket.recv(1024)
                            if not data:
                                break
                            if process.stdin:
                                process.stdin.write(data.decode())
                                process.stdin.flush()
                except Exception as e:
                    logger.debug(f"Input forwarding error: {e}")
            
            # Start forwarding threads
            output_thread = threading.Thread(target=forward_output)
            input_thread = threading.Thread(target=forward_input)
            
            output_thread.daemon = True
            input_thread.daemon = True
            
            output_thread.start()
            input_thread.start()
            
            # Wait for process to complete
            output_thread.join()
            
        except Exception as e:
            logger.error(f"Interactive forwarding error: {e}")
        finally:
            # Remove client from list
            with self.lock:
                if client_socket in job_process.interactive_clients:
                    job_process.interactive_clients.remove(client_socket)
    
    def handle_status_request(self, request: Dict) -> Dict:
        """Handle status request"""
        try:
            with self.lock:
                status = {
                    'node_id': self.node_id,
                    'available_gpus': self.available_gpus.copy(),
                    'running_jobs': list(self.running_jobs.keys()),
                    'gpu_utilization': {
                        str(i): self.get_gpu_utilization(i) for i in range(self.gpu_count)
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
                    line = process.stdout.readline()
                    if line:
                        # Send output to master server for client forwarding
                        try:
                            master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            master_sock.settimeout(2.0)
                            master_sock.connect((self.master_host, self.master_port))
                            
                            output_msg = {
                                'cmd': 'job_output',
                                'job_id': job_id,
                                'data': line,
                                'node_id': self.node_id,
                                'interactive': job_process.interactive
                            }
                            
                            master_sock.send(json.dumps(output_msg).encode())
                            master_sock.recv(1024)  # Receive response
                            master_sock.close()
                            
                        except Exception as e:
                            logger.debug(f"Failed to send job output: {e}")
                            
                time.sleep(0.05)  # Small delay to prevent flooding
            
            # Get final exit code
            exit_code = process.wait()
            
            # Send completion notification
            try:
                master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                master_sock.settimeout(10.0)
                master_sock.connect((self.master_host, self.master_port))
                
                completion_msg = {
                    'cmd': 'job_complete',
                    'job_id': job_id,
                    'exit_code': exit_code,
                    'node_id': self.node_id,
                    'interactive': job_process.interactive
                }
                
                master_sock.send(json.dumps(completion_msg).encode())
                master_sock.recv(1024)
                master_sock.close()
                
            except Exception as e:
                logger.error(f"Failed to notify job completion: {e}")
            
            # Clean up job
            with self.lock:
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    
                    # Restore GPUs
                    for gpu in job_process.gpus:
                        if gpu not in self.available_gpus:
                            self.available_gpus.append(gpu)
                    
                    # Close interactive clients
                    for client in job_process.interactive_clients:
                        try:
                            client.close()
                        except:
                            pass
                    
                    del self.running_jobs[job_id]
            
            logger.info(f"Job {job_id} completed with exit code {exit_code}")
            
        except Exception as e:
            logger.error(f"Job monitoring error for {job_id}: {e}")

    def monitor_job(self, job_id: str):
        """Monitor job completion"""
        try:
            if job_id not in self.running_jobs:
                return
            
            job_process = self.running_jobs[job_id]
            process = job_process.process
            
            # Wait for process to complete
            exit_code = process.wait()
            
            # Notify master server
            try:
                master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                master_sock.settimeout(10.0)
                master_sock.connect((self.master_host, self.master_port))
                
                completion_msg = {
                    'cmd': 'job_complete',
                    'job_id': job_id,
                    'exit_code': exit_code,
                    'node_id': self.node_id
                }
                
                master_sock.send(json.dumps(completion_msg).encode())
                master_sock.recv(1024)  # Receive response
                master_sock.close()
                
            except Exception as e:
                logger.error(f"Failed to notify master of job completion: {e}")
            
            # Clean up job
            with self.lock:
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    
                    # Restore GPUs
                    for gpu in job_process.gpus:
                        if gpu not in self.available_gpus:
                            self.available_gpus.append(gpu)
                    
                    # Close interactive clients
                    for client in job_process.interactive_clients:
                        try:
                            client.close()
                        except:
                            pass
                    
                    del self.running_jobs[job_id]
            
            logger.info(f"Job {job_id} completed with exit code {exit_code}")
            
        except Exception as e:
            logger.error(f"Job monitoring error for {job_id}: {e}")
    
    def monitor_interactive_job(self, job_id: str):
        """Monitor interactive job with real-time output streaming"""
        try:
            if job_id not in self.running_jobs:
                return
            
            job_process = self.running_jobs[job_id]
            process = job_process.process
            
            # Stream output in real-time
            while process.poll() is None:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        # Send output to master server for interactive clients
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
                            logger.debug(f"Failed to send interactive output: {e}")
                            
                time.sleep(0.1)  # Small delay to prevent flooding
            
            # Get final exit code
            exit_code = process.wait()
            
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
                logger.error(f"Failed to notify interactive completion: {e}")
            
            # Clean up job (same as regular monitor_job)
            with self.lock:
                if job_id in self.running_jobs:
                    job_process = self.running_jobs[job_id]
                    
                    # Restore GPUs
                    for gpu in job_process.gpus:
                        if gpu not in self.available_gpus:
                            self.available_gpus.append(gpu)
                    
                    # Close interactive clients
                    for client in job_process.interactive_clients:
                        try:
                            client.close()
                        except:
                            pass
                    
                    del self.running_jobs[job_id]
            
            logger.info(f"Interactive job {job_id} completed with exit code {exit_code}")
            
        except Exception as e:
            logger.error(f"Interactive job monitoring error for {job_id}: {e}")
    
    def send_heartbeat(self):
        """Send periodic heartbeat to master"""
        while self.running:
            try:
                time.sleep(10.0)  # Send heartbeat every 10 seconds
                
                with self.lock:
                    status_data = {
                        'node_id': self.node_id,
                        'available_gpus': self.available_gpus.copy(),
                        'running_jobs': list(self.running_jobs.keys())
                    }
                
                # Send to master
                try:
                    master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    master_sock.settimeout(5.0)
                    master_sock.connect((self.master_host, self.master_port))
                    
                    heartbeat_msg = {
                        'cmd': 'node_status',
                        **status_data
                    }
                    
                    master_sock.send(json.dumps(heartbeat_msg).encode())
                    master_sock.recv(1024)  # Receive response
                    master_sock.close()
                    
                except Exception as e:
                    logger.debug(f"Heartbeat failed: {e}")
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            data = client_socket.recv(8192).decode()
            if not data:
                return
            
            request = json.loads(data)
            cmd = request.get('cmd')
            
            if cmd == 'run':
                response = self.handle_run_job(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'cancel':
                response = self.handle_cancel_job(request)
                client_socket.send(json.dumps(response).encode())
                
            elif cmd == 'interactive':
                # Don't send immediate response for interactive - handle forwarding
                self.handle_interactive_request(request, client_socket)
                return  # Don't close socket
                
            elif cmd == 'status':
                response = self.handle_status_request(request)
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
    
    def start(self):
        """Start the node agent"""
        self.running = True
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # Start server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        logger.info(f"Simple Node Agent {self.node_id} started on {self.host}:{self.port}")
        logger.info(f"Master server: {self.master_host}:{self.master_port}")
        logger.info(f"Available GPUs: {self.available_gpus}")
        
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
            logger.info("Node agent shutdown requested")
        finally:
            self.running = False
            server_socket.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple Node Agent')
    parser.add_argument('--node-id', required=True, help='Node ID')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to bind to')
    parser.add_argument('--master-host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--master-port', type=int, default=8080, help='Master server port')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs on this node')
    
    args = parser.parse_args()
    
    node = SimpleNode(
        args.node_id,
        args.host,
        args.port,
        args.master_host,
        args.master_port,
        args.gpu_count
    )
    
    try:
        node.start()
    except KeyboardInterrupt:
        logger.info("Node agent stopped")

if __name__ == "__main__":
    main()
