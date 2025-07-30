"""
Multi-GPU Master Server Main Class
"""

import socket
import threading
import json
import sys
import os
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu_core.models.job_models import MessageType
from mgpu_core.utils.logging_utils import setup_logger
from mgpu_server.job_scheduler import JobScheduler
from mgpu_server.node_manager import NodeManager


logger = setup_logger(__name__)


class MasterServer:
    """Main Master Server class handling all client connections"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        
        # Initialize managers
        self.job_scheduler = JobScheduler()
        self.node_manager = NodeManager()
        
        # Connect managers
        self.job_scheduler.set_nodes(self.node_manager.get_all_nodes())
    
    def handle_client(self, client_socket: socket.socket, address):
        """Handle client connection"""
        try:
            data = client_socket.recv(8192).decode()
            if not data:
                return
            
            request = json.loads(data)
            cmd = request.get('cmd')
            
            response = self.process_request(cmd, request)
            
            # Handle interactive sessions differently
            if cmd == MessageType.SUBMIT and request.get('interactive'):
                if response.get('status') == 'ok':
                    # Send initial response
                    client_socket.send(json.dumps(response).encode())
                    
                    # Register for interactive updates
                    job_id = response['job_id']
                    if job_id not in self.job_scheduler.interactive_clients:
                        self.job_scheduler.interactive_clients[job_id] = []
                    self.job_scheduler.interactive_clients[job_id].append(client_socket)
                    
                    # Keep connection alive for interactive session
                    self.handle_interactive_client(client_socket, job_id)
                    return
                else:
                    client_socket.send(json.dumps(response).encode())
            else:
                # Regular request-response
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
    
    def process_request(self, cmd: str, request: Dict) -> Dict:
        """Process different types of requests"""
        try:
            if cmd == MessageType.SUBMIT:
                return self.job_scheduler.submit_job(request)
            
            elif cmd == MessageType.QUEUE:
                return self.job_scheduler.get_queue_status()
            
            elif cmd == MessageType.CANCEL:
                job_id = request.get('job_id')
                return self.job_scheduler.cancel_job(job_id)
            
            elif cmd == MessageType.GET_JOB_OUTPUT:
                job_id = request.get('job_id')
                from_line = request.get('from_line', 0)
                return self.job_scheduler.get_job_output(job_id, from_line)
            
            elif cmd == MessageType.NODE_REGISTER:
                return self.node_manager.register_node(request)
            
            elif cmd == MessageType.NODE_STATUS:
                return self.node_manager.handle_node_status(request)
            
            elif cmd == MessageType.JOB_COMPLETE:
                return self.job_scheduler.handle_job_completion(request)
            
            elif cmd == MessageType.INTERACTIVE_COMPLETE:
                return self.handle_interactive_completion(request)
            
            elif cmd == MessageType.JOB_OUTPUT:
                return self.job_scheduler.handle_job_output(request)
            
            else:
                return {'status': 'error', 'message': f'Unknown command: {cmd}'}
                
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_interactive_completion(self, request: Dict) -> Dict:
        """Handle interactive job completion with improved cleanup"""
        job_id = request.get('job_id')
        exit_code = request.get('exit_code', 0)
        
        logger.info(f"Interactive job {job_id} completion requested with exit code {exit_code}")
        
        # Send completion to interactive clients
        if job_id in self.job_scheduler.interactive_clients:
            completion_msg = {
                'type': 'completion',
                'job_id': job_id,
                'exit_code': exit_code
            }
            
            dead_clients = []
            for client_socket in self.job_scheduler.interactive_clients[job_id]:
                try:
                    client_socket.send(json.dumps(completion_msg).encode() + b'\n')
                except:
                    dead_clients.append(client_socket)
            
            # Clean up all clients for this job
            for client in self.job_scheduler.interactive_clients[job_id]:
                try:
                    client.close()
                except:
                    pass
            
            # Remove the job from interactive clients
            del self.job_scheduler.interactive_clients[job_id]
            logger.info(f"Cleaned up interactive clients for job {job_id}")
        
        # Also handle regular job completion
        return self.job_scheduler.handle_job_completion(request)
    
    def handle_interactive_client(self, client_socket: socket.socket, job_id: str):
        """Handle interactive client connection with proper timeout and cleanup"""
        try:
            # Keep connection alive until job completes
            while True:
                try:
                    client_socket.settimeout(1.0)
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    # Could forward input to node here if needed
                except socket.timeout:
                    # Check if job is still running
                    if job_id not in self.job_scheduler.running_jobs:
                        break
                    continue
                except:
                    break
                    
        except Exception as e:
            logger.error(f"Interactive client handler error: {e}")
        finally:
            # Remove client from interactive clients list
            if job_id in self.job_scheduler.interactive_clients:
                if client_socket in self.job_scheduler.interactive_clients[job_id]:
                    self.job_scheduler.interactive_clients[job_id].remove(client_socket)
            
            try:
                client_socket.close()
            except:
                pass
    
    def start_server(self):
        """Start the master server"""
        self.running = True
        
        # Start job scheduler
        self.job_scheduler.start_scheduler()
        
        # Start server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        logger.info(f"Master Server started on {self.host}:{self.port}")
        logger.info(f"Job scheduler initialized")
        logger.info(f"Node manager initialized")
        logger.info(f"Server ready to accept connections")
        
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
            logger.info("Shutdown signal received")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the master server"""
        logger.info("Stopping master server...")
        self.running = False
        
        # Stop job scheduler
        self.job_scheduler.stop_scheduler()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        logger.info("Master server stopped")
