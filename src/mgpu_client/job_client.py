"""
Multi-GPU Client for job submission and monitoring
"""

import socket
import json
import time
import sys
import os
from typing import Dict, List, Optional, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu_core.models.job_models import MessageType
from mgpu_core.network.network_manager import NetworkManager
from mgpu_core.utils.logging_utils import setup_logger
from mgpu_core.utils.system_utils import TimeoutConfig


logger = setup_logger(__name__)


class JobClient:
    """Client for submitting and monitoring jobs"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8080):
        self.host = host
        self.port = port
    
    def submit_job(self, gpus: int, cmd: str, interactive: bool = False, 
                   node_gpu_ids: Optional[Dict[str, List[int]]] = None, 
                   timeout_config: Optional[Dict[str, Any]] = None) -> bool:
        """Submit a job to the scheduler"""
        if timeout_config is None:
            timeout_config = TimeoutConfig.get_default_config()
        
        try:
            request = {
                'cmd': MessageType.SUBMIT,
                'user': 'user',
                'command': cmd,
                'gpus': gpus,
                'interactive': interactive
            }
            
            if node_gpu_ids:
                request['node_gpu_ids'] = node_gpu_ids
            
            # Connect to server
            sock = NetworkManager.connect_to_server(self.host, self.port, timeout_config['connection_timeout'])
            if not sock:
                print(f"Failed to connect to server at {self.host}:{self.port}")
                return False
            
            # Send request
            if not NetworkManager.send_json_message(sock, request, timeout_config['connection_timeout']):
                print("Failed to send job submission request")
                sock.close()
                return False
            
            if interactive:
                return self._handle_interactive_session(sock, timeout_config)
            else:
                return self._handle_non_interactive_session(sock, timeout_config)
                
        except Exception as e:
            logger.error(f"Job submission error: {e}")
            print(f"Error: {e}")
            return False
    
    def _handle_interactive_session(self, sock: socket.socket, timeout_config: Dict[str, Any]) -> bool:
        """Handle interactive session"""
        try:
            # Get initial response
            response = NetworkManager.receive_json_message(sock, timeout_config['connection_timeout'])
            if not response:
                print("No response from server")
                return False
            
            if response.get('status') != 'ok':
                print(f"Job submission failed: {response}")
                return False
            
            job_id = response['job_id']
            print(f"Job submission: {response}")
            print("Starting interactive session...")
            print("=" * 50)
            
            return self._monitor_interactive_output(sock, job_id, timeout_config)
            
        finally:
            try:
                sock.close()
            except:
                pass
    
    def _handle_non_interactive_session(self, sock: socket.socket, timeout_config: Dict[str, Any]) -> bool:
        """Handle non-interactive session"""
        try:
            response = NetworkManager.receive_json_message(sock, timeout_config['connection_timeout'])
            if not response:
                print("No response from server")
                return False
            
            sock.close()
            
            if response.get('status') == 'ok':
                job_id = response['job_id']
                print(f"Job submitted: {job_id}")
                return self.monitor_job_output(job_id, timeout_config)
            else:
                print(f"Job submission failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Non-interactive session error: {e}")
            return False
    
    def _monitor_interactive_output(self, sock: socket.socket, job_id: str, timeout_config: Dict[str, Any]) -> bool:
        """Monitor interactive job output"""
        session_start = time.time()
        max_session_time = timeout_config['session_timeout']
        consecutive_timeouts = 0
        max_consecutive_timeouts = timeout_config['max_consecutive_timeouts']
        
        try:
            while True:
                # Check session timeout
                if time.time() - session_start > max_session_time:
                    print(f"\nSession timed out after {max_session_time} seconds")
                    break
                
                # Set timeout for receiving data
                try:
                    sock.settimeout(1.0)
                    data = sock.recv(8192)
                    if not data:
                        print("\nConnection closed by server")
                        break
                    
                    # Reset timeout counter on successful data receive
                    consecutive_timeouts = 0
                    
                    # Process each line separately
                    lines = data.decode().strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                msg = json.loads(line)
                                if msg.get('type') == 'output':
                                    print(msg.get('data', '').rstrip())
                                elif msg.get('type') == 'completion':
                                    print("=" * 50)
                                    print(f"Job completed with exit code: {msg.get('exit_code')}")
                                    return True
                                elif msg.get('type') == 'error':
                                    print(f"ERROR: {msg.get('message')}")
                                    return False
                                else:
                                    print(f"Response: {msg}")
                            except json.JSONDecodeError:
                                print(f"Invalid JSON: {line}")
                                        
                except socket.timeout:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print(f"\nNo response from server for {max_consecutive_timeouts} seconds, ending session")
                        break
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    print("\nConnection lost")
                    break
                            
        except KeyboardInterrupt:
            print("\nInteractive session interrupted by user")
        
        return False
    
    def monitor_job_output(self, job_id: str, timeout_config: Optional[Dict[str, Any]] = None) -> bool:
        """Monitor non-interactive job output"""
        if timeout_config is None:
            timeout_config = TimeoutConfig.get_default_config()
        
        print(f"Monitoring job {job_id} output...")
        print("=" * 50)
        
        shown_lines = 0
        max_wait_time = timeout_config['max_wait_time']
        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            try:
                current_time = time.time()
                
                # Check if we've exceeded maximum wait time
                if current_time - start_time > max_wait_time:
                    print("=" * 50)
                    print(f"Job monitoring timed out after {max_wait_time} seconds")
                    print("The job may still be running on the server.")
                    break
                
                # Request job status and output
                request = {
                    'cmd': MessageType.GET_JOB_OUTPUT,
                    'job_id': job_id,
                    'from_line': shown_lines
                }
                
                sock = NetworkManager.connect_to_server(self.host, self.port, timeout_config['connection_timeout'])
                if not sock:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("Cannot connect to server, stopping monitoring")
                        break
                    time.sleep(5.0)
                    continue
                
                if not NetworkManager.send_json_message(sock, request, timeout_config['connection_timeout']):
                    sock.close()
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("Failed to send request, stopping monitoring")
                        break
                    time.sleep(3.0)
                    continue
                
                response = NetworkManager.receive_json_message(sock, timeout_config['connection_timeout'])
                sock.close()
                
                if response and response.get('status') == 'ok':
                    # Reset failure counter on successful response
                    consecutive_failures = 0
                    
                    # Show new output lines
                    output_lines = response.get('output', [])
                    new_lines = output_lines[shown_lines:]
                    
                    if new_lines:
                        for line in new_lines:
                            print(line.rstrip())
                        shown_lines = len(output_lines)
                    
                    # Check if job is completed
                    job_status = response.get('job_status')
                    if job_status in ['completed', 'failed', 'cancelled']:
                        print("=" * 50)
                        print(f"Job completed with status: {job_status}")
                        if response.get('exit_code') is not None:
                            print(f"Exit code: {response.get('exit_code')}")
                        return job_status == 'completed'
                    
                    # Check if job is unknown
                    if job_status == 'unknown':
                        print("=" * 50)
                        print(f"Job {job_id} not found on server")
                        return False
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive failures, stopping monitoring")
                        break
                        
                time.sleep(2.0)
                
            except KeyboardInterrupt:
                print("\nOutput monitoring interrupted by user")
                break
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Unexpected error occurred, stopping monitoring: {e}")
                    break
                time.sleep(3.0)
        
        return False
    
    def get_queue_status(self) -> bool:
        """Get queue status"""
        try:
            request = {'cmd': MessageType.QUEUE}
            
            sock = NetworkManager.connect_to_server(self.host, self.port, 10.0)
            if not sock:
                print(f"Failed to connect to server at {self.host}:{self.port}")
                return False
            
            if not NetworkManager.send_json_message(sock, request, 10.0):
                print("Failed to send queue request")
                sock.close()
                return False
            
            response = NetworkManager.receive_json_message(sock, 10.0)
            sock.close()
            
            if response and response.get('status') == 'ok':
                queued = response.get('queue', [])
                running = response.get('running', [])
                nodes = response.get('nodes', {})
                
                print("=== Queue Status ===")
                print(f"Queued jobs: {len(queued)}")
                for job in queued:
                    print(f"  {job['id']}: {job['cmd'][:50]}...")
                
                print(f"Running jobs: {len(running)}")
                for job in running:
                    print(f"  {job['id']}: {job['cmd'][:50]}... (Node: {job.get('assigned_node')})")
                
                print("Node status:")
                for node_id, node_info in nodes.items():
                    print(f"  {node_id}: GPUs {node_info['available_gpus']} available")
                return True
            else:
                print(f"Error: {response.get('message') if response else 'No response'}")
                return False
                
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            print(f"Error: {e}")
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            request = {'cmd': MessageType.CANCEL, 'job_id': job_id}
            
            sock = NetworkManager.connect_to_server(self.host, self.port, 10.0)
            if not sock:
                print(f"Failed to connect to server at {self.host}:{self.port}")
                return False
            
            if not NetworkManager.send_json_message(sock, request, 10.0):
                print("Failed to send cancel request")
                sock.close()
                return False
            
            response = NetworkManager.receive_json_message(sock, 10.0)
            sock.close()
            
            print(f"Cancel result: {response}")
            return response and response.get('status') == 'ok'
            
        except Exception as e:
            logger.error(f"Cancel job error: {e}")
            print(f"Error: {e}")
            return False
