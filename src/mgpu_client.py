import socket
import json
import argparse
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Client:
    """Simplified Client for Job Submission"""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
    
    def submit_job(self, gpus, cmd, interactive=False, node_gpu_ids=None, timeout_config=None):
        """Submit a job"""
        # Set default timeout configuration
        if timeout_config is None:
            timeout_config = {
                'session_timeout': 7200,  # 2 hours
                'connection_timeout': 30,  # 30 seconds
                'max_wait_time': 300,  # 5 minutes
                'max_consecutive_timeouts': 30  # 30 consecutive timeouts
            }
        
        try:
            request = {
                'cmd': 'submit',
                'user': 'user',
                'command': cmd,
                'gpus': gpus,
                'interactive': interactive
            }
            
            if node_gpu_ids:
                request['node_gpu_ids'] = node_gpu_ids
            
            # Send request
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_config['connection_timeout'])
            sock.connect((self.host, self.port))
            sock.send(json.dumps(request).encode())
            
            if interactive:
                # Handle interactive session
                response_data = sock.recv(8192).decode()
                result = json.loads(response_data)
                
                if result.get('status') == 'ok':
                    job_id = result['job_id']
                    print(f"Job submission: {result}")
                    
                    # Start interactive session with improved timeout handling
                    print("Starting interactive session...")
                    print("=" * 50)
                    
                    session_start = time.time()
                    max_session_time = 7200  # 2 hours maximum (increased for GPU tests)
                    consecutive_timeouts = 0
                    max_consecutive_timeouts = 60  # 60 seconds of consecutive timeouts before giving up (increased for GPU tests)
                    
                    try:
                        while True:
                            # Check session timeout
                            if time.time() - session_start > max_session_time:
                                print("\nSession timed out after 2 hours")
                                break
                            
                            # Set timeout for receiving data
                            sock.settimeout(1.0)
                            try:
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
                                                sock.close()
                                                return
                                            elif msg.get('type') == 'error':
                                                print(f"ERROR: {msg.get('message')}")
                                                sock.close()
                                                return
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
                    finally:
                        try:
                            sock.close()
                        except:
                            pass
                else:
                    print(f"Job submission: {result}")
                    sock.close()
            else:
                # Non-interactive job - get response and monitor output
                response_data = sock.recv(8192).decode()
                result = json.loads(response_data)
                sock.close()
                
                if result.get('status') == 'ok':
                    job_id = result['job_id']
                    print(f"Job submitted: {job_id}")
                    
                    # Monitor job output until completion
                    self.monitor_job_output(job_id, timeout_config)
                else:
                    print(f"Job submission failed: {result}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def monitor_job_output(self, job_id, timeout_config=None):
        """Monitor non-interactive job output with timeout"""
        # Set default timeout configuration
        if timeout_config is None:
            timeout_config = {
                'max_wait_time': 300,  # 5 minutes
                'connection_timeout': 30  # 30 seconds
            }
        
        print(f"Monitoring job {job_id} output...")
        print("=" * 50)
        
        shown_lines = 0
        max_wait_time = timeout_config['max_wait_time']
        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5  # Allow 5 consecutive failures before giving up
        
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
                    'cmd': 'get_job_output',
                    'job_id': job_id,
                    'from_line': shown_lines
                }
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout_config.get('connection_timeout', 30))  # Use configurable timeout
                sock.connect((self.host, self.port))
                sock.send(json.dumps(request).encode())
                
                response_data = sock.recv(8192).decode()
                result = json.loads(response_data)
                sock.close()
                
                if result.get('status') == 'ok':
                    # Reset failure counter on successful response
                    consecutive_failures = 0
                    
                    # Show new output lines
                    output_lines = result.get('output', [])
                    new_lines = output_lines[shown_lines:]
                    
                    if new_lines:  # Only print if there are new lines
                        for line in new_lines:
                            print(line.rstrip())
                        shown_lines = len(output_lines)
                    
                    # Check if job is completed
                    job_status = result.get('job_status')
                    if job_status in ['completed', 'failed', 'cancelled']:
                        print("=" * 50)
                        print(f"Job completed with status: {job_status}")
                        if result.get('exit_code') is not None:
                            print(f"Exit code: {result.get('exit_code')}")
                        break
                    
                    # Check if job is unknown (might have been lost)
                    if job_status == 'unknown':
                        print("=" * 50)
                        print(f"Job {job_id} not found on server")
                        print("The job may have been cancelled or the server restarted")
                        break
                        
                else:
                    print(f"Error getting job output: {result.get('message', 'Unknown error')}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("=" * 50)
                        print(f"Too many consecutive failures ({consecutive_failures}), stopping monitoring")
                        break
                        
                time.sleep(2.0)  # Poll every 2 seconds (reduced from 1 second)
                
            except KeyboardInterrupt:
                print("\nOutput monitoring interrupted by user")
                break
            except socket.timeout:
                print("Request timed out, retrying...")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("=" * 50)
                    print("Too many timeouts, stopping monitoring")
                    break
                time.sleep(3.0)  # Wait longer after timeout
            except (ConnectionRefusedError, OSError) as e:
                print(f"Connection error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("=" * 50)
                    print("Cannot connect to server, stopping monitoring")
                    break
                time.sleep(5.0)  # Wait longer after connection error
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("=" * 50)
                    print(f"Unexpected error occurred, stopping monitoring: {e}")
                    break
                time.sleep(3.0)  # Wait after unexpected error
    
    def get_queue_status(self):
        """Get queue status"""
        try:
            request = {'cmd': 'queue'}
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.host, self.port))
            sock.send(json.dumps(request).encode())
            
            response_data = sock.recv(8192).decode()
            result = json.loads(response_data)
            sock.close()
            
            if result.get('status') == 'ok':
                queued = result.get('queue', [])
                running = result.get('running', [])
                nodes = result.get('nodes', {})
                
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
            else:
                print(f"Error: {result.get('message')}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def cancel_job(self, job_id):
        """Cancel a job"""
        try:
            request = {'cmd': 'cancel', 'job_id': job_id}
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((self.host, self.port))
            sock.send(json.dumps(request).encode())
            
            response_data = sock.recv(8192).decode()
            result = json.loads(response_data)
            sock.close()
            
            print(f"Cancel result: {result}")
            
        except Exception as e:
            print(f"Error: {e}")

def parse_node_gpu_ids(node_gpu_str):
    """Parse node-gpu mapping string like 'node1:0,1;node2:2,3'"""
    if not node_gpu_str:
        return None
    
    result = {}
    for mapping in node_gpu_str.split(';'):
        if ':' in mapping:
            node, gpus_str = mapping.split(':', 1)
            gpus = [int(g.strip()) for g in gpus_str.split(',')]
            result[node] = gpus
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Simple GPU Scheduler Client')
    parser.add_argument('--host', default='127.0.0.1', help='Master server host')
    parser.add_argument('--port', type=int, default=8080, help='Master server port')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a job')
    submit_parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs needed')
    submit_parser.add_argument('--interactive', action='store_true', help='Interactive session')
    submit_parser.add_argument('--node-gpu-ids', help='Specific node-GPU mapping (e.g., node1:0,1;node2:2)')
    submit_parser.add_argument('--session-timeout', type=int, default=7200, help='Session timeout in seconds (default: 7200 = 2 hours)')
    submit_parser.add_argument('--connection-timeout', type=int, default=30, help='Connection timeout in seconds (default: 30)')
    submit_parser.add_argument('--max-wait-time', type=int, default=300, help='Maximum wait time for job output in seconds (default: 300 = 5 minutes)')
    submit_parser.add_argument('--max-consecutive-timeouts', type=int, default=30, help='Maximum consecutive timeouts before giving up (default: 30)')
    submit_parser.add_argument('cmd', help='Command to execute')
    
    # Queue command
    queue_parser = subparsers.add_parser('queue', help='Show queue status')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='Job ID to cancel')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = Client(args.host, args.port)
    
    if args.command == 'submit':
        node_gpu_ids = parse_node_gpu_ids(args.node_gpu_ids)
        
        # Create timeout configuration
        timeout_config = {
            'session_timeout': args.session_timeout,
            'connection_timeout': args.connection_timeout,
            'max_wait_time': args.max_wait_time,
            'max_consecutive_timeouts': args.max_consecutive_timeouts
        }
        
        client.submit_job(args.gpus, args.cmd, args.interactive, node_gpu_ids, timeout_config)
        
    elif args.command == 'queue':
        client.get_queue_status()
        
    elif args.command == 'cancel':
        client.cancel_job(args.job_id)

if __name__ == "__main__":
    main()
