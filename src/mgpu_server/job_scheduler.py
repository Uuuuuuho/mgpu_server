"""
Job Scheduler for Multi-GPU Master Server
"""

import json
import queue
import threading
import time
import uuid
import sys
import os
from typing import Dict, List, Optional, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu_core.models.job_models import SimpleJob, NodeInfo
from mgpu_core.network.network_manager import NetworkManager
from mgpu_core.utils.logging_utils import setup_logger


logger = setup_logger(__name__)


class JobScheduler:
    """Handles job scheduling and queue management"""
    
    def __init__(self):
        self.job_queue = queue.Queue()
        self.running_jobs = {}  # job_id -> SimpleJob
        self.completed_jobs = {}  # job_id -> SimpleJob
        self.job_outputs = {}  # job_id -> List[str]
        self.interactive_clients = {}  # job_id -> List[socket]
        self.lock = threading.RLock()
        self.running = False
        self.nodes = {}  # Will be set by master
    
    def set_nodes(self, nodes: Dict[str, NodeInfo]):
        """Set nodes reference from master"""
        self.nodes = nodes
    
    def submit_job(self, request: Dict) -> Dict:
        """Handle job submission"""
        try:
            logger.info(f"Received submit request: {request}")
            
            # Create job
            job = SimpleJob(
                id=request.get('job_id', str(uuid.uuid4())[:8].upper()),
                user=request.get('user', 'unknown'),
                cmd=request.get('command', ''),
                gpus_needed=request.get('gpus', 1),
                node_gpu_ids=request.get('node_gpu_ids'),
                priority=request.get('priority', 0),
                interactive=request.get('interactive', False)
            )
            
            logger.info(f"Created job {job.id} with node_gpu_ids: {job.node_gpu_ids}")
            
            # Add to queue
            self.job_queue.put(job)
            logger.info(f"Job {job.id} submitted: {job.cmd[:50]}...")
            
            return {'status': 'ok', 'job_id': job.id, 'message': 'Job submitted'}
            
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        try:
            with self.lock:
                queued_jobs = []
                temp_queue = queue.Queue()
                
                # Extract jobs from queue (temporarily)
                while not self.job_queue.empty():
                    try:
                        job = self.job_queue.get_nowait()
                        queued_jobs.append(job.to_dict())
                        temp_queue.put(job)
                    except queue.Empty:
                        break
                
                # Put jobs back in queue
                while not temp_queue.empty():
                    self.job_queue.put(temp_queue.get())
                
                running_jobs = [job.to_dict() for job in self.running_jobs.values()]
                
                # Node status
                nodes_status = {}
                for node_id, node in self.nodes.items():
                    nodes_status[node_id] = {
                        'available_gpus': node.available_gpus.copy(),
                        'running_jobs': len(node.running_jobs),
                        'total_gpus': node.gpu_count,
                        'failure_count': getattr(node, 'failure_count', 0)
                    }
                
                return {
                    'status': 'ok',
                    'queue': queued_jobs,
                    'running': running_jobs,
                    'nodes': nodes_status
                }
                
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a job"""
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            with self.lock:
                # Check if job is running
                if job_id in self.running_jobs:
                    job = self.running_jobs[job_id]
                    
                    # Send cancel request to node
                    if job.assigned_node and job.assigned_node in self.nodes:
                        node = self.nodes[job.assigned_node]
                        cancel_msg = {'cmd': 'cancel', 'job_id': job_id}
                        response = NetworkManager.send_to_node(node, cancel_msg)
                        
                        if response and response.get('status') == 'ok':
                            # Move to completed jobs
                            job.status = 'cancelled'
                            job.end_time = time.time()
                            self.completed_jobs[job_id] = job
                            del self.running_jobs[job_id]
                            
                            # Free up node resources
                            if job.assigned_gpus:
                                for gpu in job.assigned_gpus:
                                    if gpu not in node.available_gpus:
                                        node.available_gpus.append(gpu)
                            
                            return {'status': 'ok', 'message': f'Job {job_id} cancelled'}
                        else:
                            return {'status': 'error', 'message': 'Failed to cancel job on node'}
                    else:
                        return {'status': 'error', 'message': 'Job node not found'}
                
                # Check if job is in queue
                temp_queue = queue.Queue()
                found = False
                
                while not self.job_queue.empty():
                    try:
                        job = self.job_queue.get_nowait()
                        if job.id == job_id:
                            found = True
                            job.status = 'cancelled'
                            self.completed_jobs[job_id] = job
                        else:
                            temp_queue.put(job)
                    except queue.Empty:
                        break
                
                # Put remaining jobs back
                while not temp_queue.empty():
                    self.job_queue.put(temp_queue.get())
                
                if found:
                    return {'status': 'ok', 'message': f'Job {job_id} cancelled from queue'}
                else:
                    return {'status': 'error', 'message': f'Job {job_id} not found'}
                    
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def find_available_node(self, job: SimpleJob) -> Optional[str]:
        """Find available node for job"""
        logger.info(f"Finding node for job {job.id}, node_gpu_ids: {job.node_gpu_ids}")
        
        # If specific node-gpu mapping is requested
        if job.node_gpu_ids:
            for node_id, requested_gpus in job.node_gpu_ids.items():
                if node_id not in self.nodes:
                    logger.warning(f"Requested node {node_id} not found")
                    continue
                
                node = self.nodes[node_id]
                if all(gpu in node.available_gpus for gpu in requested_gpus):
                    logger.info(f"Node {node_id} selected for job {job.id}")
                    return node_id
                else:
                    logger.warning(f"Node {node_id} doesn't have required GPUs {requested_gpus}")
        
        # Find any node with enough GPUs
        logger.info(f"Auto-selecting node with {job.gpus_needed} GPUs")
        for node_id, node in self.nodes.items():
            if len(node.available_gpus) >= job.gpus_needed:
                logger.info(f"Node {node_id} selected for job {job.id}")
                return node_id
        
        logger.info(f"No available node found")
        return None
    
    def schedule_jobs(self):
        """Job scheduler thread"""
        while self.running:
            try:
                job = self.job_queue.get(timeout=1.0)
                
                # Find available node
                node_id = self.find_available_node(job)
                if not node_id:
                    # Put job back in queue
                    self.job_queue.put(job)
                    time.sleep(1.0)
                    continue
                
                # Assign job to node
                node = self.nodes[node_id]
                
                # Determine GPUs to assign
                if job.node_gpu_ids and node_id in job.node_gpu_ids:
                    assigned_gpus = job.node_gpu_ids[node_id]
                else:
                    assigned_gpus = node.available_gpus[:job.gpus_needed]
                
                # Reserve GPUs
                for gpu in assigned_gpus:
                    if gpu in node.available_gpus:
                        node.available_gpus.remove(gpu)
                
                # Update job
                job.assigned_node = node_id
                job.assigned_gpus = assigned_gpus
                job.status = 'running'
                job.start_time = time.time()
                
                # Create debug command
                debug_cmd = self.create_debug_command(job.cmd, node_id, job.id)
                
                # Send to node
                run_request = {
                    'cmd': 'run',
                    'job_id': job.id,
                    'command': debug_cmd,
                    'gpus': assigned_gpus,
                    'interactive': job.interactive
                }
                
                response = NetworkManager.send_to_node(node, run_request)
                
                if response and response.get('status') == 'ok':
                    # Move to running jobs
                    with self.lock:
                        self.running_jobs[job.id] = job
                        node.running_jobs.append(job.id)
                    
                    logger.info(f"Job {job.id} started on node {node_id} with GPUs {assigned_gpus}")
                else:
                    # Restore GPUs on failure
                    for gpu in assigned_gpus:
                        if gpu not in node.available_gpus:
                            node.available_gpus.append(gpu)
                    
                    logger.error(f"Failed to start job {job.id} on node {node_id}")
                    
                    # Retry job
                    job.retry_count += 1
                    if job.retry_count < 3:
                        self.job_queue.put(job)
                    else:
                        job.status = 'failed'
                        with self.lock:
                            self.completed_jobs[job.id] = job
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def create_debug_command(self, original_cmd: str, node_id: str, job_id: str) -> str:
        """Create command with debug information to track actual execution location"""
        debug_prefix = f'''echo "=== JOB EXECUTION DEBUG INFO ==="
echo "Job ID: {job_id}"
echo "Target Node ID: {node_id}"
echo "Actual Hostname: $(hostname)"
echo "Actual IP: $(hostname -I | cut -d' ' -f1 || echo 'N/A')"
echo "Process ID: $$"
echo "Working Directory: $(pwd)"
echo "Timestamp: $(date)"
echo "Port Check: $(netstat -tlnp 2>/dev/null | grep :808 | head -3 || echo 'No 808x ports')"
echo "=============================="
'''
        return f"{debug_prefix}\n{original_cmd}"
    
    def handle_job_completion(self, request: Dict) -> Dict:
        """Handle job completion from node"""
        job_id = request.get('job_id')
        exit_code = request.get('exit_code', 0)
        node_id = request.get('node_id')
        
        if job_id not in self.running_jobs:
            return {'status': 'error', 'message': 'Job not found'}
        
        try:
            with self.lock:
                job = self.running_jobs[job_id]
                job.status = 'completed' if exit_code == 0 else 'failed'
                job.exit_code = exit_code
                job.end_time = time.time()
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.running_jobs[job_id]
                
                # Free node resources
                if job.assigned_node and job.assigned_node in self.nodes:
                    node = self.nodes[job.assigned_node]
                    if job.id in node.running_jobs:
                        node.running_jobs.remove(job.id)
                    
                    if job.assigned_gpus:
                        for gpu in job.assigned_gpus:
                            if gpu not in node.available_gpus:
                                node.available_gpus.append(gpu)
                
                logger.info(f"Job {job_id} completed with exit code {exit_code}")
                return {'status': 'ok', 'message': 'Job completion processed'}
                
        except Exception as e:
            logger.error(f"Job completion error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_job_output(self, job_id: str, from_line: int = 0) -> Dict:
        """Get job output for non-interactive jobs"""
        if not job_id:
            return {'status': 'error', 'message': 'job_id required'}
        
        try:
            with self.lock:
                # Check running jobs
                if job_id in self.running_jobs:
                    job = self.running_jobs[job_id]
                    output = self.job_outputs.get(job_id, [])
                    return {
                        'status': 'ok',
                        'job_status': job.status,
                        'output': output,
                        'exit_code': job.exit_code
                    }
                
                # Check completed jobs
                if job_id in self.completed_jobs:
                    job = self.completed_jobs[job_id]
                    output = self.job_outputs.get(job_id, [])
                    return {
                        'status': 'ok',
                        'job_status': job.status,
                        'output': output,
                        'exit_code': job.exit_code
                    }
                
                return {
                    'status': 'ok',
                    'job_status': 'unknown',
                    'output': [],
                    'exit_code': None
                }
                
        except Exception as e:
            logger.error(f"Get job output error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def handle_job_output(self, request: Dict) -> Dict:
        """Handle job output from nodes"""
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
                dead_clients = []
                for client_socket in self.interactive_clients[job_id]:
                    try:
                        message = {'type': 'output', 'data': data}
                        client_socket.send(json.dumps(message).encode() + b'\n')
                    except:
                        dead_clients.append(client_socket)
                
                # Remove dead clients
                for client in dead_clients:
                    self.interactive_clients[job_id].remove(client)
            
            return {'status': 'ok', 'message': 'Output received'}
            
        except Exception as e:
            logger.error(f"Job output error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_scheduler(self):
        """Start the job scheduler"""
        self.running = True
        scheduler_thread = threading.Thread(target=self.schedule_jobs)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Job scheduler started")
    
    def stop_scheduler(self):
        """Stop the job scheduler"""
        self.running = False
        logger.info("Job scheduler stopped")
