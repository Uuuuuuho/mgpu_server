#!/usr/bin/env python3
import os
import socket
import threading
import subprocess
import uuid
import json
import time
import random
import string
import argparse
from collections import deque
import psutil
import select

SOCKET_PATH = '/tmp/mgpu_scheduler.sock'
MAX_JOB_TIME = 600  # Maximum occupation time (seconds), can be passed as argument in main if needed

class Job:
    def __init__(self, user, gpus, mem, cmd, time_limit=None, priority=0, gpu_ids=None, env_setup_cmd=None, client_socket=None):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.user = user
        self.gpus = gpus
        self.mem = mem  # If None, server will auto-allocate
        self.cmd = cmd
        self.status = 'queued'
        self.proc = None
        self.start_time = None  # Job execution start time
        self.time_limit = time_limit  # Per-user time limit (seconds)
        self.priority = priority
        self.gpu_ids = gpu_ids  # User-requested specific GPU IDs
        self.env_setup_cmd = env_setup_cmd  # User-requested environment setup command
        self.client_socket = client_socket  # Socket to stream output back to user

    def to_dict(self):
        return {
            'id': self.id,
            'user': self.user,
            'gpus': self.gpus,
            'mem': self.mem,
            'cmd': self.cmd,
            'status': self.status,
            'gpu_ids': self.gpu_ids,
            'env_setup_cmd': self.env_setup_cmd
        }

def get_available_gpus():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        mems = [int(x) for x in out.decode().strip().split('\n')]
        return mems
    except Exception:
        return []

class Scheduler:
    def __init__(self):
        self.job_queue = deque()
        self.running_jobs = {}
        self.lock = threading.Lock()

    def submit_job(self, job):
        with self.lock:
            self.job_queue.append(job)
            return job.id

    def _kill_proc_tree(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        except Exception:
            pass

    def cancel_job(self, job_id):
        with self.lock:
            # 큐에서 먼저 제거
            for job in list(self.job_queue):
                if job.id == job_id:
                    self.job_queue.remove(job)
                    # 혹시 실행 중인 job이 있으면 프로세스 트리 전체 kill
                    if job_id in self.running_jobs:
                        proc = self.running_jobs[job_id].proc
                        if proc:
                            try:
                                self._kill_proc_tree(proc.pid)
                            except Exception as e:
                                print(f"[DEBUG] Failed to kill proc tree: {e}")
                        del self.running_jobs[job_id]
                    return True
            # 큐에 없고 실행 중인 경우
            if job_id in self.running_jobs:
                proc = self.running_jobs[job_id].proc
                if proc:
                    try:
                        self._kill_proc_tree(proc.pid)
                    except Exception as e:
                        print(f"[DEBUG] Failed to kill proc tree: {e}")
                del self.running_jobs[job_id]
                return True
        return False

    def get_queue(self):
        with self.lock:
            return [job.to_dict() if getattr(job, 'status', '') != 'error' else {**job.to_dict(), 'error_msg': getattr(job, 'error_msg', '')} for job in self.job_queue]

    def get_running(self):
        with self.lock:
            return [job.to_dict() for job in self.running_jobs.values()]

    def _stream_output_to_client(self, job, proc):
        """Stream job output back to the client terminal"""
        try:
            print(f"[DEBUG] Starting output streaming for job {job.id}")
            
            # Read output line by line in real-time
            while proc.poll() is None and job.client_socket:
                try:
                    # Read stdout line by line (stderr is merged into stdout)
                    line = proc.stdout.readline()
                    if line and job.client_socket:
                        # print(f"[DEBUG] Streaming line: {line.strip()}")  # Debug output
                        msg = json.dumps({'type': 'output', 'data': line})
                        try:
                            job.client_socket.send((msg + '\n').encode())
                        except (BrokenPipeError, ConnectionResetError, OSError) as e:
                            print(f"[DEBUG] Client disconnected: {e}")
                            print(f"[DEBUG] Canceling job {job.id} due to client disconnection")
                            self._cancel_job_due_to_disconnect(job, proc)
                            return
                    elif not line:
                        # No more output available, small delay
                        time.sleep(0.001)
                        
                except Exception as e:
                    print(f"[DEBUG] Error reading output: {e}")
                    print(f"[DEBUG] Canceling job {job.id} due to streaming error")
                    self._cancel_job_due_to_disconnect(job, proc)
                    return
            
            # Check if client disconnected while job was running
            if job.client_socket and proc.poll() is None:
                try:
                    # Try to send a small test message to check connection
                    job.client_socket.send(b'')
                except (BrokenPipeError, ConnectionResetError, OSError):
                    print(f"[DEBUG] Client disconnected during job execution")
                    print(f"[DEBUG] Canceling job {job.id} due to client disconnection")
                    self._cancel_job_due_to_disconnect(job, proc)
                    return
            
            print(f"[DEBUG] Job {job.id} finished with exit code {proc.returncode}")
            
            # Send job completion message
            if job.client_socket:
                try:
                    completion_msg = json.dumps({'type': 'completion', 'job_id': job.id, 'exit_code': proc.returncode})
                    job.client_socket.send((completion_msg + '\n').encode())
                    print(f"[DEBUG] Sent completion message for job {job.id}")
                except Exception as e:
                    print(f"[DEBUG] Error sending completion: {e}")
                finally:
                    try:
                        job.client_socket.close()
                    except:
                        pass
                    job.client_socket = None
                    
        except Exception as e:
            print(f"[DEBUG] Error streaming output: {e}")
            print(f"[DEBUG] Canceling job {job.id} due to streaming exception")
            self._cancel_job_due_to_disconnect(job, proc)

    def _cancel_job_due_to_disconnect(self, job, proc):
        """Cancel a job when client disconnects"""
        try:
            print(f"[DEBUG] Killing process tree for job {job.id}")
            self._kill_proc_tree(proc.pid)
            
            # Remove from running jobs
            with self.lock:
                if job.id in self.running_jobs:
                    del self.running_jobs[job.id]
                    
            # Close client socket if still open
            if job.client_socket:
                try:
                    job.client_socket.close()
                except:
                    pass
                job.client_socket = None
                
            print(f"[DEBUG] Job {job.id} canceled due to client disconnection")
            
        except Exception as e:
            print(f"[DEBUG] Error canceling job {job.id}: {e}")

    def try_run_jobs(self, max_job_time=None):
        with self.lock:
            available = get_available_gpus()
            if not available:
                return
            max_mem = max(available) if available else 0
            min_mem = min(available) if available else 0
            used = [0]*len(available)
            # 현재 실행 중인 작업의 GPU 메모리 점유량 반영
            for running_job in self.running_jobs.values():
                if running_job.mem is not None and running_job.gpus > 0:
                    # 이미 할당된 GPU 인덱스 추정 (가장 단순하게 앞에서부터 할당)
                    for i in range(running_job.gpus):
                        if i < len(used):
                            used[i] += running_job.mem
            # User priority table (can be loaded from config or set here)
            user_priority = {
                # 'username': priority (higher is more important)
                # Example:
                # 'alice': 10,
                # 'bob': 5,
            }
            # Sort queue by user-supplied priority (descending), then FIFO
            self.job_queue = deque(sorted(self.job_queue, key=lambda job: (-getattr(job, 'priority', 0), job.id)))
            for job in list(self.job_queue):
                job_mem = job.mem if job.mem is not None else min_mem
                if job_mem > max_mem or job_mem < 1:
                    job.status = 'error'
                    job.error_msg = f"요청 메모리({job_mem}MB)가 허용 범위({min_mem}~{max_mem}MB)를 벗어났습니다."
                    continue
                # GPU allocation: prefer idle GPUs, then those with most free memory
                gpu_status = [(i, available[i] - used[i], used[i]) for i in range(len(available))]
                gpu_status.sort(key=lambda x: (x[2] > 0, -x[1]))

                if job.gpu_ids:
                    # Validate requested GPU IDs (convert to int if needed)
                    candidate_idxs = [int(i) for i in job.gpu_ids if int(i) < len(available) and available[int(i)] - used[int(i)] >= job_mem]
                else:
                    candidate_idxs = [i for i, free, u in gpu_status if free >= job_mem]

                if len(candidate_idxs) >= job.gpus:
                    selected_idxs = candidate_idxs[:job.gpus]
                    for idx in selected_idxs:
                        used[idx] += job_mem
                    
                    # Build command with CUDA_VISIBLE_DEVICES and force unbuffered output
                    cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in selected_idxs)}"
                    home_dir = os.path.expanduser(f'~{job.user}')
                    env_setup_cmd = getattr(job, 'env_setup_cmd', None)
                    
                    # Build full command with unbuffered output
                    if env_setup_cmd:
                        cmd = f"cd {home_dir} && {env_setup_cmd} && PYTHONUNBUFFERED=1 {cuda_env} {job.cmd}"
                    else:
                        cmd = f"cd {home_dir} && PYTHONUNBUFFERED=1 {cuda_env} {job.cmd}"
                    
                    # Start process with output pipes for streaming
                    if job.client_socket:
                        # Interactive mode - stream output to client
                        # Use unbuffered output and pty for real-time streaming
                        proc = subprocess.Popen([
                            'sudo', '-u', job.user, 'bash', '-lc', cmd
                        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                        bufsize=0, universal_newlines=True, preexec_fn=os.setsid)
                        
                        # Start output streaming thread
                        threading.Thread(
                            target=self._stream_output_to_client, 
                            args=(job, proc), 
                            daemon=True
                        ).start()
                    else:
                        # Background mode - no output streaming
                        proc = subprocess.Popen([
                            'sudo', '-u', job.user, 'bash', '-lc', cmd
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                    
                    job.proc = proc
                    job.status = 'running'
                    job.start_time = time.time()
                    self.running_jobs[job.id] = job
                    self.job_queue.remove(job)

    def reap_jobs(self):
        with self.lock:
            finished = [jid for jid, job in self.running_jobs.items() if job.proc.poll() is not None]
            for jid in finished:
                del self.running_jobs[jid]
    
    def check_disconnected_clients(self):
        """Check for disconnected interactive clients and cancel their jobs"""
        with self.lock:
            disconnected_jobs = []
            for jid, job in self.running_jobs.items():
                # Only check interactive jobs (those with client_socket)
                if job.client_socket and job.proc and job.proc.poll() is None:
                    try:
                        # Try to send empty data to test connection
                        job.client_socket.send(b'')
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print(f"[DEBUG] Detected disconnected client for job {jid}")
                        disconnected_jobs.append(jid)
            
            # Cancel disconnected jobs
            for jid in disconnected_jobs:
                job = self.running_jobs[jid]
                try:
                    print(f"[DEBUG] Canceling job {jid} due to client disconnection")
                    self._kill_proc_tree(job.proc.pid)
                except Exception as e:
                    print(f"[DEBUG] Error killing job {jid}: {e}")
                
                # Clean up
                if job.client_socket:
                    try:
                        job.client_socket.close()
                    except:
                        pass
                    job.client_socket = None
                
                del self.running_jobs[jid]

def handle_client(conn, scheduler, max_job_time):
    try:
        data = conn.recv(4096)
        req = json.loads(data.decode())
        cmd = req.get('cmd')
        if cmd == 'submit':
            available = get_available_gpus()
            max_mem = max(available) if available else 0
            min_mem = min(available) if available else 0
            mem = req.get('mem')
            if mem is not None:
                if mem > max_mem or mem < 1:
                    job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                    msg = f"요청 메모리({mem}MB)가 허용 범위({min_mem}~{max_mem}MB)를 벗어났습니다."
                    conn.send(json.dumps({'status':'fail','job_id':job_id,'msg':msg}).encode())
                    return
            time_limit = req.get('time_limit')
            priority = req.get('priority', 0)
            gpu_ids = req.get('gpu_ids')
            env_setup_cmd = req.get('env_setup_cmd')
            
            # Check if this is an interactive job (client wants output streaming)
            interactive = req.get('interactive', False)
            client_socket = conn if interactive else None
            
            job = Job(req['user'], req['gpus'], mem, req['cmdline'], time_limit, priority, gpu_ids, env_setup_cmd, client_socket)
            job_id = scheduler.submit_job(job)
            
            # Send initial response
            response = {'status':'ok','job_id':job_id}
            if interactive:
                response['interactive'] = True
            response_data = json.dumps(response).encode()
            conn.send(response_data)
            
            # For non-interactive jobs, ensure data is sent before closing
            if not interactive:
                try:
                    conn.shutdown(socket.SHUT_WR)  # Signal we're done sending
                except:
                    pass
                conn.close()
            # For interactive jobs, connection stays open for streaming
            
        elif cmd == 'queue':
            queue = scheduler.get_queue()
            running = scheduler.get_running()
            response_data = json.dumps({'status':'ok','queue':queue,'running':running}).encode()
            conn.send(response_data)
            try:
                conn.shutdown(socket.SHUT_WR)
            except:
                pass
            conn.close()
        elif cmd == 'cancel':
            ok = scheduler.cancel_job(req['job_id'])
            response_data = json.dumps({'status':'ok' if ok else 'fail'}).encode()
            conn.send(response_data)
            try:
                conn.shutdown(socket.SHUT_WR)
            except:
                pass
            conn.close()
        else:
            response_data = json.dumps({'status':'fail','msg':'unknown command'}).encode()
            conn.send(response_data)
            try:
                conn.shutdown(socket.SHUT_WR)
            except:
                pass
            conn.close()
    except Exception as e:
        try:
            response_data = json.dumps({'status':'fail','msg':str(e)}).encode()
            conn.send(response_data)
            conn.shutdown(socket.SHUT_WR)
        except:
            pass
        conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-job-time', type=int, default=None, help='모든 작업의 최대 점유시간(초). 미설정시 무제한')
    args = parser.parse_args()
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    scheduler = Scheduler()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCKET_PATH)
    s.listen(5)
    print('mgpu_scheduler_server started')
    def bg():
        while True:
            scheduler.try_run_jobs(args.max_job_time)
            scheduler.reap_jobs()
            scheduler.check_disconnected_clients()
            time.sleep(2)
    threading.Thread(target=bg, daemon=True).start()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle_client, args=(conn, scheduler, args.max_job_time), daemon=True).start()

if __name__ == "__main__":
    main()
