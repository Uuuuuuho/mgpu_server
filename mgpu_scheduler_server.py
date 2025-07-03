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

SOCKET_PATH = '/tmp/mgpu_scheduler.sock'
MAX_JOB_TIME = 600  # 최대 점유시간(초), 필요시 main에서 인자로 받을 수 있음

class Job:
    def __init__(self, user, gpus, mem, cmd, time_limit=None, priority=0, gpu_ids=None, env_setup_cmd=None):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.user = user
        self.gpus = gpus
        self.mem = mem  # None이면 서버에서 자동 할당
        self.cmd = cmd
        self.status = 'queued'
        self.proc = None
        self.start_time = None  # 실행 시작 시간
        self.time_limit = time_limit  # 유저별 시간 제한(초)
        self.priority = priority
        self.gpu_ids = gpu_ids  # 사용자가 요청한 특정 GPU ID
        self.env_setup_cmd = env_setup_cmd  # 사용자가 요청한 환경설정 명령어

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
                    env = os.environ.copy()
                    # Build CUDA_VISIBLE_DEVICES string
                    cuda_env = f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in selected_idxs)}"
                    env_setup_cmd = getattr(job, 'env_setup_cmd', None)
                    cmd = f"{env_setup_cmd} && {cuda_env} {job.cmd}"
                    # Save log file to server user's home directory
                    server_home = os.path.expanduser('~')
                    log_file = os.path.join(server_home, f'.mgpu_job_{job.id}.log')
                    with open(log_file, 'a') as lf:
                        proc = subprocess.Popen([
                            'sudo', '-u', job.user, 'bash', '-lc', cmd
                        ], stdout=lf, stderr=lf, preexec_fn=os.setsid)
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
            job = Job(req['user'], req['gpus'], mem, req['cmdline'], time_limit, priority, gpu_ids, env_setup_cmd)
            job_id = scheduler.submit_job(job)
            conn.send(json.dumps({'status':'ok','job_id':job_id}).encode())
        elif cmd == 'queue':
            queue = scheduler.get_queue()
            running = scheduler.get_running()
            conn.send(json.dumps({'status':'ok','queue':queue,'running':running}).encode())
        elif cmd == 'cancel':
            ok = scheduler.cancel_job(req['job_id'])
            conn.send(json.dumps({'status':'ok' if ok else 'fail'}).encode())
        else:
            conn.send(json.dumps({'status':'fail','msg':'unknown command'}).encode())
    except Exception as e:
        conn.send(json.dumps({'status':'fail','msg':str(e)}).encode())
    finally:
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
            time.sleep(2)
    threading.Thread(target=bg, daemon=True).start()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle_client, args=(conn, scheduler, args.max_job_time), daemon=True).start()

if __name__ == "__main__":
    main()
