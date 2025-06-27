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

SOCKET_PATH = '/tmp/mgpu_scheduler.sock'
MAX_JOB_TIME = 600  # 최대 점유시간(초), 필요시 main에서 인자로 받을 수 있음

class Job:
    def __init__(self, user, gpus, mem, cmd, time_limit=None):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.user = user
        self.gpus = gpus
        self.mem = mem  # None이면 서버에서 자동 할당
        self.cmd = cmd
        self.status = 'queued'
        self.proc = None
        self.start_time = None  # 실행 시작 시간
        self.time_limit = time_limit  # 유저별 시간 제한(초)

    def to_dict(self):
        return {
            'id': self.id,
            'user': self.user,
            'gpus': self.gpus,
            'mem': self.mem,
            'cmd': self.cmd,
            'status': self.status
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

    def cancel_job(self, job_id):
        with self.lock:
            for job in list(self.job_queue):
                if job.id == job_id:
                    self.job_queue.remove(job)
                    return True
            if job_id in self.running_jobs:
                proc = self.running_jobs[job_id].proc
                if proc:
                    proc.terminate()
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
            now = time.time()
            # 실행 중인 작업의 점유시간 체크 및 선점
            for job in list(self.running_jobs.values()):
                limit = job.time_limit if job.time_limit is not None else max_job_time
                if limit is not None and job.start_time and now - job.start_time > limit:
                    print(f"[INFO] Job {job.id}({job.user}) 점유시간({limit}s) 초과로 큐 뒤로 이동(context switch)")
                    if job.proc:
                        job.proc.terminate()
                    job.status = 'timeout'
                    job.start_time = None
                    self.job_queue.append(self.job_queue.pop() if len(self.job_queue) else job)  # 맨 뒤로
                    del self.running_jobs[job.id]
            for job in list(self.job_queue):
                # mem이 None이면 최소 메모리로 자동 할당
                job_mem = job.mem if job.mem is not None else min_mem
                if job_mem > max_mem or job_mem < 1:
                    job.status = 'error'
                    job.error_msg = f"요청 메모리({job_mem}MB)가 허용 범위({min_mem}~{max_mem}MB)를 벗어났습니다."
                    continue
                idxs = [i for i, m in enumerate(available) if m >= job_mem]
                if len(idxs) >= job.gpus:
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in idxs[:job.gpus])
                    # 유저 홈 디렉토리에서 명령 실행
                    home_dir = os.path.expanduser(f'~{job.user}')
                    cmd = f'cd {home_dir} && {job.cmd}'
                    proc = subprocess.Popen([
                        'sudo', '-u', job.user, 'bash', '-lc', cmd
                    ], env=env)
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
            job = Job(req['user'], req['gpus'], mem, req['cmdline'], time_limit)
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
