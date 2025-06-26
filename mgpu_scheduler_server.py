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
from collections import deque

SOCKET_PATH = '/tmp/mgpu_scheduler.sock'

class Job:
    def __init__(self, user, gpus, mem, cmd):
        self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.user = user
        self.gpus = gpus
        self.mem = mem
        self.cmd = cmd
        self.status = 'queued'
        self.proc = None

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

    def try_run_jobs(self):
        with self.lock:
            available = get_available_gpus()
            if not available:
                return
            max_mem = max(available) if available else 0
            min_mem = min(available) if available else 0
            used = [0]*len(available)
            for job in self.running_jobs.values():
                for i in range(job.gpus):
                    used[i] += 1
            for job in list(self.job_queue):
                # 메모리 제약 체크
                if job.mem > max_mem or job.mem < 1:
                    job.status = 'error'
                    job.error_msg = f"요청 메모리({job.mem}MB)가 허용 범위({min_mem}~{max_mem}MB)를 벗어났습니다."
                    continue
                idxs = [i for i, m in enumerate(available) if m >= job.mem]
                if len(idxs) >= job.gpus:
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in idxs[:job.gpus])
                    proc = subprocess.Popen(job.cmd, shell=True, env=env)
                    job.proc = proc
                    job.status = 'running'
                    self.running_jobs[job.id] = job
                    self.job_queue.remove(job)

    def reap_jobs(self):
        with self.lock:
            finished = [jid for jid, job in self.running_jobs.items() if job.proc.poll() is not None]
            for jid in finished:
                del self.running_jobs[jid]

def handle_client(conn, scheduler):
    try:
        data = conn.recv(4096)
        req = json.loads(data.decode())
        cmd = req.get('cmd')
        if cmd == 'submit':
            # 제출 시점에 GPU 상태 확인 및 메모리 제약 체크
            available = get_available_gpus()
            max_mem = max(available) if available else 0
            min_mem = min(available) if available else 0
            if req['mem'] > max_mem or req['mem'] < 1:
                job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                msg = f"요청 메모리({req['mem']}MB)가 허용 범위({min_mem}~{max_mem}MB)를 벗어났습니다."
                conn.send(json.dumps({'status':'fail','job_id':job_id,'msg':msg}).encode())
                return
            job = Job(req['user'], req['gpus'], req['mem'], req['cmdline'])
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
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    scheduler = Scheduler()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCKET_PATH)
    s.listen(5)
    print('mgpu_scheduler_server started')
    def bg():
        while True:
            scheduler.try_run_jobs()
            scheduler.reap_jobs()
            time.sleep(2)
    threading.Thread(target=bg, daemon=True).start()
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle_client, args=(conn, scheduler), daemon=True).start()

if __name__ == "__main__":
    main()
