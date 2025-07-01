#!/usr/bin/env python3
import sys
import os
import socket
import json
import getpass

def main():
    if '--gpus' not in sys.argv or '--' not in sys.argv:
        print('Usage: mgpu_srun --gpus <N> [--mem <MB>] [--time-limit <sec>] [--priority <N>] -- <command>')
        sys.exit(1)
    gpus = int(sys.argv[sys.argv.index('--gpus')+1])
    mem = None
    if '--mem' in sys.argv:
        mem = int(sys.argv[sys.argv.index('--mem')+1])
    time_limit = None
    if '--time-limit' in sys.argv:
        time_limit = int(sys.argv[sys.argv.index('--time-limit')+1])
    priority = 0
    if '--priority' in sys.argv:
        priority = int(sys.argv[sys.argv.index('--priority')+1])
    cmd_idx = sys.argv.index('--')+1
    cmdline = ' '.join(sys.argv[cmd_idx:])
    user = getpass.getuser()
    req = {'cmd':'submit','user':user,'gpus':gpus,'cmdline':cmdline, 'priority': priority}
    if mem is not None:
        req['mem'] = mem
    if time_limit is not None:
        req['time_limit'] = time_limit
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/tmp/mgpu_scheduler.sock')
    s.send(json.dumps(req).encode())
    resp = json.loads(s.recv(4096).decode())
    if resp['status'] == 'ok':
        print(f"Job submitted. ID: {resp['job_id']} (priority={priority})")
    else:
        print(f"Submit failed: {resp.get('msg','')} (ID: {resp.get('job_id','')})")

if __name__ == "__main__":
    main()
