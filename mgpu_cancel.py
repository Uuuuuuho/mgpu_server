#!/usr/bin/env python3
import sys
import socket
import json

def main():
    if len(sys.argv) < 2:
        print('Usage: mgpu_cancel <job_id>')
        sys.exit(1)
    job_id = sys.argv[1]
    req = {'cmd':'cancel','job_id':job_id}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/tmp/mgpu_scheduler.sock')
    s.send(json.dumps(req).encode())
    resp = json.loads(s.recv(4096).decode())
    if resp['status'] == 'ok':
        print('Job cancelled.')
    else:
        print('Cancel failed.')

if __name__ == "__main__":
    main()
