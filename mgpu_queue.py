#!/usr/bin/env python3
import socket
import json

def main():
    req = {'cmd':'queue'}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/tmp/mgpu_scheduler.sock')
    s.send(json.dumps(req).encode())
    resp = json.loads(s.recv(4096).decode())
    if resp['status'] == 'ok':
        print('--- Running Jobs ---')
        for job in resp['running']:
            print(f"ID: {job['id']} | User: {job['user']} | GPUs: {job['gpus']} | Mem: {job['mem']} | CMD: {job['cmd']} | Status: running")
        print('--- Queue ---')
        for job in resp['queue']:
            print(f"ID: {job['id']} | User: {job['user']} | GPUs: {job['gpus']} | Mem: {job['mem']} | CMD: {job['cmd']} | Status: queued")
    else:
        print('Query failed:', resp.get('msg',''))

if __name__ == "__main__":
    main()
