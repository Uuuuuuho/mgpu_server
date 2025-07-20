#!/usr/bin/env python3
import sys
import os
import socket
import json
import getpass

def main():
    # Parse command line arguments for job submission
    if '--gpu-ids' not in sys.argv or '--' not in sys.argv:
        print('Usage: mgpu_srun --gpu-ids <ID1,ID2,...> [--mem <MB>] [--time-limit <sec>] [--priority <N>] [--env-setup-cmd <CMD>] [--interactive] [--background] -- <command>')
        sys.exit(1)
    gpu_ids = sys.argv[sys.argv.index('--gpu-ids')+1].split(',')
    gpus = len(gpu_ids)
    mem = None
    if '--mem' in sys.argv:
        mem = int(sys.argv[sys.argv.index('--mem')+1])
    time_limit = None
    if '--time-limit' in sys.argv:
        time_limit = int(sys.argv[sys.argv.index('--time-limit')+1])
    priority = 0
    if '--priority' in sys.argv:
        priority = int(sys.argv[sys.argv.index('--priority')+1])
    env_setup_cmd = None
    if '--env-setup-cmd' in sys.argv:
        env_setup_cmd = sys.argv[sys.argv.index('--env-setup-cmd')+1]
    
    # Default to interactive mode unless --background is specified
    interactive = '--background' not in sys.argv
    cmd_idx = sys.argv.index('--')+1
    cmdline = ' '.join(sys.argv[cmd_idx:])
    user = getpass.getuser()
    req = {'cmd':'submit','user':user,'gpus':gpus,'gpu_ids':gpu_ids,'cmdline':cmdline, 'priority': priority, 'interactive': interactive}
    if mem is not None:
        req['mem'] = mem
    if time_limit is not None:
        req['time_limit'] = time_limit
    if env_setup_cmd is not None:
        req['env_setup_cmd'] = env_setup_cmd
    # Connect to the scheduler server and submit the job
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect('/tmp/mgpu_scheduler.sock')
    s.send(json.dumps(req).encode())
    resp = json.loads(s.recv(4096).decode())
    if resp['status'] == 'ok':
        print(f"Job submitted. ID: {resp['job_id']} (priority={priority})")
        
        # If interactive mode, listen for output
        if interactive and resp.get('interactive'):
            print("Waiting for job to start...")
            job_id = resp['job_id']
            try:
                s.settimeout(None)  # Remove timeout for streaming
                buffer = ""
                while True:
                    try:
                        data = s.recv(4096)
                        if not data:
                            print("[DEBUG] No more data from server")
                            break
                        
                        buffer += data.decode('utf-8', errors='ignore')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                try:
                                    msg = json.loads(line)
                                    if msg['type'] == 'output':
                                        print(msg['data'], end='', flush=True)
                                    elif msg['type'] == 'completion':
                                        print(f"\nJob {msg['job_id']} completed with exit code {msg['exit_code']}")
                                        s.close()
                                        return
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, just print the line
                                    print(f"[DEBUG] Non-JSON line: {line}")
                    except (ConnectionResetError, ConnectionAbortedError, socket.error) as e:
                        print(f"\nConnection to server lost: {e}")
                        break
            except KeyboardInterrupt:
                print("\nUser interrupted. Canceling job...")
                try:
                    # Send cancel request
                    cancel_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    cancel_socket.connect('/tmp/mgpu_scheduler.sock')
                    cancel_req = {'cmd': 'cancel', 'job_id': job_id}
                    cancel_socket.send(json.dumps(cancel_req).encode())
                    cancel_resp = json.loads(cancel_socket.recv(4096).decode())
                    if cancel_resp['status'] == 'ok':
                        print(f"Job {job_id} canceled successfully.")
                    else:
                        print(f"Failed to cancel job {job_id}")
                    cancel_socket.close()
                except Exception as e:
                    print(f"Error canceling job: {e}")
            finally:
                try:
                    s.close()
                except:
                    pass
        else:
            print("Job queued. Use mgpu_queue to check status.")
    else:
        print(f"Submit failed: {resp.get('msg','')} (ID: {resp.get('job_id','')})")
    s.close()

if __name__ == "__main__":
    main()
