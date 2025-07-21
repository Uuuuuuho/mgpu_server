#!/usr/bin/env python3
"""
Multi-Node GPU Scheduler - Enhanced Client
Supports both single-node and multi-node job submission
"""
import sys
import os
import socket
import json
import getpass
import random
import string

def generate_job_id():
    """고유한 작업 ID 생성"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def parse_node_requirements(args):
    """노드 요구사항 파싱"""
    requirements = {}
    
    # 기존 단일 노드 방식
    if '--gpu-ids' in args:
        gpu_ids_idx = args.index('--gpu-ids')
        gpu_ids = args[gpu_ids_idx + 1].split(',')
        requirements['gpu_ids'] = [int(x) for x in gpu_ids]
        requirements['total_gpus'] = len(gpu_ids)
        return requirements, 'single'
    
    # 멀티 노드 방식
    distributed_type = 'single'
    
    if '--nodes' in args:
        nodes_idx = args.index('--nodes')
        requirements['nodes'] = int(args[nodes_idx + 1])
        distributed_type = 'pytorch'  # 기본값
    
    if '--gpus-per-node' in args:
        gpus_idx = args.index('--gpus-per-node')
        requirements['gpus_per_node'] = int(args[gpus_idx + 1])
    
    if '--nodelist' in args:
        nodelist_idx = args.index('--nodelist')
        requirements['nodelist'] = args[nodelist_idx + 1].split(',')
    
    if '--exclude' in args:
        exclude_idx = args.index('--exclude')
        requirements['exclude'] = args[exclude_idx + 1].split(',')
    
    # 분산 실행 타입
    if '--mpi' in args:
        distributed_type = 'mpi'
    elif '--distributed' in args or '--pytorch-distributed' in args:
        distributed_type = 'pytorch'
    
    # 총 GPU 수 계산
    if 'nodes' in requirements and 'gpus_per_node' in requirements:
        requirements['total_gpus'] = requirements['nodes'] * requirements['gpus_per_node']
        distributed_type = 'pytorch' if distributed_type == 'single' else distributed_type
    elif 'gpu_ids' in requirements:
        requirements['total_gpus'] = len(requirements['gpu_ids'])
    else:
        requirements['total_gpus'] = 1
    
    return requirements, distributed_type

def print_usage():
    """사용법 출력"""
    print("""
Usage: mgpu_srun [OPTIONS] -- <command>

Single Node Options:
  --gpu-ids <ID1,ID2,...>     Specific GPU IDs to use
  --mem <MB>                  Memory requirement per GPU
  --time-limit <sec>          Job time limit
  --priority <N>              Job priority (higher = more priority)
  --env-setup-cmd <CMD>       Environment setup command
  --background                Run in background mode (default: interactive)

Multi-Node Options:
  --nodes <N>                 Number of nodes to use
  --gpus-per-node <N>         GPUs per node
  --nodelist <node1,node2>    Specific nodes to use
  --exclude <node1,node2>     Nodes to exclude
  --mpi                       Use MPI for distributed execution
  --distributed               Use PyTorch distributed execution
  --pytorch-distributed       Same as --distributed

Examples:
  # Single node with specific GPUs
  mgpu_srun --gpu-ids 0,1 -- python train.py
  
  # Multi-node distributed training
  mgpu_srun --nodes 2 --gpus-per-node 4 --distributed -- \\
    torchrun --nnodes=2 --nproc_per_node=4 train.py
  
  # MPI distributed execution
  mgpu_srun --nodes 4 --gpus-per-node 2 --mpi -- \\
    mpirun -np 8 python mpi_train.py
  
  # Specific nodes
  mgpu_srun --nodelist node001,node002 --gpus-per-node 2 -- python train.py
""")

def main():
    # 인자 파싱
    if len(sys.argv) < 2 or '--' not in sys.argv:
        print_usage()
        sys.exit(1)
    
    try:
        cmd_idx = sys.argv.index('--')
    except ValueError:
        print("Error: Missing '--' separator before command")
        print_usage()
        sys.exit(1)
    
    args = sys.argv[1:cmd_idx]
    cmdline = ' '.join(sys.argv[cmd_idx+1:])
    
    if not cmdline:
        print("Error: No command specified")
        print_usage()
        sys.exit(1)
    
    # 노드 요구사항 파싱
    try:
        node_requirements, distributed_type = parse_node_requirements(args)
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        print_usage()
        sys.exit(1)
    
    # 기타 옵션 파싱
    mem = None
    if '--mem' in args:
        mem_idx = args.index('--mem')
        mem = int(args[mem_idx + 1])
    
    time_limit = None
    if '--time-limit' in args:
        time_idx = args.index('--time-limit')
        time_limit = int(args[time_idx + 1])
    
    priority = 0
    if '--priority' in args:
        priority_idx = args.index('--priority')
        priority = int(args[priority_idx + 1])
    
    env_setup_cmd = None
    if '--env-setup-cmd' in args:
        env_idx = args.index('--env-setup-cmd')
        env_setup_cmd = args[env_idx + 1]
    
    # 기본값: 인터랙티브 모드
    interactive = '--background' not in args
    
    # 작업 요청 구성
    user = getpass.getuser()
    job_id = generate_job_id()
    
    # 마스터 서버 연결 (항상 TCP 연결 사용)
    master_host = os.environ.get('MGPU_MASTER_HOST', 'localhost')
    master_port = int(os.environ.get('MGPU_MASTER_PORT', '8080'))
    
    s = None
    try:
        # 항상 TCP 연결 사용 (단일/멀티 노드 자동 처리)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((master_host, master_port))
        
        req = {
            'cmd': 'submit',
            'job_id': job_id,
            'user': user,
            'cmdline': cmdline,
            'node_requirements': node_requirements,
            'total_gpus': node_requirements.get('total_gpus', 1),
            'distributed_type': distributed_type,
            'priority': priority,
            'interactive': interactive
        }
        
        # 옵션 추가
        if mem is not None:
            req['mem'] = mem
        if time_limit is not None:
            req['time_limit'] = time_limit
        if env_setup_cmd is not None:
            req['env_setup_cmd'] = env_setup_cmd
        
        # 요청 전송
        s.send(json.dumps(req).encode())
        resp = json.loads(s.recv(4096).decode())
        
        if resp['status'] == 'ok':
            print(f"Job submitted. ID: {resp['job_id']} (priority={priority})")
            
            print(f"Distributed type: {distributed_type}")
            if 'nodes' in node_requirements:
                print(f"Nodes requested: {node_requirements['nodes']}")
            if 'gpus_per_node' in node_requirements:
                print(f"GPUs per node: {node_requirements['gpus_per_node']}")
            
            # 인터랙티브 모드 처리
            if interactive and resp.get('interactive'):
                print("Waiting for job to start...")
                try:
                    s.settimeout(None)
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
                                            if s:
                                                s.close()
                                            return
                                    except json.JSONDecodeError:
                                        print(f"[DEBUG] Non-JSON line: {line}")
                        except (ConnectionResetError, ConnectionAbortedError, socket.error) as e:
                            print(f"\nConnection to server lost: {e}")
                            break
                except KeyboardInterrupt:
                    print("\nUser interrupted. Canceling job...")
                    try:
                        # 취소 요청 전송 (항상 TCP 연결 사용)
                        cancel_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        cancel_socket.connect((master_host, master_port))
                        
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
                    if s:
                        try:
                            s.close()
                        except:
                            pass
            else:
                print("Job queued. Use mgpu_queue to check status.")
        else:
            print(f"Submit failed: {resp.get('msg', resp.get('message', ''))}")
    
    except ConnectionRefusedError:
        print(f"Error: Cannot connect to master server at {master_host}:{master_port}")
        print("Make sure mgpu_master_server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if s:
            try:
                s.close()
            except:
                pass

if __name__ == "__main__":
    main()
