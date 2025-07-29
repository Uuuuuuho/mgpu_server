# Multi-GPU Scheduler

A simplified, reliable multi-GPU job scheduling system for distributed computing environments.

## Overview

The Multi-GPU Scheduler is a lightweight job scheduling system designed to manage GPU resources efficiently. This version prioritizes **simplicity, stability, and maintainability** over complex features, providing a clean foundation for GPU resource management.

## System Architecture

### Current Implementation

**Note**: This system currently supports **single-node operation** with **localhost execution**. Multi-node support requires additional node agent implementation.

#### Available Components

1. **Simple Master Server** (`mgpu_simple_master.py`)
   - **Currently working** implementation
   - Handles job queuing and local execution
   - Socket-based API for client communication
   - Supports localhost GPU allocation

2. **Simple Node Agent** (`mgpu_simple_node.py`)
   - **Reference implementation** for remote nodes
   - Can be deployed on compute nodes
   - Communicates with master server
   - **Requires manual setup** for multi-node clusters

3. **Client Interface** (`mgpu_simple_client.py`)
   - **Currently working** command-line client
   - Simple interface: submit, queue, cancel
   - Works with the simple master server

#### Future Development

4. **Enhanced Master Server** (`mgpu_master_server.py`)
   - **Under development** - simplified architecture
   - Will replace complex legacy system
   - Currently incomplete

5. **Enhanced Client** (`mgpu_srun_multinode.py`)
   - **Under development** - improved interface
   - Will provide better user experience
   - Currently incomplete

## Key Features

### Simplified Design Philosophy
- **Clean codebase**: Easy to understand and maintain
- **Minimal dependencies**: Reduces complexity and potential issues
- **Proven patterns**: Based on working simple implementations
- **Robust error handling**: Graceful failure recovery

### Resource Management
- **GPU allocation**: Automatic assignment based on availability
- **Node management**: Support for single and multi-node setups
- **Resource cleanup**: Automatic cleanup on job completion
- **Localhost-first**: Works out of the box on single machines

### Job Management
- **Simple submission**: Submit jobs with minimal configuration
- **Queue status**: Real-time view of job and resource status
- **Job control**: Cancel and monitor running jobs
- **Status tracking**: Clear job state management

## Quick Start

### Prerequisites

- Python 3.7+
- PyYAML
- Basic CUDA setup (for GPU jobs)

### Installation

```bash
git clone <repository-url>
cd multigpu_scheduler
pip install -r requirements.txt
```

### Virtual Environment Setup

For PyTorch and GPU-enabled workloads, use a dedicated virtual environment:

```bash
# Activate the virtual environment (example path)
source venv/bin/activate

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Note**: Replace `venv/bin/activate` with your actual virtual environment path.

### Basic Usage

**Currently working with the simple implementation:**

#### Step 1: Start the Master Server

```bash
# Using virtual environment Python
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

#### Step 2: Start Node Agent (for multi-node or local execution)

```bash
# Start node agent with proper node ID
venv/bin/python src/mgpu_simple_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080
```

#### Step 3: Submit Jobs

**Interactive job with real-time output (recommended for testing):**
```bash
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python /path/to/your_script.py"
```

**Non-interactive job (background execution):**
```bash
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 "venv/bin/python /path/to/your_script.py"
```

**Example with PyTorch test:**
```bash
# Interactive PyTorch test execution
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"
```

#### Step 4: Monitor and Manage Jobs

```bash
# Check queue status
venv/bin/python src/mgpu_simple_client.py queue

# Cancel a job
venv/bin/python src/mgpu_simple_client.py cancel <job_id>
```

**Job submission with timeout options:**
```bash
# Interactive job with custom timeouts
venv/bin/python src/mgpu_simple_client.py submit --interactive --gpus 1 \
  --session-timeout 3600 \
  --max-consecutive-timeouts 60 \
  "venv/bin/python long_running_script.py"

# Non-interactive job with custom timeouts
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 \
  --max-wait-time 600 \
  --connection-timeout 15 \
  "venv/bin/python batch_job.py"
```

3. **Check queue status:**
```bash
venv/bin/python src/mgpu_simple_client.py queue
```

4. **Cancel a job:**
```bash
venv/bin/python src/mgpu_simple_client.py cancel <job_id>
```

### Multi-node Setup (Manual Node Connection)

**How multiple nodes connect to the master server:**

The current working implementation (`mgpu_simple_*`) supports multi-node setups through manual node agent deployment:

1. **Start master server on the head node:**
```bash
# Using virtual environment with proper host binding
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

2. **On each worker node, connect to the master:**
```bash
# Replace <MASTER_IP> with actual master server IP
venv/bin/python src/mgpu_simple_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node001
venv/bin/python src/mgpu_simple_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node002
# ... repeat for each worker node
```

3. **Submit jobs from any client:**
```bash
# Client can run from any machine that can reach the master
venv/bin/python src/mgpu_simple_client.py --host <MASTER_IP> --port 8080 submit --gpus 2 "venv/bin/python distributed_script.py"
```

**Key Points:**
- **Node agents must be started manually** on each worker machine
- **Master server coordinates** all GPU resources across connected nodes
- **Automatic load balancing** - jobs are assigned to available GPUs across all connected nodes
- **Network accessibility** - all nodes must be able to reach the master server's IP and port
- **Virtual environment consistency** - ensure the same Python environment is available on all nodes

### Tested Example: PyTorch GPU Workload

Here's a complete working example that has been tested:

1. **Start the system:**
```bash
# Terminal 1: Start master server
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080

# Terminal 2: Start node agent
venv/bin/python src/mgpu_simple_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080
```

2. **Submit and monitor PyTorch test:**
```bash
# Terminal 3: Submit interactive PyTorch test
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"
```

**Expected output:**
```
Job submission: {'status': 'ok', 'job_id': 'F68FAFEC', 'message': 'Job submitted'}
Starting interactive session...
==================================================
PyTorch CUDA Load Test
Start time: 2025-07-28 22:32:49

==================================================
CUDA Environment Check
==================================================
PyTorch version: 2.7.1+cu126
CUDA available: True
CUDA version: 12.6
Number of available GPUs: 1
GPU 0: NVIDIA GeForce RTX 3060
GPU 0 memory: 12.0 GB
...
```

3. **Check system status:**
```bash
# Check queue and node status
venv/bin/python src/mgpu_simple_client.py queue
```

## Configuration

### Single-node Configuration (Current)

For **localhost-only** operation (current working setup):

**No configuration file needed** - the simple master server works out of the box:

```bash
# Step 1: Start master server
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080

# Step 2: Start local node agent
venv/bin/python src/mgpu_simple_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080

# Step 3: Submit jobs
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python your_script.py"
```

The system will automatically detect available GPUs on localhost.

### Multi-node Configuration (Advanced Setup)

**For advanced users** wanting to set up multiple nodes:

1. **Start master server on main node:**
```bash
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

2. **Start node agents on compute nodes:**
```bash
# On node 1 (ensure virtual environment is available)
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node001

# On node 2  
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node002
```

3. **Configure firewall** to allow communication on port 8080

**Important Requirements:**
- Same virtual environment must be accessible on all nodes
- Python executable path should be consistent: `venv/bin/python`
- Network connectivity between all nodes
- Shared filesystem (optional but recommended for job scripts)

## Usage Examples

### Job Submission (Current Working Commands)

```bash
# Simple single-GPU job
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 "venv/bin/python train.py"

# Multi-GPU job on localhost
venv/bin/python src/mgpu_simple_client.py submit --gpus 2 "venv/bin/python distributed_train.py"

# Interactive job with output streaming (recommended for testing)
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python interactive_script.py"

# Interactive job with custom timeouts for long-running GPU workloads
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive \
  --session-timeout 14400 \
  --max-consecutive-timeouts 120 \
  "venv/bin/python gpu_training.py"

# Non-interactive job with custom monitoring timeouts
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 \
  --max-wait-time 1800 \
  --connection-timeout 20 \
  "venv/bin/python batch_processing.py"

# Real tested example: PyTorch CUDA test
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"
```

### Queue Management

```bash
# View all jobs and node status
venv/bin/python src/mgpu_simple_client.py queue

# Cancel a specific job
venv/bin/python src/mgpu_simple_client.py cancel <job_id>
```

### Advanced: Multi-node Jobs

**다른 노드의 GPU에 job 요청하는 방법:**

#### 1. 자동 스케줄링 (추천)
마스터 서버가 자동으로 사용 가능한 GPU에 할당:
```bash
# 단일 GPU job - 자동으로 가장 사용 가능한 노드에 할당
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 "venv/bin/python script.py"

# 다중 GPU job - 여러 노드에 자동 분산
venv/bin/python src/mgpu_simple_client.py submit --gpus 4 "venv/bin/python distributed_training.py"

# 인터랙티브 모드로 다른 노드에서 실행
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"
```

#### 2. 특정 노드/GPU 지정 (고급 사용자)
```bash
# 특정 노드의 특정 GPU에 job 제출
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --node-gpu-ids "node001:0" "venv/bin/python script.py"

# 특정 노드의 여러 GPU 사용
venv/bin/python src/mgpu_simple_client.py submit --gpus 2 --node-gpu-ids "node001:0,1" "venv/bin/python multi_gpu_script.py"

# 여러 노드에 걸친 분산 job
venv/bin/python src/mgpu_simple_client.py submit --gpus 4 --node-gpu-ids "node001:0,1;node002:0,1" "venv/bin/python distributed_training.py"

# 특정 노드에서 인터랙티브 실행
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive --node-gpu-ids "node002:0" "venv/bin/python debug_script.py"
```

#### 3. 실제 멀티노드 설정 예시

**마스터 노드 (192.168.1.100):**
```bash
# 마스터 서버 시작 - 모든 네트워크 인터페이스에서 수신
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

**워커 노드들:**
```bash
# 노드 1 (192.168.1.101)
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node001

# 노드 2 (192.168.1.102) 
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node002

# 노드 3 (192.168.1.103)
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node003
```

**클라이언트에서 job 제출:**
```bash
# 어느 머신에서든 마스터 서버에 연결하여 job 제출 가능
venv/bin/python src/mgpu_simple_client.py --host 192.168.1.100 --port 8080 submit --gpus 2 "venv/bin/python distributed_script.py"

# 특정 노드 지정
venv/bin/python src/mgpu_simple_client.py --host 192.168.1.100 --port 8080 submit --gpus 1 --node-gpu-ids "node002:0" "venv/bin/python script.py"
```

#### 4. 노드 상태 확인
```bash
# 모든 노드와 GPU 상태 확인
venv/bin/python src/mgpu_simple_client.py --host 192.168.1.100 --port 8080 queue

# 출력 예시:
# Jobs:
#   F68FAFEC: running on node002:gpu0
#   A23B4C5D: queued (waiting for 2 GPUs)
# 
# Nodes:
#   node001: 2 GPUs (1 free, 1 busy)
#   node002: 1 GPU (0 free, 1 busy) 
#   node003: 4 GPUs (4 free, 0 busy)
```

## Command Reference

### mgpu_simple_client.py (Current Working Client)

**submit** - Submit a new job
```bash
submit [--gpus N] [--interactive] [--node-gpu-ids "node:gpu_list"] \
       [--session-timeout SECONDS] [--connection-timeout SECONDS] \
       [--max-wait-time SECONDS] [--max-consecutive-timeouts COUNT] \
       "command"
```

**Timeout Options:**
- `--session-timeout`: Maximum session duration in seconds (default: 7200 = 2 hours)
- `--connection-timeout`: Socket connection timeout in seconds (default: 30)
- `--max-wait-time`: Maximum wait time for job output in seconds (default: 300 = 5 minutes)
- `--max-consecutive-timeouts`: Maximum consecutive timeouts before giving up (default: 30)

**queue** - Show queue status
```bash
queue
```

**cancel** - Cancel a job
```bash
cancel <job_id>
```

### mgpu_simple_master.py (Current Working Server)

**Start the master server:**
```bash
venv/bin/python src/mgpu_simple_master.py [--host HOST] [--port PORT]

# Example: Bind to all interfaces on port 8080
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

### mgpu_simple_node.py (For Multi-node Setup)

**Start a node agent:**
```bash
venv/bin/python src/mgpu_simple_node.py --master-host HOST --master-port PORT --node-id NODE_NAME

# Example: Connect to local master
venv/bin/python src/mgpu_simple_node.py --master-host 127.0.0.1 --master-port 8080 --node-id node001
```

## Development

### Architecture Principles

1. **Simplicity First**: Keep code simple and readable
2. **Reliability**: Prefer stable, proven approaches
3. **Maintainability**: Easy to debug and extend
4. **Testability**: Clear interfaces for testing

### Code Structure

**Currently working implementations:**
```
src/
├── mgpu_simple_master.py      # ✅ Working master server
├── mgpu_simple_client.py      # ✅ Working client interface  
├── mgpu_simple_node.py        # ✅ Working node agent
├── mgpu_master_server.py      # 🚧 Under development (simplified)
├── mgpu_srun_multinode.py     # 🚧 Under development (enhanced client)
└── mgpu_*.py                  # 📚 Legacy complex implementations
```

**Recommended for new users:** Start with `mgpu_simple_*` files - they are stable and fully functional.

### Testing

```bash
# Basic functionality test with virtual environment
venv/bin/python test/test_torch_load.py

# Test through the scheduler (recommended)
# 1. Start master and node first, then:
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"

# Run test suite
venv/bin/python test/run_tests.py
```

### Quick Test Procedure

**Verified working setup:**

1. **Start the system:**
```bash
# Terminal 1
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080

# Terminal 2  
venv/bin/python src/mgpu_simple_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080
```

2. **Test PyTorch functionality:**
```bash
# Terminal 3
venv/bin/python src/mgpu_simple_client.py submit --gpus 1 --interactive "venv/bin/python test/test_torch_load.py"
```

3. **Verify output includes:**
   - PyTorch version and CUDA availability
   - GPU device information
   - Successful completion of all test phases

## Troubleshooting

### Node 연결 문제 진단

**다른 노드에서 연결이 안될 때 단계별 확인 방법:**

#### 1. 마스터 서버 상태 확인 (마스터 노드에서)
```bash
# 마스터 서버 프로세스 확인
ps aux | grep mgpu_simple_master

# 포트 바인딩 확인 - 0.0.0.0:8080 으로 바인딩되어야 함
netstat -tlnp | grep 8080
# 또는
ss -tlnp | grep 8080

# 기대하는 출력: tcp 0 0 0.0.0.0:8080 0.0.0.0:* LISTEN
# 만약 127.0.0.1:8080 으로만 바인딩되어 있다면 외부 연결 불가
```

#### 2. 네트워크 연결 테스트 (워커 노드에서)
```bash
# 마스터 노드 ping 테스트
ping -c 3 192.168.1.100

# 포트 연결 테스트
telnet 192.168.1.100 8080
# 또는
nc -v 192.168.1.100 8080

# 성공시: Connected to 192.168.1.100
# 실패시: Connection refused / Connection timed out
```

#### 3. 방화벽 확인
**마스터 노드에서:**
```bash
# Ubuntu/Debian
sudo ufw status
sudo ufw allow 8080

# CentOS/RHEL/Rocky
sudo firewall-cmd --list-all
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# 방화벽 완전 비활성화 (테스트용)
sudo ufw disable  # Ubuntu
sudo systemctl stop firewalld  # CentOS
```

#### 4. 마스터 서버 재시작 (올바른 바인딩으로)
```bash
# 잘못된 시작 (외부 연결 불가)
venv/bin/python src/mgpu_simple_master.py --host 127.0.0.1 --port 8080

# 올바른 시작 (모든 인터페이스에서 수신)
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

#### 5. 노드 에이전트 연결 테스트
**워커 노드에서:**
```bash
# 상세한 오류 메시지와 함께 노드 에이전트 시작
venv/bin/python src/mgpu_simple_node.py \
  --master-host 192.168.1.100 \
  --master-port 8080 \
  --node-id node001 \
  --verbose

# 연결 성공시: "Connected to master server"
# 연결 실패시: "Connection refused" 또는 "Connection timeout"
```

#### 6. 네트워크 경로 및 라우팅 확인
```bash
# 라우팅 테이블 확인
ip route

# 특정 IP로의 경로 추적
traceroute 192.168.1.100

# 네트워크 인터페이스 확인
ip addr show
```

#### 7. 포트 스캔으로 서비스 확인
```bash
# nmap으로 마스터 노드 포트 확인
nmap -p 8080 192.168.1.100

# 출력 예시:
# 8080/tcp open  http-proxy  (연결 가능)
# 8080/tcp filtered http-proxy  (방화벽 차단)
# 8080/tcp closed http-proxy  (서비스 미실행)
```

#### 8. 로그 및 오류 메시지 확인
```bash
# 마스터 서버 로그 (터미널에서 확인)
# 노드 연결시 보이는 상세 메시지:
# 🔗 Node node001 connected from 192.168.1.101:8081
#    └─ 🎮 2 GPU(s) detected:
#       ├─ GPU 0: NVIDIA GeForce RTX 3060 (12288 MB)
#       ├─ GPU 1: NVIDIA GeForce RTX 4090 (24576 MB)

# 노드 에이전트 로그 (각 워커 노드에서)
# ✅ Registration successful
# Simple Node Agent node001 started on 0.0.0.0:8081

# 노드 에이전트 오류 메시지 확인
# - "Connection refused": 마스터 서버 미실행 또는 포트 문제
# - "No route to host": 네트워크 연결 문제
# - "Connection timed out": 방화벽 또는 네트워크 지연
# - "Registration failed": 마스터 서버 연결은 되지만 등록 실패
```

#### 9. 단계별 문제 해결
```bash
# Step 1: 마스터 서버 올바른 시작
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080

# Step 2: 방화벽 설정 (마스터 노드)
sudo ufw allow 8080

# Step 3: 연결 테스트 (워커 노드)
telnet 192.168.1.100 8080

# Step 4: 노드 에이전트 시작 (워커 노드)
venv/bin/python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node001

# Step 5: 연결 확인 (어느 노드에서든)
venv/bin/python src/mgpu_simple_client.py --host 192.168.1.100 --port 8080 queue
```

#### 10. 일반적인 오류 메시지와 해결책
```bash
# "Connection refused"
→ 마스터 서버가 실행되지 않았거나 잘못된 바인딩
→ 해결: --host 0.0.0.0 으로 마스터 서버 재시작

# "No route to host"
→ 네트워크 설정 문제 또는 IP 주소 오류
→ 해결: ping 테스트 및 네트워크 설정 확인

# "Connection timed out"
→ 방화벽이 포트를 차단하고 있음
→ 해결: 방화벽에서 8080 포트 허용

# "Name or service not known"
→ 호스트명 해석 실패
→ 해결: IP 주소 직접 사용 또는 /etc/hosts 설정
```

### 기타 일반적인 문제들

1. **Server won't start**
   - Check if port 8080 is available: `netstat -tlnp | grep 8080`
   - Verify Python environment: `which python`
   - Check dependencies: `pip list | grep -i yaml`

2. **Jobs not running**
   - Verify GPU availability: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Check node registration: `venv/bin/python src/mgpu_simple_client.py queue`

3. **Virtual environment issues**
   - Ensure same venv path on all nodes
   - Check PyTorch installation consistency
   - Verify CUDA libraries availability

### Debug Mode

Enable detailed logging:
```bash
# For current working implementation
venv/bin/python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080 --verbose

# Add verbose output to node agent
venv/bin/python src/mgpu_simple_node.py --master-host IP --master-port 8080 --node-id nodeXXX --verbose
```

## Migration Guide

### From Complex v1.0 System

If you're migrating from the previous complex system:

1. **Backup your current setup**
2. **Update configuration** to the new simplified YAML format
3. **Test with simple jobs** before production use
4. **Review and simplify** your job submission scripts

### Configuration Changes

- Simplified YAML structure
- Removed complex scheduling options
- Focus on core functionality

## Contributing

1. Follow the simplicity principle
2. Write clear, readable code
3. Add tests for new features
4. Update documentation

## Version History

### v2.0.0 - Simplified Architecture
- Complete rewrite focusing on simplicity and reliability
- Removed complex features that caused stability issues
- Clean, maintainable codebase
- Improved error handling and recovery

### v1.0.0 - Complex System (Legacy)
- Feature-rich but complex architecture
- Multiple threading and streaming features
- Known stability and maintenance issues

## License

[Specify your license here]

## Support

For issues:
1. Check this README and troubleshooting section
2. Review system logs for errors
3. Test with the simple reference implementations
4. Report issues with clear reproduction steps

---

**Note**: This simplified version prioritizes stability and maintainability. If you need advanced features, consider implementing them incrementally on top of this solid foundation.
