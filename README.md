# Multi-GPU Scheduler

A modular, reliable multi-GPU job scheduling system for distributed computing environments.

## Overview

The Multi-GPU Scheduler is a lightweight job scheduling system designed to manage GPU resources efficiently across single-node and multi-node environments. This version prioritizes **modularity, stability, and maintainability** with a clean, Single Responsibility Principle-based architecture.

## System Architecture

### Current Implementation

The system features a **modular architecture** with clear separation of concerns:

#### Core Components

1. **mgpu_core/** - Shared foundation components
   - **models/**: Data models (SimpleJob, NodeInfo, JobProcess, MessageType)
   - **network/**: Network communication utilities (NetworkManager)
   - **utils/**: System utilities (GPUManager, IPManager, TimeoutConfig, logging)

2. **mgpu_server/** - Master server components
   - **job_scheduler.py**: JobScheduler class for job queue management
   - **node_manager.py**: NodeManager class for node registration and health
   - **master_server.py**: MasterServer class for client request handling

3. **mgpu_client/** - Client interface components
   - **job_client.py**: JobClient class for job submission and monitoring

4. **mgpu_node/** - Node agent components
   - **node_agent.py**: NodeAgent class for job execution on worker nodes

#### Entry Points

- **mgpu_master.py**: Master server entry point
- **mgpu_client.py**: Client interface entry point  
- **mgpu_node.py**: Node agent entry point
- **mgpu_queue.py**: Queue management entry point
- **mgpu_cancel.py**: Job cancellation entry point

## Key Features

### Modular Design Philosophy
- **Single Responsibility Principle**: Each class has one clear responsibility
- **Clean separation**: Core, server, client, and node components are separate
- **Minimal dependencies**: Reduces complexity and potential issues
- **Robust error handling**: Graceful failure recovery with comprehensive logging

### Advanced Process Management
- **Process groups**: Jobs run in separate process groups for clean termination
- **Signal handling**: Graceful termination with SIGTERM, force kill with SIGKILL
- **Psutil fallback**: Comprehensive process tree cleanup
- **Orphan prevention**: No more zombie or orphaned processes on job cancellation

### Resource Management
- **GPU allocation**: Automatic assignment based on availability
- **Multi-node support**: Seamless operation across multiple compute nodes
- **Resource cleanup**: Automatic cleanup on job completion or cancellation
- **Node health monitoring**: Automatic detection of failed nodes

### Job Management
- **Flexible submission**: Submit jobs with or without timeout constraints
- **Real-time monitoring**: Interactive and non-interactive job monitoring
- **Queue management**: Real-time view of job and resource status
- **Job control**: Cancel and monitor running jobs with proper cleanup
- **Status tracking**: Clear job state management across all nodes

## Quick Start

### Prerequisites

- Python 3.7+
- PyYAML
- psutil (for process management)
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
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify PyTorch installation (if needed)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Basic Usage

#### Step 1: Start the Master Server

```bash
# Start master server (binds to all interfaces)
python src/mgpu_master.py --host 0.0.0.0 --port 8080
```

#### Step 2: Start Node Agents

```bash
# Start local node agent
python src/mgpu_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080

# For multi-node setup, start agents on each compute node:
# python src/mgpu_node.py --node-id node002 --master-host <MASTER_IP> --master-port 8080
```

#### Step 3: Submit Jobs

**Interactive job with real-time output:**
```bash
python src/mgpu_client.py submit --gpus 1 --interactive "python your_script.py"
```

**Non-interactive job (background execution):**
```bash
python src/mgpu_client.py submit --gpus 1 "python your_script.py"
```

**Job with unlimited execution time:**
```bash
# No timeout constraints - job runs until completion
python src/mgpu_client.py submit --gpus 1 "python long_running_script.py"
```

**Job with custom timeout settings:**
```bash
# Interactive job with custom timeouts
python src/mgpu_client.py submit --interactive --gpus 1 \
  --session-timeout 3600 \
  --max-consecutive-timeouts 60 \
  "python training_script.py"
```

#### Step 4: Monitor and Manage Jobs

```bash
# Check queue status and node information
python src/mgpu_client.py queue

# Monitor specific job output
python src/mgpu_client.py monitor <job_id>

# Cancel a job (with proper process tree cleanup)
python src/mgpu_client.py cancel <job_id>
```

### Multi-node Setup

**Setting up a distributed multi-node cluster:**

#### 1. Master Node Setup
```bash
# Start master server on the head node (replace with actual IP)
python src/mgpu_master.py --host 0.0.0.0 --port 8080
```

#### 2. Worker Node Setup
```bash
# On each worker node, connect to the master (replace <MASTER_IP> with actual IP)
python src/mgpu_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node001
python src/mgpu_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node002
# ... repeat for each worker node with unique node IDs
```

#### 3. Client Job Submission
```bash
# Submit jobs from any machine that can reach the master
python src/mgpu_client.py --host <MASTER_IP> --port 8080 submit --gpus 2 "python distributed_script.py"
```

**Key Points:**
- **Automatic load balancing**: Jobs are assigned to available GPUs across all connected nodes
- **Network accessibility**: All nodes must be able to reach the master server's IP and port
- **Environment consistency**: Ensure compatible Python environments on all nodes
- **Shared filesystem**: Recommended for job scripts and data access

### Example: Complete Multi-node Setup

**Master node (192.168.1.100):**
```bash
# Start master server - binds to all network interfaces
python src/mgpu_master.py --host 0.0.0.0 --port 8080
```

**Worker nodes:**
```bash
# Worker node 1 (192.168.1.101)
python src/mgpu_node.py --master-host 192.168.1.100 --master-port 8080 --node-id worker01

# Worker node 2 (192.168.1.102) 
python src/mgpu_node.py --master-host 192.168.1.100 --master-port 8080 --node-id worker02

# Worker node 3 (192.168.1.103)
python src/mgpu_node.py --master-host 192.168.1.100 --master-port 8080 --node-id worker03
```

**Client job submission from any machine:**
```bash
# Automatic GPU allocation across available nodes
python src/mgpu_client.py --host 192.168.1.100 --port 8080 submit --gpus 4 "python multi_gpu_training.py"

# Specific node-GPU assignment
python src/mgpu_client.py --host 192.168.1.100 --port 8080 submit --gpus 2 --node-gpu-ids "worker01:0,1" "python script.py"

# Check cluster status
python src/mgpu_client.py --host 192.168.1.100 --port 8080 queue
```

## Configuration

### Single-node Configuration

For **localhost-only** operation (default setup):

**No configuration file needed** - the system works out of the box:

```bash
# Step 1: Start master server
python src/mgpu_master.py --host 0.0.0.0 --port 8080

# Step 2: Start local node agent
python src/mgpu_node.py --node-id node001 --master-host 127.0.0.1 --master-port 8080

# Step 3: Submit jobs
python src/mgpu_client.py submit --gpus 1 --interactive "python your_script.py"
```

The system will automatically detect available GPUs on localhost.

### Multi-node Configuration

**For distributed clusters:**

1. **Configure firewall** to allow communication on port 8080 (or your chosen port)
2. **Ensure network connectivity** between all nodes
3. **Verify Python environment** compatibility across nodes
4. **Optional**: Set up shared filesystem for job scripts and data

**Network Requirements:**
- All worker nodes must be able to connect to master server IP:port
- Firewall rules must allow incoming connections on the master server port
- DNS resolution or IP connectivity between nodes

**Environment Requirements:**
- Compatible Python versions across all nodes  
- Required packages installed on all nodes
- CUDA drivers and libraries available on GPU nodes

## Usage Examples

### Job Submission Options

```bash
# Simple single-GPU job (no timeout limits)
python src/mgpu_client.py submit --gpus 1 "python train.py"

# Multi-GPU job with automatic node allocation
python src/mgpu_client.py submit --gpus 4 "python distributed_train.py"

# Interactive job with real-time output streaming
python src/mgpu_client.py submit --gpus 1 --interactive "python interactive_script.py"

# Job with custom timeout settings
python src/mgpu_client.py submit --gpus 1 --interactive \
  --session-timeout 3600 \
  --max-consecutive-timeouts 120 \
  "python gpu_training.py"

# Job with specific node-GPU assignment
python src/mgpu_client.py submit --gpus 2 --node-gpu-ids "worker01:0,1" "python script.py"

# Cross-node distributed job
python src/mgpu_client.py submit --gpus 4 --node-gpu-ids "worker01:0,1;worker02:0,1" "python multi_node_training.py"
```

### Queue Management

```bash
# View all jobs and cluster status
python src/mgpu_client.py queue

# Monitor specific job output (unlimited time if no --max-wait-time specified)
python src/mgpu_client.py monitor <job_id>

# Monitor with custom timeout
python src/mgpu_client.py monitor <job_id> --max-wait-time 1800

# Cancel a job (with complete process tree cleanup)
python src/mgpu_client.py cancel <job_id>

# View queue status from remote machine
python src/mgpu_client.py --host <MASTER_IP> --port 8080 queue
```

### Advanced Usage Examples

**1. Automatic Job Scheduling (Recommended)**
```bash
# Single GPU job - automatically assigned to best available node
python src/mgpu_client.py submit --gpus 1 "python script.py"

# Multi-GPU job - automatically distributed across available nodes
python src/mgpu_client.py submit --gpus 4 "python distributed_training.py"

# Interactive mode with automatic assignment
python src/mgpu_client.py submit --gpus 1 --interactive "python debug_script.py"
```

**2. Specific Node/GPU Assignment (Advanced Users)**
```bash
# Specific node and GPU assignment
python src/mgpu_client.py submit --gpus 1 --node-gpu-ids "worker01:0" "python script.py"

# Multiple GPUs on specific node
python src/mgpu_client.py submit --gpus 2 --node-gpu-ids "worker01:0,1" "python multi_gpu_script.py"

# Distributed job across multiple nodes
python src/mgpu_client.py submit --gpus 4 --node-gpu-ids "worker01:0,1;worker02:0,1" "python distributed_training.py"

# Interactive execution on specific node
python src/mgpu_client.py submit --gpus 1 --interactive --node-gpu-ids "worker02:0" "python debug_script.py"
```

**3. Timeout Management**
```bash
# Unlimited execution time (default behavior)
python src/mgpu_client.py submit --gpus 1 "python long_running_job.py"

# With session timeout only
python src/mgpu_client.py submit --gpus 1 --session-timeout 7200 "python training.py"

# With all timeout options
python src/mgpu_client.py submit --gpus 1 --interactive \
  --session-timeout 3600 \
  --connection-timeout 30 \
  --max-wait-time 600 \
  --max-consecutive-timeouts 60 \
  "python monitored_job.py"
```

## Command Reference

### mgpu_client.py - Job Management Client

**submit** - Submit a new job
```bash
python src/mgpu_client.py submit [OPTIONS] "command"

Options:
  --gpus N                          Number of GPUs required (default: 1)
  --interactive                     Enable interactive mode with real-time output
  --node-gpu-ids "node:gpu_list"    Specific node-GPU mapping (e.g., "node1:0,1;node2:2")
  --session-timeout SECONDS         Maximum session duration (unlimited if not specified)
  --connection-timeout SECONDS      Socket connection timeout (unlimited if not specified)  
  --max-wait-time SECONDS           Maximum wait time for job output (unlimited if not specified)
  --max-consecutive-timeouts COUNT  Maximum consecutive timeouts (unlimited if not specified)
```

**queue** - Show cluster and job status
```bash
python src/mgpu_client.py queue
```

**cancel** - Cancel a job with complete process cleanup
```bash
python src/mgpu_client.py cancel <job_id>
```

**monitor** - Monitor job output
```bash
python src/mgpu_client.py monitor <job_id> [--max-wait-time SECONDS]
```

**Global Options:**
```bash
--host HOST        Master server hostname/IP (default: 127.0.0.1)
--port PORT        Master server port (default: 8080)
--verbose          Enable detailed logging
```

### mgpu_master.py - Master Server

**Start the master server:**
```bash
python src/mgpu_master.py [OPTIONS]

Options:
  --host HOST        Bind address (default: 127.0.0.1, use 0.0.0.0 for all interfaces)
  --port PORT        Listen port (default: 8080)
  --verbose          Enable detailed logging
```

### mgpu_node.py - Node Agent

**Start a node agent:**
```bash
python src/mgpu_node.py [OPTIONS]

Options:
  --master-host HOST    Master server hostname/IP (required)
  --master-port PORT    Master server port (default: 8080)
  --node-id NODE_ID     Unique node identifier (required)
  --verbose             Enable detailed logging
```

### mgpu_queue.py - Queue Management

**Queue operations:**
```bash
python src/mgpu_queue.py [OPTIONS]

Options:
  --host HOST        Master server hostname/IP (default: 127.0.0.1)
  --port PORT        Master server port (default: 8080)
```

### mgpu_cancel.py - Job Cancellation

**Cancel jobs:**
```bash
python src/mgpu_cancel.py [OPTIONS] <job_id>

Options:
  --host HOST        Master server hostname/IP (default: 127.0.0.1)
  --port PORT        Master server port (default: 8080)
```

## Development

### Architecture Principles

1. **Single Responsibility Principle**: Each class has one clear, well-defined responsibility
2. **Modular Design**: Clear separation between core, server, client, and node components
3. **Reliability**: Robust error handling with graceful degradation
4. **Maintainability**: Clean, readable code with comprehensive logging
5. **Testability**: Clear interfaces and dependency injection for testing

### Code Structure

**Core Components:**
```
src/
├── mgpu_core/                 # ✅ Shared foundation components
│   ├── models/               # Data models and message types
│   ├── network/              # Network communication utilities
│   └── utils/                # System utilities and helpers
├── mgpu_server/              # ✅ Master server components  
│   ├── job_scheduler.py      # Job queue and scheduling logic
│   ├── node_manager.py       # Node registration and health monitoring
│   └── master_server.py      # Client request handling
├── mgpu_client/              # ✅ Client interface components
│   └── job_client.py         # Job submission and monitoring
├── mgpu_node/                # ✅ Worker node components
│   └── node_agent.py         # Job execution and process management
└── Entry Points:             # ✅ Command-line interfaces
    ├── mgpu_master.py        # Master server entry point
    ├── mgpu_client.py        # Client interface entry point
    ├── mgpu_node.py          # Node agent entry point
    ├── mgpu_queue.py         # Queue management entry point
    └── mgpu_cancel.py        # Job cancellation entry point
```

### Building and Packaging

**Create standalone executables:**
```bash
# Build all components
./build.sh

# Build specific components
python -m PyInstaller mgpu_master.spec
python -m PyInstaller mgpu_client.spec  
python -m PyInstaller mgpu_node.spec
```

**Built executables will be available in:**
```
build/mgpu_master/mgpu_master
build/mgpu_client/mgpu_client
build/mgpu_node/mgpu_node
```

### Testing

```bash
# Run comprehensive test suite
python test/run_tests.py

# Test individual components
python test/test_torch_load.py
python test/test_gpu_functionality.py
python test/test_integration.py

# Test through the scheduler (recommended integration test)
# 1. Start master and node agents first, then:
python src/mgpu_client.py submit --gpus 1 --interactive "python test/test_torch_load.py"
```

### Testing Checklist

**Basic Functionality:**
1. ✅ Master server starts and binds correctly
2. ✅ Node agents connect and register with master
3. ✅ Jobs submit successfully and execute on appropriate nodes
4. ✅ Interactive and non-interactive modes work correctly
5. ✅ Job cancellation properly cleans up processes
6. ✅ Queue status shows accurate information
7. ✅ Multi-node job distribution works correctly

**Process Management:**
1. ✅ Jobs run in separate process groups
2. ✅ Process tree cleanup on cancellation
3. ✅ No orphaned or zombie processes
4. ✅ Graceful termination with SIGTERM/SIGKILL progression

**Network and Timeout:**
1. ✅ Unlimited execution time when no timeouts specified
2. ✅ Custom timeout settings work correctly
3. ✅ Network reconnection and error handling
4. ✅ Multi-node communication and fault tolerance

## Troubleshooting

### Multi-node Connection Diagnostics

**Step-by-step troubleshooting for node connection issues:**

#### 1. Master Server Status Check (on master node)
```bash
# Check master server process
ps aux | grep mgpu_master

# Check port binding - should bind to 0.0.0.0:8080 for external connections
netstat -tlnp | grep 8080
# or
ss -tlnp | grep 8080

# Expected output: tcp 0 0 0.0.0.0:8080 0.0.0.0:* LISTEN
# If only 127.0.0.1:8080, external connections will fail
```

#### 2. Network Connectivity Test (from worker nodes)
```bash
# Test master node reachability
ping -c 3 192.168.1.100

# Test port connectivity
telnet 192.168.1.100 8080
# or
nc -v 192.168.1.100 8080

# Success: "Connected to 192.168.1.100"
# Failure: "Connection refused" / "Connection timed out"
```

#### 3. Firewall Configuration
**On master node:**
```bash
# Ubuntu/Debian
sudo ufw status
sudo ufw allow 8080

# CentOS/RHEL/Rocky Linux
sudo firewall-cmd --list-all
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Temporary disable for testing (be careful!)
sudo ufw disable  # Ubuntu
sudo systemctl stop firewalld  # CentOS
```

#### 4. Restart Master Server with Correct Binding
```bash
# Incorrect (only local connections)
python src/mgpu_master.py --host 127.0.0.1 --port 8080

# Correct (all network interfaces)
python src/mgpu_master.py --host 0.0.0.0 --port 8080
```

#### 5. Node Agent Connection Test
**On worker nodes:**
```bash
# Start node agent with verbose logging
python src/mgpu_node.py \
  --master-host 192.168.1.100 \
  --master-port 8080 \
  --node-id worker01 \
  --verbose

# Success: "Node registration successful"
# Failure: "Connection refused" or "Connection timeout"
```

#### 6. Network Routing and Path Verification
```bash
# Check routing table
ip route

# Trace route to master
traceroute 192.168.1.100

# Check network interfaces
ip addr show

# Test with different master IP/hostname
python src/mgpu_node.py --master-host master.example.com --master-port 8080 --node-id worker01
```

#### 7. Port Scanning for Service Verification
```bash
# Scan master node ports
nmap -p 8080 192.168.1.100

# Expected outputs:
# 8080/tcp open  http-proxy  (service running and accessible)
# 8080/tcp filtered http-proxy  (firewall blocking)
# 8080/tcp closed http-proxy  (service not running)
```

#### 8. Log Analysis and Error Messages
```bash
# Master server logs (in terminal output):
# Success: "Node worker01 connected from 192.168.1.101"
# Success: "GPU detection: 2 GPU(s) found on worker01"

# Node agent logs (in terminal output):
# Success: "Connected to master server at 192.168.1.100:8080"
# Success: "Node registration successful"

# Common error messages:
# "Connection refused": Master server not running or wrong port
# "No route to host": Network connectivity issues  
# "Connection timed out": Firewall blocking or network delays
# "Registration failed": Connected but master rejected registration
```

#### 9. Complete Multi-node Setup Verification
```bash
# Step 1: Start master server (bind to all interfaces)
python src/mgpu_master.py --host 0.0.0.0 --port 8080

# Step 2: Configure firewall (on master node)
sudo ufw allow 8080

# Step 3: Test connectivity (from worker node)
telnet 192.168.1.100 8080

# Step 4: Start node agent (on worker node)
python src/mgpu_node.py --master-host 192.168.1.100 --master-port 8080 --node-id worker01

# Step 5: Verify cluster status (from any machine)
python src/mgpu_client.py --host 192.168.1.100 --port 8080 queue
```

#### 10. Common Error Messages and Solutions
```bash
# "Connection refused"
→ Master server not running or incorrect binding
→ Solution: Restart master with --host 0.0.0.0

# "No route to host"  
→ Network configuration issues or wrong IP address
→ Solution: Check ping connectivity and network settings

# "Connection timed out"
→ Firewall blocking the port
→ Solution: Allow port 8080 in firewall rules

# "Name or service not known"
→ DNS hostname resolution failure
→ Solution: Use IP address directly or configure /etc/hosts

# "Registration failed"
→ Master server rejects node registration
→ Solution: Check master server logs for specific error details
```

### Common Issues and Solutions

1. **Server won't start**
   - Check if port 8080 is available: `netstat -tlnp | grep 8080`
   - Verify Python environment: `which python`
   - Check dependencies: `pip list | grep -E "(yaml|psutil)"`

2. **Jobs not executing**
   - Verify GPU availability: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Check node registration: `python src/mgpu_client.py queue`
   - Review node agent logs for errors

3. **Process cleanup issues**
   - Verify psutil installation: `pip install psutil`
   - Check system permissions for process signals
   - Review job cancellation logs for cleanup details

4. **Environment compatibility**
   - Ensure consistent Python versions across nodes
   - Verify required packages on all nodes: `pip list`
   - Check CUDA driver compatibility on GPU nodes

### Debug Mode

Enable detailed logging for troubleshooting:
```bash
# Master server with verbose output
python src/mgpu_master.py --host 0.0.0.0 --port 8080 --verbose

# Node agent with verbose output  
python src/mgpu_node.py --master-host IP --master-port 8080 --node-id nodeXXX --verbose

# Client with verbose output
python src/mgpu_client.py --verbose submit --gpus 1 "python script.py"
```

## Migration Guide

### From Legacy v1.0 System

If you're migrating from a previous version:

1. **Update entry points**: Use new modular entry points (`mgpu_master.py`, `mgpu_client.py`, `mgpu_node.py`)
2. **Review timeout behavior**: New system supports unlimited execution by default
3. **Update job scripts**: No changes needed for job scripts themselves
4. **Test with simple jobs**: Verify functionality before production use

### Configuration Changes

- **Simplified command structure**: Cleaner argument parsing
- **Modular architecture**: Components are now clearly separated
- **Enhanced process management**: Improved job cleanup and termination
- **Flexible timeout handling**: Support for unlimited execution time

## Contributing

### Development Guidelines

1. **Follow Single Responsibility Principle**: Each class should have one clear purpose
2. **Maintain modular structure**: Keep core, server, client, and node components separate
3. **Write comprehensive tests**: Add tests for new features and bug fixes
4. **Update documentation**: Keep README and code comments current
5. **Use consistent coding style**: Follow existing patterns and conventions

### Code Review Process

1. **Test thoroughly**: Verify single-node and multi-node functionality
2. **Check process management**: Ensure proper cleanup of job processes
3. **Validate timeout handling**: Test unlimited and limited timeout scenarios
4. **Review error handling**: Ensure graceful failure recovery
5. **Update tests**: Add or modify tests as needed

## Version History

### v2.0.0 - Modular Architecture
- **Complete modular rewrite** following Single Responsibility Principle
- **Enhanced process management** with proper process tree cleanup
- **Flexible timeout handling** with unlimited execution support
- **Improved error handling** and logging throughout the system
- **Clean separation** of core, server, client, and node components
- **Robust multi-node support** with automatic load balancing

### v1.0.0 - Legacy System
- Initial implementation with basic GPU scheduling
- Simple architecture suitable for small-scale deployments
- Limited process management and error handling

## License

[Specify your license here]

## Support

For issues and support:

1. **Check troubleshooting section** in this README
2. **Review system logs** for detailed error messages
3. **Test with verbose logging** enabled for debugging
4. **Verify network connectivity** for multi-node setups
5. **Check process management** for job cleanup issues

### Reporting Issues

When reporting issues, please include:

- **System configuration**: OS, Python version, GPU setup
- **Network setup**: Single-node vs multi-node configuration
- **Error messages**: Complete error logs with timestamps
- **Reproduction steps**: Clear steps to reproduce the issue
- **Expected vs actual behavior**: What should happen vs what actually happens

### Getting Help

For questions and assistance:

- **Documentation**: Review this README thoroughly
- **Test cases**: Check the test directory for examples
- **Verbose logging**: Enable detailed logging for troubleshooting
- **Community**: Engage with other users for tips and best practices

---

**Note**: This modular v2.0 system provides a solid, maintainable foundation for GPU resource management. The architecture supports both simple single-node setups and complex multi-node clusters while maintaining clean, readable code that's easy to debug and extend.
