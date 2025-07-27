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

### Basic Usage

**Currently working with the simple implementation:**

1. **Start the simple master server:**
```bash
python src/mgpu_simple_master.py --host localhost --port 8080
```

2. **Submit a job (simple client):**
```bash
python src/mgpu_simple_client.py submit --gpus 1 "python your_script.py"
```

**Job submission with timeout options:**
```bash
# Interactive job with custom timeouts
python src/mgpu_simple_client.py submit --interactive --gpus 1 \
  --session-timeout 3600 \
  --max-consecutive-timeouts 60 \
  "python long_running_script.py"

# Non-interactive job with custom timeouts
python src/mgpu_simple_client.py submit --gpus 1 \
  --max-wait-time 600 \
  --connection-timeout 15 \
  "python batch_job.py"
```

3. **Check queue status:**
```bash
python src/mgpu_simple_client.py queue
```

4. **Cancel a job:**
```bash
python src/mgpu_simple_client.py cancel <job_id>
```

### Multi-node Setup (Manual Node Connection)

**How multiple nodes connect to the master server:**

The current working implementation (`mgpu_simple_*`) supports multi-node setups through manual node agent deployment:

1. **Start master server on the head node:**
```bash
python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

2. **On each worker node, connect to the master:**
```bash
# Replace <MASTER_IP> with actual master server IP
python src/mgpu_simple_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node001
python src/mgpu_simple_node.py --master-host <MASTER_IP> --master-port 8080 --node-id node002
# ... repeat for each worker node
```

3. **Submit jobs from any client:**
```bash
# Client can run from any machine that can reach the master
python src/mgpu_simple_client.py --host <MASTER_IP> --port 8080 submit --gpus 2 "python distributed_script.py"
```

**Key Points:**
- **Node agents must be started manually** on each worker machine
- **Master server coordinates** all GPU resources across connected nodes
- **Automatic load balancing** - jobs are assigned to available GPUs across all connected nodes
- **Network accessibility** - all nodes must be able to reach the master server's IP and port

## Configuration

### Single-node Configuration (Current)

For **localhost-only** operation (current working setup):

**No configuration file needed** - the simple master server works out of the box:

```bash
python src/mgpu_simple_master.py --host localhost --port 8080
```

The system will automatically detect available GPUs on localhost.

### Multi-node Configuration (Advanced Setup)

**For advanced users** wanting to set up multiple nodes:

1. **Start master server on main node:**
```bash
python src/mgpu_simple_master.py --host 0.0.0.0 --port 8080
```

2. **Start node agents on compute nodes:**
```bash
# On node 1
python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node001

# On node 2  
python src/mgpu_simple_node.py --master-host 192.168.1.100 --master-port 8080 --node-id node002
```

3. **Configure firewall** to allow communication on port 8080

## Usage Examples

### Job Submission (Current Working Commands)

```bash
# Simple single-GPU job
python src/mgpu_simple_client.py submit --gpus 1 "python train.py"

# Multi-GPU job on localhost
python src/mgpu_simple_client.py submit --gpus 2 "python distributed_train.py"

# Interactive job with output streaming
python src/mgpu_simple_client.py submit --gpus 1 --interactive "python interactive_script.py"

# Interactive job with custom timeouts for long-running GPU workloads
python src/mgpu_simple_client.py submit --gpus 1 --interactive \
  --session-timeout 14400 \
  --max-consecutive-timeouts 120 \
  "python gpu_training.py"

# Non-interactive job with custom monitoring timeouts
python src/mgpu_simple_client.py submit --gpus 1 \
  --max-wait-time 1800 \
  --connection-timeout 20 \
  "python batch_processing.py"
```

### Queue Management

```bash
# View all jobs and node status
python src/mgpu_simple_client.py queue

# Cancel a specific job
python src/mgpu_simple_client.py cancel <job_id>
```

### Advanced: Multi-node Jobs

**Only if you have set up node agents on multiple machines:**

```bash
# Submit job to specific node
python src/mgpu_simple_client.py submit --gpus 1 --node-gpu-ids "node001:0" "python script.py"

# Multi-node distributed job
python src/mgpu_simple_client.py submit --gpus 4 --node-gpu-ids "node001:0,1;node002:0,1" "python distributed_training.py"
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
python src/mgpu_simple_master.py [--host HOST] [--port PORT]
```

### mgpu_simple_node.py (For Multi-node Setup)

**Start a node agent:**
```bash
python src/mgpu_simple_node.py --master-host HOST --master-port PORT --node-id NODE_NAME
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
# Basic functionality test
python test/test_torch_load.py

# Run test suite
python test/run_tests.py
```

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if port 8080 is available
   - Verify configuration file syntax
   - Check Python dependencies

2. **Jobs not running**
   - Verify GPU availability
   - Check CUDA installation
   - Review server logs

3. **Connection errors**
   - Ensure master server is running
   - Check firewall settings
   - Verify network connectivity

### Debug Mode

Enable detailed logging:
```bash
python src/mgpu_master_server.py --config cluster_config.yaml --debug
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
