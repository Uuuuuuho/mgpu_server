# Multi-GPU Scheduler

## Overview

This project provides a multi-user, multi-GPU job scheduling system for efficient and fair GPU resource sharing across **single or multiple nodes**. It offers a Slurm-like CLI and supports resource-aware scheduling, queueing, user/job cancellation, per-job and global time limits, flexible environment setup, distributed multi-node execution, and **real-time interactive job execution**.

## Project Structure

```
multigpu_scheduler/
├── src/                     # Source code
│   ├── mgpu_master_server.py     # Multi-node master server with interactive support
│   ├── mgpu_node_agent.py        # Node agent with interactive job streaming
│   ├── mgpu_srun_multinode.py    # Unified job submission client with interactive mode
│   ├── mgpu_queue.py            # Queue status viewer
│   └── mgpu_cancel.py           # Job cancellation tool
├── test/                    # Test scripts and utilities
├── build-config/            # PyInstaller configuration files
├── docs/                    # Documentation
├── dist/                    # Built binaries (after build)
├── cluster_config.yaml      # Multi-node cluster configuration
├── cluster_config_localhost.yaml # Single-node testing configuration
├── build_and_run.sh         # Build script
├── Makefile                 # Build automation
└── README.md               # This file
```

---

## Features

### Core Features
- **Multi-user, multi-node support**: Handle multiple users submitting jobs across multiple compute nodes
- **Slurm-style CLI**: Familiar interface with `mgpu_srun_multinode`, `mgpu_queue`, `mgpu_cancel`
- **Interactive job execution**: Real-time output streaming for debugging and monitoring
- **Specific GPU allocation**: Users can specify exact GPU IDs on specific nodes
- **Resource-aware scheduling**: Automatic allocation based on GPU availability and memory requirements
- **Fair scheduling**: Jobs are queued fairly and started automatically when resources become available

### Advanced Features
- **Node-specific GPU mapping**: `--node-gpu-ids node001:0,1;node002:2,3` for precise control
- **Distributed training support**: Built-in support for PyTorch distributed and MPI execution
- **Real-time monitoring**: Live output streaming for interactive jobs
- **Automatic environment setup**: CUDA_VISIBLE_DEVICES and distributed training variables
- **Per-job and global time limits**: Prevent resource hogging with configurable timeouts
- **MPI-based distributed execution** support
- **Node-specific job placement** (--nodelist, --exclude options)
- **Cluster-wide resource monitoring** and load balancing
- **Fault tolerance** with node failure detection and recovery
- **Network topology-aware scheduling** for optimal performance

---

## Installation & Build

### For Production Use
1. Python 3.8 or higher required
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```
3. Build binaries:
   ```bash
   ./build_and_run.sh
   # OR using Makefile
   make build
   ```
   Executables will be created in the `dist/` directory.

4. Optional: Clean build artifacts
   ```bash
   make clean        # Remove build/dist directories
   make clean-all    # Also remove spec files
   make regenerate-specs  # Regenerate PyInstaller spec files
   ```

### For Development
To run directly from source code:
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python src/mgpu_scheduler_server.py

# Submit jobs (in another terminal)
python src/mgpu_srun.py --gpu-ids 0 -- python test/test_output.py

# Check queue status
python src/mgpu_queue.py

# Cancel a job
python src/mgpu_cancel.py <job_id>
```

---

## Quick Start

### 1. Start Master Server
```bash
# Start the master server (handles job scheduling)
python src/mgpu_master_server.py --config cluster_config_localhost.yaml --port 8080
```

### 2. Start Node Agent
```bash
# Start node agent (handles job execution)
python src/mgpu_node_agent.py --node-id node001 --gpu-count 1
```

### 3. Submit Jobs
```bash
# Submit a simple job
python src/mgpu_srun_multinode.py submit "echo 'Hello GPU World'"

# Submit an interactive job (see real-time output)
python src/mgpu_srun_multinode.py submit --interactive "python train.py"

# Submit job with specific GPU allocation
python src/mgpu_srun_multinode.py submit --interactive --node-gpu-ids "node001:0" "python gpu_test.py"

# Check queue status
python src/mgpu_queue.py

# Cancel a job
python src/mgpu_cancel.py JOB_ID
```

---

## Usage Examples

### Interactive Jobs (Real-time Output)
```bash
# Run with real-time output streaming
python src/mgpu_srun_multinode.py submit --interactive -- python train.py

# Multi-GPU training with specific GPU assignment
python src/mgpu_srun_multinode.py submit --interactive --node-gpu-ids "node001:0,1" -- \
  python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### Multi-Node Distributed Training
```bash
# PyTorch distributed training across multiple nodes
python src/mgpu_srun_multinode.py submit --nodes 2 --gpus-per-node 4 --distributed -- \
  torchrun --nnodes=2 --nproc_per_node=4 --master_addr=\$MASTER_ADDR --master_port=29500 train.py

# MPI distributed execution
python src/mgpu_srun_multinode.py submit --nodes 4 --gpus-per-node 2 --mpi -- \
  mpirun -np 8 python mpi_train.py
```

### Specific Node and GPU Selection
```bash
# Use specific nodes and GPUs
python src/mgpu_srun_multinode.py submit --node-gpu-ids "node001:0,1;node002:2,3" -- python train.py

# Exclude certain nodes
python src/mgpu_srun_multinode.py submit --nodes 2 --exclude node003,node004 -- python train.py
```

# Single node with specific GPUs (same as single-node mode)
./dist/mgpu_srun_multinode --gpu-ids 0,1,2,3 -- python train.py

# Specify exact GPUs on specific nodes (NEW)
./dist/mgpu_srun_multinode --node-gpu-ids "node001:0,1,2;node002:1,3,4" -- python train.py

# Mix with distributed training - exact GPU control
./dist/mgpu_srun_multinode --node-gpu-ids "node001:0,1;node002:2,3" --distributed -- \
  torchrun --nnodes=2 --nproc_per_node=2 --master_addr=\$MASTER_ADDR --master_port=29500 train.py
```

### Environment Variables for Multi-Node
```bash
export MGPU_MASTER_HOST=master.example.com
export MGPU_MASTER_PORT=8080
```

---

## Command Line Options

### mgpu_srun_multinode (Multi-Node Job Submission)

#### Resource Allocation Options
- `--gpu-ids`: Comma-separated GPU IDs for single-node jobs (e.g., `0,1,2`)
- `--nodes`: Number of nodes to allocate for multi-node jobs
- `--gpus-per-node`: Number of GPUs per node (default: 1)
- `--nodelist`: Comma-separated list of specific node IDs to use
- `--exclude`: Comma-separated list of node IDs to exclude
- `--node-gpu-ids`: **NEW** - Specify exact GPUs on specific nodes using format `node1:gpu1,gpu2;node2:gpu3,gpu4`

#### Job Control Options
- `--mem`: Memory requirement per GPU (MB)
- `--time-limit`: Job time limit (seconds)
- `--priority`: Job priority (higher = sooner, default: 0)
- `--env-setup-cmd`: Environment setup command to run before job
- `--interactive`: Run job interactively (stream output in real-time)
- `--background`: Submit job and exit immediately

#### Distributed Execution Options
- `--mpi`: Use MPI for distributed execution
- `--distributed`: Use PyTorch distributed execution

#### Connection Options
- `--master-host`: Master server hostname (default: localhost or $MGPU_MASTER_HOST)
- `--master-port`: Master server port (default: 8080 or $MGPU_MASTER_PORT)

### Examples of --node-gpu-ids Usage

```bash
# Use GPUs 0,1 on node001 and GPUs 2,3 on node002
mgpu_srun_multinode --node-gpu-ids "node001:0,1;node002:2,3" -- python train.py

# Use specific GPUs on three nodes for large distributed training
mgpu_srun_multinode --node-gpu-ids "node001:0,1,2,3;node002:0,1,2,3;node003:4,5,6,7" --distributed -- \
  torchrun --nnodes=3 --nproc_per_node=4 --master_addr=\$MASTER_ADDR --master_port=29500 train.py

# Mix different GPU counts per node
mgpu_srun_multinode --node-gpu-ids "gpu-server-1:0,1;gpu-server-2:3,4,5" -- python inference.py
```

**Note**: When using `--node-gpu-ids`, the `--nodes`, `--gpus-per-node`, and `--nodelist` options are ignored as the node-GPU mapping is explicitly specified.

---

## Interactive vs Background Mode
- **Interactive mode** (`--interactive`): See job output in real-time on your terminal. You can detach with Ctrl+C while keeping the job running.
- **Background mode** (default): Job runs silently. Check status with `mgpu_queue` or view logs from job output files.

---

## How It Works & Policies
- The server communicates with clients via UNIX domain socket (`/tmp/mgpu_scheduler.sock`)
- Jobs are executed in the user's home directory with proper user privileges
- Users specify exact GPU IDs they want to use (e.g., `--gpu-ids 0,2,3`)
- `CUDA_VISIBLE_DEVICES` is automatically set in the command environment to match requested GPUs
- Inside jobs, PyTorch/CUDA will see GPUs renumbered as `cuda:0`, `cuda:1`, etc., mapped to the physical GPUs requested
- Jobs are only started if there is enough GPU memory available on the requested GPUs
- If resources are insufficient, jobs wait in the queue and are started automatically when resources are freed
- Context switch (move to end of queue) occurs if per-job or global time limit is exceeded
- Optional environment setup commands are executed before the main job command

---

## GPU ID Mapping

### Single-Node Jobs
When you specify `--gpu-ids 1,3`, your job will:
1. Set `CUDA_VISIBLE_DEVICES=1,3` in the environment
2. Inside your job, `cuda:0` maps to physical GPU 1, `cuda:1` maps to physical GPU 3
3. Use `cuda:0`, `cuda:1`, etc. in your training scripts (not the physical GPU IDs)

### Multi-Node Jobs with --node-gpu-ids
When you specify `--node-gpu-ids "node001:1,3;node002:0,2"`:
1. On node001: `CUDA_VISIBLE_DEVICES=1,3` → your job sees `cuda:0` (physical GPU 1), `cuda:1` (physical GPU 3)
2. On node002: `CUDA_VISIBLE_DEVICES=0,2` → your job sees `cuda:0` (physical GPU 0), `cuda:1` (physical GPU 2)
3. Each node processes see their own local GPU indices starting from 0

### Important Notes
- Always use local GPU IDs (`cuda:0`, `cuda:1`, etc.) in your training scripts
- The scheduler automatically handles physical GPU mapping and sets `CUDA_VISIBLE_DEVICES`
- For PyTorch distributed training, use `--nproc_per_node` matching the number of GPUs assigned to each node

---

## Notes and Cautions
- The server must be run as root. User jobs are executed with the target user's privileges using setuid/setgid (no sudo required)
- Always use local GPU IDs (`cuda:0`, `cuda:1`, etc.) in your training scripts, not physical GPU IDs
- Environment setup commands (if provided) are executed before setting CUDA_VISIBLE_DEVICES and running the job
- For distributed training with torchrun, specify the number of processes that matches your GPU count

---

## Troubleshooting

### "No module named 'psutil'" Error
If you get this error when running the server binary:

1. **Rebuild with explicit dependencies**:
   ```bash
   # Make sure you're in your virtual environment
   source venv/bin/activate
   pip install psutil pyinstaller
   ./build_and_run.sh
   ```

2. **Alternative: Run from source**:
   ```bash
   # Instead of using the binary, run directly from Python
   python mgpu_scheduler_server.py --max-job-time 3600
   ```

3. **Check dependencies**:
   ```bash
   # Verify psutil is installed in your environment
   python -c "import psutil; print('psutil version:', psutil.__version__)"
   ```

### Permission Issues
- Ensure the server is run as root: `sudo ./dist/mgpu_scheduler_server`
- Make sure binary files are executable: `chmod +x dist/*`

### GPU Detection Issues
- Verify nvidia-smi works: `nvidia-smi`
- Check CUDA installation and GPU visibility

---

## License
MIT
