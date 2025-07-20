# Multi-GPU Scheduler

## Overview

This project provides a multi-user, multi-GPU job scheduling system for efficient and fair GPU resource sharing across **single or multiple nodes**. It offers a Slurm-like CLI (`mgpu_srun`, `mgpu_queue`, `mgpu_cancel`) and supports resource-aware scheduling, queueing, user/job cancellation, per-job and global time limits, flexible environment setup, and **distributed multi-node execution**.

## Project Structure

```
multigpu_scheduler/
├── src/                     # Source code
│   ├── mgpu_scheduler_server.py  # Single-node scheduler server
│   ├── mgpu_master_server.py     # Multi-node master server (NEW)
│   ├── mgpu_node_agent.py        # Node agent for multi-node (NEW)
│   ├── mgpu_srun.py             # Single-node job submission client
│   ├── mgpu_srun_multinode.py   # Multi-node job submission client (NEW)
│   ├── mgpu_queue.py            # Queue status viewer
│   └── mgpu_cancel.py           # Job cancellation tool
├── test/                    # Test scripts
│   ├── test_output.py           # Output streaming test
│   ├── test_cancellation.py    # Job cancellation test
│   └── README.md               # Test documentation
├── build-config/            # PyInstaller configuration files
├── docs/                    # Documentation
│   └── multi-node-design.md    # Multi-node architecture design
├── dist/                    # Built binaries (after build)
├── cluster_config.yaml      # Multi-node cluster configuration
├── build_and_run.sh         # Build script
├── Makefile                 # Build automation
└── README.md               # This file
```

---

## Features

### Single-Node Features
- Multiple users can submit jobs concurrently
- Slurm-style CLI: `mgpu_srun` (job submission), `mgpu_queue` (queue/status), `mgpu_cancel` (job cancellation)
- Specific GPU selection: users can specify exact GPU IDs to use
- Resource allocation based on GPU memory requirements
- Fair scheduling: jobs are queued if resources are insufficient, and started automatically when available
- Per-job and global time limits, with context switching (move to end of queue)
- Interactive mode with real-time output streaming
- Automatic job cancellation when client disconnects

### Multi-Node Features (NEW)
- **Multi-node distributed job execution** across GPU clusters
- **Automatic node selection** and resource allocation
- **PyTorch distributed training** support with automatic environment setup
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

## Usage

### Single-Node Mode

#### Start the Server (as root)
```bash
sudo ./dist/mgpu_scheduler_server --max-job-time 3600
```

#### Client Examples
```bash
# Submit an interactive job (see output in real-time)
./dist/mgpu_srun --gpu-ids 0 -- python train.py

# Submit a background job
./dist/mgpu_srun --gpu-ids 0,1 --mem 8000 --time-limit 600 --background -- python train.py
```

### Multi-Node Mode (NEW)

#### 1. Setup Cluster Configuration
```bash
# Copy and modify cluster configuration
cp cluster_config.yaml /etc/mgpu_cluster.yaml
# Edit the file with your cluster node information
```

#### 2. Start Master Server
```bash
# On the master node
sudo ./dist/mgpu_master_server --config /etc/mgpu_cluster.yaml --port 8080
```

#### 3. Start Node Agents
```bash
# On each compute node
sudo ./dist/mgpu_node_agent --node-id node001 --master-host master.example.com --master-port 8080 --agent-port 8081
```

#### 4. Submit Multi-Node Jobs
```bash
# PyTorch distributed training across 2 nodes, 4 GPUs per node
./dist/mgpu_srun_multinode --nodes 2 --gpus-per-node 4 --distributed -- \
  torchrun --nnodes=2 --nproc_per_node=4 --master_addr=\$MASTER_ADDR --master_port=29500 train.py

# MPI distributed execution
./dist/mgpu_srun_multinode --nodes 4 --gpus-per-node 2 --mpi -- \
  mpirun -np 8 python mpi_train.py

# Specific node selection
./dist/mgpu_srun_multinode --nodelist node001,node002 --gpus-per-node 2 -- python train.py

# Single node with specific GPUs (same as single-node mode)
./dist/mgpu_srun_multinode --gpu-ids 0,1,2,3 -- python train.py
```

### Environment Variables for Multi-Node
```bash
export MGPU_MASTER_HOST=master.example.com
export MGPU_MASTER_PORT=8080
```

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
When you specify `--gpu-ids 1,3`, your job will:
1. Set `CUDA_VISIBLE_DEVICES=1,3` in the environment
2. Inside your job, `cuda:0` maps to physical GPU 1, `cuda:1` maps to physical GPU 3
3. Use `cuda:0`, `cuda:1`, etc. in your training scripts (not the physical GPU IDs)

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
