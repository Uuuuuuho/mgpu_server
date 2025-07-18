# Multi-GPU Scheduler

## Overview

This project provides a multi-user, multi-GPU job scheduling system for efficient and fair GPU resource sharing. It offers a Slurm-like CLI (`mgpu_srun`, `mgpu_queue`, `mgpu_cancel`) and supports resource-aware scheduling, queueing, user/job cancellation, per-job and global time limits, and flexible environment setup.

## Project Structure

```
multigpu_scheduler/
├── src/                     # Source code
│   ├── mgpu_scheduler_server.py  # Main scheduler server
│   ├── mgpu_srun.py             # Job submission client
│   ├── mgpu_queue.py            # Queue status viewer
│   └── mgpu_cancel.py           # Job cancellation tool
├── test/                    # Test scripts
│   ├── test_output.py           # Output streaming test
│   ├── test_cancellation.py    # Job cancellation test
│   └── README.md               # Test documentation
├── build-config/            # PyInstaller configuration files
│   ├── mgpu_scheduler_server.spec
│   ├── mgpu_srun.spec
│   ├── mgpu_queue.spec
│   └── mgpu_cancel.spec
├── dist/                    # Built binaries (after build)
├── build_and_run.sh         # Build script
├── Makefile                 # Build automation
└── README.md               # This file
```

---

## Features
- Multiple users can submit jobs concurrently
- Slurm-style CLI: `mgpu_srun` (job submission), `mgpu_queue` (queue/status), `mgpu_cancel` (job cancellation)
- Specific GPU selection: users can specify exact GPU IDs to use
- Resource allocation based on GPU memory requirements
- Fair scheduling: jobs are queued if resources are insufficient, and started automatically when available
- Per-job and global time limits, with context switching (move to end of queue)
- Unique job IDs, with queue/running status and cancellation support
- Flexible environment setup: users can specify custom environment activation commands
- Robust logging: job logs are saved to the server user's home directory
- Proper CUDA_VISIBLE_DEVICES handling for distributed training (torchrun, etc.)

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

### Start the Server (as root)
```bash
sudo ./dist/mgpu_scheduler_server --max-job-time 3600
```
- `--max-job-time`: (optional) Maximum allowed time per job (seconds)

### Client Examples
```bash
# Submit an interactive job (see output in real-time)
./dist/mgpu_srun --gpu-ids 0 --interactive -- python train.py

# Submit a background job (runs silently)
./dist/mgpu_srun --gpu-ids 0,1 --mem 8000 --time-limit 600 -- torchrun --nproc_per_node=2 train.py

# Submit a job with custom environment setup
./dist/mgpu_srun --gpu-ids 2 --env-setup-cmd "source venv/bin/activate" -- python train.py

# Submit a job with conda environment activation
./dist/mgpu_srun --gpu-ids 1 --env-setup-cmd "conda activate myenv" -- python inference.py

# Submit a job with priority (higher number = higher priority)
./dist/mgpu_srun --gpu-ids 0 --priority 10 -- python important_task.py

# View queue and running jobs
./dist/mgpu_queue

# Cancel a job
./dist/mgpu_cancel <job_id>
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
