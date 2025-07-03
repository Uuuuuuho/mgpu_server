# Multi-GPU Scheduler

## Overview

This project provides a multi-user, multi-GPU job scheduling system for efficient and fair GPU resource sharing. It offers a Slurm-like CLI (`mgpu_srun`, `mgpu_queue`, `mgpu_cancel`) and supports resource-aware scheduling, queueing, user/job cancellation, per-job and global time limits, and automatic Python venv activation.

---

## Features
- Multiple users can submit jobs concurrently
- Slurm-style CLI: `mgpu_srun` (job submission), `mgpu_queue` (queue/status), `mgpu_cancel` (job cancellation)
- Resource allocation based on requested GPU count and memory
- Fair scheduling: jobs are queued if resources are insufficient, and started automatically when available
- Per-job and global time limits, with context switching (move to end of queue)
- Unique job IDs, with queue/running status and cancellation support
- Automatic activation of Python venv in user's home directory (if present)
- Robust logging: job logs are saved to the server user's home directory

---

## Installation & Build

1. Python 3.8 or higher required
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```
3. Build binaries:
   ```bash
   ./build_and_run.sh
   ```
   Executables will be created in the `dist/` directory.

---

## Usage

### Start the Server (as root)
```bash
sudo ./dist/mgpu_scheduler_server --max-job-time 3600
```
- `--max-job-time`: (optional) Maximum allowed time per job (seconds)

### Client Examples
```bash
# Submit a job (1 GPU, 8000MB memory, 600s time limit)
./dist/mgpu_srun --gpus 1 --mem 8000 --time-limit 600 -- python train.py

# Submit a job (without memory option, venv auto-activation if present)
./dist/mgpu_srun --gpus 1 -- python train.py

# View queue and running jobs
./dist/mgpu_queue

# Cancel a job
./dist/mgpu_cancel <job_id>
```

---

## How It Works & Policies
- The server communicates with clients via UNIX domain socket (`/tmp/mgpu_scheduler.sock`)
- When running a job, it is executed in the user's home directory, and if `venv/bin/activate` exists, it is automatically activated
- Jobs are only started if there is enough GPU memory available, considering running jobs
- If resources are insufficient, jobs wait in the queue and are started automatically when resources are freed
- Context switch (move to end of queue) occurs if per-job or global time limit is exceeded

---

## Notes and Cautions
- The server must be run as root. User jobs are executed with the target user's privileges using setuid/setgid (no sudo required)
- Each user can create a Python venv in their home directory for automatic activation
- To support automatic conda activation or other environments, modify the code as needed

---

## License
MIT
