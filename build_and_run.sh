#!/bin/bash
# Build and run script for Python server/client binaries on Ubuntu 22.x
# Usage: ./build_and_run.sh

set -e

# 1. Install pyinstaller if needed
if ! command -v pyinstaller &> /dev/null; then
  echo "[INFO] pyinstaller is not installed. Installing now."
  pip install pyinstaller
fi

# 2. Build binaries
echo "[INFO] Building single-node components..."
for f in src/mgpu_scheduler_server.py src/mgpu_srun.py src/mgpu_queue.py src/mgpu_cancel.py; do
  if [ -f "$f" ]; then
    echo "[INFO] Building: $f"
    # Add explicit hidden imports for dependencies
    if [ "$f" = "src/mgpu_scheduler_server.py" ]; then
      pyinstaller --onefile --hidden-import=psutil --hidden-import=select "$f"
    else
      pyinstaller --onefile "$f"
    fi
  fi
done

echo "[INFO] Building multi-node components..."
for f in src/mgpu_master_server.py src/mgpu_node_agent.py src/mgpu_srun_multinode.py; do
  if [ -f "$f" ]; then
    echo "[INFO] Building: $f"
    # Multi-node components need additional dependencies
    if [ "$f" = "src/mgpu_master_server.py" ] || [ "$f" = "src/mgpu_node_agent.py" ]; then
      pyinstaller --onefile --hidden-import=psutil --hidden-import=yaml --hidden-import=dataclasses "$f"
    else
      pyinstaller --onefile "$f"
    fi
  fi
done

# 3. Run instructions
cat <<EOF

[INFO] Build complete! Executables are generated in the dist/ directory.

=== Single Node Mode ===
Run server:
  sudo ./dist/mgpu_scheduler_server

Client examples:
  ./dist/mgpu_srun --gpu-ids 0,1 -- python train.py
  ./dist/mgpu_queue
  ./dist/mgpu_cancel <job_id>

=== Multi-Node Mode ===
Run master server:
  sudo ./dist/mgpu_master_server --config cluster_config.yaml

Run node agent (on each compute node):
  sudo ./dist/mgpu_node_agent --node-id node001 --master-host <MASTER_IP> --master-port 8080

Multi-node client examples:
  ./dist/mgpu_srun_multinode --nodes 2 --gpus-per-node 4 --distributed -- torchrun train.py
  ./dist/mgpu_srun_multinode --nodelist node001,node002 --gpus-per-node 2 -- python train.py

EOF

