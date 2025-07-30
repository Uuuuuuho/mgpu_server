#!/bin/bash
# Build and run script for Python server/client binaries on Ubuntu 22.x environment
# Usage: ./build.sh

set -e

# 1. Install pyinstaller (if needed)
if ! command -v pyinstaller &> /dev/null; then
    echo "[INFO] pyinstaller is not installed. Proceeding with installation."
    pip install pyinstaller
fi

# 2. Build new modular binaries
echo "[INFO] Building modular Multi-GPU Scheduler components..."

# List of files to build with their target names
declare -A build_files=(
    ["src/mgpu_master.py"]="mgpu_master"
    ["src/mgpu_node.py"]="mgpu_node"
    ["src/mgpu_client.py"]="mgpu_client"
    ["src/mgpu_queue.py"]="mgpu_queue"
    ["src/mgpu_cancel.py"]="mgpu_cancel"
    ["src/mgpu_srun.py"]="mgpu_srun"
)

# Build each component
for file in "${!build_files[@]}"; do
    if [ -f "$file" ]; then
        target_name="${build_files[$file]}"
        echo "[INFO] Building: $file -> $target_name"
        
        # Add explicit hidden imports for dependencies
        if [[ "$file" == *"master"* ]]; then
            pyinstaller --onefile --name="$target_name" \
                --hidden-import=mgpu_core \
                --hidden-import=mgpu_server \
                --hidden-import=psutil \
                --hidden-import=select \
                "$file"
        elif [[ "$file" == *"node"* ]]; then
            pyinstaller --onefile --name="$target_name" \
                --hidden-import=mgpu_core \
                --hidden-import=mgpu_node \
                --hidden-import=psutil \
                "$file"
        elif [[ "$file" == *"client"* ]]; then
            pyinstaller --onefile --name="$target_name" \
                --hidden-import=mgpu_core \
                --hidden-import=mgpu_client \
                "$file"
        else
            pyinstaller --onefile --name="$target_name" "$file"
        fi
    else
        echo "[WARNING] File not found: $file"
    fi
done

# 3. Execution guide
cat <<EOF

[INFO] Build completed! Executable files are created in the dist/ directory.

=== NEW MODULAR ARCHITECTURE ===

Master Server:
  ./dist/mgpu_master --host 0.0.0.0 --port 8080

Node Agent:
  ./dist/mgpu_node --node-id node001 --master-host 127.0.0.1 --gpu-count 1

Client Commands:
  ./dist/mgpu_client submit --gpus 1 "echo hello"
  ./dist/mgpu_client submit --interactive --gpus 1 "nvidia-smi"
  ./dist/mgpu_client submit --node-gpu-ids "node001:0" "echo hello"
  ./dist/mgpu_client queue
  ./dist/mgpu_client cancel <job_id>
  ./dist/mgpu_client monitor <job_id>

Legacy Commands (also available):
  ./dist/mgpu_queue
  ./dist/mgpu_cancel <job_id>
  ./dist/mgpu_srun --gpus 1 "echo hello"

=== DIRECTORY STRUCTURE ===
src/
├── mgpu_core/           # Core shared components
│   ├── models/          # Data models (SimpleJob, NodeInfo, etc.)
│   ├── network/         # Network utilities
│   └── utils/           # System utilities, logging
├── mgpu_server/         # Master server components
│   ├── job_scheduler.py # Job scheduling logic
│   ├── node_manager.py  # Node registration and management
│   └── master_server.py # Main server class
├── mgpu_client/         # Client components
│   └── job_client.py    # Job submission and monitoring
├── mgpu_node/           # Node agent components
│   └── node_agent.py    # Node job execution
├── mgpu_master.py       # Master server entry point
├── mgpu_node.py         # Node agent entry point
└── mgpu_client.py       # Client entry point

=== FEATURES ===
- Single Responsibility Principle - Each class has one purpose
- Modular Architecture - Clean separation of concerns  
- Automatic IP Detection - Nodes register with actual IPs
- Enhanced Debugging - Detailed execution tracking
- Improved Error Handling - Robust network communication
- Timeout Management - Configurable timeouts for all operations

EOF
