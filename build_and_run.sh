#!/bin/bash
# Build and run script for Python server/client binaries on Ubuntu 22.x environment
# Usage: ./build_and_run.sh

set -e

# 1. Install pyinstaller (if needed)
if ! command -v pyinstaller &> /dev/null; then
    echo "[INFO] pyinstaller is not installed. Proceeding with installation."
    pip install pyinstaller
fi

# 2. Build binaries
for f in src/mgpu_scheduler_server.py src/mgpu_srun.py src/mgpu_queue.py src/mgpu_cancel.py; do
    if [ -f "$f" ]; then
        echo "[INFO] Building: $f"
        # Get the base name without path and extension for spec file
        basename=$(basename "$f" .py)
        spec_file="build-config/${basename}.spec"
        
        # Always build directly from source (spec files can have path issues)
        # Add explicit hidden imports for dependencies
        if [ "$f" = "src/mgpu_scheduler_server.py" ]; then
            pyinstaller --onefile --hidden-import=psutil --hidden-import=select "$f"
        else
            pyinstaller --onefile "$f"
        fi
    fi
done

# 3. Execution guide
cat <<EOF

[INFO] Build completed! Executable files are created in the dist/ directory.

Server execution:
  ./dist/mgpu_scheduler_server

Client examples:
  ./dist/mgpu_srun --gpus 1 --mem 8000 -- echo hello
  ./dist/mgpu_queue
  ./dist/mgpu_cancel <job_id>

EOF
