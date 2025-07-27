#!/bin/bash
# Multi-Node GPU System Test Script
# Use this script to test your multi-node GPU cluster

echo "=== Multi-Node GPU System Test Script ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# Check prerequisites
echo "📋 Checking Prerequisites..."
echo "✓ Python version: $(python3 --version)"
echo "✓ Current directory: $(pwd)"

# Check GPU availability
echo
echo "🎮 Checking GPU Status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "✓ GPU: %s (%.1fGB total, %.1fGB free)\n", $1, $2/1024, $3/1024}'
else
    echo "❌ nvidia-smi not found"
fi

echo
echo "🔧 Testing Infrastructure..."

# Test 1: Single Node Multi-Node Server
echo "--- Test 1: Single Node Multi-Node Infrastructure ---"
python3 test/run_tests.py --test single_node_multinode 2>&1 | head -20

echo
echo "--- Test 2: Integration Tests ---"
python3 test/run_tests.py --test integration 2>&1 | tail -10

echo
echo "--- Test 3: GPU Functionality ---"
python3 test/run_tests.py --test gpu 2>&1 | grep -E "(✓|❌|GPU)"

echo
echo "=== Test Summary ==="
echo "For detailed results, run individual tests:"
echo "  python3 test/run_tests.py --test <test_name> --verbose"
echo
echo "Available tests:"
python3 test/run_tests.py --list

echo
echo "=== Manual Testing Commands ==="
echo "1. Start master server:"
echo "   python3 src/mgpu_master_server.py --config cluster_config_localhost.yaml"
echo
echo "2. Start node agent (in another terminal):"
echo "   python3 src/mgpu_node_agent.py --node-id node001"
echo
echo "3. Submit test job (in third terminal):"
echo "   python3 src/mgpu_srun_multinode.py --nodes 1 --gpus-per-node 1 -- python test/test_output.py"
echo
echo "4. Check queue status:"
echo "   python3 src/mgpu_queue.py"
