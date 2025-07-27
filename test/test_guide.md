# Multi-Node GPU System Testing Guide

## 🚀 Quick Start Testing

### 1. Basic System Check
```bash
cd /mnt/e/work/multigpu_scheduler
python3 test/run_tests.py --quick
```

### 2. Single-Node Multi-Node Server Test
```bash
python3 test/run_tests.py --test single_node_multinode
```

### 3. Full Integration Test
```bash
python3 test/run_tests.py --test integration
```

## 🏗️ Step-by-Step Testing Process

### Phase 1: Infrastructure Testing

#### Test 1: Single-Node Setup
```bash
# Start master server (in one terminal)
cd /mnt/e/work/multigpu_scheduler
python3 src/mgpu_master_server.py --config cluster_config_localhost.yaml

# Start node agent (in another terminal) 
python3 src/mgpu_node_agent.py --node-id node001

# Run basic tests (in third terminal)
python3 test/run_tests.py --test single_node_multinode --verbose
```

#### Test 2: Cluster Communication
```bash
# Test cluster connectivity
python3 test/run_tests.py --test cluster --verbose
```

### Phase 2: GPU Functionality Testing

#### Test 3: GPU Detection and Allocation
```bash
# Test GPU functionality
python3 test/run_tests.py --test gpu --verbose

# Test CUDA load
python3 test/run_tests.py --test torch_load --verbose
```

### Phase 3: Job Management Testing

#### Test 4: Job Submission and Execution
```bash
# Test output streaming
python3 test/run_tests.py --test streaming --verbose

# Test job cancellation
python3 test/run_tests.py --test cancellation --verbose
```

#### Test 5: Distributed Computing
```bash
# Test distributed PyTorch
python3 test/run_tests.py --test distributed --verbose

# Test MPI functionality
python3 test/run_tests.py --test mpi --verbose
```

### Phase 4: Performance and Stress Testing

#### Test 6: Performance Testing
```bash
# Run performance tests
python3 test/run_tests.py --test performance --verbose

# Test error handling
python3 test/run_tests.py --test error_handling --verbose
```

## 🔧 Manual Testing for Specific Scenarios

### Manual Test 1: Basic Job Submission
```bash
# Terminal 1: Start scheduler server
cd /mnt/e/work/multigpu_scheduler
python3 src/mgpu_scheduler_server.py

# Terminal 2: Submit a test job
python3 src/mgpu_srun.py --gpu-ids 0 -- python test/test_output.py

# Terminal 3: Check queue status
python3 src/mgpu_queue.py
```

### Manual Test 2: Multi-GPU Job
```bash
# Submit job with multiple GPUs
python3 src/mgpu_srun.py --gpu-ids 0,1 --interactive -- python test/test_distributed.py
```

### Manual Test 3: Job Cancellation
```bash
# Submit long-running job
python3 src/mgpu_srun.py --gpu-ids 0 --interactive -- python test/test_cancellation.py

# Press Ctrl+C after a few seconds to test cancellation
```

## 🌐 Multi-Node Testing (Actual Multi-Machine Setup)

### Prerequisites
- Multiple machines with network connectivity
- Same user account on all nodes
- SSH key-based authentication set up
- Shared filesystem (NFS recommended) or code synchronized

### Configuration Steps

#### Step 1: Update Cluster Configuration
Edit `cluster_config.yaml`:
```yaml
cluster:
  name: "gpu-cluster"
  master:
    host: "192.168.1.100"  # Your master node IP
    port: 8080

nodes:
  - node_id: "node001"
    hostname: "gpu-node-1"
    ip: "192.168.1.101"
    port: 8081
    gpu_count: 2
    
  - node_id: "node002"
    hostname: "gpu-node-2"  
    ip: "192.168.1.102"
    port: 8081
    gpu_count: 4
```

#### Step 2: Start Services on Each Node

**On Master Node (192.168.1.100):**
```bash
cd /path/to/multigpu_scheduler
python3 src/mgpu_master_server.py --config cluster_config.yaml
```

**On Compute Node 1 (192.168.1.101):**
```bash
cd /path/to/multigpu_scheduler
python3 src/mgpu_node_agent.py --node-id node001 --master-host 192.168.1.100
```

**On Compute Node 2 (192.168.1.102):**
```bash
cd /path/to/multigpu_scheduler
python3 src/mgpu_node_agent.py --node-id node002 --master-host 192.168.1.100
```

#### Step 3: Run Multi-Node Tests
```bash
# On any node with the scheduler client
export MGPU_MASTER_HOST=192.168.1.100
export MGPU_MASTER_PORT=8080

# Test cluster connectivity
python3 test/run_tests.py --test cluster

# Test distributed training across nodes
python3 test/run_tests.py --test distributed

# Submit multi-node job
python3 src/mgpu_srun_multinode.py --nodes 2 --gpus-per-node 2 -- python test/test_distributed.py
```

## 🧪 Troubleshooting Tests

### Common Issues and Solutions

#### Issue 1: Connection Refused
```bash
# Check if services are running
ps aux | grep mgpu

# Check network connectivity
telnet <master_host> 8080
telnet <node_host> 8081
```

#### Issue 2: GPU Not Found
```bash
# Check GPU availability
nvidia-smi

# Test CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Issue 3: Permission Errors
```bash
# Check socket permissions
ls -la /tmp/mgpu_scheduler.sock

# Fix permissions if needed
chmod 666 /tmp/mgpu_scheduler.sock
```

## 📊 Expected Test Results

### ✅ Successful Test Indicators
- All connectivity tests pass
- GPU detection shows available GPUs
- Jobs execute and produce expected output
- Cancellation works cleanly
- No memory leaks or hanging processes

### ❌ Failure Indicators to Watch For
- Connection timeouts
- GPU allocation failures
- Jobs hanging indefinitely
- Inconsistent output
- Process not cleaning up properly

## 🔄 Continuous Testing

### Automated Testing Setup
```bash
# Create a test script for regular validation
cat > daily_test.sh << 'EOF'
#!/bin/bash
cd /mnt/e/work/multigpu_scheduler
echo "Running daily system validation..."
python3 test/run_tests.py --quick > test_results_$(date +%Y%m%d).log 2>&1
echo "Test results saved to test_results_$(date +%Y%m%d).log"
EOF

chmod +x daily_test.sh
```

### Performance Monitoring
```bash
# Run performance benchmarks
python3 test/run_tests.py --test performance | tee performance_$(date +%Y%m%d).log
```

## 📝 Test Documentation

After running tests, document:
- System configuration used
- Test results and timing
- Any failures and their resolution
- Performance metrics
- Recommendations for optimization
