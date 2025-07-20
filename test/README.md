# Test Files

This directory contains comprehensive test scripts for the Multi-GPU Scheduler.

## Test Scripts

### Basic Functionality Tests

#### test_output.py
Tests the output streaming functionality. This script prints messages with delays to verify that logs appear in real-time in the user's terminal.

**Usage:**
```bash
python ../src/mgpu_srun.py --gpu-ids 0 -- python test_output.py
```

#### test_cancellation.py
Tests the job cancellation functionality when the user interrupts the interactive session (Ctrl+C). This script runs for 60 seconds with periodic output.

**Usage:**
```bash
python ../src/mgpu_srun.py --gpu-ids 0 -- python test_cancellation.py
# Wait a few seconds, then press Ctrl+C to test cancellation
```

### Comprehensive Test Suites

#### test_streaming.py
Comprehensive tests for output streaming, job cancellation, and background job functionality.
- Tests real-time output streaming
- Tests interactive job cancellation
- Tests background job execution

#### test_integration.py
End-to-end integration tests that verify the complete scheduler functionality:
- Server startup and connectivity
- Job submission and execution
- Job cancellation under various conditions
- Multiple concurrent jobs
- Resource constraint validation

#### test_gpu_functionality.py
GPU-specific functionality tests:
- GPU detection via nvidia-smi
- CUDA environment variable setup
- Multi-GPU allocation
- GPU memory allocation constraints

#### test_performance.py
Performance and stress testing:
- High-frequency job submissions
- Concurrent client connections
- Job cancellation under stress
- Memory pressure testing
- Queue status query performance

#### test_error_handling.py
Error handling and edge case tests:
- Invalid command handling
- Malformed request handling
- Resource limit violation handling
- Job command failure scenarios
- Server recovery after crashes
- Invalid user scenarios

#### test_distributed.py
PyTorch distributed training tests:
- Single-node distributed setup
- Multi-node distributed coordination
- Environment variable configuration
- Process group initialization

#### test_cluster.py
Multi-node cluster functionality tests:
- Cluster connectivity validation
- Job submission across nodes
- Resource monitoring
- Environment variable validation

#### test_mpi.py
MPI distributed execution tests:
- MPI environment validation
- Local MPI execution
- Scheduler-managed MPI jobs

#### test_single_node_multinode.py
Tests for running multi-node server in single-node mode:
- Master server standalone operation
- Virtual node creation
- Basic job submission and execution

#### test_torch_load.py
PyTorch CUDA load and stress tests:
- CUDA availability checks
- Matrix multiplication benchmarks
- CNN operation benchmarks
- GPU memory stress tests
- Parallel computation tests
- Continuous load tests

## Running Tests

### Quick Test Suite
Run essential tests only (faster execution):
```bash
python run_tests.py --quick
```

### Full Test Suite
Run all available tests:
```bash
python run_tests.py
```

### Individual Test Suites
Run specific test categories:
```bash
python run_tests.py --test streaming
python run_tests.py --test integration
python run_tests.py --test gpu
python run_tests.py --test performance
python run_tests.py --test error_handling
```

### List Available Tests
See all available test suites:
```bash
python run_tests.py --list
```

### Verbose Output
See real-time test output:
```bash
python run_tests.py --verbose
```

## Manual Testing

### Basic Manual Tests
1. First, start the scheduler server:
   ```bash
   cd ../src
   python mgpu_scheduler_server.py
   ```

2. In another terminal, run the test:
   ```bash
   cd test
   python ../src/mgpu_srun.py --gpu-ids 0 -- python <test_script>
   ```

3. Check the job queue status:
   ```bash
   python ../src/mgpu_queue.py
   ```

### Testing Different Modes
```bash
# Interactive mode (see real-time output)
python ../src/mgpu_srun.py --gpu-ids 0 --interactive -- python test_output.py

# Background mode (silent execution)
python ../src/mgpu_srun.py --gpu-ids 0 -- python test_output.py

# With memory constraints
python ../src/mgpu_srun.py --gpu-ids 0 --mem 2000 -- python test_torch_load.py

# With multiple GPUs
python ../src/mgpu_srun.py --gpu-ids 0,1 --interactive -- python test_distributed.py

# With priority
python ../src/mgpu_srun.py --gpu-ids 0 --priority 5 -- python test_output.py

# With environment setup
python ../src/mgpu_srun.py --gpu-ids 0 --env-setup-cmd "source venv/bin/activate" -- python test_output.py
```

## Expected Behavior

- **test_output.py**: You should see the print statements appear in your terminal in real-time as the job runs.
- **test_cancellation.py**: When you press Ctrl+C, the job should be canceled and removed from the running queue automatically.
- **test_streaming.py**: All streaming and cancellation tests should pass with proper output handling.
- **test_integration.py**: End-to-end functionality should work correctly with proper job lifecycle management.
- **test_gpu_functionality.py**: GPU allocation and CUDA environment should be set up correctly.
- **test_performance.py**: System should handle multiple concurrent jobs and high-frequency operations efficiently.
- **test_error_handling.py**: Invalid inputs and error conditions should be handled gracefully without crashing.

## Test Environment Requirements

### Minimum Requirements
- Python 3.8+
- psutil package
- Access to `/tmp` directory for Unix socket

### GPU Testing Requirements
- NVIDIA GPU with CUDA support
- nvidia-smi utility
- PyTorch with CUDA support (optional, for GPU-specific tests)

### Multi-Node Testing Requirements
- Multiple nodes with network connectivity
- SSH access between nodes
- Consistent user accounts across nodes
- NFS or shared filesystem (recommended)

### MPI Testing Requirements
- MPI implementation (OpenMPI, MPICH, etc.)
- mpi4py package (optional)
- mpirun command availability

## Troubleshooting Tests

### Common Issues
1. **Permission denied errors**: Ensure proper file permissions and user privileges
2. **Socket connection errors**: Check if scheduler server is running and socket file exists
3. **GPU not found errors**: Verify CUDA installation and GPU visibility
4. **Test timeouts**: Increase timeout values for slower systems
5. **Import errors**: Ensure all required Python packages are installed

### Debug Mode
Run tests with verbose output to see detailed execution:
```bash
python run_tests.py --verbose --test <test_name>
```

### Manual Debug
For debugging specific issues, run individual test files directly:
```bash
python test_integration.py
python test_gpu_functionality.py
python test_performance.py
python test_error_handling.py
```
