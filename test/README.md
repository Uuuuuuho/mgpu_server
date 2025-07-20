# Test Files

This directory contains test scripts for the Multi-GPU Scheduler.

## Test Scripts

### test_output.py
Tests the output streaming functionality. This script prints messages with delays to verify that logs appear in real-time in the user's terminal.

**Usage:**
```bash
python ../src/mgpu_srun.py --gpu-ids 0 -- python test_output.py
```

### test_cancellation.py
Tests the job cancellation functionality when the user interrupts the interactive session (Ctrl+C). This script runs for 60 seconds with periodic output.

**Usage:**
```bash
python ../src/mgpu_srun.py --gpu-ids 0 -- python test_cancellation.py
# Wait a few seconds, then press Ctrl+C to test cancellation
```

## Running Tests

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

## Expected Behavior

- **test_output.py**: You should see the print statements appear in your terminal in real-time as the job runs.
- **test_cancellation.py**: When you press Ctrl+C, the job should be canceled and removed from the running queue automatically.
