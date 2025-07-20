#!/usr/bin/env python3
"""
Output streaming and job cancellation test
Tests real-time output streaming and job cancellation functionality
"""
import os
import sys
import socket
import json
import time
import threading
import subprocess
import signal

def test_streaming_output():
    """Output streaming test"""
    print("=== Testing Output Streaming ===")
    
    # Test single-node server connection
    socket_path = "/tmp/mgpu_scheduler.sock"
    if os.name == 'nt':  # Windows
        socket_path = r"\\.\pipe\mgpu_scheduler"
    
    script_path = None
    try:
        # Write a simple script that generates streaming output
        test_script = """#!/usr/bin/env python3
import time
import sys

for i in range(5):
    print(f"Output line {i+1}/5", flush=True)
    time.sleep(1)

print("Test completed!", flush=True)
"""
        
        script_path = os.path.join(os.path.dirname(__file__), 'streaming_test.py')
        with open(script_path, 'w') as f:
            f.write(test_script)
        
        # Find single-node scheduler
        scheduler_path = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun.exe'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'mgpu_srun.py')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Single-node scheduler not found")
            return False
        
        # Run in interactive mode
        if scheduler_path.endswith('.py'):
            cmd = [sys.executable, scheduler_path, '--interactive', sys.executable, script_path]
        else:
            cmd = [scheduler_path, '--interactive', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✓ Streaming output test successful")
            print(f"  Execution time: {elapsed_time:.1f} seconds")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            # Check if streaming actually happened (should take about 5 seconds)
            if elapsed_time >= 4.5:  # Allow some margin
                print("✓ Output appears to be streamed (expected timing)")
            else:
                print("? Output may not be properly streamed (too fast)")
        else:
            print(f"✗ Streaming output test failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Streaming test timeout")
        return False
    except Exception as e:
        print(f"✗ Streaming test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            if script_path is not None:
                os.remove(script_path)
        except:
            pass

    return True

def test_job_cancellation():
    """Job cancellation test"""
    print("\n=== Testing Job Cancellation ===")
    
    # Write a long-running script
    long_script = """#!/usr/bin/env python3
import time
import signal
import sys

def signal_handler(sig, frame):
    print("Received signal, exiting gracefully...", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

try:
    for i in range(60):  # Run for 1 minute
        print(f"Long running task... {i+1}/60", flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrupted by user", flush=True)
    sys.exit(0)

print("Long task completed", flush=True)
"""
    
    script_path = os.path.join(os.path.dirname(__file__), 'long_task.py')
    
    try:
        with open(script_path, 'w') as f:
            f.write(long_script)
        
        # Find scheduler path
        scheduler_path = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun.exe'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'mgpu_srun.py')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Scheduler not found")
            return False
        
        # Run in interactive mode
        if scheduler_path.endswith('.py'):
            cmd = [sys.executable, scheduler_path, '--interactive', sys.executable, script_path]
        else:
            cmd = [scheduler_path, '--interactive', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        print("Will cancel after 5 seconds...")
        
        # Start process
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Cancel after 5 seconds
        time.sleep(5)
        proc.terminate()
        
        # Wait for process to exit (max 10 seconds)
        try:
            stdout, stderr = proc.communicate(timeout=10)
            
            if proc.returncode is not None:
                print("✓ Job cancellation successful")
                print(f"  Return code: {proc.returncode}")
                if stdout:
                    lines = stdout.strip().split('\n')
                    print(f"  Output lines: {len(lines)}")
                    # Show first few and last few lines
                    for line in lines[:3]:
                        print(f"    {line}")
                    if len(lines) > 3:
                        print("    ...")
                
                # Should have about 5 lines of output
                if stdout and len(stdout.strip().split('\n')) >= 3:
                    print("✓ Job was running and properly cancelled")
                else:
                    print("? Job may not have been properly running")
            else:
                print("✗ Job cancellation failed - process still running")
                proc.kill()  # Force kill
                return False
        
        except subprocess.TimeoutExpired:
            print("✗ Job cancellation timeout")
            proc.kill()
            return False
    
    except Exception as e:
        print(f"✗ Job cancellation test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            os.remove(script_path)
        except:
            pass
    
    return True

def test_background_job():
    """Background job test"""
    print("\n=== Testing Background Job ===")
    
    # Background job script
    bg_script = """#!/usr/bin/env python3
import time

for i in range(3):
    print(f"Background task {i+1}/3", flush=True)
    time.sleep(1)

print("Background task completed", flush=True)
"""
    
    script_path = os.path.join(os.path.dirname(__file__), 'bg_task.py')
    
    try:
        with open(script_path, 'w') as f:
            f.write(bg_script)
        
        # Find scheduler path
        scheduler_path = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun.exe'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'mgpu_srun.py')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Scheduler not found")
            return False
        
        # Run in background mode
        if scheduler_path.endswith('.py'):
            cmd = [sys.executable, scheduler_path, '--background', sys.executable, script_path]
        else:
            cmd = [scheduler_path, '--background', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✓ Background job test successful")
            print(f"  Command execution time: {elapsed_time:.1f} seconds")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            # Background mode should return immediately (faster than 3 seconds)
            if elapsed_time < 2.0:
                print("✓ Background mode working (immediate return)")
            else:
                print("? Background mode may not be working properly (took too long)")
        else:
            print(f"✗ Background job test failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Background job test timeout")
        return False
    except Exception as e:
        print(f"✗ Background job test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            os.remove(script_path)
        except:
            pass
    
    return True

def main():
    """Main function"""
    print("Multi-GPU Scheduler Streaming and Cancellation Test")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    tests = [
        ("Output Streaming", test_streaming_output),
        # ("Job Cancellation", test_job_cancellation),
        # ("Background Job", test_background_job)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] Running {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All streaming tests PASSED!")
        return True
    else:
        print("❌ Some streaming tests FAILED.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
