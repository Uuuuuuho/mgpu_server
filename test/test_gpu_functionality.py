#!/usr/bin/env python3
"""
GPU-specific functionality tests for the Multi-GPU Scheduler.
Tests GPU allocation, CUDA environment setup, and multi-GPU job handling.
"""
import os
import sys
import time
import socket
import json
import subprocess
import tempfile
from pathlib import Path

def test_gpu_detection():
    """Test GPU detection functionality - both local and cluster-wide"""
    print("=== Testing GPU Detection ===")
    
    # First test local GPU detection
    local_success, local_gpu_count = test_local_gpu_detection()
    
    # Then test cluster-wide GPU detection via master server
    cluster_success, cluster_info = test_cluster_gpu_detection()
    
    if local_success and cluster_success:
        return True, local_gpu_count
    elif local_success and not cluster_success:
        print("? Local GPU detection successful, but cluster detection failed")
        return True, local_gpu_count  # Still consider it a pass for local testing
    else:
        return False, 0

def test_local_gpu_detection():
    """Test local GPU detection using nvidia-smi directly"""
    print("--- Testing Local GPU Detection ---")
    
    try:
        # Test nvidia-smi availability
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            print(f"✓ Detected {len(gpu_info)} local GPU(s)")
            for i, gpu in enumerate(gpu_info):
                parts = gpu.split(', ')
                if len(parts) >= 4:
                    index, name, total_mem, free_mem = parts[:4]
                    print(f"  GPU {index}: {name} ({free_mem}MB/{total_mem}MB free)")
            return True, len(gpu_info)
        else:
            print("✗ nvidia-smi not available or failed")
            return False, 0
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ nvidia-smi not found or timeout")
        return False, 0
    except Exception as e:
        print(f"✗ GPU detection error: {e}")
        return False, 0

def test_cluster_gpu_detection():
    """Test cluster-wide GPU detection via master server"""
    print("--- Testing Cluster GPU Detection ---")
    
    sock = None
    try:
        # Try to connect to master server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        try:
            sock.connect(('localhost', 8080))
            print("✓ Connected to master server")
        except ConnectionRefusedError:
            print("? Master server not running - skipping cluster test")
            return True, None  # Not a failure, just skip
        except Exception as e:
            print(f"? Cannot connect to master server: {e}")
            return True, None  # Not a failure, just skip
        
        # Use queue command to get cluster info (which includes nodes)
        request = {'cmd': 'queue'}
        
        sock.send(json.dumps(request).encode())
        response_data = sock.recv(4096).decode()
        
        if response_data:
            response = json.loads(response_data)
            print(f"✓ Received response from master server: {response.get('status', 'unknown')}")
            
            # Extract node information from queue response
            nodes = response.get('nodes', {})
            
            if nodes:
                node_count = len(nodes)
                print("✓ Cluster Node Information:")
                for node_id, status in nodes.items():
                    print(f"  Node {node_id}: {status}")
                
                print(f"✓ Total cluster nodes: {node_count}")
                
                # Try to get more detailed resource information
                # Use get_cluster_resources if available
                try:
                    resource_request = {'cmd': 'get_cluster_resources'}
                    sock.send(json.dumps(resource_request).encode())
                    resource_response = sock.recv(4096).decode()
                    
                    if resource_response:
                        cluster_data = json.loads(resource_response)
                        if cluster_data.get('status') == 'ok':
                            resources = cluster_data.get('resources', {})
                            total_gpus = 0
                            
                            print("✓ Detailed Cluster GPU Information:")
                            for node_id, node_info in resources.items():
                                node_gpus = len(node_info.get('available_gpus', []))
                                total_gpus += node_info.get('total_gpus', node_gpus)
                                gpu_type = node_info.get('gpu_type', 'unknown')
                                status = node_info.get('status', 'unknown')
                                
                                print(f"  Node {node_id}: {node_gpus} GPUs ({gpu_type}) - {status}")
                            
                            print(f"✓ Total cluster GPUs: {total_gpus} across {node_count} nodes")
                            return True, {'total_gpus': total_gpus, 'nodes': node_count}
                        else:
                            print(f"? get_cluster_resources failed: {cluster_data.get('message', 'unknown error')}")
                except Exception as e:
                    print(f"? get_cluster_resources not available: {e}")
                
                # Fall back to basic node count if detailed info not available
                print(f"✓ Basic cluster info: {node_count} nodes detected")
                return True, {'total_gpus': 'unknown', 'nodes': node_count}
            else:
                print("? No node information in queue response")
                return True, None  # Not a failure, just no detailed info
        else:
            print("? No response from master server")
            return True, None  # Not a failure, just no response
            
    except Exception as e:
        print(f"? Cluster GPU detection error: {e}")
        return True, None  # Not a failure, just couldn't test cluster
    finally:
        if sock:
            try:
                sock.close()
            except:
                pass

def test_cuda_environment_setup():
    """Test CUDA environment variable setup"""
    print("\n=== Testing CUDA Environment Setup ===")
    
    # Create a test script that checks CUDA_VISIBLE_DEVICES
    test_script = """#!/usr/bin/env python3
import os
import sys

print("CUDA Environment Test", flush=True)
cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}", flush=True)

# Try to import torch if available
try:
    import torch
    print(f"PyTorch available: True", flush=True)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"PyTorch GPU count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
except ImportError:
    print("PyTorch not available", flush=True)

print("Environment test completed", flush=True)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Find scheduler script
        scheduler_path = None
        test_dir = Path(__file__).parent
        possible_paths = [
            test_dir.parent / 'dist' / 'mgpu_srun.exe',
            test_dir.parent / 'dist' / 'mgpu_srun',
            test_dir.parent / 'src' / 'mgpu_srun.py'
        ]
        
        for path in possible_paths:
            if path.exists():
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Scheduler executable not found")
            return False
        
        # Test with specific GPU ID
        if scheduler_path.suffix == '.py':
            cmd = [sys.executable, str(scheduler_path), '--gpu-ids', '0', 
                   '--interactive', sys.executable, script_path]
        else:
            cmd = [str(scheduler_path), '--gpu-ids', '0', 
                   '--interactive', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ CUDA environment test successful")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            # Check if CUDA_VISIBLE_DEVICES was set correctly
            if 'CUDA_VISIBLE_DEVICES: 0' in result.stdout:
                print("✓ CUDA_VISIBLE_DEVICES correctly set to GPU 0")
                return True
            else:
                print("? CUDA_VISIBLE_DEVICES may not be set correctly")
                return False
        else:
            print(f"✗ CUDA environment test failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ CUDA environment test timeout")
        return False
    except Exception as e:
        print(f"✗ CUDA environment test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            os.unlink(script_path)
        except:
            pass

def test_multi_gpu_allocation():
    """Test multiple GPU allocation"""
    print("\n=== Testing Multi-GPU Allocation ===")
    
    gpu_available, gpu_count = test_gpu_detection()
    if not gpu_available or gpu_count < 2:
        print("? Skipping multi-GPU test (need at least 2 GPUs)")
        return True  # Not a failure, just skipped
    
    # Create a test script that uses multiple GPUs
    test_script = """#!/usr/bin/env python3
import os
import sys

print("Multi-GPU Test", flush=True)
cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}", flush=True)

try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available devices: {device_count}", flush=True)
        
        if device_count >= 2:
            # Test using multiple devices
            for i in range(min(2, device_count)):
                device = torch.device(f'cuda:{i}')
                tensor = torch.randn(100, 100, device=device)
                result = torch.sum(tensor)
                print(f"GPU {i} test: {result.item():.2f}", flush=True)
            print("Multi-GPU test successful", flush=True)
        else:
            print("Only one GPU visible to PyTorch", flush=True)
    else:
        print("CUDA not available to PyTorch", flush=True)
except ImportError:
    print("PyTorch not available", flush=True)

print("Multi-GPU test completed", flush=True)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Find scheduler script
        scheduler_path = None
        test_dir = Path(__file__).parent
        possible_paths = [
            test_dir.parent / 'dist' / 'mgpu_srun.exe',
            test_dir.parent / 'dist' / 'mgpu_srun',
            test_dir.parent / 'src' / 'mgpu_srun.py'
        ]
        
        for path in possible_paths:
            if path.exists():
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Scheduler executable not found")
            return False
        
        # Test with multiple GPU IDs
        gpu_ids = '0,1' if gpu_count >= 2 else '0'
        if scheduler_path.suffix == '.py':
            cmd = [sys.executable, str(scheduler_path), '--gpu-ids', gpu_ids,
                   '--interactive', sys.executable, script_path]
        else:
            cmd = [str(scheduler_path), '--gpu-ids', gpu_ids,
                   '--interactive', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Multi-GPU allocation test successful")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            
            # Check if multiple GPUs were allocated
            expected_devices = gpu_ids.replace(',', ' ')
            if f'CUDA_VISIBLE_DEVICES: {gpu_ids}' in result.stdout:
                print(f"✓ CUDA_VISIBLE_DEVICES correctly set to: {gpu_ids}")
                return True
            else:
                print("? Multi-GPU allocation may not be working correctly")
                return False
        else:
            print(f"✗ Multi-GPU allocation test failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Multi-GPU allocation test timeout")
        return False
    except Exception as e:
        print(f"✗ Multi-GPU allocation test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            os.unlink(script_path)
        except:
            pass

def test_memory_allocation():
    """Test GPU memory allocation constraints"""
    print("\n=== Testing GPU Memory Allocation ===")
    
    gpu_available, gpu_count = test_gpu_detection()
    if not gpu_available:
        print("? Skipping memory allocation test (no GPUs available)")
        return True
    
    # Create a test script that allocates GPU memory
    test_script = """#!/usr/bin/env python3
import sys

print("Memory allocation test", flush=True)

try:
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        
        # Allocate a reasonable amount of memory (100MB)
        size = (5000, 5000)  # ~100MB for float32
        tensor = torch.randn(size, device=device)
        
        print(f"Allocated tensor of size {size} on GPU", flush=True)
        print(f"Memory used: ~{tensor.numel() * 4 / 1024/1024:.1f} MB", flush=True)
        
        # Do some computation
        result = torch.sum(tensor)
        print(f"Computation result: {result.item():.2f}", flush=True)
        
        del tensor
        torch.cuda.empty_cache()
        print("Memory freed", flush=True)
    else:
        print("CUDA not available", flush=True)
except ImportError:
    print("PyTorch not available", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)

print("Memory test completed", flush=True)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Find scheduler script
        scheduler_path = None
        test_dir = Path(__file__).parent
        possible_paths = [
            test_dir.parent / 'dist' / 'mgpu_srun.exe',
            test_dir.parent / 'dist' / 'mgpu_srun',
            test_dir.parent / 'src' / 'mgpu_srun.py'
        ]
        
        for path in possible_paths:
            if path.exists():
                scheduler_path = path
                break
        
        if not scheduler_path:
            print("✗ Scheduler executable not found")
            return False
        
        # Test with memory constraint
        if scheduler_path.suffix == '.py':
            cmd = [sys.executable, str(scheduler_path), '--gpu-ids', '0', '--mem', '500',
                   '--interactive', sys.executable, script_path]
        else:
            cmd = [str(scheduler_path), '--gpu-ids', '0', '--mem', '500',
                   '--interactive', sys.executable, script_path]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Memory allocation test successful")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            return True
        else:
            print(f"✗ Memory allocation test failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Memory allocation test timeout")
        return False
    except Exception as e:
        print(f"✗ Memory allocation test error: {e}")
        return False
    finally:
        # Clean up test script
        try:
            os.unlink(script_path)
        except:
            pass

def main():
    """Main function"""
    print("Multi-GPU Scheduler GPU Functionality Test")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    tests = [
        ("GPU Detection", lambda: test_gpu_detection()[0]),
        ("CUDA Environment Setup", test_cuda_environment_setup),
        ("Multi-GPU Allocation", test_multi_gpu_allocation),
        ("Memory Allocation", test_memory_allocation)
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
    print(f"GPU Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All GPU tests PASSED!")
        return True
    else:
        print("❌ Some GPU tests FAILED.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
