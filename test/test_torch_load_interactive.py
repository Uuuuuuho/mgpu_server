import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from datetime import datetime
import sys

def check_cuda_availability():
    """Check CUDA availability"""
    print("="*50)
    print("CUDA Environment Check")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is not available. Running in CPU mode.")
    print()
    sys.stdout.flush()  # Force output to be sent immediately

def matrix_multiplication_test(device, size=1024, iterations=50):
    """GPU load test with matrix multiplication (reduced for interactive testing)"""
    print(f"Matrix multiplication test started ({device})")
    print(f"Matrix size: {size}x{size}, iterations: {iterations}")
    sys.stdout.flush()
    
    # Generate random matrices
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Synchronize GPU memory
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        c = torch.matmul(a, b)
        if i % 10 == 0:  # More frequent progress updates
            print(f"Progress: {i+1}/{iterations}")
            sys.stdout.flush()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total execution time: {elapsed:.2f}s")
    print(f"Average computation time: {elapsed/iterations*1000:.2f}ms")
    print()
    sys.stdout.flush()
    
    return elapsed

def simple_convolution_test(device, batch_size=4, iterations=20):
    """GPU load test with CNN operations (reduced for interactive testing)"""
    print(f"Convolution operation test started ({device})")
    print(f"Batch size: {batch_size}, iterations: {iterations}")
    sys.stdout.flush()
    
    # Simple conv operation without problematic FC layer
    conv = nn.Conv2d(3, 64, 3, padding=1).to(device)
    input_data = torch.randn(batch_size, 3, 64, 64, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        with torch.no_grad():
            output = conv(input_data)
        if i % 5 == 0:  # More frequent progress updates
            print(f"Progress: {i+1}/{iterations}")
            sys.stdout.flush()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total execution time: {elapsed:.2f}s")
    print(f"Average inference time: {elapsed/iterations*1000:.2f}ms")
    print()
    sys.stdout.flush()
    
    return elapsed

def memory_stress_test(device, memory_gb=0.5):
    """GPU memory stress test (reduced for interactive testing)"""
    print(f"Memory stress test started ({device})")
    print(f"Memory to allocate: {memory_gb} GB")
    sys.stdout.flush()
    
    if device.type == 'cuda':
        # Check current memory usage
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial memory usage: {initial_memory:.2f} GB")
        sys.stdout.flush()
    
    try:
        # Create large tensor
        elements = int(memory_gb * 1024**3 / 4)  # float32 is 4 bytes
        print("Allocating memory...")
        sys.stdout.flush()
        large_tensor = torch.randn(elements, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Current memory usage: {current_memory:.2f} GB")
            sys.stdout.flush()
        
        # Perform operations in memory
        print("Performing operations in memory...")
        sys.stdout.flush()
        result = torch.sum(large_tensor)
        print(f"Sum: {result.item():.2e}")
        sys.stdout.flush()
        
        # Free memory
        print("Releasing memory...")
        sys.stdout.flush()
        del large_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"After memory release: {final_memory:.2f} GB")
            sys.stdout.flush()
        
        print("Memory stress test completed")
        sys.stdout.flush()
        
    except RuntimeError as e:
        print(f"Memory stress test failed: {e}")
        sys.stdout.flush()
    
    print()

def parallel_computation_test(device, num_streams=2):
    """Parallel computation test (reduced for interactive testing)"""
    if device.type != 'cuda':
        print("Parallel computation test is only supported on CUDA.")
        return
    
    print(f"Parallel computation test started (number of streams: {num_streams})")
    sys.stdout.flush()
    
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    tensors = []
    
    start_time = time.time()
    
    # Perform parallel operations on each stream
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            a = torch.randn(1024, 1024, device=device)  # Reduced size
            b = torch.randn(1024, 1024, device=device)  # Reduced size
            c = torch.matmul(a, b)
            tensors.append(c)
            print(f"Stream {i+1} started")
            sys.stdout.flush()
    
    # Synchronize all streams
    torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Parallel computation completed in: {elapsed:.2f}s")
    print()
    sys.stdout.flush()

def continuous_load_test(device, duration_seconds=10):
    """Continuous load test (reduced for interactive testing)"""
    print(f"Continuous load test started ({duration_seconds} seconds)")
    print("This will show progress every second...")
    sys.stdout.flush()
    
    start_time = time.time()
    iteration = 0
    last_report = start_time
    
    try:
        while time.time() - start_time < duration_seconds:
            # Perform various operations alternately
            a = torch.randn(512, 512, device=device)  # Reduced size
            b = torch.randn(512, 512, device=device)  # Reduced size
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Element-wise operations
            d = torch.sin(c) + torch.cos(c)
            
            # Reduction operations
            result = torch.sum(d)
            
            iteration += 1
            
            # Report progress every second
            current_time = time.time()
            if current_time - last_report >= 1.0:
                elapsed = current_time - start_time
                print(f"Elapsed time: {elapsed:.1f}s, iterations: {iteration}")
                sys.stdout.flush()
                last_report = current_time
        
        total_time = time.time() - start_time
        print(f"Continuous load test completed")
        print(f"Total time: {total_time:.2f}s, total iterations: {iteration}")
        print(f"Average TPS: {iteration/total_time:.1f}")
        
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\nInterrupted by user")
        print(f"Execution time: {total_time:.2f}s, iterations: {iteration}")
    
    print()
    sys.stdout.flush()

def main():
    print("PyTorch CUDA Load Test (Interactive Version)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    sys.stdout.flush()
    
    # Check CUDA availability
    check_cuda_availability()
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Running tests on CUDA device.")
    else:
        device = torch.device('cpu')
        print("Running tests on CPU device.")
    
    print()
    sys.stdout.flush()
    
    try:
        print("Starting Test 1/5: Matrix Multiplication")
        sys.stdout.flush()
        # 1. Matrix multiplication test (reduced)
        matrix_multiplication_test(device, size=1024, iterations=50)
        
        print("Starting Test 2/5: Convolution Operations")
        sys.stdout.flush()
        # 2. Simple convolution test (reduced)
        simple_convolution_test(device, batch_size=4, iterations=20)
        
        print("Starting Test 3/5: Memory Stress Test")
        sys.stdout.flush()
        # 3. Memory stress test (reduced)
        memory_stress_test(device, memory_gb=0.5)
        
        # 4. Parallel computation test (CUDA only)
        if device.type == 'cuda':
            print("Starting Test 4/5: Parallel Computation")
            sys.stdout.flush()
            parallel_computation_test(device, num_streams=2)
        
        print("Starting Test 5/5: Continuous Load Test")
        sys.stdout.flush()
        # 5. Continuous load test (reduced)
        continuous_load_test(device, duration_seconds=10)
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("All tests completed")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
