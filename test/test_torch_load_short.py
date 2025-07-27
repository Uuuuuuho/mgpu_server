import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from datetime import datetime

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

def matrix_multiplication_test(device, size=1024, iterations=100):
    """GPU load test with matrix multiplication (reduced for scheduler testing)"""
    print(f"Matrix multiplication test started ({device})")
    print(f"Matrix size: {size}x{size}, iterations: {iterations}")
    
    # Generate random matrices
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # Synchronize GPU memory
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        c = torch.matmul(a, b)
        if i % 20 == 0:
            print(f"Progress: {i+1}/{iterations}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total execution time: {elapsed:.2f}s")
    print(f"Average computation time: {elapsed/iterations*1000:.2f}ms")
    print()
    
    return elapsed

def simple_convolution_test(device, batch_size=4, iterations=50):
    """Simple convolution test (fixed tensor shapes)"""
    print(f"Simple convolution operation test started ({device})")
    print(f"Batch size: {batch_size}, iterations: {iterations}")
    
    # Simple conv operation without FC layer
    conv = nn.Conv2d(3, 64, 3, padding=1).to(device)
    input_data = torch.randn(batch_size, 3, 64, 64, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        with torch.no_grad():
            output = conv(input_data)
        if i % 10 == 0:
            print(f"Progress: {i+1}/{iterations}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total execution time: {elapsed:.2f}s")
    print(f"Average inference time: {elapsed/iterations*1000:.2f}ms")
    print()
    
    return elapsed

def memory_stress_test(device, memory_gb=0.5):
    """GPU memory stress test (reduced for scheduler testing)"""
    print(f"Memory stress test started ({device})")
    print(f"Memory to allocate: {memory_gb} GB")
    
    if device.type == 'cuda':
        # Check current memory usage
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    try:
        # Create large tensor
        elements = int(memory_gb * 1024**3 / 4)  # float32 is 4 bytes
        large_tensor = torch.randn(elements, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Current memory usage: {current_memory:.2f} GB")
        
        # Perform operations in memory
        print("Performing operations in memory...")
        result = torch.sum(large_tensor)
        print(f"Sum: {result.item():.2e}")
        
        # Free memory
        del large_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"After memory release: {final_memory:.2f} GB")
        
        print("Memory stress test completed")
        
    except RuntimeError as e:
        print(f"Memory stress test failed: {e}")
    
    print()

def main():
    print("PyTorch CUDA Load Test (Short Version)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
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
    
    try:
        # 1. Matrix multiplication test (reduced)
        matrix_multiplication_test(device, size=1024, iterations=100)
        
        # 2. Simple convolution test (fixed)
        simple_convolution_test(device, batch_size=4, iterations=50)
        
        # 3. Memory stress test (reduced)
        memory_stress_test(device, memory_gb=0.5)
        
        print("Short tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("All tests completed")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
