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

def matrix_multiplication_test(device, size=4096, iterations=100):
    """GPU load test with matrix multiplication"""
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

def convolution_test(device, batch_size=32, iterations=50):
    """GPU load test with CNN operations"""
    print(f"Convolution operation test started ({device})")
    print(f"Batch size: {batch_size}, iterations: {iterations}")
    
    # Define simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(256 * 28 * 28, 1000)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN().to(device)
    input_data = torch.randn(batch_size, 3, 224, 224, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        with torch.no_grad():
            output = model(input_data)
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

def memory_stress_test(device, memory_gb=2):
    """GPU memory stress test"""
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

def parallel_computation_test(device, num_streams=4):
    """Parallel computation test (using CUDA streams)"""
    if device.type != 'cuda':
        print("Parallel computation test is only supported on CUDA.")
        return
    
    print(f"Parallel computation test started (number of streams: {num_streams})")
    
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    tensors = []
    
    start_time = time.time()
    
    # Perform parallel operations on each stream
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            c = torch.matmul(a, b)
            tensors.append(c)
            print(f"Stream {i+1} started")
    
    # Synchronize all streams
    torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Parallel computation completed in: {elapsed:.2f}s")
    print()

def continuous_load_test(device, duration_seconds=30):
    """Continuous load test"""
    print(f"Continuous load test started ({duration_seconds} seconds)")
    print("Press Ctrl+C to interrupt.")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Perform various operations alternately
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Element-wise operations
            d = torch.sin(c) + torch.cos(c)
            
            # Reduction operations
            result = torch.sum(d)
            
            iteration += 1
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Elapsed time: {elapsed:.1f}s, iterations: {iteration}")
        
        total_time = time.time() - start_time
        print(f"Continuous load test completed")
        print(f"Total time: {total_time:.2f}s, total iterations: {iteration}")
        print(f"Average TPS: {iteration/total_time:.1f}")
        
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\nInterrupted by user")
        print(f"Execution time: {total_time:.2f}s, iterations: {iteration}")
    
    print()

def main():
    print("PyTorch CUDA Load Test")
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
        # 1. Matrix multiplication test
        matrix_multiplication_test(device, size=2048, iterations=5000)
        
        # 2. CNN operation test
        convolution_test(device, batch_size=4, iterations=300)
        
        # 3. Memory stress test
        memory_stress_test(device, memory_gb=1)
        
        # 4. Parallel computation test (CUDA only)
        if device.type == 'cuda':
            parallel_computation_test(device, num_streams=4)
        
        # 5. Continuous load test
        continuous_load_test(device, duration_seconds=20)
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("All tests completed")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()