#!/usr/bin/env python3
"""
PyTorch 2-GPU Example
Demonstrates various multi-GPU training and inference techniques using PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from datetime import datetime

def check_gpu_setup():
    """Check GPU availability and configuration"""
    print("="*60)
    print("GPU Setup Check")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        if gpu_count >= 2:
            print("✓ 2+ GPUs detected - multi-GPU examples will run")
        else:
            print("⚠ Only 1 GPU detected - some examples will be limited")
    else:
        print("✗ No CUDA GPUs available")
    print()

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=1024, hidden_size=512, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNNModel(nn.Module):
    """CNN model for image-like data"""
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def example_dataparallel():
    """Example 1: DataParallel - Simple multi-GPU training"""
    print("="*60)
    print("Example 1: DataParallel Multi-GPU Training")
    print("="*60)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping: Need at least 2 GPUs for this example")
        return
    
    # Create model and move to GPU
    model = SimpleModel(input_size=1024, hidden_size=512, output_size=10)
    
    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    # Create dummy data
    batch_size = 128
    data = torch.randn(batch_size, 1024).cuda()
    targets = torch.randint(0, 10, (batch_size,)).cuda()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Data shape: {data.shape}")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")
    print()

def example_manual_2gpu():
    """Example 2: Manual 2-GPU setup with explicit device placement"""
    print("="*60)
    print("Example 2: Manual 2-GPU Setup")
    print("="*60)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping: Need at least 2 GPUs for this example")
        return
    
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
    
    # Create two models on different GPUs
    model1 = SimpleModel(input_size=512, hidden_size=256, output_size=128).to(device0)
    model2 = SimpleModel(input_size=128, hidden_size=64, output_size=10).to(device1)
    
    # Create data on both GPUs
    batch_size = 64
    data = torch.randn(batch_size, 512).to(device0)
    targets = torch.randint(0, 10, (batch_size,)).to(device1)
    
    # Forward pass through both models
    print("Forward pass through 2-GPU pipeline...")
    start_time = time.time()
    
    with torch.no_grad():
        # Process on GPU 0
        intermediate = model1(data)
        print(f"Intermediate output shape: {intermediate.shape} (on {intermediate.device})")
        
        # Move to GPU 1 and process
        intermediate = intermediate.to(device1)
        final_output = model2(intermediate)
        print(f"Final output shape: {final_output.shape} (on {final_output.device})")
        
        # Calculate accuracy
        _, predicted = torch.max(final_output.data, 1)
        accuracy = (predicted == targets).float().mean()
        print(f"Random accuracy: {accuracy:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Pipeline completed in {elapsed:.4f}s")
    print()

def example_2gpu_training():
    """Example 3: 2-GPU training with data splitting"""
    print("="*60)
    print("Example 3: 2-GPU Training with Data Splitting")
    print("="*60)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping: Need at least 2 GPUs for this example")
        return
    
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    # Create identical models on both GPUs
    model0 = CNNModel(num_classes=10).to(device0)
    model1 = CNNModel(num_classes=10).to(device1)
    
    # Synchronize model parameters
    for p0, p1 in zip(model0.parameters(), model1.parameters()):
        p1.data.copy_(p0.data)
    
    # Create optimizers
    optimizer0 = optim.Adam(model0.parameters(), lr=0.001)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy image data
    batch_size = 32
    data = torch.randn(batch_size * 2, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size * 2,))
    
    # Split data between GPUs
    data0 = data[:batch_size].to(device0)
    data1 = data[batch_size:].to(device1)
    targets0 = targets[:batch_size].to(device0)
    targets1 = targets[batch_size:].to(device1)
    
    print(f"Training with split batches: {data0.shape} on GPU0, {data1.shape} on GPU1")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(5):
        # Forward and backward on both GPUs
        model0.train()
        model1.train()
        
        # GPU 0
        optimizer0.zero_grad()
        output0 = model0(data0)
        loss0 = criterion(output0, targets0)
        loss0.backward()
        
        # GPU 1
        optimizer1.zero_grad()
        output1 = model1(data1)
        loss1 = criterion(output1, targets1)
        loss1.backward()
        
        # Update parameters
        optimizer0.step()
        optimizer1.step()
        
        # Synchronize parameters (simple averaging)
        with torch.no_grad():
            for p0, p1 in zip(model0.parameters(), model1.parameters()):
                avg_param = (p0.data + p1.data.to(device0)) / 2
                p0.data.copy_(avg_param)
                p1.data.copy_(avg_param.to(device1))
        
        total_loss = loss0.item() + loss1.item()
        print(f"Epoch {epoch}, Total Loss: {total_loss:.4f} (GPU0: {loss0.item():.4f}, GPU1: {loss1.item():.4f})")
    
    elapsed = time.time() - start_time
    print(f"2-GPU training completed in {elapsed:.2f}s")
    print()

def example_memory_distribution():
    """Example 4: Memory distribution across 2 GPUs"""
    print("="*60)
    print("Example 4: Memory Distribution Across 2 GPUs")
    print("="*60)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping: Need at least 2 GPUs for this example")
        return
    
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    def print_memory_stats(device):
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  {device}: Allocated {allocated:.2f}GB, Cached {cached:.2f}GB")
    
    print("Initial memory usage:")
    print_memory_stats(device0)
    print_memory_stats(device1)
    
    # Allocate large tensors on both GPUs
    print("\nAllocating 1GB tensors on each GPU...")
    tensor_size = int(1024**3 / 4)  # 1GB for float32
    
    tensor0 = torch.randn(tensor_size, device=device0)
    tensor1 = torch.randn(tensor_size, device=device1)
    
    print("After allocation:")
    print_memory_stats(device0)
    print_memory_stats(device1)
    
    # Perform cross-GPU operations
    print("\nPerforming cross-GPU operations...")
    start_time = time.time()
    
    # Move data between GPUs and compute
    temp = tensor0[:100000].to(device1)
    result = torch.sum(temp + tensor1[:100000])
    
    elapsed = time.time() - start_time
    print(f"Cross-GPU operation completed in {elapsed:.4f}s")
    print(f"Result: {result.item():.2e}")
    
    # Clean up
    del tensor0, tensor1, temp
    torch.cuda.empty_cache()
    
    print("\nAfter cleanup:")
    print_memory_stats(device0)
    print_memory_stats(device1)
    print()

def example_concurrent_streams():
    """Example 5: Concurrent operations using CUDA streams"""
    print("="*60)
    print("Example 5: Concurrent Operations with CUDA Streams")
    print("="*60)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping: Need at least 2 GPUs for this example")
        return
    
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    # Create streams for concurrent execution
    stream0 = torch.cuda.Stream(device=device0)
    stream1 = torch.cuda.Stream(device=device1)
    
    # Create models
    model0 = SimpleModel().to(device0)
    model1 = SimpleModel().to(device1)
    
    # Create data
    batch_size = 64
    data0 = torch.randn(batch_size, 1024, device=device0)
    data1 = torch.randn(batch_size, 1024, device=device1)
    
    print("Running concurrent inference on both GPUs...")
    start_time = time.time()
    
    # Run inference concurrently
    with torch.cuda.stream(stream0):
        with torch.no_grad():
            output0 = model0(data0)
            result0 = torch.sum(output0)
    
    with torch.cuda.stream(stream1):
        with torch.no_grad():
            output1 = model1(data1)
            result1 = torch.sum(output1)
    
    # Wait for both streams to complete
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)
    
    elapsed = time.time() - start_time
    print(f"Concurrent inference completed in {elapsed:.4f}s")
    print(f"Results - GPU0: {result0.item():.4f}, GPU1: {result1.item():.4f}")
    print()

def benchmark_single_vs_multi_gpu():
    """Example 6: Benchmark single GPU vs multi-GPU performance"""
    print("="*60)
    print("Example 6: Single GPU vs Multi-GPU Benchmark")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return
    
    # Create data
    batch_size = 128
    data = torch.randn(batch_size, 1024)
    targets = torch.randint(0, 10, (batch_size,))
    
    # Test 1: Single GPU
    print("Testing single GPU performance...")
    model_single = SimpleModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_single.parameters())
    
    data_single = data.cuda()
    targets_single = targets.cuda()
    
    start_time = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        output = model_single(data_single)
        loss = criterion(output, targets_single)
        loss.backward()
        optimizer.step()
    
    single_gpu_time = time.time() - start_time
    print(f"Single GPU time: {single_gpu_time:.4f}s")
    
    # Test 2: Multi-GPU (if available)
    if torch.cuda.device_count() > 1:
        print("Testing multi-GPU performance...")
        model_multi = SimpleModel()
        model_multi = nn.DataParallel(model_multi).cuda()
        optimizer = optim.Adam(model_multi.parameters())
        
        data_multi = data.cuda()
        targets_multi = targets.cuda()
        
        start_time = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            output = model_multi(data_multi)
            loss = criterion(output, targets_multi)
            loss.backward()
            optimizer.step()
        
        multi_gpu_time = time.time() - start_time
        print(f"Multi-GPU time: {multi_gpu_time:.4f}s")
        print(f"Speedup: {single_gpu_time/multi_gpu_time:.2f}x")
    else:
        print("Multi-GPU test skipped: Only 1 GPU available")
    
    print()

def main():
    """Main function to run all examples"""
    print("PyTorch 2-GPU Examples")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check GPU setup
    check_gpu_setup()
    
    try:
        # Run all examples
        example_dataparallel()
        example_manual_2gpu()
        example_2gpu_training()
        example_memory_distribution()
        example_concurrent_streams()
        benchmark_single_vs_multi_gpu()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        
        print("All examples completed")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
