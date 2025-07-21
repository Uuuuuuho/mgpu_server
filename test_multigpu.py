#!/usr/bin/env python3
"""
Multi-GPU test job for GPU scheduler
"""
import time
import os

def main():
    print("=== Multi-GPU Test Job Started ===")
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    if cuda_devices != 'Not set':
        gpu_list = cuda_devices.split(',')
        print(f"Number of GPUs assigned: {len(gpu_list)}")
        print(f"GPU IDs: {gpu_list}")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    
    # Simulate multi-GPU work
    print("Initializing multi-GPU computation...")
    time.sleep(2)
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch+1}/3 ---")
        for step in range(5):
            print(f"  Step {step+1}/5: Training on all GPUs... ", end="", flush=True)
            time.sleep(0.8)
            print(f"Loss: {1.0 - (epoch*5 + step) * 0.02:.3f}")
    
    print("\n=== Multi-GPU Test Job Completed Successfully ===")

if __name__ == "__main__":
    main()
