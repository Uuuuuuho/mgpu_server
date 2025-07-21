#!/usr/bin/env python3
"""
Simple test job for GPU scheduler
"""
import time
import os

def main():
    print("=== GPU Test Job Started ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    
    # Simulate some work
    for i in range(5):
        print(f"Processing step {i+1}/5...")
        time.sleep(1)
    
    print("=== GPU Test Job Completed Successfully ===")

if __name__ == "__main__":
    main()
