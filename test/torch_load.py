#!/usr/bin/env python3
"""
Simple test script to verify the scheduler is working
"""
import time
import os

def main():
    print("=== Test Script Started ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    
    print("Running for 5 seconds...")
    for i in range(5):
        print(f"Step {i+1}/5: Working...")
        time.sleep(1)
    
    print("=== Test Script Completed ===")

if __name__ == "__main__":
    main()
