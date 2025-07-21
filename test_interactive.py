#!/usr/bin/env python3
"""
Interactive test job for GPU scheduler
"""
import time
import os
import sys

def main():
    print("=== Interactive GPU Test Job Started ===", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}", flush=True)
    print(f"Current working directory: {os.getcwd()}", flush=True)
    print(f"User: {os.environ.get('USER', 'unknown')}", flush=True)
    
    # Simulate interactive work with real-time output
    for i in range(10):
        print(f"[{i+1:2d}/10] Processing batch {i+1}... ", end="", flush=True)
        time.sleep(0.5)
        print("Done!", flush=True)
        
        if i == 4:
            print("--- Halfway checkpoint reached ---", flush=True)
    
    print("=== Interactive GPU Test Job Completed Successfully ===", flush=True)

if __name__ == "__main__":
    main()
