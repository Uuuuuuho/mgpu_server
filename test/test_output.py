#!/usr/bin/env python3
"""
Simple output streaming test for the Multi-GPU Scheduler.
Tests that job output appears in real-time in the user's terminal.
"""
import time
import sys

def main():
    """Main test function"""
    print("Starting test job...")
    sys.stdout.flush()

    for i in range(5):
        print(f"Step {i+1}: Processing...")
        sys.stdout.flush()
        time.sleep(1)

    print("Test job completed!")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
