#!/usr/bin/env python3
"""
Job cancellation test for the Multi-GPU Scheduler.
Tests the job cancellation functionality when the user interrupts 
the interactive session (Ctrl+C).
"""
import time
import sys

def main():
    """Main test function"""
    print("Starting long-running job...")
    sys.stdout.flush()

    try:
        for i in range(60):  # Run for 60 seconds
            print(f"Step {i+1}: Working... (press Ctrl+C to test cancellation)")
            sys.stdout.flush()
            time.sleep(1)
        
        print("Job completed normally!")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        print("Job was interrupted!")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
