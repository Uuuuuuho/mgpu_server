#!/usr/bin/env python3
import time
import sys

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
