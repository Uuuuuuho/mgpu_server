#!/usr/bin/env python3
import time
import sys

print("Starting test job...")
sys.stdout.flush()

for i in range(5):
    print(f"Step {i+1}: Processing...")
    sys.stdout.flush()
    time.sleep(1)

print("Test job completed!")
sys.stdout.flush()
