#!/usr/bin/env python3
"""
Multi-GPU Scheduler Master Server - New Modular Version
Replacement for mgpu_simple_master.py
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mgpu.cli import main_server

if __name__ == '__main__':
    sys.exit(main_server())
