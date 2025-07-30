#!/usr/bin/env python3
import sys
import argparse

print("Testing command parsing...")
sys.argv = ['test', 'submit', '--interactive', '--', 'python', 'test/test_torch_load.py']
print("sys.argv:", sys.argv)

parser = argparse.ArgumentParser()
parser.add_argument('action')
parser.add_argument('--interactive', action='store_true')
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()

print("Parsed args.command:", args.command)
print("Interactive:", args.interactive)

# Process command like mgpu_srun_multinode.py does
command_parts = args.command
if command_parts and command_parts[0] == '--':
    command_parts = command_parts[1:]

cmdline = ' '.join(command_parts)
print("Final cmdline:", repr(cmdline))
