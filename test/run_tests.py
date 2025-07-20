#!/usr/bin/env python3
"""
Comprehensive test runner for Multi-GPU Scheduler
Runs all test suites and provides comprehensive results
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# 테스트 스크립트들
TEST_SCRIPTS = {
    'streaming': {
        'script': 'test_streaming.py',
        'description': 'Output streaming and job cancellation tests',
        'requirements': ['scheduler server running']
    },
    'cancellation': {
        'script': 'test_cancellation.py',
        'description': 'Job cancellation and cleanup tests',
        'requirements': ['scheduler server running']
    },
    'output': {
        'script': 'test_output.py',
        'description': 'Output handling and redirection tests',
        'requirements': ['scheduler server running']
    },
    'integration': {
        'script': 'test_integration.py',
        'description': 'End-to-end integration tests',
        'requirements': ['scheduler server components']
    },
    'gpu': {
        'script': 'test_gpu_functionality.py',
        'description': 'GPU allocation and CUDA environment tests',
        'requirements': ['CUDA', 'nvidia-smi']
    },
    'performance': {
        'script': 'test_performance.py',
        'description': 'Performance and stress tests',
        'requirements': ['scheduler server running']
    },
    'error_handling': {
        'script': 'test_error_handling.py',
        'description': 'Error handling and edge case tests',
        'requirements': ['scheduler server running']
    },
    'distributed': {
        'script': 'test_distributed.py',
        'description': 'PyTorch distributed training tests',
        'requirements': ['pytorch', 'distributed training setup']
    },
    'cluster': {
        'script': 'test_cluster.py',
        'description': 'Multi-node cluster functionality tests',
        'requirements': ['master server running', 'node agents']
    },
    'mpi': {
        'script': 'test_mpi.py',
        'description': 'MPI distributed execution tests',
        'requirements': ['mpirun', 'MPI environment']
    },
    'single_node_multinode': {
        'script': 'test_single_node_multinode.py',
        'description': 'Single node multi-node server tests',
        'requirements': ['master server config']
    },
    'torch_load': {
        'script': 'test_torch_load.py',
        'description': 'PyTorch CUDA load tests',
        'requirements': ['PyTorch', 'CUDA']
    }
}

def check_test_environment():
    """테스트 환경 확인"""
    print("=== Checking Test Environment ===")
    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✓ Python version: {python_version}")
    # Test directory
    test_dir = Path(__file__).parent
    print(f"✓ Test directory: {test_dir}")
    # Check scripts exist
    missing = []
    for name, info in TEST_SCRIPTS.items():
        path = test_dir / info['script']
        if path.exists():
            print(f"✓ {info['script']}")
        else:
            print(f"✗ {info['script']} (missing)")
            missing.append(name)
    # Check scheduler executables
    schedulers = []
    sched_dir = test_dir.parent / 'dist'
    src_dir = test_dir.parent / 'src'
    files = [
        'mgpu_scheduler_server', 'mgpu_master_server', 'mgpu_node_agent',
        'mgpu_srun', 'mgpu_srun_multinode'
    ]
    for base in files:
        exe = sched_dir / base
        if exe.with_suffix('.exe').exists() or exe.exists():
            schedulers.append(str(exe))
        elif (src_dir / f"{base}.py").exists():
            schedulers.append(str(src_dir / f"{base}.py"))
    if schedulers:
        print(f"✓ Found {len(schedulers)} scheduler components")
    else:
        print("✗ No scheduler executables found")
    return not missing and bool(schedulers)

def run_test_suite(test_name, verbose=False):
    """Run a single test suite"""
    if test_name not in TEST_SCRIPTS:
        print(f"✗ Unknown test suite: {test_name}")
        return False
    info = TEST_SCRIPTS[test_name]
    script = Path(__file__).parent / info['script']
    if not script.exists():
        print(f"✗ Script not found: {info['script']}")
        return False
    print(f"\n{'='*60}\nRunning {test_name.upper()} Tests")
    print(f"Description: {info['description']}")
    print(f"Requirements: {', '.join(info['requirements'])}")
    cmd = [sys.executable, str(script)]
    if verbose:
        return subprocess.call(cmd) == 0
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0

def run_all_tests(verbose=False, quick=False):
    """Run all test suites"""
    print("Multi-GPU Scheduler Comprehensive Test Suite")
    start = time.time()
    if not check_test_environment():
        print("\n❌ Environment check failed!")
        return False
    # 퀵 모드에서는 일부 테스트만 실행
    if quick:
        test_order = ['streaming', 'cancellation', 'integration', 'gpu']
        print(f"\n🚀 Quick mode: Running {len(test_order)} essential tests")
    else:
        test_order = ['streaming', 'cancellation', 'output', 'integration', 'gpu', 'distributed', 'performance', 'error_handling', 'mpi', 'cluster', 'single_node_multinode', 'torch_load']
        print(f"\n🔬 Full mode: Running all {len(test_order)} test suites")
    
    results = {}
    for idx, name in enumerate(test_order, 1):
        print(f"\n[{idx}/{len(test_order)}] {name} Tests")
        results[name] = run_test_suite(name, verbose)
    elapsed = time.time() - start
    print(f"\nOverall time: {elapsed:.1f}s")
    passed = sum(results.values())
    total = len(results)
    print(f"\n✅ {passed}/{total} suites passed.")
    return passed == total

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Scheduler Test Runner')
    parser.add_argument('--test', choices=list(TEST_SCRIPTS.keys()) + ['all'], default='all')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--quick', '-q', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print("Available test suites:")
        for name, info in TEST_SCRIPTS.items():
            print(f"  {name:20} - {info['description']}")
        return
    if args.test == 'all':
        success = run_all_tests(args.verbose, args.quick)
    else:
        success = run_test_suite(args.test, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
