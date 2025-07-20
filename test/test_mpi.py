#!/usr/bin/env python3
"""
MPI distributed execution test
Tests MPI job execution through the scheduler
"""
import os
import sys
import subprocess
import time

def test_mpi_environment():
    """MPI 환경 테스트"""
    print("=== Testing MPI Environment ===")
    
    # mpirun 명령어 확인
    try:
        result = subprocess.run(['mpirun', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ mpirun available")
            # 첫 번째 줄만 출력 (버전 정보)
            version_line = result.stdout.strip().split('\n')[0] if result.stdout.strip() else "Unknown version"
            print(f"  Version: {version_line}")
        else:
            print("✗ mpirun not working properly")
            return False
    except FileNotFoundError:
        print("✗ mpirun not found")
        return False
    except subprocess.TimeoutExpired:
        print("✗ mpirun command timeout")
        return False
    except Exception as e:
        print(f"✗ mpirun error: {e}")
        return False
    
    # MPI 라이브러리 확인 (python)
    try:
        result = subprocess.run([sys.executable, '-c', 'import mpi4py; print("MPI4PY available")'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ mpi4py available")
        else:
            print("? mpi4py not available (optional)")
    except Exception:
        print("? mpi4py not available (optional)")
    
    return True

def create_mpi_test_script():
    """MPI 테스트 스크립트 생성"""
    mpi_script = """#!/usr/bin/env python3
import os
import socket

def main():
    # MPI 환경 변수 확인
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
    
    hostname = socket.gethostname()
    
    print(f"Hello from rank {rank} of {size} on {hostname}")
    
    # 각 프로세스가 다른 시간에 메시지 출력
    import time
    time.sleep(rank * 0.1)
    print(f"Process {rank}: MPI test completed successfully!")

if __name__ == "__main__":
    main()
"""
    
    script_path = os.path.join(os.path.dirname(__file__), 'mpi_test_script.py')
    with open(script_path, 'w') as f:
        f.write(mpi_script)
    
    return script_path

def test_local_mpi_execution():
    """로컬 MPI 실행 테스트"""
    print("\n=== Testing Local MPI Execution ===")
    
    script_path = create_mpi_test_script()
    
    try:
        # 2개 프로세스로 MPI 테스트
        cmd = ['mpirun', '-np', '2', sys.executable, script_path]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Local MPI execution successful")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print(f"✗ Local MPI execution failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ MPI execution timeout")
        return False
    except Exception as e:
        print(f"✗ MPI execution error: {e}")
        return False
    finally:
        # 테스트 스크립트 정리
        try:
            os.remove(script_path)
        except:
            pass
    
    return True

def test_scheduler_mpi_job():
    """스케줄러를 통한 MPI 작업 테스트"""
    print("\n=== Testing MPI Job through Scheduler ===")
    
    # mgpu_srun_multinode 실행 파일 경로 확인
    scheduler_path = None
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun_multinode.exe'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dist', 'mgpu_srun_multinode'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'mgpu_srun_multinode.py')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            scheduler_path = path
            break
    
    if not scheduler_path:
        print("✗ Scheduler executable not found")
        print("  Expected paths:")
        for path in possible_paths:
            print(f"    {path}")
        return False
    
    script_path = create_mpi_test_script()
    
    try:
        # 스케줄러를 통해 MPI 작업 제출
        if scheduler_path.endswith('.py'):
            cmd = [sys.executable, scheduler_path]
        else:
            cmd = [scheduler_path]
        
        cmd.extend([
            '--nodes', '1',
            '--gpus-per-node', '0',  # GPU 없이 CPU만 사용
            '--distributed-type', 'mpi',
            '--mpi-processes', '2',
            sys.executable, script_path
        ])
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Scheduler MPI job successful")
            print("Output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print(f"✗ Scheduler MPI job failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"  {line}")
            return False
    
    except subprocess.TimeoutExpired:
        print("✗ Scheduler MPI job timeout")
        return False
    except Exception as e:
        print(f"✗ Scheduler MPI job error: {e}")
        return False
    finally:
        # 테스트 스크립트 정리
        try:
            os.remove(script_path)
        except:
            pass
    
    return True

def main():
    """메인 함수"""
    print("Multi-GPU Scheduler MPI Test")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    tests = [
        ("MPI Environment", test_mpi_environment),
        ("Local MPI Execution", test_local_mpi_execution),
        ("Scheduler MPI Job", test_scheduler_mpi_job)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] Running {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All MPI tests PASSED!")
        return True
    else:
        print("❌ Some MPI tests FAILED.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
