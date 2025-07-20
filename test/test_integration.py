#!/usr/bin/env python3
"""
Comprehensive integration test for the Multi-GPU Scheduler system.
Tests end-to-end functionality including server startup, job submission, 
execution, monitoring, and cleanup.
"""
import os
import sys
import time
import socket
import json
import subprocess
import threading
import signal
import tempfile
import shutil
from pathlib import Path

class SchedulerTestRunner:
    """Test runner for comprehensive scheduler testing"""
    
    def __init__(self):
        self.server_proc = None
        self.test_dir = Path(__file__).parent
        self.src_dir = self.test_dir.parent / 'src'
        self.socket_path = '/tmp/mgpu_scheduler.sock'
        self.test_results = {}
        
    def start_server(self, timeout=10):
        """Start the scheduler server"""
        print("Starting scheduler server...")
        
        # Remove socket if it exists
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        server_script = self.src_dir / 'mgpu_scheduler_server.py'
        if not server_script.exists():
            raise FileNotFoundError(f"Server script not found: {server_script}")
        
        # Start server process
        self.server_proc = subprocess.Popen([
            sys.executable, str(server_script), '--max-job-time', '300'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(self.socket_path):
                time.sleep(0.5)  # Give it a moment to fully initialize
                print("✓ Server started successfully")
                return True
            time.sleep(0.1)
        
        raise TimeoutError("Server failed to start within timeout")
    
    def stop_server(self):
        """Stop the scheduler server"""
        if self.server_proc:
            print("Stopping scheduler server...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
                self.server_proc.wait()
            
            # Clean up socket
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
    
    def send_command(self, command):
        """Send a command to the scheduler and return response"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect(self.socket_path)
            sock.send(json.dumps(command).encode())
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            return response
        except Exception as e:
            print(f"Error sending command: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_server_connection(self):
        """Test basic server connectivity"""
        print("\n=== Testing Server Connection ===")
        
        try:
            response = self.send_command({'cmd': 'queue'})
            if response['status'] == 'ok':
                print("✓ Server connection successful")
                print(f"  Initial queue: {len(response.get('queue', []))} jobs")
                print(f"  Initial running: {len(response.get('running', []))} jobs")
                return True
            else:
                print(f"✗ Server returned error: {response}")
                return False
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
    
    def test_job_submission_and_execution(self):
        """Test job submission and execution"""
        print("\n=== Testing Job Submission and Execution ===")
        
        # Create a simple test script
        test_script = """#!/usr/bin/env python3
import time
import sys
print("Test job starting...", flush=True)
for i in range(3):
    print(f"Step {i+1}/3", flush=True)
    time.sleep(1)
print("Test job completed!", flush=True)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Submit job
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': f'python {script_path}',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] != 'ok':
                print(f"✗ Job submission failed: {response}")
                return False
            
            job_id = response['job_id']
            print(f"✓ Job submitted successfully: {job_id}")
            
            # Wait for job to start and complete
            max_wait = 30
            start_time = time.time()
            job_completed = False
            
            while time.time() - start_time < max_wait:
                response = self.send_command({'cmd': 'queue'})
                if response['status'] == 'ok':
                    running_jobs = {job['id']: job for job in response['running']}
                    queued_jobs = {job['id']: job for job in response['queue']}
                    
                    if job_id in running_jobs:
                        print(f"  Job {job_id} is running...")
                    elif job_id in queued_jobs:
                        print(f"  Job {job_id} is queued...")
                    else:
                        print(f"✓ Job {job_id} completed")
                        job_completed = True
                        break
                
                time.sleep(1)
            
            if job_completed:
                return True
            else:
                print(f"✗ Job did not complete within {max_wait} seconds")
                return False
                
        finally:
            # Clean up test script
            try:
                os.unlink(script_path)
            except:
                pass
    
    def test_job_cancellation(self):
        """Test job cancellation functionality"""
        print("\n=== Testing Job Cancellation ===")
        
        # Create a long-running test script
        test_script = """#!/usr/bin/env python3
import time
import signal
import sys

def signal_handler(sig, frame):
    print("Job cancelled!", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("Long job starting...", flush=True)
for i in range(60):  # 60 seconds
    print(f"Working... {i+1}/60", flush=True)
    time.sleep(1)
print("Long job completed!", flush=True)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Submit long job
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': f'python {script_path}',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] != 'ok':
                print(f"✗ Job submission failed: {response}")
                return False
            
            job_id = response['job_id']
            print(f"✓ Long job submitted: {job_id}")
            
            # Wait for job to start running
            start_time = time.time()
            job_running = False
            
            while time.time() - start_time < 10:
                response = self.send_command({'cmd': 'queue'})
                if response['status'] == 'ok':
                    running_jobs = {job['id']: job for job in response['running']}
                    if job_id in running_jobs:
                        print(f"✓ Job {job_id} is running")
                        job_running = True
                        break
                time.sleep(0.5)
            
            if not job_running:
                print(f"✗ Job failed to start running")
                return False
            
            # Cancel the job
            time.sleep(2)  # Let it run a bit
            cancel_response = self.send_command({'cmd': 'cancel', 'job_id': job_id})
            
            if cancel_response['status'] != 'ok':
                print(f"✗ Job cancellation failed: {cancel_response}")
                return False
            
            print(f"✓ Cancellation command sent")
            
            # Verify job is no longer running
            time.sleep(2)
            response = self.send_command({'cmd': 'queue'})
            if response['status'] == 'ok':
                running_jobs = {job['id']: job for job in response['running']}
                if job_id not in running_jobs:
                    print(f"✓ Job {job_id} successfully cancelled")
                    return True
                else:
                    print(f"✗ Job {job_id} still running after cancellation")
                    return False
            
            return False
                
        finally:
            # Clean up test script
            try:
                os.unlink(script_path)
            except:
                pass
    
    def test_multiple_jobs(self):
        """Test multiple job submission and queue management"""
        print("\n=== Testing Multiple Jobs ===")
        
        job_ids = []
        
        try:
            # Submit multiple jobs
            for i in range(3):
                job_request = {
                    'cmd': 'submit',
                    'user': os.getenv('USER', 'testuser'),
                    'gpus': 1,
                    'mem': 1000,
                    'cmdline': f'echo "Job {i+1}" && sleep 2',
                    'interactive': False,
                    'priority': i  # Different priorities
                }
                
                response = self.send_command(job_request)
                if response['status'] == 'ok':
                    job_ids.append(response['job_id'])
                    print(f"✓ Job {i+1} submitted: {response['job_id']}")
                else:
                    print(f"✗ Job {i+1} submission failed: {response}")
                    return False
            
            # Check queue status
            response = self.send_command({'cmd': 'queue'})
            if response['status'] == 'ok':
                total_jobs = len(response['running']) + len(response['queue'])
                print(f"✓ Total jobs in system: {total_jobs}")
                print(f"  Running: {len(response['running'])}")
                print(f"  Queued: {len(response['queue'])}")
                
                # Wait for all jobs to complete
                max_wait = 30
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    response = self.send_command({'cmd': 'queue'})
                    if response['status'] == 'ok':
                        all_job_ids = set()
                        all_job_ids.update(job['id'] for job in response['running'])
                        all_job_ids.update(job['id'] for job in response['queue'])
                        
                        remaining_jobs = [jid for jid in job_ids if jid in all_job_ids]
                        if not remaining_jobs:
                            print("✓ All jobs completed")
                            return True
                        
                        print(f"  Waiting for {len(remaining_jobs)} jobs to complete...")
                    
                    time.sleep(2)
                
                print(f"✗ Not all jobs completed within {max_wait} seconds")
                return False
            
            return False
            
        except Exception as e:
            print(f"✗ Multiple jobs test failed: {e}")
            return False
    
    def test_resource_constraints(self):
        """Test resource constraint handling"""
        print("\n=== Testing Resource Constraints ===")
        
        try:
            # Submit job with very high memory requirement (should fail)
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 999999,  # Very high memory
                'cmdline': 'echo "This should not run"',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] == 'fail':
                print("✓ High memory job correctly rejected")
                
                # Submit normal job (should succeed)
                job_request['mem'] = 1000
                response = self.send_command(job_request)
                if response['status'] == 'ok':
                    print("✓ Normal memory job accepted")
                    return True
                else:
                    print(f"✗ Normal job failed: {response}")
                    return False
            else:
                print(f"✗ High memory job was not rejected: {response}")
                return False
                
        except Exception as e:
            print(f"✗ Resource constraints test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("="*60)
        print("Multi-GPU Scheduler Integration Test Suite")
        print("="*60)
        
        tests = [
            ("Server Connection", self.test_server_connection),
            ("Job Submission and Execution", self.test_job_submission_and_execution),
            ("Job Cancellation", self.test_job_cancellation),
            ("Multiple Jobs", self.test_multiple_jobs),
            ("Resource Constraints", self.test_resource_constraints)
        ]
        
        passed = 0
        total = len(tests)
        
        try:
            self.start_server()
            
            for test_name, test_func in tests:
                print(f"\n[{passed+1}/{total}] Running {test_name}...")
                try:
                    if test_func():
                        print(f"✓ {test_name} PASSED")
                        self.test_results[test_name] = 'PASSED'
                        passed += 1
                    else:
                        print(f"✗ {test_name} FAILED")
                        self.test_results[test_name] = 'FAILED'
                except Exception as e:
                    print(f"✗ {test_name} ERROR: {e}")
                    self.test_results[test_name] = f'ERROR: {e}'
        
        finally:
            self.stop_server()
        
        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST RESULTS")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status_icon = "✓" if result == 'PASSED' else "✗"
            print(f"{status_icon} {test_name:30} {result}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            print(f"\n❌ {total - passed} test(s) failed.")
            return False

def main():
    """Main function"""
    runner = SchedulerTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
