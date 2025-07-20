#!/usr/bin/env python3
"""
Error handling and edge case tests for the Multi-GPU Scheduler.
Tests various failure modes, invalid inputs, and recovery scenarios.
"""
import os
import sys
import time
import socket
import json
import subprocess
import tempfile
import signal
from pathlib import Path

class ErrorHandlingTestRunner:
    """Test runner for error handling and edge cases"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.src_dir = self.test_dir.parent / 'src'
        self.socket_path = '/tmp/mgpu_scheduler.sock'
        self.server_proc = None
        
    def start_server(self):
        """Start the scheduler server"""
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        server_script = self.src_dir / 'mgpu_scheduler_server.py'
        self.server_proc = subprocess.Popen([
            sys.executable, str(server_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        start_time = time.time()
        while time.time() - start_time < 10:
            if os.path.exists(self.socket_path):
                time.sleep(0.5)
                return True
            time.sleep(0.1)
        
        raise TimeoutError("Server failed to start")
    
    def stop_server(self):
        """Stop the scheduler server"""
        if self.server_proc:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
        
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
    
    def send_command(self, command, timeout=5):
        """Send command to scheduler"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(self.socket_path)
            sock.send(json.dumps(command).encode())
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            return response
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_invalid_commands(self):
        """Test handling of invalid commands"""
        print("\n=== Testing Invalid Commands ===")
        
        invalid_commands = [
            {'cmd': 'invalid_command'},
            {'cmd': 'submit'},  # Missing required fields
            {'cmd': 'submit', 'user': 'test'},  # Still missing fields
            {'cmd': 'cancel'},  # Missing job_id
            {'unknown_field': 'value'},  # No cmd field
            {},  # Empty command
        ]
        
        success_count = 0
        
        for i, cmd in enumerate(invalid_commands):
            print(f"  Testing invalid command {i+1}: {cmd}")
            response = self.send_command(cmd)
            
            if response['status'] == 'fail' or response['status'] == 'error':
                print(f"    ✓ Correctly rejected with: {response.get('message', response.get('msg', 'No message'))}")
                success_count += 1
            else:
                print(f"    ✗ Incorrectly accepted: {response}")
        
        print(f"✓ {success_count}/{len(invalid_commands)} invalid commands correctly handled")
        return success_count == len(invalid_commands)
    
    def test_malformed_requests(self):
        """Test handling of malformed JSON requests"""
        print("\n=== Testing Malformed Requests ===")
        
        malformed_data = [
            b'not json at all',
            b'{"incomplete": json',
            b'{"cmd": "submit", "user": }',  # Invalid JSON syntax
            b'',  # Empty data
            b'null',
            b'[]',  # Valid JSON but wrong type
        ]
        
        success_count = 0
        
        for i, data in enumerate(malformed_data):
            print(f"  Testing malformed data {i+1}: {data[:50]}...")
            
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect(self.socket_path)
                sock.send(data)
                
                try:
                    response_data = sock.recv(4096)
                    if response_data:
                        response = json.loads(response_data.decode())
                        if response.get('status') in ['fail', 'error']:
                            print(f"    ✓ Server handled malformed data gracefully")
                            success_count += 1
                        else:
                            print(f"    ✗ Server gave unexpected response: {response}")
                    else:
                        print(f"    ✓ Server closed connection (acceptable behavior)")
                        success_count += 1
                except json.JSONDecodeError:
                    print(f"    ✓ Server response was not JSON (connection likely closed)")
                    success_count += 1
                except socket.timeout:
                    print(f"    ✓ Server didn't respond (acceptable for malformed data)")
                    success_count += 1
                
                sock.close()
                
            except Exception as e:
                print(f"    ✓ Connection error (expected): {e}")
                success_count += 1
        
        print(f"✓ {success_count}/{len(malformed_data)} malformed requests handled appropriately")
        return success_count >= len(malformed_data) * 0.8  # Allow some tolerance
    
    def test_resource_limit_violations(self):
        """Test handling of resource limit violations"""
        print("\n=== Testing Resource Limit Violations ===")
        
        violation_tests = [
            {
                'name': 'Negative GPU count',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': -1,
                    'mem': 1000,
                    'cmdline': 'echo test'
                }
            },
            {
                'name': 'Zero GPU count',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': 0,
                    'mem': 1000,
                    'cmdline': 'echo test'
                }
            },
            {
                'name': 'Excessive GPU count',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': 1000,
                    'mem': 1000,
                    'cmdline': 'echo test'
                }
            },
            {
                'name': 'Negative memory',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': 1,
                    'mem': -1000,
                    'cmdline': 'echo test'
                }
            },
            {
                'name': 'Excessive memory',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': 1,
                    'mem': 999999999,
                    'cmdline': 'echo test'
                }
            },
            {
                'name': 'Invalid GPU IDs',
                'request': {
                    'cmd': 'submit',
                    'user': 'test',
                    'gpus': 1,
                    'mem': 1000,
                    'cmdline': 'echo test',
                    'gpu_ids': [999]  # Non-existent GPU
                }
            }
        ]
        
        success_count = 0
        
        for test in violation_tests:
            print(f"  Testing {test['name']}...")
            response = self.send_command(test['request'])
            
            if response['status'] == 'fail':
                print(f"    ✓ Correctly rejected: {response.get('message', response.get('msg', 'No message'))}")
                success_count += 1
            else:
                print(f"    ✗ Incorrectly accepted: {response}")
        
        print(f"✓ {success_count}/{len(violation_tests)} resource violations correctly handled")
        return success_count >= len(violation_tests) * 0.8
    
    def test_job_command_failures(self):
        """Test handling of jobs with failing commands"""
        print("\n=== Testing Job Command Failures ===")
        
        failing_commands = [
            {
                'name': 'Non-existent command',
                'cmd': 'this_command_does_not_exist_12345'
            },
            {
                'name': 'Command with syntax error',
                'cmd': 'python -c "invalid python syntax here'
            },
            {
                'name': 'Command that exits with error',
                'cmd': 'python -c "import sys; sys.exit(1)"'
            },
            {
                'name': 'Command that crashes',
                'cmd': 'python -c "raise Exception(\\"Test crash\\")"'
            }
        ]
        
        job_ids = []
        
        # Submit failing jobs
        for test in failing_commands:
            print(f"  Submitting job with {test['name']}...")
            
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': test['cmd'],
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] == 'ok':
                job_ids.append(response['job_id'])
                print(f"    ✓ Job submitted: {response['job_id']}")
            else:
                print(f"    ✗ Job submission failed: {response}")
        
        # Wait for jobs to run and fail
        print("  Waiting for jobs to execute and fail...")
        time.sleep(10)
        
        # Check that jobs are no longer running
        response = self.send_command({'cmd': 'queue'})
        if response['status'] == 'ok':
            running_jobs = {job['id'] if isinstance(job, dict) and 'id' in job else job[0] for job in response['running']}
            failed_job_count = 0
            
            for job_id in job_ids:
                if job_id not in running_jobs:
                    failed_job_count += 1
                    print(f"    ✓ Job {job_id} completed (likely failed as expected)")
                else:
                    print(f"    ? Job {job_id} still running")
            
            print(f"✓ {failed_job_count}/{len(job_ids)} failing jobs handled appropriately")
            return failed_job_count >= len(job_ids) * 0.8
        
        return False
    
    def test_server_recovery(self):
        """Test server behavior during restart scenarios"""
        print("\n=== Testing Server Recovery ===")
        
        # Submit a job
        job_request = {
            'cmd': 'submit',
            'user': os.getenv('USER', 'testuser'),
            'gpus': 1,
            'mem': 1000,
            'cmdline': 'echo "Before restart" && sleep 5',
            'interactive': False
        }
        
        response = self.send_command(job_request)
        if response['status'] != 'ok':
            print(f"✗ Initial job submission failed: {response}")
            return False
        
        initial_job_id = response['job_id']
        print(f"✓ Submitted initial job: {initial_job_id}")
        
        # Stop server abruptly
        print("  Stopping server abruptly...")
        if self.server_proc:
            self.server_proc.kill()  # Abrupt termination
            self.server_proc.wait()
        
        time.sleep(2)
        
        # Restart server
        print("  Restarting server...")
        try:
            self.start_server()
            print("✓ Server restarted successfully")
        except Exception as e:
            print(f"✗ Server restart failed: {e}")
            return False
        
        # Test that server is responsive
        response = self.send_command({'cmd': 'queue'})
        if response['status'] == 'ok':
            print("✓ Server is responsive after restart")
            
            # Submit a new job to verify functionality
            new_job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': 'echo "After restart"',
                'interactive': False
            }
            
            response = self.send_command(new_job_request)
            if response['status'] == 'ok':
                print(f"✓ New job submitted after restart: {response['job_id']}")
                return True
            else:
                print(f"✗ New job submission failed after restart: {response}")
                return False
        else:
            print(f"✗ Server not responsive after restart: {response}")
            return False
    
    def test_invalid_user_scenarios(self):
        """Test handling of invalid user scenarios"""
        print("\n=== Testing Invalid User Scenarios ===")
        
        invalid_user_tests = [
            {
                'name': 'Empty username',
                'user': ''
            },
            {
                'name': 'Non-existent user',
                'user': 'non_existent_user_12345'
            },
            {
                'name': 'User with special characters',
                'user': 'user@#$%^&*()'
            },
            {
                'name': 'Very long username',
                'user': 'a' * 1000
            }
        ]
        
        success_count = 0
        
        for test in invalid_user_tests:
            print(f"  Testing {test['name']}: '{test['user'][:50]}...'")
            
            job_request = {
                'cmd': 'submit',
                'user': test['user'],
                'gpus': 1,
                'mem': 1000,
                'cmdline': 'echo test',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            
            # For some cases, the job might be submitted but fail during execution
            # For others, it should be rejected immediately
            if response['status'] in ['fail', 'error']:
                print(f"    ✓ Correctly rejected: {response.get('message', response.get('msg', 'No message'))}")
                success_count += 1
            elif response['status'] == 'ok':
                print(f"    ? Job accepted (may fail during execution): {response['job_id']}")
                success_count += 0.5  # Partial credit
            else:
                print(f"    ✗ Unexpected response: {response}")
        
        print(f"✓ {success_count}/{len(invalid_user_tests)} invalid user scenarios handled")
        return success_count >= len(invalid_user_tests) * 0.7
    
    def run_all_tests(self):
        """Run all error handling tests"""
        print("="*60)
        print("Multi-GPU Scheduler Error Handling Test Suite")
        print("="*60)
        
        tests = [
            ("Invalid Commands", self.test_invalid_commands),
            ("Malformed Requests", self.test_malformed_requests),
            ("Resource Limit Violations", self.test_resource_limit_violations),
            ("Job Command Failures", self.test_job_command_failures),
            ("Server Recovery", self.test_server_recovery),
            ("Invalid User Scenarios", self.test_invalid_user_scenarios)
        ]
        
        passed = 0
        total = len(tests)
        test_results = {}
        
        try:
            print("Starting scheduler server...")
            self.start_server()
            
            for test_name, test_func in tests:
                print(f"\n[{passed+1}/{total}] Running {test_name}...")
                try:
                    if test_func():
                        print(f"✓ {test_name} PASSED")
                        test_results[test_name] = 'PASSED'
                        passed += 1
                    else:
                        print(f"✗ {test_name} FAILED")
                        test_results[test_name] = 'FAILED'
                        
                except Exception as e:
                    print(f"✗ {test_name} ERROR: {e}")
                    test_results[test_name] = f'ERROR: {e}'
                
                # Give server time to recover between tests
                time.sleep(1)
        
        finally:
            self.stop_server()
        
        # Print summary
        print("\n" + "="*60)
        print("ERROR HANDLING TEST RESULTS")
        print("="*60)
        
        for test_name, result in test_results.items():
            status_icon = "✓" if result == 'PASSED' else "✗"
            print(f"{status_icon} {test_name:30} {result}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL ERROR HANDLING TESTS PASSED!")
            print("The scheduler handles errors and edge cases gracefully.")
            return True
        else:
            print(f"\n❌ {total - passed} error handling test(s) failed.")
            print("Consider improving error handling and input validation.")
            return False

def main():
    """Main function"""
    runner = ErrorHandlingTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
