#!/usr/bin/env python3
"""
Performance and stress testing for the Multi-GPU Scheduler.
Tests system behavior under various load conditions and edge cases.
"""
import os
import sys
import time
import socket
import json
import subprocess
import threading
import tempfile
import concurrent.futures
from pathlib import Path

class PerformanceTestRunner:
    """Performance and stress test runner"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.src_dir = self.test_dir.parent / 'src'
        self.socket_path = '/tmp/mgpu_scheduler.sock'
        self.server_proc = None
        
    def start_server(self):
        """Start the scheduler server for testing"""
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        
        server_script = self.src_dir / 'mgpu_scheduler_server.py'
        self.server_proc = subprocess.Popen([
            sys.executable, str(server_script), '--max-job-time', '60'
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
    
    def send_command(self, command, timeout=10):
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
    
    def test_high_frequency_submissions(self):
        """Test rapid job submissions"""
        print("\n=== Testing High Frequency Job Submissions ===")
        
        num_jobs = 20
        submission_times = []
        job_ids = []
        
        print(f"Submitting {num_jobs} jobs rapidly...")
        
        start_time = time.time()
        
        for i in range(num_jobs):
            job_start = time.time()
            
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': f'echo "Job {i}" && sleep 1',
                'interactive': False
            }
            
            response = self.send_command(job_request, timeout=5)
            job_end = time.time()
            
            submission_times.append(job_end - job_start)
            
            if response['status'] == 'ok':
                job_ids.append(response['job_id'])
            else:
                print(f"Job {i} submission failed: {response}")
        
        total_time = time.time() - start_time
        
        print(f"✓ Submitted {len(job_ids)}/{num_jobs} jobs successfully")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average submission time: {sum(submission_times)/len(submission_times)*1000:.1f}ms")
        print(f"  Max submission time: {max(submission_times)*1000:.1f}ms")
        print(f"  Submission rate: {len(job_ids)/total_time:.1f} jobs/sec")
        
        # Wait for some jobs to start processing
        time.sleep(5)
        
        # Check queue status
        response = self.send_command({'cmd': 'queue'})
        if response['status'] == 'ok':
            running_count = len(response['running'])
            queued_count = len(response['queue'])
            print(f"  Jobs running: {running_count}")
            print(f"  Jobs queued: {queued_count}")
        
        return len(job_ids) >= num_jobs * 0.9  # 90% success rate
    
    def test_concurrent_client_connections(self):
        """Test multiple concurrent client connections"""
        print("\n=== Testing Concurrent Client Connections ===")
        
        num_clients = 10
        
        def submit_job(client_id):
            """Submit a job from a specific client"""
            try:
                job_request = {
                    'cmd': 'submit',
                    'user': os.getenv('USER', 'testuser'),
                    'gpus': 1,
                    'mem': 1000,
                    'cmdline': f'echo "Client {client_id}" && sleep 2',
                    'interactive': False
                }
                
                start_time = time.time()
                response = self.send_command(job_request, timeout=10)
                end_time = time.time()
                
                return {
                    'client_id': client_id,
                    'success': response['status'] == 'ok',
                    'response_time': end_time - start_time,
                    'job_id': response.get('job_id', None)
                }
            except Exception as e:
                return {
                    'client_id': client_id,
                    'success': False,
                    'error': str(e),
                    'response_time': None
                }
        
        print(f"Starting {num_clients} concurrent clients...")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = [executor.submit(submit_job, i) for i in range(num_clients)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        successful_clients = [r for r in results if r['success']]
        response_times = [r['response_time'] for r in successful_clients if r['response_time']]
        
        print(f"✓ {len(successful_clients)}/{num_clients} clients successful")
        print(f"  Total time: {total_time:.2f}s")
        if response_times:
            print(f"  Average response time: {sum(response_times)/len(response_times)*1000:.1f}ms")
            print(f"  Max response time: {max(response_times)*1000:.1f}ms")
        
        for result in results:
            if not result['success']:
                print(f"  Client {result['client_id']} failed: {result.get('error', 'Unknown error')}")
        
        return len(successful_clients) >= num_clients * 0.8  # 80% success rate
    
    def test_job_cancellation_stress(self):
        """Test job cancellation under stress"""
        print("\n=== Testing Job Cancellation Stress ===")
        
        # Submit multiple long-running jobs
        job_ids = []
        num_jobs = 5
        
        print(f"Submitting {num_jobs} long-running jobs...")
        
        for i in range(num_jobs):
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': f'echo "Long job {i}" && sleep 30',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] == 'ok':
                job_ids.append(response['job_id'])
            else:
                print(f"Job {i} submission failed: {response}")
        
        print(f"✓ Submitted {len(job_ids)} jobs")
        
        # Wait for jobs to start
        time.sleep(3)
        
        # Cancel all jobs rapidly
        print("Cancelling all jobs rapidly...")
        
        start_time = time.time()
        cancellation_results = []
        
        for job_id in job_ids:
            cancel_start = time.time()
            response = self.send_command({'cmd': 'cancel', 'job_id': job_id})
            cancel_end = time.time()
            
            cancellation_results.append({
                'job_id': job_id,
                'success': response['status'] == 'ok',
                'time': cancel_end - cancel_start
            })
        
        total_cancel_time = time.time() - start_time
        
        successful_cancellations = [r for r in cancellation_results if r['success']]
        
        print(f"✓ Cancelled {len(successful_cancellations)}/{len(job_ids)} jobs")
        print(f"  Total cancellation time: {total_cancel_time:.2f}s")
        
        if cancellation_results:
            cancel_times = [r['time'] for r in cancellation_results]
            print(f"  Average cancellation time: {sum(cancel_times)/len(cancel_times)*1000:.1f}ms")
        
        # Verify jobs are actually cancelled
        time.sleep(2)
        response = self.send_command({'cmd': 'queue'})
        if response['status'] == 'ok':
            remaining_jobs = len(response['running']) + len(response['queue'])
            print(f"  Remaining jobs in system: {remaining_jobs}")
            
            if remaining_jobs == 0:
                print("✓ All jobs successfully removed from system")
                return True
            else:
                print("? Some jobs may still be in the system")
                return len(successful_cancellations) >= len(job_ids) * 0.8
        
        return False
    
    def test_memory_pressure(self):
        """Test system behavior under memory pressure"""
        print("\n=== Testing Memory Pressure ===")
        
        # Submit jobs with varying memory requirements
        job_configs = [
            {'mem': 100, 'desc': 'low memory'},
            {'mem': 1000, 'desc': 'medium memory'},
            {'mem': 2000, 'desc': 'high memory'},
            {'mem': 5000, 'desc': 'very high memory'},
            {'mem': 50000, 'desc': 'excessive memory (should fail)'}
        ]
        
        results = []
        
        for config in job_configs:
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': config['mem'],
                'cmdline': f'echo "Memory test: {config["desc"]}" && sleep 2',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            results.append({
                'config': config,
                'success': response['status'] == 'ok',
                'response': response
            })
            
            print(f"  {config['desc']} ({config['mem']}MB): {'✓' if response['status'] == 'ok' else '✗'}")
        
        # Count successful submissions (last one should fail)
        successful_jobs = [r for r in results if r['success']]
        failed_jobs = [r for r in results if not r['success']]
        
        print(f"✓ {len(successful_jobs)} jobs accepted, {len(failed_jobs)} rejected")
        
        # The last job (excessive memory) should be rejected
        excessive_mem_result = results[-1]
        if not excessive_mem_result['success']:
            print("✓ Excessive memory job correctly rejected")
            return True
        else:
            print("✗ Excessive memory job was incorrectly accepted")
            return False
    
    def test_queue_status_performance(self):
        """Test queue status query performance under load"""
        print("\n=== Testing Queue Status Performance ===")
        
        # Submit several jobs to populate the queue
        num_jobs = 15
        job_ids = []
        
        print(f"Populating queue with {num_jobs} jobs...")
        
        for i in range(num_jobs):
            job_request = {
                'cmd': 'submit',
                'user': os.getenv('USER', 'testuser'),
                'gpus': 1,
                'mem': 1000,
                'cmdline': f'echo "Queue test {i}" && sleep 5',
                'interactive': False
            }
            
            response = self.send_command(job_request)
            if response['status'] == 'ok':
                job_ids.append(response['job_id'])
        
        print(f"✓ Submitted {len(job_ids)} jobs")
        
        # Perform multiple rapid queue status queries
        num_queries = 50
        query_times = []
        
        print(f"Performing {num_queries} rapid queue queries...")
        
        for i in range(num_queries):
            start_time = time.time()
            response = self.send_command({'cmd': 'queue'}, timeout=5)
            end_time = time.time()
            
            if response['status'] == 'ok':
                query_times.append(end_time - start_time)
            else:
                print(f"Query {i} failed: {response}")
        
        if query_times:
            avg_time = sum(query_times) / len(query_times)
            max_time = max(query_times)
            min_time = min(query_times)
            
            print(f"✓ {len(query_times)}/{num_queries} queries successful")
            print(f"  Average query time: {avg_time*1000:.1f}ms")
            print(f"  Max query time: {max_time*1000:.1f}ms")
            print(f"  Min query time: {min_time*1000:.1f}ms")
            
            # Performance benchmark: queries should complete quickly
            if avg_time < 0.1:  # 100ms average
                print("✓ Query performance is good")
                return True
            else:
                print("? Query performance may be slow")
                return avg_time < 0.5  # 500ms tolerance
        
        return False
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("="*60)
        print("Multi-GPU Scheduler Performance Test Suite")
        print("="*60)
        
        tests = [
            ("High Frequency Submissions", self.test_high_frequency_submissions),
            ("Concurrent Client Connections", self.test_concurrent_client_connections),
            ("Job Cancellation Stress", self.test_job_cancellation_stress),
            ("Memory Pressure", self.test_memory_pressure),
            ("Queue Status Performance", self.test_queue_status_performance)
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
                    start_time = time.time()
                    success = test_func()
                    end_time = time.time()
                    
                    if success:
                        print(f"✓ {test_name} PASSED ({end_time-start_time:.1f}s)")
                        test_results[test_name] = 'PASSED'
                        passed += 1
                    else:
                        print(f"✗ {test_name} FAILED ({end_time-start_time:.1f}s)")
                        test_results[test_name] = 'FAILED'
                        
                except Exception as e:
                    print(f"✗ {test_name} ERROR: {e}")
                    test_results[test_name] = f'ERROR: {e}'
                
                # Clean up between tests
                time.sleep(2)
        
        finally:
            self.stop_server()
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST RESULTS")
        print("="*60)
        
        for test_name, result in test_results.items():
            status_icon = "✓" if result == 'PASSED' else "✗"
            print(f"{status_icon} {test_name:30} {result}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL PERFORMANCE TESTS PASSED!")
            print("The scheduler performs well under various load conditions.")
            return True
        else:
            print(f"\n❌ {total - passed} performance test(s) failed.")
            print("Consider optimizing scheduler performance.")
            return False

def main():
    """Main function"""
    runner = PerformanceTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
