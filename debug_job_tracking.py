#!/usr/bin/env python3
"""
Job Tracking Debug Tool
실제 작업이 어느 노드에서 실행되는지 정확히 추적하는 도구
"""

import os
import socket
import json
import time
import subprocess

def create_debug_command(original_cmd, node_id, job_id):
    """
    원본 명령어에 디버깅 정보를 추가한 명령어 생성
    """
    debug_info = f"""
echo "=== JOB EXECUTION DEBUG INFO ==="
echo "Job ID: {job_id}"
echo "Assigned Node ID: {node_id}"
echo "Actual Hostname: $(hostname)"
echo "Actual IP Address: $(hostname -I | cut -d' ' -f1)"
echo "Process ID: $$"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "Timestamp: $(date)"
echo "==============================="
    """
    
    return f"{debug_info.strip()}\n{original_cmd}"

def analyze_node_mapping():
    """
    현재 시스템의 노드 매핑 분석
    """
    analysis = {
        "hostname": subprocess.getoutput("hostname"),
        "ip_address": subprocess.getoutput("hostname -I | cut -d' ' -f1"),
        "all_interfaces": subprocess.getoutput("ip addr show | grep 'inet ' | awk '{print $2}'"),
        "listening_ports": subprocess.getoutput("netstat -tlnp 2>/dev/null | grep LISTEN | grep -E '808[0-9]'"),
        "process_info": subprocess.getoutput("ps aux | grep mgpu_simple"),
    }
    return analysis

def verify_job_execution_location(job_id, expected_node, actual_output):
    """
    작업 실행 위치 검증
    """
    verification = {
        "job_id": job_id,
        "expected_node": expected_node,
        "analysis": {}
    }
    
    # 출력에서 디버그 정보 추출
    lines = actual_output.split('\n')
    for line in lines:
        if "Assigned Node ID:" in line:
            verification["analysis"]["assigned_node"] = line.split(":")[1].strip()
        elif "Actual Hostname:" in line:
            verification["analysis"]["actual_hostname"] = line.split(":")[1].strip()
        elif "Actual IP Address:" in line:
            verification["analysis"]["actual_ip"] = line.split(":")[1].strip()
        elif "Process ID:" in line:
            verification["analysis"]["process_id"] = line.split(":")[1].strip()
    
    # 일치성 검사
    verification["location_match"] = (
        verification.get("analysis", {}).get("assigned_node") == expected_node
    )
    
    return verification

def main():
    print("=== Node Mapping Analysis ===")
    analysis = analyze_node_mapping()
    
    for key, value in analysis.items():
        print(f"{key}:")
        print(f"  {value}")
        print()
    
    print("=== Recommendations ===")
    print("1. 작업 제출 시 debug_command를 사용하여 실행 위치 추적")
    print("2. 클라이언트와 노드 모두에서 동일한 hostname/IP 정보 확인")
    print("3. 로그에서 'Assigned Node ID'와 'Actual Hostname' 비교")

if __name__ == "__main__":
    main()
