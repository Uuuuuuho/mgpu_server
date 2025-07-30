# 특정 노드 할당 문제 디버깅 가이드

## 📋 문제 상황
클라이언트가 특정 노드(예: node2)에 작업을 할당했지만, 실제로는 다른 노드(예: node1)에서 실행되는 것처럼 보이는 문제

## 🔍 진단 절차

### 1단계: 시스템 상태 확인
```bash
# 클러스터 전체 상태 확인
python3 diagnostic_client.py health

# 출력 예시:
# === Cluster Health Check ===
# Total nodes: 2
#   node001: 🟢 Healthy
#     Available GPUs: [0]
#     Last seen: 5.2s ago
#   node002: 🟢 Healthy  
#     Available GPUs: [0, 1]
#     Last seen: 3.1s ago
```

### 2단계: 특정 노드 할당 테스트
```bash
# node002에 작업 할당 테스트
python3 diagnostic_client.py test-node node002 --gpu 0

# 출력 분석:
# Expected Node: node002
# Actual Hostname: [실제 실행된 호스트명]
# Actual IP: [실제 IP 주소]
# ✅ MATCH 또는 ❌ MISMATCH
```

### 3단계: 네트워크 구성 확인
```bash
# 포트 바인딩 확인
netstat -tlnp | grep 808

# 프로세스 확인
ps aux | grep mgpu_simple

# 호스트명 확인
hostname
hostname -I
```

## 🛠 개선된 기능들

### A. 자동 디버그 명령어 삽입
마스터 서버가 자동으로 모든 작업에 디버그 정보를 추가합니다:
```python
def create_debug_command(self, original_cmd: str, node_id: str, job_id: str) -> str:
    debug_prefix = '''echo "=== JOB EXECUTION DEBUG INFO ==="
echo "Job ID: {job_id}"
echo "Target Node ID: {node_id}"
echo "Actual Hostname: $(hostname)"
...'''
    return f"{debug_prefix}\n{original_cmd}"
```

### B. 노드 건강성 모니터링
```python
def get_node_health_status(self, node_id: str) -> Dict[str, Any]:
    # 실패 횟수, 마지막 하트비트, GPU 가용성 등 종합 분석
    # 노드가 비정상 상태인지 자동 감지
```

### C. 스케줄링 진단
```python
def diagnose_scheduling_issue(self, job: SimpleJob) -> Dict[str, Any]:
    # 작업이 스케줄링되지 않는 이유 자동 분석
    # 권장 해결책 제시
```

## 🔧 문제 해결 방법

### 케이스 1: hostname vs node_id 불일치
**증상**: node002로 할당했지만 hostname이 다름
**해결책**: 
1. 노드 등록 시 실제 hostname 확인
2. /etc/hosts 파일에서 매핑 확인
3. DNS 설정 검토

### 케이스 2: 네트워크 라우팅 문제
**증상**: 포트 8083으로 보낸 요청이 8081로 라우팅됨
**해결책**:
1. 방화벽 규칙 확인
2. Docker/container 포트 매핑 검토
3. 프록시 설정 확인

### 케이스 3: 클라이언트 모니터링 오류
**증상**: 실제로는 올바른 노드에서 실행되지만 잘못 추적됨
**해결책**:
1. 디버그 출력으로 실제 실행 위치 확인
2. 클라이언트 연결 상태 검토
3. stdout/stderr 리다이렉션 확인

## 📊 로그 분석

### 마스터 서버 로그 패턴
```
INFO - ✅ Node node002 selected for job ABC123
INFO - Job ABC123 started on node002 with GPUs [0]
```

### 작업 출력에서 확인할 정보
```
=== JOB EXECUTION DEBUG INFO ===
Job ID: ABC123
Target Node ID: node002
Actual Hostname: node002-hostname
Actual IP: 192.168.1.100
============================
```

## 🚀 사용 예시

### 정상 케이스
```bash
$ python3 diagnostic_client.py test-node node002
=== Testing Node Assignment: node002 ===
Cluster Status: ok
Available Nodes:
  node001: GPUs=[0]
  node002: GPUs=[0, 1]
Submitting test job to node002:GPU0
Job submitted: A1B2C3
Job completed!
Output:
  Expected Node: node002
  Actual Hostname: node002-host
  Actual IP: 192.168.1.100

=== Execution Location Analysis ===
Expected Node: node002
Actual Hostname: node002-host
Actual IP: 192.168.1.100
✅ MATCH: Job executed on expected node
```

### 문제 케이스
```bash
$ python3 diagnostic_client.py test-node node002
...
❌ MISMATCH: Job executed on different node!
This indicates a potential issue with node assignment
```

## 📈 성능 모니터링

개선된 시스템은 다음 메트릭을 제공합니다:
- 노드별 실패 횟수 추적
- 작업 재시도 통계
- 리소스 가용성 히스토리
- 스케줄링 성공률

## 🎯 결론

이 디버깅 시스템을 통해:
1. **정확한 실행 위치 추적**: 디버그 정보로 실제 실행 컨텍스트 확인
2. **자동 문제 진단**: 스케줄링 실패 원인 자동 분석
3. **실시간 모니터링**: 노드 상태 및 클러스터 건강성 실시간 추적
4. **구체적인 해결책 제시**: 문제 유형별 맞춤형 해결 방안
