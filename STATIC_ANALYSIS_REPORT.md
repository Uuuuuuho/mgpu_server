# Multi-GPU Scheduler 정적 분석 보고서

## 1. 시스템 아키텍처 분석

### 현재 구현 상태
✅ **완성된 기능들:**
- 노드 등록 및 GPU 정보 로깅
- 특정 노드 지정 작업 스케줄링
- 노드 실패 추적 및 복구
- 재시도 메커니즘 with 지수 백오프
- 대체 노드 선택 (fallback)
- 인터랙티브 작업 지원

### 핵심 구조
```
SimpleMaster (포트 8080)
├── Job Queue Management
├── Node Registry (nodes: Dict[str, NodeInfo])
├── Scheduler Thread
└── Client Handler

SimpleNode (포트 808x)
├── GPU Detection (nvidia-smi)
├── Job Execution
├── Auto Registration
└── Heartbeat/Status Reporting
```

## 2. 발견된 주요 문제점

### 🔴 Critical Issues

#### Issue #1: 작업 실행 위치 추적 불명확성
**문제**: 클라이언트가 "node2에 할당했지만 node1에서 실행"이라고 보고
**원인**: 
- hostname vs logical node_id 불일치 가능성
- 클라이언트 측 모니터링 로직 부족
- 실제 실행 컨텍스트 정보 부족

**해결방안**:
```python
# 개선된 디버그 명령어 생성
def create_debug_command(original_cmd, node_id, job_id):
    debug_prefix = f"""
echo "=== DEBUG: Job {job_id} ==="
echo "Target Node: {node_id}"
echo "Actual Host: $(hostname)"
echo "Actual IP: $(hostname -I)"
echo "=========================="
"""
    return debug_prefix + original_cmd
```

#### Issue #2: 노드 상태 동기화
**문제**: 마스터와 노드 간 상태 불일치 가능성
```python
# 현재 구현 (mgpu_simple_master.py:282-304)
if hasattr(node, 'failure_count') and node.failure_count >= 3:
    logger.error(f"Node {node_id} marked as unhealthy")
```

**개선점**: 노드 상태를 더 정교하게 관리 필요

### 🟡 Medium Issues

#### Issue #3: 스케줄링 로직 복잡성
**현재 로직**:
1. 특정 노드 요청 확인
2. 노드 건강성 체크
3. GPU 가용성 확인
4. 실패 시 fallback to 자동 선택

**개선 가능점**:
- 더 명확한 우선순위 정책
- 로드 밸런싱 고려
- 노드 성능 메트릭 추가

#### Issue #4: 에러 처리 및 복구
```python
# 현재 재시도 로직 (mgpu_simple_master.py:395-408)
if job.retry_count >= 5:  # Maximum 5 retries
    job.status = 'failed'
    job.exit_code = -1
    self.completed_jobs[job.id] = job
```

**개선점**: 재시도 정책을 더 유연하게 설정

## 3. 코드 품질 분석

### ✅ 잘 구현된 부분
1. **자동 노드 등록**: GPU 정보와 함께 자동 등록
2. **실패 추적**: failure_count 기반 노드 상태 관리
3. **재시도 메커니즘**: 지수 백오프와 최대 재시도 제한
4. **로깅**: 상세한 디버그 로깅 구현

### 🔧 개선 필요 부분

#### A. 동시성 및 Thread Safety
```python
# 현재 구현
self.lock = threading.RLock()
with self.lock:
    # 임계 구역
```
**상태**: 기본적인 동시성 보호는 구현됨
**개선점**: 더 세밀한 락 관리 가능

#### B. 설정 관리
```python
# 하드코딩된 상수들
if job.retry_count >= 5:  # Maximum 5 retries
if node.failure_count >= 3:  # Unhealthy threshold
```
**개선점**: 설정 파일로 분리 필요

#### C. 메트릭 및 모니터링
**현재**: 기본적인 상태 로깅만 구현
**필요**: 
- 작업 성공률 추적
- 노드 성능 메트릭
- 시스템 헬스 체크

## 4. 특정 노드 할당 문제 분석

### 문제 상황 재구성
1. 클라이언트: `--node-gpu-ids "node002:0"` 요청
2. 마스터: node002 선택하고 작업 전송
3. 실제: node001에서 실행되는 것처럼 보임

### 가능한 원인들

#### 원인 A: hostname 불일치
```bash
# node002의 실제 hostname이 다를 수 있음
hostname  # 실제 시스템 이름
```

#### 원인 B: 네트워크 라우팅
- node002:8083으로 보낸 요청이 실제로는 node001:8081로 라우팅
- Docker/container 환경에서 포트 매핑 문제

#### 원인 C: 클라이언트 모니터링 오류
- 클라이언트가 잘못된 연결로 출력 모니터링
- stdout/stderr 리다이렉션 문제

## 5. 권장 해결책

### 단기 해결책
1. **디버그 명령어 사용**:
```python
debug_cmd = f"echo 'EXECUTING ON: $(hostname)' && {original_cmd}"
```

2. **네트워크 확인**:
```bash
netstat -tlnp | grep 808  # 포트 바인딩 확인
```

3. **프로세스 추적**:
```python
# 작업 실행 시 PID와 호스트 정보 로깅
```

### 장기 해결책
1. **End-to-End 추적 시스템**
2. **서비스 디스커버리 개선**
3. **헬스 체크 자동화**
4. **메트릭 대시보드**

## 6. 결론

현재 시스템은 **기능적으로는 올바르게 구현**되어 있습니다. 사용자가 보고한 문제는 다음 중 하나일 가능성이 높습니다:

1. **모니터링/추적 문제**: 실제로는 올바른 노드에서 실행되지만 잘못 추적됨
2. **환경 설정 문제**: hostname/IP 매핑이나 네트워크 구성 문제
3. **클라이언트 측 버그**: 출력 수집 로직의 문제

**추천 조치**: 
- debug_job_tracking.py 도구 사용
- 상세한 로깅으로 실행 컨텍스트 확인
- 네트워크 구성 검증
