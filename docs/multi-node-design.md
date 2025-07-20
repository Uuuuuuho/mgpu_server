# Multi-Node GPU Scheduler Design

## 현재 아키텍처 vs 멀티노드 아키텍처

### 현재 (Single Node):
```
mgpu_srun → mgpu_scheduler_server → Local GPUs
```

### 목표 (Multi-Node):
```
mgpu_srun → mgpu_master_server → mgpu_node_agent1 → Node1 GPUs
                                → mgpu_node_agent2 → Node2 GPUs
                                → mgpu_node_agent3 → Node3 GPUs
```

## 주요 컴포넌트

### 1. mgpu_master_server.py (새로 작성)
- 클라이언트 요청 처리
- 노드별 리소스 관리 및 스케줄링
- 노드 에이전트와 통신
- 전역 큐 관리

### 2. mgpu_node_agent.py (새로 작성)
- 각 노드에서 실행
- 로컬 GPU 리소스 모니터링
- 마스터 서버와 통신
- 실제 잡 실행

### 3. mgpu_srun.py (수정)
- 노드 지정 옵션 추가 (--nodes, --nodelist)
- 분산 실행 지원

## 네트워크 통신

### Master ↔ Node Agents
- TCP 소켓 통신 (HTTP/gRPC/ZeroMQ 등)
- 하트비트 및 상태 모니터링
- 리소스 정보 동기화

### Client ↔ Master
- 현재와 동일 (UNIX socket 또는 TCP)

## 데이터 구조 변경

### Node 정보:
```python
class Node:
    def __init__(self, node_id, hostname, ip, port, gpu_count):
        self.node_id = node_id
        self.hostname = hostname
        self.ip = ip
        self.port = port
        self.gpu_count = gpu_count
        self.status = 'online'  # online, offline, maintenance
        self.last_heartbeat = time.time()
```

### Job 정보 확장:
```python
class Job:
    def __init__(self, ...):
        # 기존 필드들...
        self.node_list = []  # 실행할 노드 목록
        self.distributed = False  # 분산 실행 여부
        self.master_node = None  # 마스터 노드 (분산 실행시)
```

## 스케줄링 알고리즘

### 1. 리소스 수집
```python
def collect_cluster_resources(self):
    cluster_resources = {}
    for node in self.nodes:
        try:
            resources = self.get_node_resources(node)
            cluster_resources[node.node_id] = resources
        except Exception:
            node.status = 'offline'
    return cluster_resources
```

### 2. 노드 선택 알고리즘
```python
def select_nodes(self, job_requirements):
    # 1. 요청된 GPU 수에 따라 노드 조합 찾기
    # 2. 네트워크 토폴로지 고려 (같은 랙/스위치 우선)
    # 3. 부하 분산 고려
    # 4. 사용자 제약사항 고려 (--nodelist, --exclude)
```

## 구현 단계

### Phase 1: 기본 멀티노드 지원
1. `mgpu_node_agent.py` 구현
2. `mgpu_master_server.py` 구현  
3. 네트워크 통신 프로토콜 정의
4. 기본 스케줄링 로직

### Phase 2: 고급 기능
1. 분산 실행 지원 (MPI, torchrun 등)
2. 노드 장애 처리
3. 로드 밸런싱
4. 네트워크 토폴로지 인식

### Phase 3: 운영 기능
1. 클러스터 모니터링 대시보드
2. 노드 관리 명령어
3. 설정 관리
4. 로깅 및 감사

## 설정 파일 예시

### cluster_config.yaml
```yaml
cluster:
  name: "gpu-cluster"
  master:
    host: "master.example.com"
    port: 8080
  
nodes:
  - node_id: "node001"
    hostname: "gpu001.example.com"
    ip: "192.168.1.10"
    port: 8081
    gpu_count: 8
    gpu_type: "A100"
    
  - node_id: "node002"
    hostname: "gpu002.example.com" 
    ip: "192.168.1.11"
    port: 8081
    gpu_count: 8
    gpu_type: "A100"

networking:
  heartbeat_interval: 30
  timeout: 60
  
scheduling:
  default_policy: "fill_first"  # fill_first, spread, custom
```

## 명령어 확장

### 노드 지정 옵션:
```bash
# 특정 노드에서 실행
mgpu_srun --nodelist=node001,node002 --gpu-ids 0,1 -- python train.py

# 노드 수 지정
mgpu_srun --nodes=2 --gpus-per-node=4 -- python distributed_train.py

# 특정 노드 제외
mgpu_srun --exclude=node003 --gpu-ids 0,1,2,3 -- python train.py
```

### 분산 실행 지원:
```bash
# MPI 실행
mgpu_srun --nodes=2 --gpus-per-node=4 --mpi -- mpirun -np 8 python train.py

# PyTorch 분산 실행
mgpu_srun --nodes=2 --gpus-per-node=4 --distributed -- \
  torchrun --nnodes=2 --nproc_per_node=4 train.py
```

## 보안 고려사항

1. **인증**: 노드 간 상호 인증
2. **암호화**: 네트워크 통신 암호화
3. **권한**: 노드별 접근 권한 관리
4. **감사**: 모든 작업 로깅

## 성능 고려사항

1. **네트워크 대역폭**: 인터커넥트 성능
2. **지연시간**: 스케줄링 응답 시간
3. **확장성**: 노드 추가시 성능 영향
4. **장애 복구**: 노드 장애시 복구 시간

## 호환성

기존 단일 노드 명령어는 그대로 지원하되, 내부적으로 로컬 노드에서만 실행되도록 처리.
