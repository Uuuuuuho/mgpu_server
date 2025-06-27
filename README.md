# Multi-GPU Scheduler

## 소개

이 프로젝트는 여러 사용자가 동시에 GPU 자원을 효율적으로 사용할 수 있도록 지원하는 멀티 GPU 작업 스케줄러입니다. Slurm의 srun과 유사한 CLI를 제공하며, GPU 메모리/개수 기반 자원 할당, 대기열 관리, 공정성(fairness), 작업별 시간제한, 자동 가상환경 활성화 등 다양한 기능을 지원합니다.

---

## 주요 기능
- 여러 사용자가 동시에 작업 제출 가능
- `mgpu_srun` 명령어로 작업 제출 (Slurm srun 스타일)
- GPU 개수/메모리 요구량 기반 자원 할당
- 자원이 부족하면 대기열에 삽입, 정책에 따라 순서 변경 가능
- 작업별/전체 시간제한, context switch 지원
- 작업별 고유 ID 부여, 대기열/실행중 작업 조회 및 취소 가능
- 서버가 유저 홈의 venv 자동 활성화 (존재 시)
- 실행 중인 작업의 GPU 메모리 점유량을 고려한 안전한 스케줄링

---

## 설치 및 빌드

1. Python 3.8 이상 필요
2. 의존성 설치 (서버/클라이언트 모두 필요)
   ```bash
   pip install pyinstaller
   ```
3. 바이너리 빌드
   ```bash
   ./build_and_run.sh
   ```
   빌드 후 `dist/` 폴더에 실행 파일이 생성됩니다.

---

## 사용법

### 서버 실행
```bash
./dist/mgpu_scheduler_server --max-job-time 3600
```
- `--max-job-time`: (선택) 모든 작업의 최대 점유시간(초)

### 클라이언트 예시
```bash
# 작업 제출 (GPU 1개, 메모리 8000MB, 600초 제한)
./dist/mgpu_srun --gpus 1 --mem 8000 --time-limit 600 -- python train.py

# 작업 제출 (메모리 옵션 없이, 홈의 venv 자동 활성화)
./dist/mgpu_srun --gpus 1 -- python train.py

# 대기열/실행중 작업 조회
./dist/mgpu_queue

# 작업 취소
./dist/mgpu_cancel <job_id>
```

---

## 동작 방식 및 정책
- 서버는 UNIX 도메인 소켓(`/tmp/mgpu_scheduler.sock`)으로 클라이언트와 통신
- 작업 실행 시, 유저 홈 디렉토리에서 실행하며, `venv/bin/activate`가 있으면 자동 활성화
- 실행 중인 작업의 GPU 메모리 점유량을 고려하여, 실제로 자원이 충분할 때만 작업 실행
- 자원이 부족하면 대기열에서 대기, 자원 해제 시 자동 실행
- 작업별/전체 시간제한 초과 시 context switch (대기열 맨 뒤로 이동)

---

## 참고 및 주의사항
- 서버 실행 계정은 sudo 권한이 필요하며, `/etc/sudoers`에서 비밀번호 없이 실행 가능해야 함
- 각 사용자는 자신의 홈 디렉토리에 venv를 만들어 두면 자동 활성화됨
- conda 환경 자동 활성화 등 추가 정책은 코드 수정으로 확장 가능

---

## 라이선스
MIT
