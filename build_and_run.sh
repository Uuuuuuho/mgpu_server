#!/bin/bash
# Ubuntu 22.x 환경에서 Python 서버/클라이언트 바이너리 빌드 및 실행 스크립트
# 사용법: ./build_and_run.sh

set -e

# 1. pyinstaller 설치 (필요시)
if ! command -v pyinstaller &> /dev/null; then
    echo "[INFO] pyinstaller가 설치되어 있지 않습니다. 설치를 진행합니다."
    pip install pyinstaller
fi

# 2. 바이너리 빌드
for f in mgpu_scheduler_server.py mgpu_srun.py mgpu_queue.py mgpu_cancel.py; do
    if [ -f "$f" ]; then
        echo "[INFO] 빌드: $f"
        pyinstaller --onefile "$f"
    fi
done

# 3. 실행 안내
cat <<EOF

[INFO] 빌드 완료! 실행 파일은 dist/ 디렉토리에 생성됩니다.

서버 실행:
  ./dist/mgpu_scheduler_server

클라이언트 예시:
  ./dist/mgpu_srun --gpus 1 --mem 8000 -- echo hello
  ./dist/mgpu_queue
  ./dist/mgpu_cancel <job_id>

EOF
