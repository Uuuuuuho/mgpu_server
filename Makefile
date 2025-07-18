# Makefile for Multi-GPU Scheduler

.PHONY: build clean test install dev-install help regenerate-specs

# Default target
help:
	@echo "Available targets:"
	@echo "  build           - Build binary executables using PyInstaller"
	@echo "  clean           - Remove build artifacts"
	@echo "  clean-all       - Remove build artifacts and spec files"
	@echo "  regenerate-specs - Regenerate PyInstaller spec files"
	@echo "  test            - Run test scripts"
	@echo "  install         - Install production dependencies"
	@echo "  dev-install     - Install development dependencies"
	@echo "  help            - Show this help message"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev-install:
	pip install -r requirements.txt
	pip install pyinstaller

# Build binary executables
build: dev-install
	./build_and_run.sh

# Clean build artifacts
clean:
	rm -rf build/ dist/ __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Clean build artifacts including spec files
clean-all: clean
	rm -rf build-config/

# Regenerate PyInstaller spec files
regenerate-specs:
	pyinstaller --onefile --specpath build-config --hidden-import=psutil --hidden-import=select src/mgpu_scheduler_server.py --name mgpu_scheduler_server --noconfirm
	pyinstaller --onefile --specpath build-config src/mgpu_srun.py --name mgpu_srun --noconfirm
	pyinstaller --onefile --specpath build-config src/mgpu_queue.py --name mgpu_queue --noconfirm
	pyinstaller --onefile --specpath build-config src/mgpu_cancel.py --name mgpu_cancel --noconfirm
	@echo "Spec files regenerated in build-config/"

# Run tests (requires server to be running)
test:
	@echo "Make sure the server is running: python src/mgpu_scheduler_server.py"
	@echo "Running output streaming test..."
	python src/mgpu_srun.py --gpu-ids 0 -- python test/test_output.py
	@echo ""
	@echo "To test cancellation manually, run:"
	@echo "python src/mgpu_srun.py --gpu-ids 0 -- python test/test_cancellation.py"
	@echo "Then press Ctrl+C after a few seconds"

# Development server (runs from source)
dev-server:
	python src/mgpu_scheduler_server.py

# Development client examples
dev-test-output:
	python src/mgpu_srun.py --gpu-ids 0 -- python test/test_output.py

dev-test-cancel:
	python src/mgpu_srun.py --gpu-ids 0 -- python test/test_cancellation.py

dev-queue:
	python src/mgpu_queue.py
