#!/usr/bin/env python3
"""
Multi-node distributed training test script
Tests PyTorch distributed functionality across multiple nodes
"""
import os
import sys
import time
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """분산 환경 설정"""
    # 환경 변수에서 분산 설정 정보 가져오기
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"[Node {rank}] Setting up distributed training...")
    print(f"[Node {rank}] Rank: {rank}, World Size: {world_size}")
    print(f"[Node {rank}] Master: {master_addr}:{master_port}")
    
    # CUDA 디바이스 설정
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # 분산 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    
    print(f"[Node {rank}] Distributed setup complete. Device: {device}")
    return rank, world_size, device

def cleanup_distributed():
    """분산 환경 정리"""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """간단한 테스트 모델"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

def test_distributed_training():
    """분산 훈련 테스트"""
    try:
        rank, world_size, device = setup_distributed()
        
        # 모델 생성 및 DDP 래핑
        model = SimpleModel().to(device)
        model = DDP(model, device_ids=[device.index])
        
        # 옵티마이저 설정
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        print(f"[Node {rank}] Starting distributed training simulation...")
        
        # 훈련 루프 시뮬레이션
        for epoch in range(5):
            # 더미 데이터 생성
            data = torch.randn(32, 10).to(device)
            target = torch.randn(32, 1).to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 모든 노드에서 loss 수집 (테스트용)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            avg_loss = loss.item() / world_size
            
            if rank == 0:  # 마스터 노드에서만 출력
                print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}")
            
            time.sleep(1)  # 진행 상황을 보기 위한 딜레이
        
        print(f"[Node {rank}] Training completed successfully!")
        
        # 동기화 테스트
        dist.barrier()
        if rank == 0:
            print("All nodes completed training. Test PASSED!")
        
    except Exception as e:
        print(f"[Node {rank}] Error in distributed training: {e}")
        sys.exit(1)
    finally:
        cleanup_distributed()

def test_single_node():
    """단일 노드 모드 테스트"""
    print("Testing single-node mode (no distributed setup)...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Starting single-node training simulation...")
    
    for epoch in range(3):
        data = torch.randn(32, 10).to(device)
        target = torch.randn(32, 1).to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/3, Loss: {loss.item():.4f}")
        time.sleep(1)
    
    print("Single-node training completed successfully!")

def main():
    print("=== Multi-Node GPU Scheduler Test ===")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # 환경 변수 출력
    env_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"{var}: {value}")
    
    print("-" * 50)
    
    # 분산 모드인지 확인
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("Detected distributed mode - running distributed test")
        test_distributed_training()
    else:
        print("No distributed environment detected - running single-node test")
        test_single_node()

if __name__ == "__main__":
    main()
