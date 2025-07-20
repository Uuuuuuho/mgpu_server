import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from datetime import datetime

def check_cuda_availability():
    """CUDA 사용 가능성 확인"""
    print("="*50)
    print("CUDA 환경 확인")
    print("="*50)
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    print()

def matrix_multiplication_test(device, size=4096, iterations=100):
    """행렬 곱셈으로 GPU 부하 테스트"""
    print(f"행렬 곱셈 테스트 시작 ({device})")
    print(f"행렬 크기: {size}x{size}, 반복 횟수: {iterations}")
    
    # 랜덤 행렬 생성
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # GPU 메모리 동기화
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        c = torch.matmul(a, b)
        if i % 20 == 0:
            print(f"진행률: {i+1}/{iterations}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"총 실행 시간: {elapsed:.2f}초")
    print(f"평균 연산 시간: {elapsed/iterations*1000:.2f}ms")
    print()
    
    return elapsed

def convolution_test(device, batch_size=32, iterations=50):
    """CNN 연산으로 GPU 부하 테스트"""
    print(f"합성곱 연산 테스트 시작 ({device})")
    print(f"배치 크기: {batch_size}, 반복 횟수: {iterations}")
    
    # 간단한 CNN 모델 정의
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(256 * 28 * 28, 1000)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN().to(device)
    input_data = torch.randn(batch_size, 3, 224, 224, device=device)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for i in range(iterations):
        with torch.no_grad():
            output = model(input_data)
        if i % 10 == 0:
            print(f"진행률: {i+1}/{iterations}")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"총 실행 시간: {elapsed:.2f}초")
    print(f"평균 추론 시간: {elapsed/iterations*1000:.2f}ms")
    print()
    
    return elapsed

def memory_stress_test(device, memory_gb=2):
    """GPU 메모리 부하 테스트"""
    print(f"메모리 부하 테스트 시작 ({device})")
    print(f"할당할 메모리: {memory_gb} GB")
    
    if device.type == 'cuda':
        # 현재 메모리 사용량 확인
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"초기 메모리 사용량: {initial_memory:.2f} GB")
    
    try:
        # 대용량 텐서 생성
        elements = int(memory_gb * 1024**3 / 4)  # float32는 4바이트
        large_tensor = torch.randn(elements, device=device, dtype=torch.float32)
        
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"현재 메모리 사용량: {current_memory:.2f} GB")
        
        # 메모리에서 연산 수행
        print("메모리에서 연산 수행 중...")
        result = torch.sum(large_tensor)
        print(f"합계: {result.item():.2e}")
        
        # 메모리 해제
        del large_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"메모리 해제 후: {final_memory:.2f} GB")
        
        print("메모리 부하 테스트 완료")
        
    except RuntimeError as e:
        print(f"메모리 부하 테스트 실패: {e}")
    
    print()

def parallel_computation_test(device, num_streams=4):
    """병렬 연산 테스트 (CUDA 스트림 사용)"""
    if device.type != 'cuda':
        print("병렬 연산 테스트는 CUDA에서만 지원됩니다.")
        return
    
    print(f"병렬 연산 테스트 시작 (스트림 수: {num_streams})")
    
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    tensors = []
    
    start_time = time.time()
    
    # 각 스트림에서 병렬로 연산 수행
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            c = torch.matmul(a, b)
            tensors.append(c)
            print(f"스트림 {i+1} 시작")
    
    # 모든 스트림 동기화
    torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"병렬 연산 완료 시간: {elapsed:.2f}초")
    print()

def continuous_load_test(device, duration_seconds=30):
    """지속적인 부하 테스트"""
    print(f"지속적인 부하 테스트 시작 ({duration_seconds}초간)")
    print("Ctrl+C를 눌러 중단할 수 있습니다.")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # 다양한 연산을 번갈아가며 수행
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            
            # 행렬 곱셈
            c = torch.matmul(a, b)
            
            # 요소별 연산
            d = torch.sin(c) + torch.cos(c)
            
            # Reduction 연산
            result = torch.sum(d)
            
            iteration += 1
            
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"진행 시간: {elapsed:.1f}s, 반복 횟수: {iteration}")
        
        total_time = time.time() - start_time
        print(f"지속적인 부하 테스트 완료")
        print(f"총 시간: {total_time:.2f}초, 총 반복: {iteration}회")
        print(f"평균 TPS: {iteration/total_time:.1f}")
        
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\n사용자에 의해 중단됨")
        print(f"실행 시간: {total_time:.2f}초, 반복 횟수: {iteration}회")
    
    print()

def main():
    print("PyTorch CUDA 부하 테스트")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # CUDA 사용 가능성 확인
    check_cuda_availability()
    
    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA 디바이스에서 테스트를 실행합니다.")
    else:
        device = torch.device('cpu')
        print("CPU 디바이스에서 테스트를 실행합니다.")
    
    print()
    
    try:
        # 1. 행렬 곱셈 테스트
        matrix_multiplication_test(device, size=2048, iterations=5000)
        
        # 2. CNN 연산 테스트
        convolution_test(device, batch_size=4, iterations=300)
        
        # 3. 메모리 부하 테스트
        memory_stress_test(device, memory_gb=1)
        
        # 4. 병렬 연산 테스트 (CUDA만)
        if device.type == 'cuda':
            parallel_computation_test(device, num_streams=4)
        
        # 5. 지속적인 부하 테스트
        continuous_load_test(device, duration_seconds=20)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("모든 테스트 완료")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()