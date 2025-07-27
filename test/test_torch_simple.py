#!/usr/bin/env python3

import sys
import traceback

def main():
    try:
        import torch
        print(f"PyTorch imported successfully: {torch.__version__}")
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            
            device = torch.device('cuda')
            print(f"Using device: {device}")
            
            # Simple test
            print("Creating tensors...")
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            
            print("Performing matrix multiplication...")
            c = torch.matmul(a, b)
            
            print("Computing sum...")
            result = torch.sum(c).item()
            print(f"Result: {result:.2f}")
            
            print("Test completed successfully!")
        else:
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)
            result = torch.sum(c).item()
            print(f"CPU Result: {result:.2f}")
            print("CPU test completed successfully!")
            
    except ImportError as e:
        print(f"Import error: {e}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
