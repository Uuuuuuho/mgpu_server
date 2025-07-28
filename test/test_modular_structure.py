#!/usr/bin/env python3
"""
Test the new modular structure
"""

import sys
import os
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test core modules
        from mgpu.core.models import SimpleJob, JobProcess, NodeInfo, MessageType
        from mgpu.core.config import Config
        print("✓ Core modules imported successfully")
        
        # Test utils modules  
        from mgpu.utils.logging import setup_logger
        from mgpu.utils.network import connect_to_server, send_json_message
        from mgpu.utils.gpu import get_available_gpus
        print("✓ Utils modules imported successfully")
        
        # Test main components
        from mgpu.client.client import SimpleClient
        from mgpu.server.master import SimpleMaster
        from mgpu.node.agent import SimpleNode
        print("✓ Main components imported successfully")
        
        # Test CLI
        from mgpu.cli import main_client, main_server, main_node
        print("✓ CLI modules imported successfully")
        
        # Test main package
        from mgpu import SimpleClient, SimpleMaster, SimpleNode, Config
        print("✓ Main package imports work")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without starting servers"""
    print("\nTesting basic functionality...")
    
    try:
        from mgpu import SimpleJob, Config, MessageType
        
        # Test job creation
        job = SimpleJob(
            id="test_job",
            user="test_user", 
            command="echo 'test'",
            gpus=1
        )
        print(f"✓ Created job: {job.id}")
        
        # Test config
        config = Config.get_default_timeout_config()
        print(f"✓ Got default config: {config}")
        
        # Test message types
        assert MessageType.SUBMIT == 'submit'
        assert MessageType.QUEUE == 'queue'
        print("✓ Message types working")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

def test_entry_points():
    """Test that entry point files exist and have correct structure"""
    print("\nTesting entry point files...")
    
    entry_points = [
        'src/mgpu_client.py',
        'src/mgpu_server.py', 
        'src/mgpu_node.py'
    ]
    
    for entry_point in entry_points:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', entry_point)
        if os.path.exists(full_path):
            print(f"✓ {entry_point} exists")
        else:
            print(f"✗ {entry_point} missing")
            return False
    
    return True

if __name__ == '__main__':
    print("Multi-GPU Scheduler - Modular Structure Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_basic_functionality()
    all_passed &= test_entry_points()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! The modular structure is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the module structure.")
        sys.exit(1)
