#!/usr/bin/env python3
"""
Test validation script - checks that all test files are complete and executable.
"""
import os
import sys
import ast
import subprocess
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_file_executable(file_path):
    """Check if a file is executable"""
    return os.access(file_path, os.X_OK)

def check_test_completeness(file_path):
    """Check if test file appears to be complete"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for basic test structure
        has_main = 'def main():' in content or 'if __name__ == "__main__":' in content
        has_docstring = '"""' in content or "'''" in content
        has_print_statements = 'print(' in content
        
        issues = []
        if not has_main:
            issues.append("No main function or main guard found")
        if not has_docstring:
            issues.append("No docstring found")
        if not has_print_statements:
            issues.append("No print statements found (may be incomplete)")
        
        # Check for TODO or incomplete markers
        incomplete_markers = ['TODO', 'FIXME', 'NotImplemented', '...']
        for marker in incomplete_markers:
            if marker in content:
                issues.append(f"Contains incomplete marker: {marker}")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def check_test_imports(file_path):
    """Check if test file has necessary imports"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse imports
        tree = ast.parse(content)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        # Check for common required imports
        expected_imports = {
            'os', 'sys', 'time', 'subprocess'
        }
        
        missing_imports = expected_imports - imports
        if missing_imports and len(content) > 1000:  # Only check substantial files
            return False, f"Missing common imports: {missing_imports}"
        
        return True, None
        
    except Exception as e:
        return False, f"Error parsing imports: {e}"

def main():
    """Main validation function"""
    print("Multi-GPU Scheduler Test Validation")
    print("=" * 50)
    
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob('test_*.py'))
    
    if not test_files:
        print("❌ No test files found!")
        return False
    
    print(f"Found {len(test_files)} test files to validate\n")
    
    all_passed = True
    
    for test_file in sorted(test_files):
        print(f"Validating {test_file.name}...")
        file_passed = True
        
        # Check syntax
        syntax_ok, syntax_error = check_python_syntax(test_file)
        if syntax_ok:
            print("  ✓ Syntax valid")
        else:
            print(f"  ✗ Syntax error: {syntax_error}")
            file_passed = False
        
        # Check if executable
        if check_file_executable(test_file):
            print("  ✓ File is executable")
        else:
            print("  ? File is not executable (you may need to chmod +x)")
        
        # Check completeness
        complete_ok, complete_issues = check_test_completeness(test_file)
        if complete_ok:
            print("  ✓ Test appears complete")
        else:
            print(f"  ? Potential issues:")
            for issue in complete_issues:
                print(f"    - {issue}")
        
        # Check imports
        imports_ok, import_error = check_test_imports(test_file)
        if imports_ok:
            print("  ✓ Imports look good")
        else:
            print(f"  ? Import issue: {import_error}")
        
        # Try to run with --help or basic check
        try:
            result = subprocess.run([
                sys.executable, str(test_file), '--help'
            ], capture_output=True, text=True, timeout=5)
            
            # If --help doesn't work, try importing the module
            if result.returncode != 0:
                result = subprocess.run([
                    sys.executable, '-c', f'import sys; sys.path.append("{test_dir}"); import {test_file.stem}'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    print("  ✓ Module can be imported")
                else:
                    print(f"  ? Import test failed: {result.stderr.strip()}")
            else:
                print("  ✓ Help option works")
        
        except subprocess.TimeoutExpired:
            print("  ? Basic execution test timed out")
        except Exception as e:
            print(f"  ? Execution test error: {e}")
        
        if not file_passed:
            all_passed = False
        
        print()
    
    # Check for run_tests.py
    run_tests_file = test_dir / 'run_tests.py'
    if run_tests_file.exists():
        print("✓ Main test runner (run_tests.py) exists")
        
        # Check if it references all test files
        with open(run_tests_file, 'r') as f:
            runner_content = f.read()
        
        referenced_tests = []
        for test_file in test_files:
            if test_file.name in runner_content:
                referenced_tests.append(test_file.name)
        
        print(f"✓ Test runner references {len(referenced_tests)}/{len(test_files)} test files")
        
        missing_refs = [tf.name for tf in test_files if tf.name not in runner_content]
        if missing_refs:
            print(f"  ? Not referenced in runner: {missing_refs}")
    else:
        print("✗ Main test runner (run_tests.py) not found")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed validation!")
        print("\nTo run tests:")
        print("  python run_tests.py --quick    # Quick test suite")
        print("  python run_tests.py           # Full test suite")
        print("  python run_tests.py --list    # List all tests")
    else:
        print("❌ Some validation issues found.")
        print("Please review the issues above and fix any problems.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
