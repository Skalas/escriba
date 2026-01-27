#!/usr/bin/env python3
"""
Test script to verify integration with audio-capture CLI.

This script tests reading PCM audio data from the Swift CLI's stdout.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_list_command():
    """Test the --list command."""
    print("Testing --list command...")
    try:
        result = subprocess.run(
            ['swift', 'run', 'audio-capture', '--list'],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"Exit code: {result.returncode}")
        print(f"Stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except FileNotFoundError:
        print("Swift not found. Make sure Swift is installed.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_capture_short():
    """Test capturing a short amount of audio."""
    print("\nTesting audio capture (5 seconds)...")
    print("Make sure you have audio playing on your system!")
    
    try:
        # Try to use the built executable first
        executable = Path(__file__).parent / ".build" / "release" / "audio-capture"
        if not executable.exists():
            executable = Path(__file__).parent / ".build" / "debug" / "audio-capture"
        
        if not executable.exists():
            print("Executable not found. Please build first:")
            print("  cd swift-audio-capture && swift build -c release")
            return False
        
        process = subprocess.Popen(
            [str(executable), '--sample-rate', '16000', '--channels', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        print("Capturing audio for 5 seconds...")
        total_bytes = 0
        chunks_received = 0
        
        start_time = time.time()
        while time.time() - start_time < 5.0:
            chunk = process.stdout.read(32000)  # Read ~1 second of audio at 16kHz mono
            if chunk:
                total_bytes += len(chunk)
                chunks_received += 1
                print(f"Received chunk {chunks_received}: {len(chunk)} bytes (total: {total_bytes} bytes)")
            else:
                time.sleep(0.1)
        
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        
        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
        if stderr_output:
            print(f"\nStderr output:\n{stderr_output}")
        
        print(f"\nTotal bytes received: {total_bytes}")
        print(f"Expected for 5 seconds at 16kHz mono: {5 * 16000 * 2} bytes (32000 bytes)")
        
        if total_bytes > 0:
            print("✅ Audio capture test passed!")
            return True
        else:
            print("❌ No audio data received")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("audio-capture Integration Test")
    print("=" * 60)
    
    # Test 1: List command
    test1_passed = test_list_command()
    
    # Test 2: Short capture (only if list works)
    test2_passed = False
    if test1_passed:
        print("\n" + "=" * 60)
        test2_passed = test_capture_short()
    else:
        print("\nSkipping capture test (list command failed)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  List command: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Audio capture: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
