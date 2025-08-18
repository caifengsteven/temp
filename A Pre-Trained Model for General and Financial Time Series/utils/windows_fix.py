"""
Windows compatibility utilities for Delphyne project
"""

import sys
import os
import locale


def setup_windows_encoding():
    """
    Setup proper encoding for Windows console to handle Unicode characters.
    Call this at the beginning of your scripts if you encounter encoding issues.
    """
    if sys.platform.startswith('win'):
        try:
            # Try to set UTF-8 encoding for stdout/stderr
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            
            # Set environment variable for Python to use UTF-8
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            print("Windows encoding configured for UTF-8")
            
        except Exception as e:
            print(f"Warning: Could not configure UTF-8 encoding: {e}")
            print("Emojis and special characters may not display correctly")
    
    return True


def safe_print(message: str, use_fallback: bool = True):
    """
    Safely print messages that might contain Unicode characters.
    
    Args:
        message: Message to print
        use_fallback: Whether to use ASCII fallback on encoding errors
    """
    try:
        print(message)
    except UnicodeEncodeError:
        if use_fallback:
            # Replace problematic characters with ASCII equivalents
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
        else:
            # Just print without special characters
            import re
            ascii_message = re.sub(r'[^\x00-\x7F]+', '***', message)
            print(ascii_message)


def get_system_info():
    """Get system information for debugging encoding issues."""
    info = {
        'platform': sys.platform,
        'encoding': sys.stdout.encoding,
        'locale': locale.getpreferredencoding(),
        'python_version': sys.version,
    }
    
    if sys.platform.startswith('win'):
        try:
            import subprocess
            result = subprocess.run(['chcp'], capture_output=True, text=True, shell=True)
            info['windows_codepage'] = result.stdout.strip()
        except:
            info['windows_codepage'] = 'unknown'
    
    return info


if __name__ == "__main__":
    print("Windows Compatibility Check")
    print("=" * 40)
    
    # Show system info
    info = get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 40)
    
    # Test encoding setup
    setup_windows_encoding()
    
    # Test Unicode printing
    print("\nTesting Unicode characters:")
    test_messages = [
        "Basic ASCII text",
        "Unicode test: café, naïve, résumé",
        "Emojis: 🎉 ✅ ❌ 🚀",
        "Math symbols: α β γ δ ∑ ∫",
        "Chinese: 你好世界",
        "Japanese: こんにちは",
    ]
    
    for msg in test_messages:
        print(f"Testing: ", end="")
        safe_print(msg)
    
    print("\nIf you see garbled characters or errors above,")
    print("consider running your Python scripts with:")
    print("  set PYTHONIOENCODING=utf-8")
    print("  python your_script.py")
    print("\nOr use the safe_print() function for problematic output.")
