import subprocess
import time
import os
import sys

def start_mysql_service():
    """
    Try different methods to start MySQL service
    """
    print("Attempting to start MySQL service...")
    
    # Method 1: Try to start the service using net start
    try:
        result = subprocess.run(['net', 'start', 'MySQL57'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("MySQL service started successfully using 'net start'")
            return True
        else:
            print(f"Failed to start with 'net start': {result.stderr}")
    except Exception as e:
        print(f"Error with 'net start': {e}")
    
    # Method 2: Try to start using sc command
    try:
        result = subprocess.run(['sc', 'start', 'MySQL57'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("MySQL service started successfully using 'sc start'")
            return True
        else:
            print(f"Failed to start with 'sc start': {result.stderr}")
    except Exception as e:
        print(f"Error with 'sc start': {e}")
    
    # Method 3: Try to start MySQL directly
    mysql_paths = [
        r"C:\Program Files\MySQL\MySQL Server 5.7\bin\mysqld.exe",
        r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe",
        r"C:\MySQL\bin\mysqld.exe",
        r"C:\xampp\mysql\bin\mysqld.exe"
    ]
    
    for mysql_path in mysql_paths:
        if os.path.exists(mysql_path):
            print(f"Found MySQL at: {mysql_path}")
            try:
                # Start MySQL as a background process
                subprocess.Popen([mysql_path, '--console'], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
                print("MySQL started directly")
                time.sleep(5)  # Give it time to start
                return True
            except Exception as e:
                print(f"Error starting MySQL directly: {e}")
    
    print("Could not start MySQL service. Please start it manually.")
    print("\nManual steps:")
    print("1. Open Command Prompt as Administrator")
    print("2. Run: net start MySQL57")
    print("3. Or start MySQL Workbench and connect to start the service")
    return False

def check_mysql_status():
    """
    Check if MySQL is running
    """
    try:
        result = subprocess.run(['sc', 'query', 'MySQL57'], 
                              capture_output=True, text=True, shell=True)
        if 'RUNNING' in result.stdout:
            print("MySQL service is running")
            return True
        else:
            print("MySQL service is not running")
            return False
    except Exception as e:
        print(f"Error checking MySQL status: {e}")
        return False

def main():
    print("MySQL Service Manager")
    print("=" * 30)
    
    # Check current status
    if check_mysql_status():
        print("MySQL is already running!")
        return
    
    # Try to start MySQL
    if start_mysql_service():
        print("Waiting for MySQL to fully start...")
        time.sleep(10)
        
        # Verify it's running
        if check_mysql_status():
            print("MySQL is now running successfully!")
        else:
            print("MySQL may have started but status is unclear")
    else:
        print("Failed to start MySQL automatically")
        print("\nPlease try these manual steps:")
        print("1. Open Command Prompt as Administrator")
        print("2. Run: net start MySQL57")
        print("3. Or use MySQL Workbench to start the service")

if __name__ == "__main__":
    main()
