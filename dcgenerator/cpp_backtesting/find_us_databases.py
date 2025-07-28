#!/usr/bin/env python3
"""
Find US market database files in the specified directory.
"""

import os
import glob

def find_databases():
    print("=== US Market Database Finder ===")
    print()
    
    # Possible base paths
    base_paths = [
        "F:\\BaiduNetdiskDownload\\1分钟",
        "F:\\BaiduNetdiskDownload\\1分钟\\",
        "F:\\BaiduNetdiskDownload",
        "F:\\BaiduNetdiskDownload\\"
    ]
    
    found_files = []
    
    for base_path in base_paths:
        print(f"Checking path: {base_path}")
        
        if os.path.exists(base_path):
            print(f"  ✓ Path exists")
            
            # List all files in directory
            try:
                all_files = os.listdir(base_path)
                print(f"  Found {len(all_files)} files/folders")
                
                # Look for database files
                db_files = [f for f in all_files if f.lower().endswith(('.db', '.sqlite', '.sqlite3'))]
                
                if db_files:
                    print(f"  Database files found:")
                    for db_file in db_files:
                        full_path = os.path.join(base_path, db_file)
                        file_size = os.path.getsize(full_path)
                        print(f"    - {db_file} ({file_size:,} bytes)")
                        found_files.append(full_path)
                else:
                    print(f"  No database files found")
                
                # Show first few files for reference
                if all_files:
                    print(f"  Sample files:")
                    for i, file in enumerate(all_files[:5]):
                        print(f"    - {file}")
                    if len(all_files) > 5:
                        print(f"    ... and {len(all_files) - 5} more")
                        
            except PermissionError:
                print(f"  ✗ Permission denied")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ Path does not exist")
        
        print()
    
    print("=== Summary ===")
    if found_files:
        print(f"Found {len(found_files)} database file(s):")
        for file_path in found_files:
            print(f"  {file_path}")
        
        print()
        print("To use with C++ program, update the path in us_market_dc_test.cpp")
        print("or create symbolic links with expected names:")
        for file_path in found_files:
            filename = os.path.basename(file_path)
            if "etf" in filename.lower():
                print(f"  mklink US_ETF_1min.db \"{file_path}\"")
            elif "stock" in filename.lower():
                print(f"  mklink US_stock_1min.db \"{file_path}\"")
    else:
        print("No database files found!")
        print()
        print("Please check:")
        print("1. Files exist in F:\\BaiduNetdiskDownload\\1分钟\\")
        print("2. Files have .db, .sqlite, or .sqlite3 extension")
        print("3. Path is accessible and correct")

if __name__ == "__main__":
    find_databases()
