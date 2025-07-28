import os

print("=== Simple Database Finder ===")
print()

# Try the main path
main_path = "F:\\BaiduNetdiskDownload"
print(f"Checking: {main_path}")

if os.path.exists(main_path):
    print("  Path exists!")
    
    # List all items
    items = os.listdir(main_path)
    print(f"  Found {len(items)} items:")
    
    for item in items:
        item_path = os.path.join(main_path, item)
        if os.path.isdir(item_path):
            print(f"    [DIR]  {item}")
        else:
            print(f"    [FILE] {item}")
    
    print()
    
    # Look for database files in main directory
    db_files = [f for f in items if f.lower().endswith(('.db', '.sqlite'))]
    if db_files:
        print("Database files in main directory:")
        for db in db_files:
            print(f"  {db}")
    
    # Check subdirectories for database files
    print("Checking subdirectories...")
    for item in items:
        item_path = os.path.join(main_path, item)
        if os.path.isdir(item_path):
            try:
                sub_items = os.listdir(item_path)
                sub_dbs = [f for f in sub_items if f.lower().endswith(('.db', '.sqlite'))]
                if sub_dbs:
                    print(f"  {item}/ contains:")
                    for db in sub_dbs:
                        full_path = os.path.join(item_path, db)
                        size = os.path.getsize(full_path)
                        print(f"    {db} ({size:,} bytes)")
            except:
                print(f"  {item}/ - cannot access")

else:
    print("  Path does not exist!")
    print("  Please check if F:\\BaiduNetdiskDownload exists")

print()
print("=== Done ===")
