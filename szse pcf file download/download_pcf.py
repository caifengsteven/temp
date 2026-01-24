#!/usr/bin/env python3
"""
Script to download PCF (申购赎回清单) files from SZSE (Shenzhen Stock Exchange)
URL: https://www.szse.cn/www/disclosure/fund/currency/
"""

import os
import re
import requests
import argparse
from datetime import datetime, timedelta
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.szse.cn"
REPORT_BASE_URL = "https://reportdocs.static.szse.cn"
API_URL = "https://www.szse.cn/api/report/ShowReport/data"


def get_date_range(days_back=1):
    """Get default date range (today and yesterday by default)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def fetch_file_list(start_date, end_date, page=1, pagesize=100):
    """Fetch the list of PCF files from the SZSE API."""
    params = {
        "SHOWTYPE": "JSON",
        "CATALOGID": "sgshqd",
        "txtStart": start_date,
        "txtEnd": end_date,
        "pageno": page,
        "pagesize": pagesize,
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.szse.cn/www/disclosure/fund/currency/",
    }
    
    response = requests.get(API_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_file_urls(data):
    """Parse the API response and extract file URLs."""
    files = []
    if not data or not data[0].get("data"):
        return files

    for item in data[0]["data"]:
        jjdm = item.get("jjdm", "")

        # Extract the txt file path from encode-open attribute
        # These files are at reportdocs.static.szse.cn
        txt_match = re.search(r"encode-open='([^']+)'", jjdm)
        if txt_match:
            txt_path = txt_match.group(1)
            files.append({
                "url": f"{REPORT_BASE_URL}{txt_path}",
                "filename": os.path.basename(txt_path),
                "type": "txt"
            })

        # Extract PCF download files from the download link
        pcf_match = re.search(r"filename=([^&]+)", jjdm)
        if pcf_match:
            filenames = unquote(pcf_match.group(1)).split(";")
            for fname in filenames:
                if fname:
                    files.append({
                        "url": f"{REPORT_BASE_URL}/files/text/ETFDown/{fname}.txt",
                        "filename": f"{fname}.txt",
                        "type": "pcf"
                    })

    return files


def download_file(file_info, output_dir, headers, convert_to_utf8=True):
    """Download a single file."""
    url = file_info["url"]
    filename = file_info["filename"]
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return ("skipped", f"Skipped (exists): {filename}")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Check if we got HTML (error page) instead of actual data
        content = response.content
        if b'<!DOCTYPE' in content[:100] or b'<html' in content[:100]:
            return ("failed", f"Failed (got HTML error page): {filename}")

        # Convert from GBK to UTF-8 if requested
        if convert_to_utf8:
            try:
                text = content.decode('gbk')
                content = text.encode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass  # Keep original content if conversion fails

        with open(filepath, "wb") as f:
            f.write(content)

        return ("success", f"Downloaded: {filename}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return ("not_found", f"Not found (404): {filename}")
        return ("failed", f"Failed: {filename} - {e}")
    except Exception as e:
        return ("failed", f"Failed: {filename} - {e}")


def main():
    parser = argparse.ArgumentParser(description="Download PCF files from SZSE")
    parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", "-d", type=int, default=1, help="Days back from today (default: 1)")
    parser.add_argument("--output", "-o", default="downloads", help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Number of download workers")
    parser.add_argument("--type", "-t", choices=["all", "txt", "pcf"], default="txt", 
                        help="File type to download (default: txt)")
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = get_date_range(args.days)
    
    print(f"Fetching PCF files from {start_date} to {end_date}...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Fetch all pages
    all_files = []
    page = 1
    
    while True:
        print(f"Fetching page {page}...")
        data = fetch_file_list(start_date, end_date, page=page, pagesize=100)
        
        if not data or not data[0].get("data"):
            break
        
        metadata = data[0].get("metadata", {})
        total_pages = metadata.get("pagecount", 0)
        record_count = metadata.get("recordcount", 0)
        
        if page == 1:
            print(f"Total records: {record_count}, Total pages: {total_pages}")
        
        files = parse_file_urls(data)
        
        # Filter by type
        if args.type != "all":
            files = [f for f in files if f["type"] == args.type]
        
        all_files.extend(files)

        if page >= total_pages:
            break
        page += 1

    # Deduplicate files by URL
    seen_urls = set()
    unique_files = []
    for f in all_files:
        if f["url"] not in seen_urls:
            seen_urls.add(f["url"])
            unique_files.append(f)

    print(f"\nFound {len(all_files)} total entries, {len(unique_files)} unique files to download")
    
    if not unique_files:
        print("No files found for the specified date range.")
        return

    all_files = unique_files

    # Download files
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.szse.cn/www/disclosure/fund/currency/",
    }
    
    print(f"Downloading to: {os.path.abspath(args.output)}")
    
    # Statistics
    stats = {"success": 0, "skipped": 0, "not_found": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, f, args.output, headers): f for f in all_files}

        for i, future in enumerate(as_completed(futures), 1):
            status, message = future.result()
            stats[status] += 1
            print(f"[{i}/{len(all_files)}] {message}")

    print("\n" + "=" * 50)
    print("Download Summary:")
    print(f"  Successfully downloaded: {stats['success']}")
    print(f"  Skipped (already exist): {stats['skipped']}")
    print(f"  Not found (404):         {stats['not_found']}")
    print(f"  Failed (other errors):   {stats['failed']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

