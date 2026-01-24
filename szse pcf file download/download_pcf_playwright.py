"""
Download PCF files from SZSE using Playwright browser automation.
This script automates the browser to set date range, iterate through pages, and download all files.
"""
import os
import time
import argparse
from playwright.sync_api import sync_playwright


def download_file_via_browser(page, context, link, output_dir: str) -> bool:
    """Download a file by clicking the link and capturing the new page content."""
    try:
        encode_open = link.get_attribute("encode-open")
        if not encode_open:
            return False

        filename = encode_open.split("/")[-1]
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  Skipping (exists): {filename}")
            return True

        # Click the link and wait for new page
        with context.expect_page() as new_page_info:
            link.click()

        new_page = new_page_info.value
        new_page.wait_for_load_state("domcontentloaded", timeout=30000)
        time.sleep(1)

        # Get the content from the new page
        content = new_page.content()

        # Extract the text content (the page shows the file content)
        # The content is usually in a <pre> tag or just plain text
        body = new_page.locator("body")
        text_content = body.inner_text()

        # Save the content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)

        print(f"  Downloaded: {filename}")
        new_page.close()
        return True

    except Exception as e:
        print(f"  Error downloading: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download PCF files from SZSE")
    parser.add_argument("--start", default="2026-01-08", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-09", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="downloads", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.abspath(args.output)
    
    print(f"Starting browser automation...")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Output directory: {output_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        page.set_default_timeout(120000)  # 2 minute default timeout

        url = "https://www.szse.cn/www/disclosure/fund/currency/"
        print(f"Navigating to {url}...")
        try:
            page.goto(url, timeout=120000)  # 2 minute timeout
        except Exception as e:
            print(f"Initial navigation error (may be ok): {e}")

        # Wait for the table to be visible
        print("Waiting for page content...")
        page.wait_for_selector("a[encode-open]", timeout=60000)
        time.sleep(2)
        
        # Set date range
        print(f"Setting date range: {args.start} to {args.end}")
        
        # Clear and set start date
        start_input = page.locator("#sgshqd_tab1_txtStart")
        start_input.click()
        start_input.fill(args.start)
        time.sleep(0.5)
        
        # Clear and set end date
        end_input = page.locator("#sgshqd_tab1_txtEnd")
        end_input.click()
        end_input.fill(args.end)
        time.sleep(0.5)
        
        # Click search button
        search_btn = page.locator("button:has-text('查询')")
        search_btn.click()
        print("Clicked search button, waiting for results...")
        time.sleep(3)
        
        # Get total pages from pagination
        total_pages = 1
        try:
            last_page_link = page.locator(".pagination a.last")
            if last_page_link.count() > 0:
                total_pages = int(last_page_link.inner_text())
        except:
            pass
        print(f"Total pages: {total_pages}")
        
        total_downloads = 0

        # Iterate through all pages and download files
        for page_num in range(1, total_pages + 1):
            print(f"\n=== Page {page_num}/{total_pages} ===")

            # Get all download links on current page
            links = page.query_selector_all("a[encode-open]")
            num_links = len(links)
            print(f"Found {num_links} files on this page")

            # First, collect all the encode-open attributes
            file_attrs = []
            for link in links:
                encode_open = link.get_attribute("encode-open")
                if encode_open:
                    file_attrs.append(encode_open)

            # Download each file on this page
            for i, encode_open in enumerate(file_attrs):
                print(f"  Processing file {i+1}/{len(file_attrs)}...")
                filename = encode_open.split("/")[-1]
                filepath = os.path.join(output_path, filename)

                if os.path.exists(filepath):
                    print(f"    Skipping (exists): {filename}")
                    total_downloads += 1
                    continue

                # Find the link with this encode-open attribute and click it
                try:
                    link = page.query_selector(f'a[encode-open="{encode_open}"]')
                    if link and download_file_via_browser(page, context, link, output_path):
                        total_downloads += 1
                except Exception as e:
                    print(f"    Error: {e}")

            # Go to next page if not on last page
            if page_num < total_pages:
                next_btn = page.locator(".pagination li.next a")
                if next_btn.count() > 0:
                    next_btn.click()
                    time.sleep(2)
                    # Wait for new content to load
                    page.wait_for_selector("a[encode-open]", timeout=30000)

        print(f"\n=== Download Complete ===")
        print(f"Total files downloaded: {total_downloads}")
        
        browser.close()

if __name__ == "__main__":
    main()

