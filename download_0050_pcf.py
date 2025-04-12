import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_yuanta_pcf(etf_code='0050', download_dir=None, chromedriver_path=None):
    """
    Download PCF file from Yuanta ETFs website for a specified ETF code.
    
    Args:
        etf_code (str): The ETF code, default is '0050'
        download_dir (str): Directory to save the downloaded file, default is user's Downloads folder
        chromedriver_path (str): Path to chromedriver executable
    
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    # Set download directory
    if download_dir is None:
        download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    
    os.makedirs(download_dir, exist_ok=True)
    
    # Configure Chrome options
    chrome_options = Options()
    
    # Set download preferences
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Initialize the WebDriver
    try:
        if chromedriver_path:
            # Use specified ChromeDriver
            driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
        else:
            # Try to use the Chrome service without specifying the path (uses system PATH)
            driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error initializing Chrome driver: {str(e)}")
        print("\nPlease try one of these solutions:")
        print("1. Download ChromeDriver manually from https://chromedriver.chromium.org/downloads")
        print("2. Provide the path to chromedriver.exe using the chromedriver_path parameter")
        print("3. Install an older version of Chrome that matches available ChromeDriver versions")
        return None
    
    try:
        # Navigate to the PCF page for the specified ETF
        url = f"https://www.yuantaetfs.com/tradeInfo/pcf/{etf_code}"
        print(f"Navigating to {url}...")
        driver.get(url)
        
        # Wait for the page to load and find the export button
        export_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '匯出excel')]"))
        )
        
        # Get page information before clicking
        try:
            etf_name = driver.find_element(By.XPATH, "//div[contains(@class, 'fund-name')]").text
            date_info = driver.find_element(By.XPATH, "//div[contains(text(), '上傳時間')]").text
            
            print(f"Found ETF: {etf_name}")
            print(f"{date_info}")
        except:
            print("Could not find ETF details, but continuing with download...")
        
        # Click the export button
        print("Clicking the 'Export excel' button...")
        export_button.click()
        
        # Wait for the download to complete (adjust timeout as needed)
        print("Waiting for download to complete...")
        time.sleep(5)
        
        # Try to find the most recently downloaded file in the download directory
        files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
        if files:
            latest_file = max(files, key=os.path.getctime)
            # Check if the file was just downloaded (within last 10 seconds)
            if time.time() - os.path.getctime(latest_file) < 10:
                print(f"Successfully downloaded to: {latest_file}")
                return latest_file
        
        print("Download may have completed but couldn't verify the file location.")
        return None
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None
    
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    # Option 1: Try to use the default ChromeDriver installation
    downloaded_file = download_yuanta_pcf('0050')
    
    # Option 2: If the above fails, specify a path to chromedriver.exe
    if not downloaded_file:
        # Update this path to where you downloaded chromedriver.exe
        chromedriver_path = "C:/path/to/your/chromedriver.exe"  # Example path, update this
        print(f"\nRetrying with specified ChromeDriver path: {chromedriver_path}")
        downloaded_file = download_yuanta_pcf('0050', chromedriver_path=chromedriver_path)
    
    if downloaded_file:
        print(f"PCF file downloaded successfully: {downloaded_file}")
    else:
        print("Failed to download the PCF file.")