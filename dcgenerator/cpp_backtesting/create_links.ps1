# PowerShell script to create symbolic links for US market databases
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "Creating Symbolic Links for US Market Databases (PowerShell)" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host ""

# Define source paths
$sourcePath = "F:\BaiduNetdiskDownload\US stock ane etf 1mins"
$etfSource = Join-Path $sourcePath "US_ETF_1min.db"
$stockSource = Join-Path $sourcePath "US_stock_1min.db"

# Define target paths (current directory)
$etfTarget = "US_ETF_1min.db"
$stockTarget = "US_stock_1min.db"

Write-Host "Source directory: $sourcePath"
Write-Host ""

# Check if source files exist
if (-not (Test-Path $etfSource)) {
    Write-Host "ERROR: US_ETF_1min.db not found at $etfSource" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path $stockSource)) {
    Write-Host "ERROR: US_stock_1min.db not found at $stockSource" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Source files found:" -ForegroundColor Green
Write-Host "  ETF Database: $etfSource"
Write-Host "  Stock Database: $stockSource"
Write-Host ""

# Remove existing links/files if they exist
if (Test-Path $etfTarget) {
    Write-Host "Removing existing $etfTarget..."
    Remove-Item $etfTarget -Force
}

if (Test-Path $stockTarget) {
    Write-Host "Removing existing $stockTarget..."
    Remove-Item $stockTarget -Force
}

# Try to create symbolic links
Write-Host "Creating symbolic links..." -ForegroundColor Yellow

try {
    # Create symbolic link for ETF database
    New-Item -ItemType SymbolicLink -Path $etfTarget -Target $etfSource -Force
    Write-Host "✓ Created symbolic link: $etfTarget" -ForegroundColor Green
    
    # Create symbolic link for Stock database
    New-Item -ItemType SymbolicLink -Path $stockTarget -Target $stockSource -Force
    Write-Host "✓ Created symbolic link: $stockTarget" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "=======================================================" -ForegroundColor Green
    Write-Host "Symbolic links created successfully!" -ForegroundColor Green
    Write-Host "You can now run: us_market_dc_test.exe --explore" -ForegroundColor Green
    Write-Host "=======================================================" -ForegroundColor Green
    
} catch {
    Write-Host "Failed to create symbolic links. Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "This might be due to insufficient privileges." -ForegroundColor Yellow
    Write-Host "Try running PowerShell as Administrator, or use copy_databases.bat instead." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
