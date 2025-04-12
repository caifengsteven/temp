import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import traceback
import sys

# Import WindPy - WIND's Python API
try:
    from WindPy import w
except ImportError:
    print("ERROR: WindPy module not found. Please ensure that:")
    print("1. WIND terminal is installed on your machine")
    print("2. WindPy is properly installed and accessible to Python")
    print("3. Your WIND subscription is active")
    exit(1)

def initialize_wind():
    """Initialize WIND API connection"""
    print("Initializing WIND API connection...")
    result = w.start()
    
    if result.ErrorCode != 0:
        print(f"Failed to connect to WIND. Error code: {result.ErrorCode}")
        print(f"Error message: {result.Data[0] if hasattr(result, 'Data') and result.Data else 'Unknown error'}")
        return False
    
    print("Successfully connected to WIND API")
    return True

def format_wind_date(date_str=None):
    """Format date for WIND API"""
    if not date_str:
        # Get latest trading day
        today = datetime.now()
        # If weekend, adjust to Friday
        if today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            days_to_subtract = today.weekday() - 4  # Go back to Friday
            today = today - timedelta(days=days_to_subtract)
        return today.strftime("%Y-%m-%d")
    
    try:
        # Parse and validate date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {date_str}. Using latest trading day.")
        return format_wind_date(None)

def check_etf_exists(etf_code):
    """Check if the ETF code exists in WIND database using minimal fields"""
    # Format ETF code with appropriate suffix
    formatted_code = format_etf_code(etf_code)
    
    if not formatted_code:
        return None
    
    # Query only basic ETF info to verify existence - using minimal fields
    print(f"Verifying ETF code: {formatted_code}")
    etf_info = w.wss(formatted_code, "sec_name")
    
    if etf_info.ErrorCode != 0:
        print(f"Error: ETF code '{formatted_code}' not found in WIND database.")
        print(f"Error code: {etf_info.ErrorCode}")
        print(f"Error message: {etf_info.Data[0] if hasattr(etf_info, 'Data') and etf_info.Data else 'Unknown error'}")
        return None
    
    # Ensure we have valid data
    if not hasattr(etf_info, 'Data') or len(etf_info.Data) < 1 or not etf_info.Data[0]:
        print(f"Error: No valid data returned for ETF {formatted_code}")
        return None
    
    return {
        'code': formatted_code,
        'name': etf_info.Data[0][0] if etf_info.Data[0] else "Unknown"
    }

def format_etf_code(etf_code):
    """Format ETF code with appropriate exchange suffix"""
    if not etf_code:
        return None
    
    # If already has exchange suffix
    if '.' in etf_code:
        return etf_code
    
    # Add suffix based on code pattern
    code = etf_code.strip()
    if code.startswith('51') or code.startswith('56') or code.startswith('58'):
        return f"{code}.SH"  # Shanghai
    elif code.startswith('1') or code.startswith('3'):
        return f"{code}.SZ"  # Shenzhen
    else:
        # Try to determine by checking existence
        sh_code = f"{code}.SH"
        sh_check = w.wss(sh_code, "sec_name")
        if sh_check.ErrorCode == 0 and hasattr(sh_check, 'Data') and sh_check.Data and sh_check.Data[0]:
            return sh_code
        
        sz_code = f"{code}.SZ"
        sz_check = w.wss(sz_code, "sec_name")
        if sz_check.ErrorCode == 0 and hasattr(sz_check, 'Data') and sz_check.Data and sz_check.Data[0]:
            return sz_code
        
        print(f"Warning: Could not determine exchange for code '{code}'")
        return None

def get_latest_trading_day():
    """Get latest trading day from WIND"""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        trading_day = w.tdaysoffset(0, today_str, "")
        
        if trading_day.ErrorCode == 0 and hasattr(trading_day, 'Data') and trading_day.Data:
            # Check if Data[0] is a datetime object, not a list
            if isinstance(trading_day.Data[0], datetime):
                return trading_day.Data[0].strftime("%Y-%m-%d")
            elif isinstance(trading_day.Data[0], list) and trading_day.Data[0] and isinstance(trading_day.Data[0][0], datetime):
                return trading_day.Data[0][0].strftime("%Y-%m-%d")
        
        # Fallback to business day logic
        return format_wind_date(None)
    except Exception as e:
        print(f"Error getting latest trading day: {str(e)}")
        # Fallback to current date with business day logic
        return format_wind_date(None)

def try_get_wind_data(method_name, func, *args, **kwargs):
    """Generic function to try getting data from WIND with error handling"""
    try:
        print(f"Trying {method_name}...")
        result = func(*args, **kwargs)
        
        if result.ErrorCode != 0:
            print(f"{method_name} failed. Error code: {result.ErrorCode}")
            error_msg = result.Data[0] if hasattr(result, 'Data') and result.Data else 'No data'
            print(f"Error message: {error_msg}")
            return None
        
        if not hasattr(result, 'Data') or len(result.Data) == 0 or len(result.Data[0]) == 0:
            print(f"{method_name}: No data returned")
            return None
            
        return result
    except Exception as e:
        print(f"{method_name} exception: {str(e)}")
        traceback.print_exc()  # Print detailed stack trace
        return None

def get_etf_components_simplified(etf_code, date=None):
    """Simplified method to get ETF components, focusing on reliability over completeness"""
    if date is None:
        date = get_latest_trading_day()
    
    print(f"Retrieving components for ETF {etf_code} as of {date} using simplified approach...")
    
    # Try first method: etfcomponents
    result = try_get_wind_data(
        "etfcomponents method", 
        w.wset,
        "etfcomponents", 
        f"date={date};windcode={etf_code}"
    )
    
    if result is not None:
        # Get the field names from result.Fields
        fields = result.Fields if hasattr(result, 'Fields') else []
        
        # Create a dictionary mapping field names to data columns
        data_dict = {}
        for i, field in enumerate(fields):
            if i < len(result.Data):
                data_dict[field] = result.Data[i]
        
        # Convert to DataFrame
        if data_dict:
            return pd.DataFrame(data_dict)
    
    # Try second method: constituents
    print("First method failed, trying to get constituents...")
    
    # Get the tracking index for the ETF
    index_result = try_get_wind_data(
        "Get tracking index", 
        w.wss,
        etf_code, 
        "fund_trackindexcode"
    )
    
    if index_result is not None and index_result.Data[0] and index_result.Data[0][0]:
        index_code = index_result.Data[0][0]
        print(f"ETF tracks index: {index_code}")
        
        # Get index constituents
        const_result = try_get_wind_data(
            "Index constituents", 
            w.wset,
            "indexconstituent", 
            f"date={date};windcode={index_code}"
        )
        
        if const_result is not None:
            # Get the field names from result.Fields
            fields = const_result.Fields if hasattr(const_result, 'Fields') else []
            
            # Create a dictionary mapping field names to data columns
            data_dict = {}
            for i, field in enumerate(fields):
                if i < len(const_result.Data):
                    data_dict[field] = const_result.Data[i]
            
            # Convert to DataFrame
            if data_dict:
                df = pd.DataFrame(data_dict)
                # Add a source column
                df['source'] = 'index tracking'
                return df
    
    # Try third method: basic portfolio data (might be limited)
    print("Second method failed, trying basic portfolio data...")
    
    try:
        # Get direct component data from fund NAV
        portfolio_data = w.wss(etf_code, "prt_stocktoasset,nav_date,prt_fundtoasset")
        
        if portfolio_data.ErrorCode == 0 and hasattr(portfolio_data, 'Data') and portfolio_data.Data:
            print("Retrieved basic portfolio allocation information")
            # Create a simple dataframe with this limited information
            allocation_data = {
                'Asset Type': ['Stocks', 'Other Funds', 'Cash/Other'],
                'Allocation (%)': [
                    portfolio_data.Data[0][0] if portfolio_data.Data[0] else 0,
                    portfolio_data.Data[2][0] if len(portfolio_data.Data) > 2 and portfolio_data.Data[2] else 0,
                    100 - (
                        (portfolio_data.Data[0][0] if portfolio_data.Data[0] else 0) + 
                        (portfolio_data.Data[2][0] if len(portfolio_data.Data) > 2 and portfolio_data.Data[2] else 0)
                    )
                ],
                'As of Date': [
                    portfolio_data.Data[1][0] if len(portfolio_data.Data) > 1 and portfolio_data.Data[1] else date,
                    portfolio_data.Data[1][0] if len(portfolio_data.Data) > 1 and portfolio_data.Data[1] else date,
                    portfolio_data.Data[1][0] if len(portfolio_data.Data) > 1 and portfolio_data.Data[1] else date
                ]
            }
            
            # Mark this as a special case - allocation only, not components
            df = pd.DataFrame(allocation_data)
            df['is_allocation_only'] = True
            print("Note: Only allocation data available, not individual components")
            return df
    except Exception as e:
        print(f"Third method exception: {str(e)}")
    
    return None

def get_etf_basic_info(etf_code):
    """Get only the most basic information about the ETF that's likely to be available"""
    try:
        # Use only the most essential and commonly available fields
        basic_fields = [
            "sec_name", "fund_setupdate", "fund_mgrcomp", 
            "fund_type", "nav", "price"
        ]
        
        print(f"Retrieving basic information for ETF {etf_code}...")
        etf_info = w.wss(etf_code, ",".join(basic_fields))
        
        if etf_info.ErrorCode != 0:
            print(f"Error retrieving basic ETF info. Error code: {etf_info.ErrorCode}")
            return None
        
        if not hasattr(etf_info, 'Data') or len(etf_info.Data) < len(basic_fields):
            print("Error: Incomplete ETF information retrieved")
            return None
        
        # Create dictionary of ETF information
        info_dict = {
            'code': etf_code,
            'name': etf_info.Data[0][0] if etf_info.Data[0] else "Unknown",
            'setup_date': etf_info.Data[1][0] if len(etf_info.Data) > 1 and etf_info.Data[1] else None,
            'fund_manager': etf_info.Data[2][0] if len(etf_info.Data) > 2 and etf_info.Data[2] else "Unknown",
            'fund_type': etf_info.Data[3][0] if len(etf_info.Data) > 3 and etf_info.Data[3] else None,
            'nav': etf_info.Data[4][0] if len(etf_info.Data) > 4 and etf_info.Data[4] else None,
            'price': etf_info.Data[5][0] if len(etf_info.Data) > 5 and etf_info.Data[5] else None
        }
        
        # Try to get tracking index in a separate call
        try:
            index_result = w.wss(etf_code, "fund_trackindexcode,fund_trackindexname")
            if index_result.ErrorCode == 0 and hasattr(index_result, 'Data') and index_result.Data:
                info_dict['tracking_index_code'] = index_result.Data[0][0] if index_result.Data[0] else None
                info_dict['tracking_index_name'] = index_result.Data[1][0] if len(index_result.Data) > 1 and index_result.Data[1] else None
        except Exception:
            # Ignore errors in getting tracking index
            pass
            
        # Try to get net assets in a separate call
        try:
            assets_result = w.wss(etf_code, "netasset")
            if assets_result.ErrorCode == 0 and hasattr(assets_result, 'Data') and assets_result.Data:
                info_dict['net_assets'] = assets_result.Data[0][0] if assets_result.Data[0] else None
        except Exception:
            # Ignore errors in getting net assets
            pass
        
        return info_dict
    
    except Exception as e:
        print(f"Error retrieving basic ETF info: {str(e)}")
        return None

def export_etf_data(components_df, etf_code, etf_info=None, export_folder="ETF_Data", include_charts=True):
    """
    Export ETF data to files - simplified version for reliability
    """
    if components_df is None or components_df.empty:
        print("No data to export")
        return False
        
    # Create a folder for exports if it doesn't exist
    os.makedirs(export_folder, exist_ok=True)
    
    # Extract ETF code without exchange suffix for filenames
    clean_code = etf_code.split('.')[0]
    
    # Get current date for filename
    current_date = datetime.now().strftime("%Y%m%d")
    
    try:
        # Export to CSV (most reliable format)
        csv_file = os.path.join(export_folder, f"{clean_code}_data_{current_date}.csv")
        components_df.to_csv(csv_file, index=False)
        print(f"Exported ETF data to CSV: {csv_file}")
        
        # Try to export to Excel
        try:
            excel_file = os.path.join(export_folder, f"{clean_code}_data_{current_date}.xlsx")
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Export components/allocation
                if 'is_allocation_only' in components_df.columns and components_df['is_allocation_only'].iloc[0]:
                    components_df.to_excel(writer, sheet_name='Allocation', index=False)
                else:
                    components_df.to_excel(writer, sheet_name='Components', index=False)
                
                # Export ETF info if available
                if etf_info:
                    # Convert ETF info to DataFrame
                    etf_info_df = pd.DataFrame(list(etf_info.items()), columns=['Field', 'Value'])
                    etf_info_df.to_excel(writer, sheet_name='ETF_Info', index=False)
            
            print(f"Exported ETF data to Excel: {excel_file}")
            
        except Exception as e:
            print(f"Could not export to Excel: {str(e)}")
            print("Data is still available in CSV format")
        
        # Create simple visualization if possible and if charts are requested
        if include_charts:
            try:
                # Check if this is allocation-only data
                if 'is_allocation_only' in components_df.columns and components_df['is_allocation_only'].iloc[0]:
                    # Create pie chart of allocation
                    plt.figure(figsize=(10, 7))
                    plt.pie(
                        components_df['Allocation (%)'], 
                        labels=components_df['Asset Type'],
                        autopct='%1.1f%%'
                    )
                    plt.title(f"Asset Allocation - {etf_info['name'] if etf_info else etf_code}")
                    plt.axis('equal')
                    
                    # Save chart
                    pie_file = os.path.join(export_folder, f"{clean_code}_allocation_{current_date}.png")
                    plt.savefig(pie_file)
                    plt.close()
                    print(f"Exported allocation chart: {pie_file}")
                    
                else:
                    # Determine weight column
                    if 'i_weight' in components_df.columns:
                        weight_col = 'i_weight'
                    elif 'weight' in components_df.columns:
                        weight_col = 'weight'
                    else:
                        weight_col = None
                    
                    if weight_col:
                        # Determine name column
                        if 'sec_name' in components_df.columns:
                            name_col = 'sec_name'
                        elif 'name' in components_df.columns:
                            name_col = 'name'
                        else:
                            name_col = None
                        
                        if name_col:
                            # Create top holdings chart
                            plt.figure(figsize=(12, 8))
                            
                            # Sort and get top 10
                            top_df = components_df.sort_values(weight_col, ascending=False).head(10)
                            
                            plt.barh(top_df[name_col], top_df[weight_col])
                            plt.xlabel('Weight (%)')
                            plt.ylabel('Component')
                            plt.title(f"Top 10 Holdings - {etf_info['name'] if etf_info else etf_code}")
                            plt.tight_layout()
                            
                            # Save chart
                            chart_file = os.path.join(export_folder, f"{clean_code}_top_holdings_{current_date}.png")
                            plt.savefig(chart_file)
                            plt.close()
                            print(f"Exported top holdings chart: {chart_file}")
            
            except Exception as e:
                print(f"Could not create visualizations: {str(e)}")
        
        return True  # Successfully exported
    
    except Exception as e:
        print(f"Error exporting data: {str(e)}")
        traceback.print_exc()  # Add stack trace for export errors
        return False

def read_etf_list(filename):
    """Read list of ETF codes from a file"""
    try:
        with open(filename, 'r') as f:
            # Read lines and strip whitespace
            etf_codes = [line.strip() for line in f.readlines()]
            
            # Filter out empty lines and comments
            etf_codes = [code for code in etf_codes if code and not code.startswith('#')]
            
            print(f"Successfully read {len(etf_codes)} ETF codes from {filename}")
            return etf_codes
    except Exception as e:
        print(f"Error reading ETF list from {filename}: {str(e)}")
        return []

def process_etf(etf_code, date=None, export_folder="ETF_Data", include_charts=True):
    """Process a single ETF - get info, components, and export"""
    try:
        # Format ETF code if needed
        if '.' not in etf_code:
            formatted_code = format_etf_code(etf_code)
            if not formatted_code:
                print(f"Could not determine proper format for ETF code {etf_code}")
                return False
            etf_code = formatted_code
        
        # Verify ETF exists
        etf_exists = check_etf_exists(etf_code)
        if not etf_exists:
            print(f"ETF {etf_code} not found in WIND database")
            return False
        
        print(f"Processing ETF: {etf_exists['name']} ({etf_code})")
        
        # Get ETF information
        etf_info = get_etf_basic_info(etf_code)
        if not etf_info:
            print(f"Could not retrieve information for ETF {etf_code}")
            etf_info = {'code': etf_code, 'name': etf_exists['name']}
        
        # Get ETF components
        components_df = get_etf_components_simplified(etf_code, date)
        if components_df is None or components_df.empty:
            print(f"No component data found for ETF {etf_code}")
            return False
        
        # Export data
        return export_etf_data(components_df, etf_code, etf_info, export_folder, include_charts)
    
    except Exception as e:
        print(f"Error processing ETF {etf_code}: {str(e)}")
        traceback.print_exc()
        return False

def batch_process_etfs(etf_codes, date=None, export_folder="ETF_Data", include_charts=True):
    """Process multiple ETFs in batch"""
    results = {
        'successful': [],
        'failed': []
    }
    
    total_etfs = len(etf_codes)
    
    print(f"Starting batch processing of {total_etfs} ETFs...")
    
    # Create export folder if it doesn't exist
    os.makedirs(export_folder, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(export_folder, f"batch_process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w') as log:
        log.write(f"ETF Batch Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total ETFs to process: {total_etfs}\n")
        log.write("=" * 50 + "\n\n")
        
        # Process each ETF
        for i, etf_code in enumerate(etf_codes, 1):
            start_time = time.time()
            
            log.write(f"[{i}/{total_etfs}] Processing ETF: {etf_code}\n")
            print(f"\n[{i}/{total_etfs}] Processing ETF: {etf_code}")
            
            success = process_etf(etf_code, date, export_folder, include_charts)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if success:
                results['successful'].append(etf_code)
                status = "SUCCESS"
            else:
                results['failed'].append(etf_code)
                status = "FAILED"
            
            log.write(f"  Status: {status} (took {duration:.2f} seconds)\n")
            log.write("-" * 50 + "\n")
            
            print(f"  Status: {status} (took {duration:.2f} seconds)")
            
            # Add a small delay between requests to avoid overwhelming the WIND API
            if i < total_etfs:
                time.sleep(1)
    
    # Write summary to log
    with open(log_file, 'a') as log:
        log.write("\n\n" + "=" * 50 + "\n")
        log.write("BATCH PROCESSING SUMMARY\n")
        log.write(f"Total ETFs: {total_etfs}\n")
        log.write(f"Successfully processed: {len(results['successful'])}\n")
        log.write(f"Failed: {len(results['failed'])}\n")
        
        if results['failed']:
            log.write("\nFailed ETFs:\n")
            for etf in results['failed']:
                log.write(f"  - {etf}\n")
    
    # Print summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print(f"Total ETFs: {total_etfs}")
    print(f"Successfully processed: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Log file: {log_file}")
    
    return results

def main():
    """Main function to extract and process ETF component information from a list"""
    print("===== WIND ETF Data Batch Extractor =====")
    
    # Initialize WIND API connection
    if not initialize_wind():
        return
    
    try:
        # Check command line arguments for ETF list file
        if len(sys.argv) > 1:
            etf_list_file = sys.argv[1]
        else:
            # Ask for ETF list file
            etf_list_file = input("Enter path to file containing ETF codes (one per line): ").strip()
            
            # Default file if none provided
            if not etf_list_file:
                etf_list_file = "etf_list.txt"
                print(f"Using default file: {etf_list_file}")
        
        # Check if file exists
        if not os.path.exists(etf_list_file):
            # If default file doesn't exist, create a sample one
            if etf_list_file == "etf_list.txt":
                with open(etf_list_file, 'w') as f:
                    f.write("# List of ETF codes to process (one per line)\n")
                    f.write("# Lines starting with # are comments\n")
                    f.write("513050.SH\n")  # Example ETF
                print(f"Created sample ETF list file: {etf_list_file}")
                print("Please edit this file to add your desired ETFs, then run the script again.")
                return
            else:
                print(f"Error: File {etf_list_file} not found.")
                return
        
        # Read ETF codes from file
        etf_codes = read_etf_list(etf_list_file)
        
        if not etf_codes:
            print("No valid ETF codes found in the file. Please check the file content.")
            return
        
        # Ask for date (optional)
        date_input = input("Enter date for component data (YYYY-MM-DD) or press Enter for latest: ").strip()
        
        # Ask for export folder
        export_folder = input("Enter export folder path or press Enter for default (ETF_Data): ").strip()
        if not export_folder:
            export_folder = "ETF_Data"
        
        # Ask if charts should be included
        include_charts = input("Generate charts? (y/n, default: y): ").strip().lower() != 'n'
        
        # Process ETFs in batch
        batch_process_etfs(etf_codes, date_input if date_input else None, export_folder, include_charts)
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Close WIND connection
        try:
            w.close()
            print("WIND connection closed")
        except:
            pass

if __name__ == "__main__":
    main()