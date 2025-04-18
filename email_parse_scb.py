import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import extract_msg  # Library to extract .msg files
import random
import string

class FXTradeEmailParser:
    """Parse FX trade confirmation emails and convert to CSV format."""
    
    def __init__(self):
        """Initialize the parser with regex patterns for extracting trade data."""
        self.patterns = {
            # Common patterns
            'currency_pair': r'(?:Currency Pair|vs\.)\s*:?\s*([A-Z]{3}[/]?[A-Z]{3})',
            'tenor': r'Tenor\s*:\s*(\w+)',
            'trade_rate': r'(?:Trade Rate|at)\s*:?\s*([\d.]+)',
            'spot_rate': r'Spot Rate\s*:\s*([\d.]+)',
            'fwd_pts': r'Fwd Pts\s*:\s*([-\d.]+)',
            'value_date': r'Value Date\s*:?\s*(\d{1,2}\s+\w{3,}\s+\d{4}|\d{4}-\w+-\d{1,2})',
            'fixing_date': r'Fixing (?:Date|TAIFX1 Fixing Date)\s*:?\s*(\d{1,2}\s+\w{3,}\s+\d{4}|\d{4}-\w+-\d{1,2})',
            'fixing_mechanism': r'Fixing Mechanism\s*:\s*([A-Z0-9\s]+)',
            
            # SC.com specific patterns
            'jefferies_transaction': r'(?:Jefferies|JEFFERIES).*(?:Buys|Sells|Buy|Sell)[^:]*:?\s*([\d,]+(?:\.\d+)?)\s+([A-Z]{3})\s+and\s+(?:Buys|Sells|Buy|Sell)[^:]*:?\s*([\d,]+(?:\.\d+)?)\s+([A-Z]{3})',
            'scb_transaction': r'SCB\s+(?:Buys|Sells|Buy|Sell)[^:]*:?\s*([\d,]+(?:\.\d+)?)\s+([A-Z]{3})\s+and\s+(?:Buys|Sells|Buy|Sell)[^:]*:?\s*([\d,]+(?:\.\d+)?)\s+([A-Z]{3})',
            
            # CA-CIB specific patterns
            'cacib_transaction': r'(?:JEFFERIES|Jefferies).*(?:Sell|Buy)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?(?:\.00)?)\s+(?:vs\.|against)\s+([A-Z]{3})',
            'cacib_rate': r'at\s+([\d.]+)',
            'cacib_location': r'Facing\s+([A-Z\-]+\s+[A-Z]+\s+[A-Z]+)'
        }
    
    def generate_random_string(self, length=6):
        """Generate random alphanumeric string for reference numbers."""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    def parse_email(self, email_text):
        """
        Parse email text to extract FX trade information.
        
        Args:
            email_text (str): The raw email text
            
        Returns:
            dict: Extracted trade information or None if not a valid trade email
        """
        # Check if this is a trade confirmation email
        if not any(phrase in email_text for phrase in ["Thank you for the trade", "To confirm", "Trade Confirmation"]):
            return None
            
        trade_data = {}
        
        # Determine the source of the email
        if any(source in email_text for source in ["CA-CIB", "CACIB", "ca-cib.com"]):
            trade_data['source'] = 'CACIB'
        else:
            trade_data['source'] = 'SCBL'
        
        # Extract basic trade information
        for key, pattern in self.patterns.items():
            if key not in ['jefferies_transaction', 'scb_transaction', 'cacib_transaction']:
                match = re.search(pattern, email_text)
                if match:
                    trade_data[key] = match.group(1)
        
        # Source-specific parsing
        if trade_data['source'] == 'CACIB':
            # Parse CA-CIB specific format
            cacib_match = re.search(self.patterns['cacib_transaction'], email_text)
            rate_match = re.search(self.patterns['cacib_rate'], email_text)
            
            if cacib_match:
                currency1 = cacib_match.group(1)  # Usually TWD
                amount1 = cacib_match.group(2).replace(',', '')
                currency2 = cacib_match.group(3)  # Usually USD
                
                # Determine if Jefferies is selling or buying
                if "Sell" in email_text.split("JEFFERIES")[1].split("vs")[0]:
                    # Jefferies selling TWD
                    trade_data['bs'] = "B"  # We're buying
                else:
                    # Jefferies buying TWD
                    trade_data['bs'] = "S"  # We're selling
                
                trade_data['position'] = "TWD"
                trade_data['counter'] = "USD"
                
                # Set amounts based on which currency is which
                if currency1 == "TWD":
                    trade_data['position_amount'] = amount1
                    # Counter amount might be calculated from the rate
                    if 'trade_rate' in trade_data:
                        rate = float(trade_data['trade_rate'])
                        trade_data['counter_amount'] = str(float(amount1) / rate)
                else:
                    trade_data['counter_amount'] = amount1
                    # Position amount might be calculated from the rate
                    if 'trade_rate' in trade_data and rate_match:
                        rate = float(rate_match.group(1))
                        trade_data['position_amount'] = str(float(amount1) * rate)
                        trade_data['trade_rate'] = rate_match.group(1)
                
            # Extract location/branch if available
            location_match = re.search(self.patterns['cacib_location'], email_text)
            if location_match:
                trade_data['location'] = location_match.group(1)
                
        else:
            # Parse SC.com specific format
            # Extract Jefferies transaction details
            jefferies_match = re.search(self.patterns['jefferies_transaction'], email_text)
            scb_match = re.search(self.patterns['scb_transaction'], email_text)
            
            if jefferies_match and scb_match:
                # Get Jefferies transaction details
                jeff_amount1 = jefferies_match.group(1).replace(',', '')
                jeff_ccy1 = jefferies_match.group(2)  # Usually USD
                jeff_amount2 = jefferies_match.group(3).replace(',', '')
                jeff_ccy2 = jefferies_match.group(4)  # Usually TWD
                
                # Get SCB transaction details
                scb_amount1 = scb_match.group(1).replace(',', '')
                scb_ccy1 = scb_match.group(2)  # Usually TWD
                scb_amount2 = scb_match.group(3).replace(',', '')
                scb_ccy2 = scb_match.group(4)  # Usually USD
                
                # Jefferies is the counterparty we're focused on
                if "Buys" in email_text.split("Jefferies")[1].split("and")[0] or "Buy" in email_text.split("Jefferies")[1].split("and")[0]:
                    # If Jefferies is buying USD, we're selling USD to them (S)
                    if jeff_ccy1 == "USD":
                        trade_data['bs'] = "S"
                    else:
                        trade_data['bs'] = "B"
                else:
                    # If Jefferies is selling USD, we're buying USD from them (B)
                    if jeff_ccy1 == "USD":
                        trade_data['bs'] = "B"
                    else:
                        trade_data['bs'] = "S"
                
                # Always use the TWD and USD from SCB's side for consistency
                if scb_ccy1 == "TWD":
                    trade_data['position'] = "TWD"
                    trade_data['position_amount'] = scb_amount1
                    trade_data['counter'] = "USD"
                    trade_data['counter_amount'] = scb_amount2
                else:
                    trade_data['position'] = "TWD"
                    trade_data['position_amount'] = scb_amount2
                    trade_data['counter'] = "USD"
                    trade_data['counter_amount'] = scb_amount1
            elif jefferies_match:
                # Only Jefferies transaction found (simpler format)
                if "Sell" in email_text and "TWD" in email_text:
                    # Jefferies selling TWD
                    trade_data['bs'] = "B"  # We're buying TWD
                    trade_data['position'] = "TWD"
                    
                    # Try to extract amounts directly
                    twdMatch = re.search(r'([A-Z]{3})\s+([\d,]+(?:\.\d+)?)', email_text)
                    if twdMatch and twdMatch.group(1) == "TWD":
                        trade_data['position_amount'] = twdMatch.group(2).replace(',', '')
                    
                    trade_data['counter'] = "USD"
                else:
                    # Jefferies buying TWD or selling USD
                    trade_data['bs'] = "S"  # We're selling TWD
                    trade_data['position'] = "TWD"
                    trade_data['counter'] = "USD"
        
        # For CACIB format, often need to extract numbers directly from text
        if trade_data['source'] == 'CACIB' and ('position_amount' not in trade_data or 'counter_amount' not in trade_data):
            amount_matches = re.findall(r'([\d,]+(?:\.\d+)?)', email_text)
            if len(amount_matches) >= 2:
                # Assume first large number is TWD amount, second is USD or rate
                for amount in amount_matches:
                    amount = amount.replace(',', '')
                    if float(amount) > 100000:  # Likely TWD amount
                        trade_data['position_amount'] = amount
                    elif float(amount) < 100 and 'trade_rate' not in trade_data:
                        trade_data['trade_rate'] = amount
                    elif 1000 < float(amount) < 100000:  # Likely USD amount
                        trade_data['counter_amount'] = amount
        
        # Convert dates to required format (if needed)
        for date_field in ['value_date', 'fixing_date']:
            if date_field in trade_data:
                try:
                    # Try parsing different date formats
                    if re.match(r'\d{4}-\w+-\d{1,2}', trade_data[date_field]):
                        date_obj = datetime.strptime(trade_data[date_field], '%Y-%b-%d')
                    else:  # Format like "23 Apr 2025"
                        date_obj = datetime.strptime(trade_data[date_field], '%d %b %Y')
                    
                    trade_data[date_field] = date_obj.strftime('%Y%m%d')
                except ValueError:
                    # Keep original format if parsing fails
                    pass
                    
        # Set additional fields based on example
        trade_data['fx_type'] = "NDF"  # Assuming NDF from the example
        trade_data['trader'] = "JHKLFX"  # From example
        
        return trade_data
    
    def format_to_csv_rows(self, trade_data):
        """
        Format parsed trade data into two CSV rows - main and back-to-back.
        
        Args:
            trade_data (dict): Parsed trade information
            
        Returns:
            list: List of two dictionaries (CSV rows) - main trade and back-to-back trade
        """
        if not trade_data:
            return []
            
        # Create timestamp for reference number
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        
        # Generate random strings for reference numbers
        random_ref1 = self.generate_random_string()
        random_ref2 = self.generate_random_string()
        
        # Map the extracted data to first CSV row (main trade)
        main_row = {
            'ACTION': 'Add',
            'REFERENCE_NUMBER': f"{timestamp}-cal1-{random_ref1}",
            'FX_TYPE': trade_data.get('fx_type', 'NDF'),
            'COUNTERPARTY': 'CALL' if trade_data.get('source') == 'CACIB' else 'SCBL',
            'TRADER': trade_data.get('trader', 'JHKLFX'),
            'BS': trade_data.get('bs', ''),
            'POSITION': trade_data.get('position', 'TWD'),
            'COUNTER': trade_data.get('counter', 'USD'),
            'POSITION_RATE': trade_data.get('position_amount', ''),
            'TERMS': trade_data.get('trade_rate', ''),
            'EMPTY_COLUMN': '',  # New empty column after TERMS
            'TRANSACTION_DATE': datetime.now().strftime('%Y%m%d'),
            'VALUE_DATE': trade_data.get('value_date', ''),
            'PORTFOLIO': 'JBFSI',  # From example
            'FIXING_DATE': trade_data.get('fixing_date', ''),
            'AUXILIARY_DATA#1.FIELD': 'PTMMM',  # From example
            'AUXILIARY_DATA#1.VALUE': 'N',      # From example
            'AUXILIARY_DATA#2.FIELD': 'AFFILIATE BOOK',  # From example
            'AUXILIARY_DATA#2.VALUE': '33206'   # From example
        }
        
        # Create the back-to-back row (second line)
        back_to_back_row = main_row.copy()
        
        # Modify the back-to-back specific fields
        back_to_back_row['REFERENCE_NUMBER'] = f"{timestamp}-cal2-{random_ref2}"
        back_to_back_row['COUNTERPARTY'] = 'FXN99041'  # Back-to-back counterparty from example
        
        # Flip the buy/sell direction for the back-to-back trade
        if main_row['BS'] == 'B':
            back_to_back_row['BS'] = 'S'
        else:
            back_to_back_row['BS'] = 'B'
        
        return [main_row, back_to_back_row]


class FXTradeConverterApp:
    """GUI application for converting trade emails to CSV format."""
    
    def __init__(self, root):
        """Initialize the application UI."""
        self.root = root
        self.root.title("FX Trade Email to CSV Converter")
        self.root.geometry("800x600")
        
        self.parser = FXTradeEmailParser()
        self.email_files = []
        self.output_dir = os.path.expanduser("~/Documents")
        
        self.create_ui()
        
    def create_ui(self):
        """Create the application UI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Email selection area
        select_frame = ttk.LabelFrame(main_frame, text="Select Outlook Message Files", padding="10")
        select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(select_frame, text="Select .msg Files", command=self.select_msg_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text="Select Output Directory", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Display selected files
        list_frame = ttk.LabelFrame(main_frame, text="Selected Files", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar to listbox
        list_scroll = ttk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_list = tk.Listbox(list_frame, yscrollcommand=list_scroll.set)
        self.file_list.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        list_scroll.config(command=self.file_list.yview)
        
        # Output path display
        output_frame = ttk.Frame(main_frame, padding="5")
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        self.output_path_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=50, state="readonly").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Process button
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Process Emails", command=self.process_emails).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection).pack(side=tk.RIGHT, padx=5)
        
        # Status area
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=780).pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=5)
        
        # Add a test area for debugging
        test_frame = ttk.LabelFrame(main_frame, text="Test Email Parser", padding="10")
        test_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(test_frame, text="Test Sample", command=self.test_sample).pack(side=tk.LEFT, padx=5)
        
    def select_msg_files(self):
        """Open file dialog to select Outlook .msg files."""
        filetypes = [
            ("Outlook Message files", "*.msg"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Outlook Message Files",
            filetypes=filetypes
        )
        
        if files:
            self.email_files = list(files)
            self.update_file_list()
            self.status_var.set(f"Selected {len(self.email_files)} files")
            
    def select_output_dir(self):
        """Open directory dialog to select output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir
        )
        
        if directory:
            self.output_dir = directory
            self.output_path_var.set(directory)
            self.status_var.set(f"Output directory set to: {directory}")
            
    def update_file_list(self):
        """Update the listbox with selected files."""
        self.file_list.delete(0, tk.END)
        for file in self.email_files:
            self.file_list.insert(tk.END, os.path.basename(file))
            
    def clear_selection(self):
        """Clear the selected files."""
        self.email_files = []
        self.update_file_list()
        self.status_var.set("Selection cleared")
        self.progress_var.set(0)
    
    def test_sample(self):
        """Test the parser with sample data."""
        # Sample CA-CIB format
        cacib_sample = """
        To confirm,
        
        JEFFERIES FINANCIAL SERVICES INC Sell TWD 150,000,000.00 vs. USD
        at 32.5920 (32.5920 + 0.0)
        Value Date 23 Apr 2025
        Fixing TAIFX1 Fixing Date 21 Apr 2025
        Facing CA-CIB UNITED KINGDOM BRANCH | 2155
        """
        
        # Sample SC.com format
        sc_sample = """
        Hi team
        
        Thank you for the trade. Please find below details:
        
        Currency Pair : USDTWD
        Tenor : 1M
        Value Date : 2025-May-19
        Trade Rate : 32.125
        Spot Rate : 32.4176
        Fwd Pts : -292.6
        Fixing Date : 2025-May-15
        Fixing Mechanism : TWD TAIFX1
        
        Jefferies Bache Financial Serv (OBO) Buys Amount, Ccy : 5,000,000 USD and Sells Amount, Ccy : 160,625,000 TWD
        SCB Buys Amount, Ccy : 160,625,000 TWD and Sells Amount, Ccy : 5,000,000 USD
        """
        
        # Test both samples
        print("\nTesting CA-CIB sample:")
        cacib_data = self.parser.parse_email(cacib_sample)
        if cacib_data:
            print("Parsed CA-CIB data:")
            for key, value in cacib_data.items():
                print(f"  {key}: {value}")
            
            cacib_rows = self.parser.format_to_csv_rows(cacib_data)
            if cacib_rows:
                print("\nGenerated CSV rows for CA-CIB:")
                for i, row in enumerate(cacib_rows):
                    print(f"Row {i+1}:")
                    for key, value in row.items():
                        print(f"  {key}: {value}")
        
        print("\nTesting SC.com sample:")
        sc_data = self.parser.parse_email(sc_sample)
        if sc_data:
            print("Parsed SC.com data:")
            for key, value in sc_data.items():
                print(f"  {key}: {value}")
            
            sc_rows = self.parser.format_to_csv_rows(sc_data)
            if sc_rows:
                print("\nGenerated CSV rows for SC.com:")
                for i, row in enumerate(sc_rows):
                    print(f"Row {i+1}:")
                    for key, value in row.items():
                        print(f"  {key}: {value}")
        
        self.status_var.set("Test completed - see console for results")
        
    def process_emails(self):
        """Process selected Outlook .msg files and convert to CSV."""
        if not self.email_files:
            messagebox.showwarning("No Files", "Please select Outlook message files to process.")
            return
            
        # Check if output directory exists
        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {e}")
                return
                
        # Initialize progress
        total_files = len(self.email_files)
        self.progress_var.set(0)
        processed = 0
        successful = 0
        
        # Prepare CSV output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.output_dir, f"fx_trades_{timestamp}.csv")
        
        try:
            with open(csv_file, 'w', newline='') as f:
                # Define fieldnames with the new empty column
                fieldnames = [
                    'ACTION', 'REFERENCE_NUMBER', 'FX_TYPE', 'COUNTERPARTY', 'TRADER', 
                    'BS', 'POSITION', 'COUNTER', 'POSITION_RATE', 'TERMS', 
                    'EMPTY_COLUMN',  # New empty column after TERMS
                    'TRANSACTION_DATE', 'VALUE_DATE', 'PORTFOLIO', 'FIXING_DATE',
                    'AUXILIARY_DATA#1.FIELD', 'AUXILIARY_DATA#1.VALUE',
                    'AUXILIARY_DATA#2.FIELD', 'AUXILIARY_DATA#2.VALUE'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each Outlook msg file
                for msg_file in self.email_files:
                    self.status_var.set(f"Processing: {os.path.basename(msg_file)}")
                    self.root.update()
                    
                    try:
                        # Use extract_msg to read the .msg file
                        msg = extract_msg.Message(msg_file)
                        email_text = msg.body
                        
                        # Parse email
                        trade_data = self.parser.parse_email(email_text)
                        
                        # Convert to CSV rows (main and back-to-back)
                        if trade_data:
                            csv_rows = self.parser.format_to_csv_rows(trade_data)
                            if csv_rows:
                                for row in csv_rows:
                                    writer.writerow(row)
                                successful += 1
                                source = "CA-CIB" if trade_data.get('source') == 'CACIB' else "SC.com"
                                self.status_var.set(f"Processed {os.path.basename(msg_file)} successfully - created main trade and back-to-back entries. Source: {source}")
                                
                                # Debug information - print parsed data
                                print(f"Successfully parsed {os.path.basename(msg_file)}:")
                                for key, value in trade_data.items():
                                    print(f"  {key}: {value}")
                        else:
                            self.status_var.set(f"No trade data found in {os.path.basename(msg_file)}")
                            self.root.update()
                                
                    except Exception as e:
                        self.status_var.set(f"Error processing {os.path.basename(msg_file)}: {e}")
                        self.root.update()
                        
                    # Update progress
                    processed += 1
                    self.progress_var.set((processed / total_files) * 100)
                    self.root.update()
                    
            # Show completion message
            self.status_var.set(f"Completed: {successful} out of {total_files} files processed successfully")
            messagebox.showinfo("Processing Complete", 
                               f"Processed {total_files} files.\n"
                               f"Successfully converted: {successful}\n"
                               f"Output saved to: {csv_file}")
                               
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_var.set(f"Error: {e}")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = FXTradeConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()