import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import extract_msg  # Library to extract .msg files
import tempfile

class FXTradeEmailParser:
    """Parse FX trade confirmation emails and convert to CSV format."""
    
    def __init__(self):
        """Initialize the parser with regex patterns for extracting trade data."""
        self.patterns = {
            'currency_pair': r'Currency Pair:\s+([A-Z]{3}/[A-Z]{3})',
            'trade_rate': r'Trade Rate:\s+([\d.]+)',
            'spot_rate': r'Spot Rate:\s+([\d.]+)',
            'fwd_pts': r'Fwd Pts:\s+([-\d.]+)',
            'value_date': r'Value Date:\s+(\d{4}-\w+-\d{2})',
            'fixing_date': r'Fixing Date:\s+(\d{4}-\w+-\d{2})',
            'transaction': r'([A-Z\s]+)\s+(?:Buy|Sell)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)\s+and\s+(?:Buy|Sell)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)'
        }
    
    def parse_email(self, email_text):
        """
        Parse email text to extract FX trade information.
        
        Args:
            email_text (str): The raw email text
            
        Returns:
            dict: Extracted trade information or None if not a valid trade email
        """
        # Check if this looks like a trade confirmation email
        if "Thank you for the trade" not in email_text:
            return None
            
        # Check for SC.com source (could be from domain or other identifier)
        if not any(source in email_text for source in ["SC.COM", "SC.com", "STANDARD CHARTERED", "Standard Chartered"]):
            # For this example, we'll still process but in production you might want to filter
            pass
            
        trade_data = {}
        
        # Extract basic trade information
        for key, pattern in self.patterns.items():
            if key != 'transaction':  # Handle transaction separately
                match = re.search(pattern, email_text)
                if match:
                    trade_data[key] = match.group(1)
        
        # Extract transaction details (party, direction, amounts)
        transaction_match = re.search(self.patterns['transaction'], email_text)
        if transaction_match:
            party = transaction_match.group(1).strip()
            
            # Determine buy/sell direction and currencies
            currency1 = transaction_match.group(2)
            amount1 = transaction_match.group(3).replace(',', '')
            currency2 = transaction_match.group(4)
            amount2 = transaction_match.group(5).replace(',', '')
            
            # Determine buy/sell direction
            if "Buy" in email_text and currency1 == "TWD":
                direction = "B"  # Buy TWD
                trade_data['bs'] = "B"
            elif "Sell" in email_text and currency1 == "USD":
                direction = "S"  # Sell USD
                trade_data['bs'] = "S"
            else:
                # Fallback logic
                direction = "B" if "Buy" in email_text.split(party)[1].split("and")[0] else "S"
                trade_data['bs'] = direction
            
            trade_data['party'] = party
            trade_data['position'] = "TWD"  # Based on the example
            trade_data['counter'] = "USD"   # Based on the example
            
            # Set the correct amounts based on the currencies
            if currency1 == "TWD":
                trade_data['position_amount'] = amount1
                trade_data['counter_amount'] = amount2
            else:
                trade_data['position_amount'] = amount2
                trade_data['counter_amount'] = amount1
                
        # Convert dates to required format (if needed)
        for date_field in ['value_date', 'fixing_date']:
            if date_field in trade_data:
                try:
                    date_obj = datetime.strptime(trade_data[date_field], '%Y-%b-%d')
                    trade_data[date_field] = date_obj.strftime('%Y%m%d')
                except ValueError:
                    # Keep original format if parsing fails
                    pass
                    
        # Set additional fields based on example
        trade_data['fx_type'] = "NDF"  # Assuming NDF from the example
        trade_data['trader'] = "JHKLFX"  # From example, in reality might be extracted
        
        return trade_data
    
    def format_to_csv_row(self, trade_data):
        """
        Format parsed trade data into CSV row format.
        
        Args:
            trade_data (dict): Parsed trade information
            
        Returns:
            dict: CSV row with properly mapped fields
        """
        if not trade_data:
            return None
            
        # Map the extracted data to CSV fields
        csv_row = {
            'ACTION': 'Add',
            'REFERENCE_NUMBER': datetime.now().strftime('%Y%m%d%H%M-cal'),
            'FX_TYPE': trade_data.get('fx_type', 'NDF'),
            'COUNTERPARTY': 'SCBL',  # Standard Chartered Bank Limited
            'TRADER': trade_data.get('trader', 'JHKLFX'),
            'BS': trade_data.get('bs', ''),
            'POSITION': trade_data.get('position', 'TWD'),
            'COUNTER': trade_data.get('counter', 'USD'),
            'POSITION_RATE': trade_data.get('position_amount', ''),
            'TERMS': trade_data.get('trade_rate', ''),
            'TRANSACTION_DATE': datetime.now().strftime('%Y%m%d'),
            'VALUE_DATE': trade_data.get('value_date', ''),
            'PORTFOLIO': 'JBFSI',  # From example
            'FIXING_DATE': trade_data.get('fixing_date', ''),
            'AUXILIARY_DATA#1.FIELD': 'PTMMM',  # From example
            'AUXILIARY_DATA#1.VALUE': 'N',      # From example
            'AUXILIARY_DATA#2.FIELD': 'AFFILIATE BOOK',  # From example
            'AUXILIARY_DATA#2.VALUE': '33206'   # From example
        }
        
        return csv_row


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
                # Get all possible field names from an example row or define manually
                fieldnames = [
                    'ACTION', 'REFERENCE_NUMBER', 'FX_TYPE', 'COUNTERPARTY', 'TRADER', 
                    'BS', 'POSITION', 'COUNTER', 'POSITION_RATE', 'TERMS', 
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
                        
                        # Convert to CSV row
                        if trade_data:
                            csv_row = self.parser.format_to_csv_row(trade_data)
                            if csv_row:
                                writer.writerow(csv_row)
                                successful += 1
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