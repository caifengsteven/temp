import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import re
import pandas as pd
import pyperclip
import os
import sys
import traceback
from datetime import datetime
import threading
import email
import email.policy
from email.parser import BytesParser, Parser

# Try to import extract-msg
try:
    import extract_msg
    EXTRACT_MSG_AVAILABLE = True
except ImportError:
    EXTRACT_MSG_AVAILABLE = False
    print("Warning: extract-msg package not found. Install with: pip install extract-msg")

# Try to import TkinterDnD2
try:
    import TkinterDnD2 as tkdnd
    TKDND_AVAILABLE = True
except ImportError:
    TKDND_AVAILABLE = False
    print("Warning: TkinterDnD2 not found. Drag and drop will not be available.")
    print("Install with: pip install tkinterdnd2")

class FXEmailProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("FX Transaction Email Processor")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Store processed transactions
        self.transaction_data = []
        
        # Create the main UI
        self.create_ui()
        
        # Setup drag and drop if available
        if TKDND_AVAILABLE:
            self.setup_drag_drop()
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_label = tk.Label(
            main_frame, 
            text="FX Transaction Email Processor", 
            font=("Arial", 14, "bold"),
            bg="#f0f0f0"
        )
        header_label.pack(pady=(0, 10))
        
        # Instructions
        if TKDND_AVAILABLE:
            instructions_text = "Drag .msg or .eml files or use the Open Files button to process FX transaction emails"
        else:
            instructions_text = "Click the Open Files button to select and process .msg or .eml files"
            
        instructions = tk.Label(
            main_frame,
            text=instructions_text,
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#555555"
        )
        instructions.pack(pady=(0, 10))
        
        # Drop area (only show if drag-drop is available)
        if TKDND_AVAILABLE:
            self.drop_area = tk.LabelFrame(
                main_frame, 
                text="Drop Zone", 
                bg="#ffffff",
                bd=2,
                relief=tk.GROOVE,
                padx=10,
                pady=10,
                height=100
            )
            self.drop_area.pack(fill=tk.X, pady=(0, 10))
            
            drop_label = tk.Label(
                self.drop_area, 
                text="Drag Outlook .msg or .eml files here...",
                bg="#ffffff",
                fg="#888888",
                font=("Arial", 12)
            )
            drop_label.pack(fill=tk.BOTH, expand=True)
        
        # Button to open files
        open_button = tk.Button(
            main_frame,
            text="Open Email Files",
            command=self.open_email_files,
            bg="#007BFF",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        open_button.pack(pady=(0, 10))
        
        # Results area
        results_frame = tk.LabelFrame(
            main_frame, 
            text="Extracted FX Transaction Data", 
            bg="#ffffff",
            bd=2,
            relief=tk.GROOVE
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            bg="#ffffff",
            font=("Courier New", 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg="#f0f0f0")
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Copy to clipboard button
        self.copy_button = tk.Button(
            buttons_frame,
            text="Copy to Clipboard",
            command=self.copy_to_clipboard,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.copy_button.pack(side=tk.RIGHT)
        
        # Clear button
        self.clear_button = tk.Button(
            buttons_frame,
            text="Clear Results",
            command=self.clear_results,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        self.clear_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Ready",
            bg="#f0f0f0",
            font=("Arial", 10, "italic"),
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def setup_drag_drop(self):
        """Set up drag and drop functionality (only if TkinterDnD2 is available)"""
        if TKDND_AVAILABLE:
            self.drop_area.drop_target_register("DND_Files")
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)
            self.update_status("Drag and drop initialized")
    
    def on_drop(self, event):
        """Handle files dropped onto the drop area"""
        file_paths = self.parse_drop_data(event.data)
        self.process_files(file_paths)
    
    def parse_drop_data(self, data):
        """Parse the drag and drop data to extract file paths"""
        # The format of data varies by OS and TkinterDnD version
        # Common formats include:
        # - Space-separated paths
        # - Paths with {} braces
        # - Quoted paths
        
        # Remove braces if present
        if data.startswith('{') and data.endswith('}'):
            data = data[1:-1]
        
        # Split into separate paths
        if '"' in data:
            # Handle quoted paths
            paths = []
            current_path = ""
            in_quotes = False
            
            for char in data:
                if char == '"':
                    in_quotes = not in_quotes
                    if not in_quotes and current_path:
                        paths.append(current_path)
                        current_path = ""
                elif in_quotes:
                    current_path += char
            
            if current_path:
                paths.append(current_path)
        else:
            # Simple space-separated paths
            paths = data.split(' ')
        
        # Filter out empty paths and clean up
        return [p.strip() for p in paths if p.strip()]
    
    def open_email_files(self):
        """Open file dialog to select email files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Email Files",
            filetypes=[
                ("Email files", "*.msg *.eml"), 
                ("Outlook MSG files", "*.msg"), 
                ("EML files", "*.eml"), 
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            self.process_files(file_paths)
    
    def process_files(self, file_paths):
        """Process the selected or dropped files"""
        if not file_paths:
            return
        
        # Update status
        self.update_status(f"Processing {len(file_paths)} files...")
        
        # Use threading to prevent UI freezing
        def process_thread():
            successfully_processed = 0
            
            for file_path in file_paths:
                try:
                    # Check file extension
                    file_extension = os.path.splitext(file_path.lower())[1]
                    
                    if file_extension == '.msg':
                        # Check if extract-msg is available for .msg files
                        if not EXTRACT_MSG_AVAILABLE:
                            self.update_status(f"Skipping {os.path.basename(file_path)} - extract-msg package not installed")
                            continue
                        
                        # Process .msg file
                        self.process_msg_file(file_path)
                        successfully_processed += 1
                    elif file_extension == '.eml':
                        # Process .eml file
                        self.process_eml_file(file_path)
                        successfully_processed += 1
                    else:
                        self.update_status(f"Skipping {os.path.basename(file_path)} - not a supported file type (.msg or .eml)")
                except Exception as e:
                    self.update_status(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
                    traceback.print_exc()
            
            # Update status when done
            self.update_status(f"Processed {successfully_processed} out of {len(file_paths)} files")
            
            # Update results display
            self.root.after(0, self.display_results)
        
        # Start processing thread
        threading.Thread(target=process_thread).start()
    
    def process_msg_file(self, file_path):
        """Process an individual .msg file using extract-msg"""
        try:
            # Update status
            self.update_status(f"Processing MSG: {os.path.basename(file_path)}")
            
            # Open the .msg file with extract-msg
            msg = extract_msg.Message(file_path)
            
            # Extract data
            subject = msg.subject if msg.subject else ""
            body = msg.body if msg.body else ""
            
            # Extract transaction data
            transaction = self.extract_fx_data(subject, body)
            
            # Store the transaction data
            if transaction:
                self.transaction_data.append(transaction)
                self.update_status(f"Extracted transaction data from {os.path.basename(file_path)}")
            else:
                self.update_status(f"No FX transaction data found in {os.path.basename(file_path)}")
        
        except Exception as e:
            self.update_status(f"Error processing MSG {os.path.basename(file_path)}: {str(e)}")
            traceback.print_exc()
    
    def process_eml_file(self, file_path):
        """Process an individual .eml file using built-in email module"""
        try:
            # Update status
            self.update_status(f"Processing EML: {os.path.basename(file_path)}")
            
            # Open and parse the .eml file
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=email.policy.default).parse(f)
            
            # Extract subject
            subject = msg.get('subject', '')
            
            # Extract body - handle both plain text and HTML content
            body = ""
            
            # Try to get plain text body
            if msg.get_body(preferencelist=('plain',)):
                body = msg.get_body(preferencelist=('plain',)).get_content()
            # If no plain text, try HTML
            elif msg.get_body(preferencelist=('html',)):
                html_body = msg.get_body(preferencelist=('html',)).get_content()
                # Very simple HTML to text conversion - you might want to use a more robust HTML parser
                body = re.sub(r'<[^>]+>', ' ', html_body)
            # If still no body, iterate through parts
            else:
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        body += part.get_content()
                    elif content_type == 'text/html':
                        html_part = part.get_content()
                        # Simple HTML to text conversion
                        body += re.sub(r'<[^>]+>', ' ', html_part)
            
            # Extract transaction data
            transaction = self.extract_fx_data(subject, body)
            
            # Store the transaction data
            if transaction:
                self.transaction_data.append(transaction)
                self.update_status(f"Extracted transaction data from {os.path.basename(file_path)}")
            else:
                self.update_status(f"No FX transaction data found in {os.path.basename(file_path)}")
        
        except Exception as e:
            self.update_status(f"Error processing EML {os.path.basename(file_path)}: {str(e)}")
            traceback.print_exc()
    
    def extract_fx_data(self, subject, body):
        """Extract FX transaction data from email subject and body"""
        transaction = {
            'Direction': None,  # Buy or Sell
            'USD_Amount': None,
            'TWD_Amount': None,
            'Strike': None,
            'Fixing_Date': None,
            'Value_Date': None,
            'Email_Subject': subject
        }
        
        # Extract transaction direction (Buy/Sell)
        if re.search(r'\b(?:Buy|Client\s+buy)\s+USD\b', subject, re.IGNORECASE) or re.search(r'\b(?:Buy|Client\s+buy)\s+USD\b', body, re.IGNORECASE):
            transaction['Direction'] = 'Buy USD'
        elif re.search(r'\b(?:Sell|Client\s+sell)\s+USD\b', subject, re.IGNORECASE) or re.search(r'\b(?:Sell|Client\s+sell)\s+USD\b', body, re.IGNORECASE):
            transaction['Direction'] = 'Sell USD'
        
        # Look for "Client buy USD" format
        client_buy_match = re.search(r'Client\s+buy\s+USD\s*([0-9,]+(?:\.[0-9]+)?)', body, re.IGNORECASE)
        if client_buy_match:
            transaction['Direction'] = 'Buy USD'
            transaction['USD_Amount'] = client_buy_match.group(1).replace(',', '')
        
        # Look for "Client sell TWD" format
        client_sell_match = re.search(r'Client\s+sell\s+TWD\s*([0-9,]+(?:\.[0-9]+)?)', body, re.IGNORECASE)
        if client_sell_match:
            transaction['TWD_Amount'] = client_sell_match.group(1).replace(',', '')
        
        # Alternative USD amount extraction
        if not transaction['USD_Amount']:
            usd_match = re.search(r'USD\s*([0-9,]+(?:\.[0-9]+)?)', body, re.IGNORECASE)
            if usd_match:
                transaction['USD_Amount'] = usd_match.group(1).replace(',', '')
        
        # Alternative TWD amount extraction
        if not transaction['TWD_Amount']:
            twd_match = re.search(r'TWD\s*([0-9,]+(?:\.[0-9]+)?)', body, re.IGNORECASE)
            if twd_match:
                transaction['TWD_Amount'] = twd_match.group(1).replace(',', '')
        
        # Extract strike rate
        strike_patterns = [
            r'Strike\s*:?\s*([0-9]+\.?[0-9]*)',
            r'(?:strike|rate|price)\s*(?::|rate|price|at)?\s*([0-9]+\.?[0-9]*)',
            r'(?:at|@)\s*([0-9]+\.?[0-9]*)',
            r'rate\s*(?:is|of|:)?\s*(?:USD/TWD)?\s*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in strike_patterns:
            strike_match = re.search(pattern, body, re.IGNORECASE)
            if strike_match:
                transaction['Strike'] = strike_match.group(1)
                break
        
        # If strike not found through regex, try to calculate from amounts
        if not transaction['Strike'] and transaction['USD_Amount'] and transaction['TWD_Amount']:
            try:
                usd_amount = float(transaction['USD_Amount'])
                twd_amount = float(transaction['TWD_Amount'])
                if usd_amount > 0:
                    transaction['Strike'] = f"{twd_amount/usd_amount:.4f}"
            except (ValueError, TypeError):
                pass
        
        # Extract fixing date - updated patterns
        fixing_patterns = [
            r'Fixing\s+date\s*:?\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})',  # Format: 13-May-2025
            r'Fixing\s+date\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'Fixing\s+date\s*:?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'fixing\s*date\s*(?::|is|on)?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'fixing\s*date\s*(?::|is|on)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'fixing\s*(?:date|)?\s*(?::|is|on)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'fix(?:ing|ed)\s*(?:on|at)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'fix(?:ing|ed)\s*(?:on|at)?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})'
        ]
        
        for pattern in fixing_patterns:
            fixing_match = re.search(pattern, body, re.IGNORECASE)
            if fixing_match:
                raw_date = fixing_match.group(1)
                formatted_date = self.convert_to_yyyymmdd(raw_date)
                transaction['Fixing_Date'] = formatted_date
                break
        
        # Extract value date / settlement date - updated patterns
        value_patterns = [
            r'Settlement\s+date\s*:?\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})',  # Format: 15-May-2025
            r'Settlement\s+date\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'Settlement\s+date\s*:?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'value\s*date\s*(?::|is|on)?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'value\s*date\s*(?::|is|on)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'value\s*(?:date|)?\s*(?::|is|on)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'valu(?:e|ed)\s*(?:on|at)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})',
            r'valu(?:e|ed)\s*(?:on|at)?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'settlement\s*date\s*(?::|is|on)?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
            r'settlement\s*date\s*(?::|is|on)?\s*([A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?,?\s*[0-9]{2,4})'
        ]
        
        for pattern in value_patterns:
            value_match = re.search(pattern, body, re.IGNORECASE)
            if value_match:
                raw_date = value_match.group(1)
                formatted_date = self.convert_to_yyyymmdd(raw_date)
                transaction['Value_Date'] = formatted_date
                break
        
        # Check if we have at least some data
        has_data = any([
            transaction['Direction'],
            transaction['USD_Amount'],
            transaction['TWD_Amount'],
            transaction['Strike'],
            transaction['Fixing_Date'],
            transaction['Value_Date']
        ])
        
        return transaction if has_data else None
        
    def convert_to_yyyymmdd(self, date_string):
        """Convert various date formats to 'yyyymmdd' format"""
        # Common month abbreviations and names
        month_map = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        try:
            # Clean the date string
            date_string = date_string.strip()
            
            # Try various date formats
            
            # Format: DD-MMM-YYYY (e.g., "13-May-2025")
            match = re.match(r'(\d{1,2})-([A-Za-z]{3})-(\d{4})', date_string, re.IGNORECASE)
            if match:
                day, month_abbr, year = match.groups()
                month = month_map.get(month_abbr.lower(), '01')  # Default to 01 if month not recognized
                return f"{year}{month.zfill(2)}{day.zfill(2)}"
            
            # Format: MM/DD/YYYY or DD/MM/YYYY
            match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})', date_string)
            if match:
                part1, part2, year = match.groups()
                
                # Assume MM/DD/YYYY for simplicity
                # For a more accurate conversion, you'd need to use heuristics or configuration
                month, day = part1, part2
                
                # Handle two-digit year
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                
                return f"{year}{month.zfill(2)}{day.zfill(2)}"
            
            # Format: Month DD, YYYY or Month DD YYYY
            match = re.match(r'([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})', date_string, re.IGNORECASE)
            if match:
                month_name, day, year = match.groups()
                month = month_map.get(month_name.lower()[:3], '01')  # Use first 3 chars of month name
                return f"{year}{month.zfill(2)}{day.zfill(2)}"
            
            # Format: DD Month YYYY
            match = re.match(r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})', date_string, re.IGNORECASE)
            if match:
                day, month_name, year = match.groups()
                month = month_map.get(month_name.lower()[:3], '01')
                return f"{year}{month.zfill(2)}{day.zfill(2)}"
            
            # If no format matched, return the original string
            print(f"Warning: Could not convert date format: {date_string}")
            return date_string
        
        except Exception as e:
            print(f"Error converting date {date_string}: {str(e)}")
            return date_string
    
    def display_results(self):
        """Display the extracted transaction data in the results text area"""
        if not self.transaction_data:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No transaction data extracted.")
            return
        
        # Create header row
        header = "Direction\tUSD Amount\tTWD Amount\tStrike\tFixing Date\tValue Date\tEmail Subject\n"
        
        # Create data rows
        rows = []
        for transaction in self.transaction_data:
            row_text = f"{transaction['Direction'] or 'N/A'}\t{transaction['USD_Amount'] or 'N/A'}\t{transaction['TWD_Amount'] or 'N/A'}\t{transaction['Strike'] or 'N/A'}\t{transaction['Fixing_Date'] or 'N/A'}\t{transaction['Value_Date'] or 'N/A'}\t{transaction['Email_Subject'] or 'N/A'}"
            rows.append(row_text)
        
        # Combine and display
        result_text = header + "\n".join(rows)
        
        # Update text widget
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_text)
    
    def copy_to_clipboard(self):
        """Copy the results to clipboard in tab-delimited format for Excel"""
        if not self.transaction_data:
            messagebox.showinfo("Info", "No data to copy")
            return
        
        # Get text from results
        text_to_copy = self.results_text.get(1.0, tk.END)
        
        # Copy to clipboard
        pyperclip.copy(text_to_copy)
        
        # Update status
        self.update_status("Data copied to clipboard - ready to paste into Excel")
        messagebox.showinfo("Success", "Data copied to clipboard. You can now paste it into Excel.")
    
    def clear_results(self):
        """Clear the results text area and transaction data"""
        self.transaction_data = []
        self.results_text.delete(1.0, tk.END)
        self.update_status("Results cleared")
    
    def update_status(self, message):
        """Update the status label with a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_text = f"[{timestamp}] {message}"
        
        # Update in the main thread
        def update():
            self.status_label.config(text=status_text)
        
        # If called from a non-main thread, use after() to update safely
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, update)
        else:
            update()
            
        print(message)  # Also log to console

def main():
    """Main entry point of the application"""
    # Set up root window
    if TKDND_AVAILABLE:
        # Use TkinterDnD2 for drag and drop
        root = tkdnd.TkinterDnD.Tk()
    else:
        # Use regular Tk
        root = tk.Tk()
    
    # Create the application
    app = FXEmailProcessor(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    # Add exception handling
    try:
        main()
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        messagebox.showerror("Error", f"Application error: {str(e)}")