import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import blpapi
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BloombergDataManager:
    def __init__(self):
        """Initialize Bloomberg connection"""
        self.session = None
        self.connected = False
        self.tickers_info = {}

    def connect(self):
        """Establish connection to Bloomberg API"""
        try:
            self.session = blpapi.Session()
            if not self.session.start():
                print("❌ Failed to start Bloomberg session.")
                return False

            if not self.session.openService("//blp/refdata"):
                print("❌ Failed to open //blp/refdata service.")
                return False

            self.connected = True
            print("✅ Successfully connected to Bloomberg")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Bloomberg: {e}")
            print("Please ensure Bloomberg Terminal is running and you have proper permissions.")
            return False

    def check_connection(self):
        """Check if Bloomberg connection is active"""
        if not self.connected:
            return self.connect()
        return True

    def get_historical_data(self, tickers, start_date, end_date, fields=['PX_LAST']):
        """
        Retrieve historical data from Bloomberg

        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: Bloomberg fields to retrieve

        Returns:
            DataFrame with historical data
        """
        if not self.check_connection():
            print("❌ Bloomberg connection not available. Cannot retrieve data.")
            return None

        try:
            print(f"📊 Retrieving historical data for {len(tickers)} tickers ({start_date} to {end_date})...")

            # Get the reference data service
            refDataService = self.session.getService("//blp/refdata")

            # Create a request for historical data
            request = refDataService.createRequest("HistoricalDataRequest")

            # Set the securities
            for ticker in tickers:
                request.append("securities", ticker)

            # Set the fields
            for field in fields:
                request.append("fields", field)

            # Set the date range
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))

            # Send the request
            self.session.sendRequest(request)

            # Process the response
            data_dict = {}
            dates = set()

            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        securityData = msg.getElement("securityData")
                        security = securityData.getElementAsString("security")
                        fieldData = securityData.getElement("fieldData")

                        for i in range(fieldData.numValues()):
                            row = fieldData.getValue(i)
                            date = row.getElementAsDatetime("date").strftime("%Y-%m-%d")
                            dates.add(date)

                            for field in fields:
                                if row.hasElement(field):
                                    value = row.getElementAsFloat(field)
                                    if security not in data_dict:
                                        data_dict[security] = {}
                                    if date not in data_dict[security]:
                                        data_dict[security][date] = {}
                                    data_dict[security][date][field] = value

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            # Convert to DataFrame
            if not data_dict:
                print("⚠️ No data retrieved from Bloomberg")
                return None

            # Create a multi-index DataFrame
            dates_list = sorted(list(dates))
            index = pd.DatetimeIndex(dates_list)

            if len(fields) == 1:
                # Single field - create a simple DataFrame
                data = pd.DataFrame(index=index)
                for security in data_dict:
                    values = []
                    for date in dates_list:
                        if date in data_dict[security] and fields[0] in data_dict[security][date]:
                            values.append(data_dict[security][date][fields[0]])
                        else:
                            values.append(np.nan)
                    data[security] = values
            else:
                # Multiple fields - create a multi-level column DataFrame
                columns = pd.MultiIndex.from_product([tickers, fields])
                data = pd.DataFrame(index=index, columns=columns)

                for security in data_dict:
                    for date in dates_list:
                        if date in data_dict[security]:
                            for field in fields:
                                if field in data_dict[security][date]:
                                    data.loc[date, (security, field)] = data_dict[security][date][field]

            print(f"✅ Successfully retrieved data with shape: {data.shape}")
            return data

        except Exception as e:
            print(f"❌ Error retrieving data from Bloomberg: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """Close Bloomberg connection"""
        if self.connected and self.session is not None:
            try:
                self.session.stop()
                self.connected = False
                print("✅ Bloomberg connection closed")
            except Exception as e:
                print(f"❌ Error closing Bloomberg connection: {e}")

def run_bloomberg_test():
    """
    Run a simple test to retrieve data from Bloomberg
    """
    # Initialize Bloomberg connection
    bloomberg = BloombergDataManager()
    bloomberg.connect()

    # Define agricultural commodity futures tickers with proper Bloomberg syntax
    tickers = [
        'CF1 CZC Comdty',   # CZCE cotton
        'SR1 CZC Comdty',   # CZCE sugar
        'SB1 Comdty',       # ICE eleventh sugar
        'A1 DCE Comdty',    # DCE bean
        'B1 DCE Comdty',    # DCE bean II
        'Y1 DCE Comdty',    # DCE soybean oil
        'M1 DCE Comdty',    # DCE cardamom
        'WH1 CZC Comdty',   # CZCE strong wheat
        'C1 DCE Comdty',    # DCE corn
        'KC1 Comdty',       # ICE coffee
        'CC1 Comdty',       # ICE cocoa
        'OJ1 Comdty'        # ICE frozen concentrated orange juice
    ]

    # Define date range
    start_date = '2023-01-01'
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Retrieve data from Bloomberg
    data = bloomberg.get_historical_data(tickers, start_date, end_date)

    # Close Bloomberg connection
    bloomberg.close()

    if data is not None and not data.empty:
        # Save raw data
        data.to_csv('agricultural_futures_data.csv')
        print(f"✅ Data saved to agricultural_futures_data.csv")

        # Display sample data
        print("\nSample data:")
        print(data.head())

        # Plot the data
        plt.figure(figsize=(12, 8))
        for ticker in data.columns:
            plt.plot(data.index, data[ticker], label=ticker)

        plt.title('Agricultural Commodity Futures Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('agricultural_futures_prices.png')
        print(f"✅ Plot saved to agricultural_futures_prices.png")
    else:
        print("❌ Failed to retrieve data. Exiting.")

if __name__ == "__main__":
    print("Starting Bloomberg data retrieval test...")
    try:
        run_bloomberg_test()
        print("Test completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error executing test: {e}")
        traceback.print_exc()
