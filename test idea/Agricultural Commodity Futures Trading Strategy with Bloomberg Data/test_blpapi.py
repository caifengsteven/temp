import blpapi
import pandas as pd
from datetime import datetime

def test_bloomberg_connection():
    print("Testing Bloomberg connection with blpapi directly...")
    
    try:
        # Start a session
        session = blpapi.Session()
        if not session.start():
            print("Failed to start session.")
            return
        
        print("✅ Successfully started Bloomberg session")
        
        # Open the reference data service
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return
        
        print("✅ Successfully opened reference data service")
        
        # Get the reference data service
        refDataService = session.getService("//blp/refdata")
        
        # Create a request for historical data
        request = refDataService.createRequest("HistoricalDataRequest")
        
        # Set the securities
        request.append("securities", "MSFT US Equity")
        
        # Set the fields
        request.append("fields", "PX_LAST")
        request.append("fields", "VOLUME")
        
        # Set the date range
        request.set("startDate", "20230101")
        request.set("endDate", datetime.now().strftime("%Y%m%d"))
        
        print("Sending request...")
        session.sendRequest(request)
        
        # Process the response
        print("Processing response...")
        data = []
        
        while True:
            event = session.nextEvent(500)
            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    securityData = msg.getElement("securityData")
                    security = securityData.getElementAsString("security")
                    fieldData = securityData.getElement("fieldData")
                    
                    for i in range(fieldData.numValues()):
                        row = fieldData.getValue(i)
                        date = row.getElementAsDatetime("date")
                        px_last = row.getElementAsFloat("PX_LAST") if row.hasElement("PX_LAST") else None
                        volume = row.getElementAsInteger("VOLUME") if row.hasElement("VOLUME") else None
                        
                        data.append({
                            'date': date.strftime("%Y-%m-%d"),
                            'security': security,
                            'PX_LAST': px_last,
                            'VOLUME': volume
                        })
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        if data:
            df = pd.DataFrame(data)
            print(f"✅ Successfully retrieved data with shape: {df.shape}")
            print("\nSample data:")
            print(df.head())
        else:
            print("⚠️ No data retrieved from Bloomberg")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Stop the session
        if 'session' in locals() and session.started():
            session.stop()
            print("✅ Bloomberg session stopped")

if __name__ == "__main__":
    test_bloomberg_connection()
