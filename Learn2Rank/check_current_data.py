import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def check_database_data():
    """
    Check what data is actually available in the database
    """
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='352471Cf',
            database='yuqerdata'
        )
        print("Successfully connected to database")
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return
    
    cursor = connection.cursor()
    
    try:
        # Check overall date range
        cursor.execute("SELECT MIN(tradeDate), MAX(tradeDate) FROM yq_mktstockfactorsonedayget")
        min_date, max_date = cursor.fetchone()
        print(f"Overall date range: {min_date} to {max_date}")
        
        # Check recent data availability
        cursor.execute("""
            SELECT 
                YEAR(tradeDate) as year,
                MONTH(tradeDate) as month,
                COUNT(*) as record_count,
                COUNT(DISTINCT ticker) as unique_stocks
            FROM yq_mktstockfactorsonedayget 
            WHERE tradeDate >= '2020-01-01'
            GROUP BY YEAR(tradeDate), MONTH(tradeDate)
            ORDER BY year DESC, month DESC
            LIMIT 20
        """)
        
        recent_data = cursor.fetchall()
        
        print("\nRecent data availability (last 20 months):")
        print("-" * 60)
        print(f"{'Year-Month':<12} {'Records':<10} {'Unique Stocks':<15}")
        print("-" * 60)
        
        for year, month, count, stocks in recent_data:
            print(f"{year}-{month:02d}      {count:<10} {stocks:<15}")
        
        # Check data for specific recent periods
        cursor.execute("""
            SELECT tradeDate, COUNT(*) as daily_count
            FROM yq_mktstockfactorsonedayget 
            WHERE tradeDate >= '2024-01-01'
            GROUP BY tradeDate
            ORDER BY tradeDate DESC
            LIMIT 10
        """)
        
        daily_data = cursor.fetchall()
        
        print("\nMost recent daily data:")
        print("-" * 40)
        print(f"{'Date':<12} {'Records':<10}")
        print("-" * 40)
        
        for date, count in daily_data:
            print(f"{date}   {count:<10}")
        
        # Check sample of recent data
        cursor.execute("""
            SELECT ticker, tradeDate, PE, PB, ROE
            FROM yq_mktstockfactorsonedayget 
            WHERE tradeDate >= '2024-01-01'
            ORDER BY tradeDate DESC, ticker
            LIMIT 10
        """)
        
        sample_data = cursor.fetchall()
        
        print("\nSample of recent data:")
        print("-" * 80)
        print(f"{'Ticker':<10} {'Date':<12} {'PE':<10} {'PB':<10} {'ROE':<10}")
        print("-" * 80)
        
        for ticker, date, pe, pb, roe in sample_data:
            print(f"{ticker:<10} {date}   {pe:<10} {pb:<10} {roe:<10}")
        
        # Check what's the most recent complete month
        cursor.execute("""
            SELECT 
                YEAR(tradeDate) as year,
                MONTH(tradeDate) as month,
                MAX(tradeDate) as last_date,
                COUNT(DISTINCT ticker) as stocks,
                COUNT(*) as records
            FROM yq_mktstockfactorsonedayget 
            WHERE tradeDate >= '2023-01-01'
            GROUP BY YEAR(tradeDate), MONTH(tradeDate)
            HAVING COUNT(DISTINCT ticker) >= 100
            ORDER BY year DESC, month DESC
            LIMIT 10
        """)
        
        complete_months = cursor.fetchall()
        
        print("\nComplete months with sufficient data (100+ stocks):")
        print("-" * 70)
        print(f"{'Year-Month':<12} {'Last Date':<12} {'Stocks':<8} {'Records':<10}")
        print("-" * 70)
        
        for year, month, last_date, stocks, records in complete_months:
            print(f"{year}-{month:02d}      {last_date}   {stocks:<8} {records:<10}")
        
    except mysql.connector.Error as err:
        print(f"Error querying database: {err}")
    finally:
        cursor.close()
        connection.close()

def create_working_dataset():
    """
    Create a working dataset with available data
    """
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='352471Cf',
            database='yuqerdata'
        )
        print("\nCreating working dataset...")
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return
    
    cursor = connection.cursor()
    
    try:
        # Get data from the most recent complete months
        query = """
        SELECT 
            ticker,
            tradeDate,
            PE, PB, PS, PCF,
            ROE, ROA, CurrentRatio, DebtEquityRatio,
            RSI, MACD, MA20, HBETA,
            LCAP, NetProfitGrowRate, OperatingRevenueGrowRate,
            RSTR12, Volatility, BIAS20
        FROM yq_mktstockfactorsonedayget 
        WHERE tradeDate >= '2023-01-01'
        AND ticker IS NOT NULL
        ORDER BY tradeDate DESC, ticker
        LIMIT 20000
        """
        
        print("Extracting working dataset...")
        cursor.execute(query)
        data = cursor.fetchall()
        
        # Column names
        columns = [
            'ticker', 'tradeDate',
            'PE', 'PB', 'PS', 'PCF',
            'ROE', 'ROA', 'CurrentRatio', 'DebtEquityRatio',
            'RSI', 'MACD', 'MA20', 'HBETA',
            'LCAP', 'NetProfitGrowRate', 'OperatingRevenueGrowRate',
            'RSTR12', 'Volatility', 'BIAS20'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        print(f"Working dataset shape: {df.shape}")
        print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
        print(f"Unique stocks: {df['ticker'].nunique()}")
        
        # Group by month to see data distribution
        df['tradeDate'] = pd.to_datetime(df['tradeDate'])
        df['month'] = df['tradeDate'].dt.to_period('M')
        
        monthly_summary = df.groupby('month').agg({
            'ticker': 'nunique',
            'tradeDate': 'count'
        }).rename(columns={'ticker': 'unique_stocks', 'tradeDate': 'total_records'})
        
        print("\nMonthly data summary:")
        print("-" * 40)
        print(monthly_summary)
        
        # Save working dataset
        df.to_csv('./current_results/working_dataset.csv', index=False)
        print("\nWorking dataset saved to './current_results/working_dataset.csv'")
        
        return df
        
    except mysql.connector.Error as err:
        print(f"Error creating dataset: {err}")
        return None
    finally:
        cursor.close()
        connection.close()

def main():
    """
    Main function to check current data availability
    """
    print("CHECKING CURRENT DATA AVAILABILITY")
    print("=" * 50)
    
    # Check database data
    check_database_data()
    
    # Create working dataset
    working_df = create_working_dataset()
    
    if working_df is not None:
        print("\n" + "="*60)
        print("DATA AVAILABILITY ANALYSIS COMPLETED")
        print("="*60)
        print("Next steps:")
        print("1. Use the working dataset for Learn2Rank training")
        print("2. Focus on available time periods with sufficient data")
        print("3. Adjust training parameters based on data availability")

if __name__ == "__main__":
    main()
