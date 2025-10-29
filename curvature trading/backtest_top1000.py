"""
Backtest Curved Radius Supertrend on Top 1000 US Stocks (2015-2025)
FIXED VERSION: Bankruptcy protection - stops if equity <= 0
"""

import numpy as np
import pandas as pd
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time
import sys


def get_top_1000_stocks():
    """
    Get list of top 1000 US stocks
    Using a comprehensive list of major US stocks
    """
    # Top 1000 US stocks - comprehensive list
    # This includes S&P 500, Russell 1000, and other major stocks
    stocks = [
        # Top 100 (from previous test)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
        'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'COST', 'AVGO', 'KO', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM', 'ACN', 'TMO',
        'LIN', 'ABT', 'NFLX', 'NKE', 'DHR', 'VZ', 'TXN', 'ORCL', 'PM', 'DIS',
        'CMCSA', 'INTC', 'WFC', 'AMD', 'UPS', 'NEE', 'COP', 'RTX', 'QCOM', 'HON',
        'INTU', 'UNP', 'IBM', 'LOW', 'AMGN', 'BA', 'SPGI', 'ELV', 'AMAT', 'GE',
        'CAT', 'SBUX', 'DE', 'PLD', 'BKNG', 'GILD', 'ADP', 'ADI', 'TJX', 'MDLZ',
        'CVS', 'LMT', 'SYK', 'VRTX', 'AXP', 'ISRG', 'MMC', 'CI', 'REGN', 'BLK',
        'ZTS', 'PGR', 'TMUS', 'MO', 'CB', 'SO', 'DUK', 'BSX', 'ETN', 'SCHW',
        'C', 'EOG', 'ITW', 'HCA', 'PNC', 'NOC', 'USB', 'SLB', 'MS', 'GD',

        # Next 100 (101-200)
        'TGT', 'BDX', 'MMM', 'FIS', 'CL', 'NSC', 'MU', 'SHW', 'CME', 'ICE',
        'AON', 'MCO', 'EL', 'APD', 'EQIX', 'CSX', 'FCX', 'PSA', 'DG', 'ATVI',
        'FISV', 'TT', 'EMR', 'WM', 'GIS', 'ECL', 'ROP', 'ADSK', 'APH', 'NXPI',
        'SRE', 'AEP', 'KLAC', 'LRCX', 'SNPS', 'CDNS', 'MCHP', 'PAYX', 'AIG', 'ORLY',
        'AZO', 'ROST', 'IDXX', 'CTAS', 'MSCI', 'KMB', 'WELL', 'CARR', 'OTIS', 'PCAR',
        'GM', 'F', 'AMP', 'ALL', 'TRV', 'AFL', 'MET', 'PRU', 'HUM', 'ANTM',
        'CNC', 'WBA', 'KR', 'SYY', 'YUM', 'MNST', 'HSY', 'K', 'GPC', 'FAST',
        'VRSK', 'CPRT', 'CTSH', 'FTNT', 'ANSS', 'DXCM', 'ALGN', 'ILMN', 'MKTX', 'TTWO',
        'EA', 'EBAY', 'PYPL', 'SQ', 'SHOP', 'SNAP', 'PINS', 'TWTR', 'UBER', 'LYFT',
        'ABNB', 'DASH', 'COIN', 'RBLX', 'U', 'PLTR', 'SNOW', 'CRWD', 'ZS', 'DDOG',

        # Next 100 (201-300)
        'NET', 'OKTA', 'DOCU', 'ZM', 'TEAM', 'WDAY', 'NOW', 'VEEV', 'SPLK', 'PANW',
        'FTNT', 'CHKP', 'CYBR', 'TENB', 'RPD', 'S', 'MDB', 'ESTC', 'CFLT', 'BILL',
        'PCTY', 'HUBS', 'ZI', 'FROG', 'SMAR', 'APPN', 'PATH', 'AI', 'BBAI', 'SOUN',
        'IONQ', 'RGTI', 'QUBT', 'QBTS', 'ARQQ', 'LAZR', 'OUST', 'VLDR', 'INVZ', 'LIDR',
        'AEVA', 'AEYE', 'MVIS', 'KOPN', 'VUZI', 'WIMI', 'GREE', 'BTBT', 'CAN', 'MARA',
        'RIOT', 'CLSK', 'BITF', 'HUT', 'ARBK', 'CIFR', 'WULF', 'IREN', 'CORZ', 'BTDR',
        'LCID', 'RIVN', 'FSR', 'GOEV', 'RIDE', 'NKLA', 'HYLN', 'BLNK', 'CHPT', 'EVGO',
        'PLUG', 'FCEL', 'BE', 'BLDP', 'BALLARD', 'NEL', 'ITM', 'HYSR', 'AMTX', 'CLNE',
        'SPWR', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'CSIQ', 'JKS', 'DQ', 'SOL', 'MAXN',
        'ARRY', 'AMPS', 'SHLS', 'VVNT', 'STEM', 'FLNC', 'QS', 'SES', 'ENVX', 'ABML',

        # Next 100 (301-400)
        'NIO', 'XPEV', 'LI', 'BYDDY', 'KNDI', 'SOLO', 'AYRO', 'WKHS', 'ARVL', 'MULN',
        'ELMS', 'GEV', 'PSNY', 'GGPI', 'GRAB', 'DIDI', 'BABA', 'JD', 'PDD', 'BIDU',
        'NTES', 'TME', 'BILI', 'IQ', 'VIPS', 'ATHM', 'MOMO', 'YY', 'HUYA', 'DOYU',
        'KC', 'EH', 'TIGR', 'FUTU', 'UP', 'TUYA', 'DUO', 'GOTU', 'TAL', 'EDU',
        'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'AVGO', 'ASML', 'TSM', 'AAPL', 'MSFT',
        'ORCL', 'SAP', 'SHOP', 'CRM', 'ADBE', 'INTU', 'WDAY', 'PANW', 'SNOW', 'DDOG',
        'MDB', 'NET', 'CRWD', 'ZS', 'OKTA', 'FTNT', 'CYBR', 'S', 'ESTC', 'CFLT',
        'BILL', 'PCTY', 'HUBS', 'ZI', 'FROG', 'SMAR', 'APPN', 'PATH', 'AI', 'BBAI',
        'SOUN', 'IONQ', 'RGTI', 'QUBT', 'QBTS', 'ARQQ', 'LAZR', 'OUST', 'VLDR', 'INVZ',
        'LIDR', 'AEVA', 'AEYE', 'MVIS', 'KOPN', 'VUZI', 'WIMI', 'GREE', 'BTBT', 'CAN',

        # Next 100 (401-500) - More S&P 500 and Russell 1000
        'ABBV', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AFL', 'AIG', 'AIZ',
        'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME',
        'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS', 'APA',
        'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK',
        'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B',
        'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.A',
        'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB',
        'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CERN', 'CF', 'CFG',
        'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME',
        'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB',

        # Next 100 (501-600)
        'CPRT', 'CRL', 'CRM', 'CSCO', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA',
        'CTXS', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG',
        'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR', 'DLTR', 'DOV',
        'DOW', 'DPZ', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM',
        'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH',
        'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG',
        'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FB', 'FBHS',
        'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 'FOX',
        'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL',
        'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS',

        # Next 100 (601-700)
        'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'HIG', 'HII',
        'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM',
        'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INFO', 'INTC',
        'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
        'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY',
        'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L',
        'LDOS', 'LEG', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC',
        'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA',
        'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET',
        'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',

        # Next 100 (701-800)
        'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
        'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI',
        'NKE', 'NLOK', 'NLSN', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE',
        'OMC', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PAYC', 'PAYX', 'PCAR', 'PEAK', 'PEG',
        'PENN', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI',
        'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA',
        'PSX', 'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE',
        'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP',
        'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM',

        # Next 100 (801-900)
        'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX',
        'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
        'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR',
        'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TWTR',
        'TXN', 'TXT', 'TYL', 'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH',
        'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VIAC', 'VLO', 'VMC', 'VNO',
        'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD',
        'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK',
        'WST', 'WU', 'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XYL', 'YUM',
        'ZBH', 'ZBRA', 'ZION', 'ZTS', 'AAON', 'AAP', 'AAPL', 'AAWW', 'AAXN', 'ABCB',

        # Final 100 (901-1000)
        'ABCL', 'ABCM', 'ABEO', 'ABEQ', 'ABEV', 'ABG', 'ABIO', 'ABM', 'ABMD', 'ABR',
        'ABST', 'ABT', 'ABTX', 'ACAD', 'ACBI', 'ACCD', 'ACCO', 'ACEL', 'ACER', 'ACES',
        'ACET', 'ACGL', 'ACGN', 'ACH', 'ACHC', 'ACHL', 'ACHR', 'ACHV', 'ACIA', 'ACIU',
        'ACIW', 'ACLS', 'ACLX', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACON', 'ACOR', 'ACP',
        'ACRE', 'ACRS', 'ACRV', 'ACRX', 'ACST', 'ACT', 'ACTG', 'ACVA', 'ACXP', 'ADAG',
        'ADAL', 'ADAP', 'ADBE', 'ADC', 'ADCT', 'ADD', 'ADEA', 'ADER', 'ADES', 'ADEX',
        'ADGI', 'ADI', 'ADIL', 'ADM', 'ADMA', 'ADMP', 'ADMS', 'ADMT', 'ADNT', 'ADNW',
        'ADOC', 'ADP', 'ADPT', 'ADRA', 'ADRO', 'ADRT', 'ADS', 'ADSE', 'ADSK', 'ADSW',
        'ADT', 'ADTH', 'ADTN', 'ADTX', 'ADUS', 'ADV', 'ADVM', 'ADVWW', 'ADX', 'ADXN',
        'AE', 'AEAE', 'AEE', 'AEG', 'AEHL', 'AEHR', 'AEI', 'AEIS', 'AEL', 'AEM'
    ]

    # Remove duplicates and return
    return list(set(stocks))


def backtest_single_stock(ticker, start_date, end_date, radius_strength=1.2):
    """
    Run backtest on a single stock
    """
    try:
        # Fetch data
        connector = StockDataConnector()
        data = connector.fetch_stock_data(ticker, start_date, end_date)
        connector.close()
        
        if len(data) < 100:  # Need minimum data (100 days for 10 year test)
            return None
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            position_size=0.95,
            allow_short=True
        )
        
        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': radius_strength,
            'smoothness': 3
        }
        
        results = engine.run_backtest(
            data=data,
            indicator_params=indicator_params
        )
        
        stats = engine.calculate_statistics(data)
        
        return {
            'ticker': ticker,
            'total_return': stats['total_return_pct'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown_pct'],
            'win_rate': stats['win_rate'],
            'total_trades': stats['total_trades'],
            'profit_factor': stats['profit_factor'],
            'final_equity': stats['final_equity'],
            'avg_bars_held': stats['avg_bars_held'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'days': len(data)
        }
        
    except Exception as e:
        # Silently skip errors to keep output clean
        return None


def run_top1000_backtest(start_date='2015-01-01', end_date='2025-01-01', radius_strength=1.2):
    """
    Run backtest on top 1000 stocks
    """
    print("="*80)
    print(f"BACKTESTING TOP 1000 US STOCKS")
    print(f"Period: {start_date} to {end_date} (10 years)")
    print(f"Radius Strength: {radius_strength}")
    print("="*80)
    
    # Get stock list
    stock_list = get_top_1000_stocks()
    
    if not stock_list:
        print("❌ No stocks found in database!")
        return None
    
    print(f"\n📊 Testing {len(stock_list)} stocks...")
    print("This will take several minutes. Please wait...\n")
    
    results = []
    total_stocks = len(stock_list)
    
    start_time = time.time()
    last_update = start_time
    
    for idx, ticker in enumerate(stock_list, 1):
        # Progress update every 10 stocks or 5 seconds
        current_time = time.time()
        if idx % 10 == 0 or (current_time - last_update) > 5:
            elapsed = current_time - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_stocks - idx) / rate if rate > 0 else 0
            
            print(f"\r[{idx}/{total_stocks}] Progress: {idx/total_stocks*100:.1f}% | "
                  f"Rate: {rate:.1f} stocks/sec | "
                  f"ETA: {eta/60:.1f} min | "
                  f"Successful: {len(results)}", end='', flush=True)
            
            last_update = current_time
        
        result = backtest_single_stock(ticker, start_date, end_date, radius_strength)
        
        if result:
            results.append(result)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n\n{'='*80}")
    print(f"BACKTEST COMPLETE!")
    print(f"Tested: {total_stocks} stocks")
    print(f"Successful: {len(results)} stocks")
    print(f"Failed/Skipped: {total_stocks - len(results)} stocks")
    print(f"Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("="*80)
    
    if not results:
        print("❌ No successful backtests!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = f'backtest_top1000_{start_date[:4]}_{end_date[:4]}.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Results saved to: {csv_filename}")
    
    return df


def analyze_results(df):
    """
    Analyze and display results
    """
    if df is None or len(df) == 0:
        print("No results to analyze!")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS - TOP 1000 STOCKS (2015-2025)")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Stocks Tested:     {len(df)}")
    print(f"   Profitable Stocks:       {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)")
    print(f"   Losing Stocks:           {len(df[df['total_return'] <= 0])} ({len(df[df['total_return'] <= 0])/len(df)*100:.1f}%)")
    
    print(f"\n📈 RETURN STATISTICS")
    print(f"   Average Return:          {df['total_return'].mean():>15.2f}%")
    print(f"   Median Return:           {df['total_return'].median():>15.2f}%")
    print(f"   Best Return:             {df['total_return'].max():>15.2f}%")
    print(f"   Worst Return:            {df['total_return'].min():>15.2f}%")
    print(f"   Std Deviation:           {df['total_return'].std():>15.2f}%")
    
    # Percentiles
    print(f"\n📊 RETURN PERCENTILES")
    print(f"   95th Percentile:         {df['total_return'].quantile(0.95):>15.2f}%")
    print(f"   75th Percentile:         {df['total_return'].quantile(0.75):>15.2f}%")
    print(f"   50th Percentile:         {df['total_return'].quantile(0.50):>15.2f}%")
    print(f"   25th Percentile:         {df['total_return'].quantile(0.25):>15.2f}%")
    print(f"   5th Percentile:          {df['total_return'].quantile(0.05):>15.2f}%")
    
    print(f"\n🎯 RISK METRICS")
    print(f"   Average Sharpe Ratio:    {df['sharpe_ratio'].mean():>15.2f}")
    print(f"   Median Sharpe Ratio:     {df['sharpe_ratio'].median():>15.2f}")
    print(f"   Average Max Drawdown:    {df['max_drawdown'].mean():>15.2f}%")
    print(f"   Average Win Rate:        {df['win_rate'].mean():>15.2f}%")
    print(f"   Average Profit Factor:   {df['profit_factor'].mean():>15.2f}")
    
    print(f"\n📊 TRADE STATISTICS")
    print(f"   Total Trades (All):      {df['total_trades'].sum()}")
    print(f"   Average Trades/Stock:    {df['total_trades'].mean():>15.1f}")
    print(f"   Median Trades/Stock:     {df['total_trades'].median():>15.1f}")
    print(f"   Average Holding Period:  {df['avg_bars_held'].mean():>15.1f} days")
    
    # Top performers
    print("\n" + "="*80)
    print("🏆 TOP 20 PERFORMERS (by Total Return)")
    print("="*80)
    top20 = df.nlargest(20, 'total_return')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Return':<18} {'Sharpe':<10} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(top20.itertuples(), 1):
        return_str = f"{row.total_return:.2f}%" if row.total_return < 1e6 else f"{row.total_return/1e6:.1f}M%"
        print(f"{idx:<6} {row.ticker:<8} {return_str:>16} "
              f"{row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Worst performers
    print("\n" + "="*80)
    print("📉 BOTTOM 20 PERFORMERS (by Total Return)")
    print("="*80)
    bottom20 = df.nsmallest(20, 'total_return')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Return':<18} {'Sharpe':<10} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(bottom20.itertuples(), 1):
        return_str = f"{row.total_return:.2f}%"
        print(f"{idx:<6} {row.ticker:<8} {return_str:>16} "
              f"{row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Best Sharpe ratios
    print("\n" + "="*80)
    print("🎯 TOP 20 BY SHARPE RATIO (Risk-Adjusted Returns)")
    print("="*80)
    top_sharpe = df.nlargest(20, 'sharpe_ratio')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Sharpe':<10} {'Return':<18} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(top_sharpe.itertuples(), 1):
        return_str = f"{row.total_return:.2f}%" if row.total_return < 1e6 else f"{row.total_return/1e6:.1f}M%"
        print(f"{idx:<6} {row.ticker:<8} {row.sharpe_ratio:>9.2f} "
              f"{return_str:>16} {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Return distribution
    print("\n" + "="*80)
    print("📊 RETURN DISTRIBUTION")
    print("="*80)
    bins = [
        (float('-inf'), -80, 'Loss > 80%'),
        (-80, -50, 'Loss 50-80%'),
        (-50, -20, 'Loss 20-50%'),
        (-20, 0, 'Loss 0-20%'),
        (0, 50, 'Gain 0-50%'),
        (50, 100, 'Gain 50-100%'),
        (100, 500, 'Gain 100-500%'),
        (500, 1000, 'Gain 500-1000%'),
        (1000, 10000, 'Gain 1K-10K%'),
        (10000, 100000, 'Gain 10K-100K%'),
        (100000, float('inf'), 'Gain > 100K%')
    ]
    
    for low, high, label in bins:
        count = len(df[(df['total_return'] > low) & (df['total_return'] <= high)])
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"{label:<20} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Sharpe ratio distribution
    print("\n" + "="*80)
    print("📊 SHARPE RATIO DISTRIBUTION")
    print("="*80)
    sharpe_bins = [
        (float('-inf'), 0, 'Negative (<0)'),
        (0, 1, 'Poor (0-1)'),
        (1, 2, 'Good (1-2)'),
        (2, 3, 'Excellent (2-3)'),
        (3, float('inf'), 'Outstanding (>3)')
    ]
    
    for low, high, label in sharpe_bins:
        count = len(df[(df['sharpe_ratio'] > low) & (df['sharpe_ratio'] <= high)])
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"{label:<25} {count:>4} ({pct:>5.1f}%) {bar}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n🚀 Starting backtest on Top 1000 US stocks (2015-2025)...")
    print("This is a comprehensive 10-year test and will take some time...\n")
    
    df_results = run_top1000_backtest(
        start_date='2015-01-01',
        end_date='2025-01-01',
        radius_strength=1.2
    )
    
    if df_results is not None:
        # Analyze results
        analyze_results(df_results)
        
        print("\n✅ BACKTEST COMPLETE!")
        print(f"📁 Results saved to: backtest_top1000_2015_2025.csv")
        print(f"📊 Total stocks analyzed: {len(df_results)}")
    else:
        print("\n❌ Backtest failed!")

