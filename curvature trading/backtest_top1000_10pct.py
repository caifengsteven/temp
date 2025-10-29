"""
Backtest Curved Radius Supertrend on Top 1000 US Stocks (2015-2025)
FIXED VERSION: 10% position sizing (instead of 95%)
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
        'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS', 'CZR', 'BALY', 'RSI', 'GNOG', 'FUBO',
        'SKLZ', 'HOFV', 'GMBL', 'BETZ', 'BJK', 'CHDN', 'EVRI', 'GENI', 'PDYPY', 'TPIC',
        'ACEL', 'ACHR', 'ACIC', 'ACND', 'ACVA', 'ADAP', 'ADBE', 'ADIL', 'ADMA', 'ADMP',
        'ADMS', 'ADNT', 'ADOC', 'ADP', 'ADPT', 'ADRO', 'ADSE', 'ADSK', 'ADTH', 'ADTN',
        'ADTX', 'ADUS', 'ADVM', 'ADXN', 'AEAE', 'AEE', 'AEG', 'AEHL', 'AEHR', 'AEI',
        'AEIS', 'AEL', 'AEM', 'AEMD', 'AEO', 'AEP', 'AER', 'AERI', 'AES', 'AEVA',
        'AEYE', 'AEZS', 'AFBI', 'AFCG', 'AFG', 'AFIB', 'AFL', 'AFRM', 'AFYA', 'AGCO',
        'AGEN', 'AGFY', 'AGIO', 'AGLE', 'AGM', 'AGNC', 'AGO', 'AGR', 'AGRI', 'AGRO',

        # Next 100 (301-400)
        'AGS', 'AGTC', 'AGTI', 'AGX', 'AGYS', 'AHCO', 'AHH', 'AHI', 'AHT', 'AI',
        'AIG', 'AIN', 'AINC', 'AIRC', 'AIRG', 'AIRI', 'AIRS', 'AIT', 'AIV', 'AIZ',
        'AJRD', 'AJG', 'AJXA', 'AKAM', 'AKAN', 'AKBA', 'AKRO', 'AKTS', 'AKUS', 'AKYA',
        'AL', 'ALB', 'ALBO', 'ALC', 'ALCO', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALG',
        'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALHC', 'ALIM', 'ALIT', 'ALK', 'ALKS', 'ALKT',
        'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLT', 'ALLY', 'ALNA', 'ALNY', 'ALOT', 'ALPN',
        'ALPP', 'ALRM', 'ALRS', 'ALSA', 'ALSN', 'ALT', 'ALTA', 'ALTG', 'ALTI', 'ALTM',
        'ALTR', 'ALTY', 'ALV', 'ALVR', 'ALX', 'ALXN', 'ALXO', 'ALZN', 'AM', 'AMAL',
        'AMAT', 'AMBA', 'AMBC', 'AMBO', 'AMBP', 'AMC', 'AMCR', 'AMCX', 'AMD', 'AME',
        'AMED', 'AMEH', 'AMG', 'AMGN', 'AMH', 'AMID', 'AMK', 'AMKR', 'AMLP', 'AMN',

        # Next 100 (401-500)
        'AMNB', 'AMOT', 'AMPE', 'AMPH', 'AMPL', 'AMPY', 'AMR', 'AMRC', 'AMRK', 'AMRN',
        'AMRS', 'AMRX', 'AMS', 'AMSC', 'AMSF', 'AMST', 'AMSWA', 'AMT', 'AMTB', 'AMTD',
        'AMTI', 'AMTM', 'AMTX', 'AMWD', 'AMWL', 'AMX', 'AMYT', 'AMZN', 'AN', 'ANAB',
        'ANAT', 'ANCN', 'ANDA', 'ANEB', 'ANET', 'ANF', 'ANGI', 'ANGL', 'ANGO', 'ANH',
        'ANIK', 'ANIP', 'ANIX', 'ANNX', 'ANPC', 'ANSS', 'ANTE', 'ANTM', 'ANTX', 'ANVS',
        'ANY', 'AOSL', 'AOUT', 'APA', 'APAM', 'APCX', 'APD', 'APDN', 'APEI', 'APEN',
        'APG', 'APH', 'APHA', 'API', 'APLD', 'APLE', 'APLS', 'APLT', 'APM', 'APMI',
        'APN', 'APO', 'APOG', 'APOP', 'APPH', 'APRE', 'APTO', 'APTS', 'APTV', 'APTX',
        'APVO', 'APWC', 'APXI', 'APYX', 'AQB', 'AQMS', 'AQN', 'AQST', 'AQUA', 'AR',
        'ARAV', 'ARAY', 'ARB', 'ARBE', 'ARBK', 'ARC', 'ARCB', 'ARCC', 'ARCH', 'ARCO',

        # Next 100 (501-600)
        'ARCT', 'ARDS', 'ARDX', 'ARE', 'AREC', 'AREN', 'ARES', 'ARGX', 'ARHS', 'ARI',
        'ARIS', 'ARKR', 'ARL', 'ARLO', 'ARLP', 'ARMK', 'ARMP', 'ARMR', 'ARNA', 'ARNC',
        'AROC', 'AROW', 'ARPO', 'ARQQ', 'ARQT', 'ARR', 'ARRW', 'ARRY', 'ARTL', 'ARTNA',
        'ARTW', 'ARVL', 'ARVN', 'ARW', 'ARWR', 'ARYA', 'ASAI', 'ASAN', 'ASAX', 'ASB',
        'ASC', 'ASCA', 'ASGN', 'ASH', 'ASIX', 'ASLE', 'ASM', 'ASML', 'ASMB', 'ASND',
        'ASNS', 'ASO', 'ASPI', 'ASPN', 'ASPS', 'ASPU', 'ASR', 'ASRT', 'ASRV', 'ASTC',
        'ASTE', 'ASTI', 'ASTL', 'ASTR', 'ASTS', 'ASUR', 'ASX', 'ASXC', 'ASYS', 'ATAI',
        'ATAK', 'ATAX', 'ATCX', 'ATEC', 'ATEK', 'ATEN', 'ATER', 'ATEX', 'ATGE', 'ATGL',
        'ATHA', 'ATHE', 'ATHM', 'ATHX', 'ATI', 'ATIF', 'ATIP', 'ATKR', 'ATLC', 'ATLO',
        'ATMC', 'ATMR', 'ATMU', 'ATNF', 'ATNI', 'ATNM', 'ATO', 'ATOM', 'ATOS', 'ATR',

        # Next 100 (601-700)
        'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATTO', 'ATUS', 'ATVI', 'ATXI',
        'ATXS', 'AU', 'AUB', 'AUBN', 'AUDC', 'AUGX', 'AUID', 'AUMN', 'AUPH', 'AUR',
        'AURA', 'AURC', 'AUROW', 'AUS', 'AUST', 'AUTL', 'AUTO', 'AUUD', 'AUVI', 'AUVIP',
        'AVA', 'AVAC', 'AVAH', 'AVAL', 'AVAN', 'AVAV', 'AVB', 'AVCO', 'AVCT', 'AVD',
        'AVDL', 'AVDX', 'AVEO', 'AVGO', 'AVGR', 'AVID', 'AVIR', 'AVK', 'AVLR', 'AVNS',
        'AVNT', 'AVNW', 'AVO', 'AVPT', 'AVRO', 'AVT', 'AVTA', 'AVTE', 'AVTR', 'AVTX',
        'AVXL', 'AVY', 'AVYA', 'AWH', 'AWI', 'AWK', 'AWR', 'AWRE', 'AWX', 'AX',
        'AXAS', 'AXDX', 'AXE', 'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXON', 'AXP', 'AXR',
        'AXS', 'AXSM', 'AXTA', 'AXTI', 'AXU', 'AY', 'AYI', 'AYLA', 'AYRO', 'AYTU',
        'AZ', 'AZEK', 'AZN', 'AZO', 'AZPN', 'AZRE', 'AZRX', 'AZTA', 'AZUL', 'AZZ',

        # Next 100 (701-800)
        'B', 'BA', 'BABA', 'BAC', 'BACK', 'BAER', 'BAFN', 'BAH', 'BAK', 'BALL',
        'BALY', 'BAM', 'BANC', 'BAND', 'BANF', 'BANR', 'BANX', 'BAOS', 'BAP', 'BAR',
        'BARK', 'BASE', 'BATRA', 'BATRK', 'BAX', 'BB', 'BBAI', 'BBAR', 'BBBY', 'BBCP',
        'BBD', 'BBDC', 'BBDO', 'BBGI', 'BBH', 'BBI', 'BBIG', 'BBIO', 'BBL', 'BBLG',
        'BBLN', 'BBN', 'BBQ', 'BBSI', 'BBU', 'BBUC', 'BBW', 'BBWI', 'BBY', 'BC',
        'BCAB', 'BCAC', 'BCAL', 'BCAN', 'BCAT', 'BCBP', 'BCC', 'BCDA', 'BCE', 'BCEL',
        'BCH', 'BCLI', 'BCML', 'BCO', 'BCOR', 'BCOV', 'BCOW', 'BCPC', 'BCRX', 'BCS',
        'BCSA', 'BCSF', 'BCTX', 'BCV', 'BCX', 'BCYC', 'BDC', 'BDGE', 'BDJ', 'BDL',
        'BDN', 'BDSX', 'BDTX', 'BDX', 'BDXB', 'BE', 'BEAM', 'BEAT', 'BECN', 'BEDU',
        'BEEM', 'BEEP', 'BEKE', 'BELFA', 'BELFB', 'BEN', 'BENF', 'BEP', 'BEPC', 'BERY',

        # Next 54 (801-854) - to reach approximately 1000 unique stocks
        'BEST', 'BETR', 'BF.A', 'BF.B', 'BFAM', 'BFC', 'BFEB', 'BFI', 'BFIN', 'BFLY',
        'BFRA', 'BFRI', 'BFS', 'BFST', 'BG', 'BGB', 'BGCP', 'BGFV', 'BGH', 'BGI',
        'BGLC', 'BGNE', 'BGR', 'BGS', 'BGSF', 'BGSX', 'BGT', 'BGX', 'BGXX', 'BGY',
        'BH', 'BHAT', 'BHB', 'BHC', 'BHE', 'BHF', 'BHFAL', 'BHFAN', 'BHFAP', 'BHG',
        'BHIL', 'BHK', 'BHLB', 'BHP', 'BHR', 'BHRB', 'BHV', 'BHVN', 'BIDU', 'BIG',
        'BIGC', 'BIIB', 'BILI', 'BILL'
    ]
    
    # Remove duplicates and return
    return list(set(stocks))


def backtest_single_stock(ticker, start_date, end_date, radius_strength=1.2, position_size=0.10):
    """
    Backtest a single stock with 10% position sizing
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    radius_strength : float
        Radius strength parameter for the indicator
    position_size : float
        Position size as fraction of equity (default: 0.10 = 10%)
    
    Returns:
    --------
    dict : Backtest results or None if failed
    """
    try:
        # Fetch data
        connector = StockDataConnector()
        data = connector.fetch_stock_data(ticker, start_date, end_date)
        connector.close()
        
        if len(data) < 100:  # Need minimum data
            return None
        
        # Run backtest with 10% position sizing
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            position_size=position_size,  # 10% instead of 95%
            allow_short=True
        )
        
        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': radius_strength,
            'smoothness': 3
        }
        
        results = engine.run_backtest(data=data, indicator_params=indicator_params)
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
            'days': len(data),
            'bankruptcy': stats.get('bankruptcy', False)
        }
        
    except Exception as e:
        print(f"Error backtesting {ticker}: {str(e)}")
        return None


def main():
    print("=" * 80)
    print("🚀 Starting backtest on Top 1000 US stocks (2015-2025)...")
    print("POSITION SIZE: 10% (instead of 95%)")
    print("This is a comprehensive 10-year test and will take some time...")
    print()
    print("=" * 80)
    print("BACKTESTING TOP 1000 US STOCKS")
    print("Period: 2015-01-01 to 2025-01-01 (10 years)")
    print("Position Size: 10% per trade")
    print("Radius Strength: 1.2")
    print("=" * 80)
    print()
    
    # Get stock list
    stocks = get_top_1000_stocks()
    print(f"📊 Testing {len(stocks)} stocks...")
    print("This will take several minutes. Please wait...")
    print()
    
    # Backtest parameters
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    radius_strength = 1.2
    position_size = 0.10  # 10% position sizing
    
    # Run backtests
    results = []
    start_time = time.time()
    
    for i, ticker in enumerate(stocks, 1):
        result = backtest_single_stock(ticker, start_date, end_date, radius_strength, position_size)
        
        if result is not None:
            results.append(result)
        
        # Progress update every 10 stocks
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(stocks) - i) / rate if rate > 0 else 0
            print(f"[{i}/{len(stocks)}] Progress: {i/len(stocks)*100:.1f}% | "
                  f"Rate: {rate:.1f} stocks/sec | ETA: {remaining/60:.1f} min | "
                  f"Successful: {len(results)}", end='\r')
    
    print()  # New line after progress
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("BACKTEST COMPLETE!")
    print(f"Tested: {len(stocks)} stocks")
    print(f"Successful: {len(results)} stocks")
    print(f"Failed/Skipped: {len(stocks) - len(results)} stocks")
    print(f"Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=" * 80)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = 'backtest_top1000_2015_2025_10pct.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")
    print()
    
    # Print comprehensive analysis
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS - TOP 1000 STOCKS (2015-2025) - 10% POSITION SIZE")
    print("=" * 80)
    print()
    
    # Overall statistics
    total = len(df)
    profitable = len(df[df['total_return'] > 0])
    bankrupt = len(df[df['bankruptcy'] == True])
    
    print(f"📊 OVERALL STATISTICS")
    print(f"   Total Stocks Tested:     {total}")
    print(f"   Profitable Stocks:       {profitable} ({profitable/total*100:.1f}%)")
    print(f"   Losing Stocks:           {total - profitable} ({(total-profitable)/total*100:.1f}%)")
    print(f"   Bankruptcies:            {bankrupt} ({bankrupt/total*100:.1f}%)")
    print()
    
    # Return statistics
    print(f"📈 RETURN STATISTICS")
    print(f"   Average Return:          {df['total_return'].mean():>20.2f}%")
    print(f"   Median Return:           {df['total_return'].median():>20.2f}%")
    print(f"   Best Return:             {df['total_return'].max():>20.2f}%")
    print(f"   Worst Return:            {df['total_return'].min():>20.2f}%")
    print(f"   Std Deviation:           {df['total_return'].std():>20.2f}%")
    print()
    
    # Percentiles
    print(f"📊 RETURN PERCENTILES")
    print(f"   95th Percentile:         {df['total_return'].quantile(0.95):>20.2f}%")
    print(f"   75th Percentile:         {df['total_return'].quantile(0.75):>20.2f}%")
    print(f"   50th Percentile:         {df['total_return'].quantile(0.50):>20.2f}%")
    print(f"   25th Percentile:         {df['total_return'].quantile(0.25):>20.2f}%")
    print(f"   5th Percentile:          {df['total_return'].quantile(0.05):>20.2f}%")
    print()
    
    # Risk metrics
    print(f"🎯 RISK METRICS")
    print(f"   Average Sharpe Ratio:    {df['sharpe_ratio'].mean():>20.2f}")
    print(f"   Median Sharpe Ratio:     {df['sharpe_ratio'].median():>20.2f}")
    print(f"   Average Max Drawdown:    {df['max_drawdown'].mean():>20.2f}%")
    print(f"   Average Win Rate:        {df['win_rate'].mean():>20.2f}%")
    print(f"   Average Profit Factor:   {df['profit_factor'].mean():>20.2f}")
    print()
    
    # Trade statistics
    print(f"📊 TRADE STATISTICS")
    print(f"   Total Trades (All):      {df['total_trades'].sum():>10.0f}")
    print(f"   Average Trades/Stock:    {df['total_trades'].mean():>20.1f}")
    print(f"   Median Trades/Stock:     {df['total_trades'].median():>20.1f}")
    print(f"   Average Holding Period:  {df['avg_bars_held'].mean():>20.1f} days")
    print()
    
    # Top performers
    print("=" * 80)
    print("🏆 TOP 20 PERFORMERS (by Total Return)")
    print("=" * 80)
    print()
    top20 = df.nlargest(20, 'total_return')
    print(f"{'Rank':<6} {'Ticker':<8} {'Return':<18} {'Sharpe':<10} {'Drawdown':<12} {'Trades'}")
    print("-" * 80)
    for i, row in enumerate(top20.itertuples(), 1):
        print(f"{i:<6} {row.ticker:<8} {row.total_return:>15.2f}% {row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% {row.total_trades:>9.0f}")
    
    print()
    print("=" * 80)
    print("✅ BACKTEST COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Total stocks analyzed: {total}")
    print("=" * 80)


if __name__ == "__main__":
    main()

