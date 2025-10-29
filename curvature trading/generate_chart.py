"""
Quick Chart Generator

Generate professional candlestick charts with Curved Radius Supertrend
"""

import sys
from visualize_candlestick import plot_professional_chart
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def main():
    """Generate professional chart"""
    
    # Default values
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    radius_strength = 0.5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        start_date = sys.argv[2]
    if len(sys.argv) > 3:
        end_date = sys.argv[3]
    if len(sys.argv) > 4:
        radius_strength = float(sys.argv[4])
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         PROFESSIONAL CANDLESTICK CHART GENERATOR                 ║
╚══════════════════════════════════════════════════════════════════╝

Ticker:          {ticker}
Period:          {start_date} to {end_date}
Radius Strength: {radius_strength}
""")
    
    # Generate chart
    fig, ax = plot_professional_chart(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        radius_strength=radius_strength
    )
    
    if fig is not None:
        print("\n✅ Chart generated successfully!")
        print("\nDisplaying chart...")
        plt.show()
    else:
        print("\n❌ Failed to generate chart")
        return 1
    
    return 0


if __name__ == "__main__":
    print("""
Usage:
    python generate_chart.py [TICKER] [START_DATE] [END_DATE] [RADIUS]

Examples:
    python generate_chart.py AAPL
    python generate_chart.py GOOGL 2023-01-01 2023-12-31
    python generate_chart.py TSLA 2023-01-01 2023-12-31 0.5
    """)
    
    exit(main())

