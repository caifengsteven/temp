#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMD Stock Analysis Batch Processor
This script analyzes multiple stocks using DMD with Bloomberg data.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dmd_bloomberg_analysis import BloombergDataFetcher, DMD, analyze_stock_with_dmd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_instruments(file_path):
    """Read instruments from a file
    
    Args:
        file_path: Path to the instruments file
        
    Returns:
        list: List of instruments
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Read file
        with open(file_path, 'r') as f:
            # Strip whitespace and filter out empty lines
            instruments = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(instruments)} instruments from {file_path}")
        return instruments
    
    except Exception as e:
        logger.error(f"Error reading instruments file: {e}")
        return []


def create_sample_instruments_file(file_path):
    """Create a sample instruments file
    
    Args:
        file_path: Path to the instruments file
    """
    with open(file_path, 'w') as f:
        f.write("AAPL US Equity\n")
        f.write("MSFT US Equity\n")
        f.write("AMZN US Equity\n")
        f.write("GOOGL US Equity\n")
        f.write("META US Equity\n")
    
    print(f"Sample instruments file created: {file_path}")


def generate_summary_report(output_dir):
    """Generate a summary report of DMD analysis results
    
    Args:
        output_dir: Directory containing analysis results
    """
    logger.info("Generating summary report...")
    
    # Find all result JSON files
    import json
    import glob
    
    result_files = glob.glob(os.path.join(output_dir, "*_results.json"))
    
    if not result_files:
        logger.error("No result files found. Exiting.")
        return
    
    # Load results
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.error(f"Error loading result file {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by MSE
    df.sort_values("mse", inplace=True)
    
    # Save summary to CSV
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    # Generate summary plots
    plt.figure(figsize=(12, 8))
    
    # Plot MSE distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df["mse"], kde=True)
    plt.title("Distribution of MSE")
    plt.xlabel("MSE")
    
    # Plot MAE distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df["mae"], kde=True)
    plt.title("Distribution of MAE")
    plt.xlabel("MAE")
    
    # Plot top 10 stocks by MSE
    plt.subplot(2, 2, 3)
    top10 = df.head(10)
    plt.barh(top10["security"], top10["mse"])
    plt.title("Top 10 Stocks by MSE")
    plt.xlabel("MSE")
    
    # Plot dominant period distribution
    periods = []
    for p in df["dominant_periods"]:
        if p and len(p) > 0:
            periods.append(p[0])  # First dominant period
    
    plt.subplot(2, 2, 4)
    sns.histplot(periods, kde=True)
    plt.title("Distribution of Dominant Periods")
    plt.xlabel("Period (days)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_plots.png"))
    plt.close()
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DMD Stock Analysis Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DMD Stock Analysis Summary Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This report presents the results of analyzing {len(df)} stocks using Dynamic Mode Decomposition (DMD).</p>
                <p>Analysis period: {df["start_date"].iloc[0]} to {df["end_date"].iloc[0]}</p>
                <p>DMD rank: {df["rank"].iloc[0]}</p>
            </div>
            
            <h2>Summary Plots</h2>
            <img src="summary_plots.png" alt="Summary Plots">
            
            <h2>What is DMD?</h2>
            <p>
                Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing complex dynamical systems.
                It was originally developed for fluid dynamics but has found applications in various fields, including finance.
            </p>
            <p>
                DMD works by decomposing time-series data into spatial-temporal coherent structures (modes) with
                associated frequencies and growth/decay rates. These modes can capture underlying patterns in the data
                and be used for forecasting.
            </p>
            <p>
                For stock prediction, DMD can identify cyclical patterns and trends in price movements, potentially
                providing insights that traditional time-series methods might miss.
            </p>
            
            <h2>Top 10 Stocks by Performance (Lowest MSE)</h2>
            <table>
                <tr>
                    <th>Security</th>
                    <th>MSE</th>
                    <th>MAE</th>
                    <th>Dominant Period (days)</th>
                </tr>
    """
    
    for _, row in top10.iterrows():
        dominant_period = row["dominant_periods"][0] if row["dominant_periods"] and len(row["dominant_periods"]) > 0 else "N/A"
        html_report += f"""
                <tr>
                    <td>{row["security"]}</td>
                    <td>{row["mse"]:.4f}</td>
                    <td>{row["mae"]:.4f}</td>
                    <td>{dominant_period:.2f}</td>
                </tr>
        """
    
    html_report += f"""
            </table>
            
            <h2>Stock Analysis Results</h2>
            <p>Click on a stock to view its detailed analysis:</p>
            <ul>
    """
    
    for _, row in df.iterrows():
        security_file = row["security"].replace(' ', '_')
        html_report += f"""
                <li><a href="{security_file}_prediction.png">{row["security"]}</a> - MSE: {row["mse"]:.4f}, Dominant Period: {row["dominant_periods"][0]:.2f} days</li>
        """
    
    html_report += f"""
            </ul>
            
            <h2>Conclusion</h2>
            <p>
                Dynamic Mode Decomposition (DMD) shows varying effectiveness for stock price prediction across the analyzed stocks.
                The top-performing stocks demonstrate that DMD can capture underlying dynamics in some cases, with MSE as low as
                {df["mse"].min():.4f} for the best stock.
            </p>
            <p>
                The dominant periods identified by DMD suggest that most stocks have cyclical patterns with periods
                ranging from {min(periods):.1f} to {max(periods):.1f} days.
            </p>
            <p>
                Overall, DMD appears to be most effective for stocks with clear cyclical patterns and lower volatility.
                Further refinement of the DMD approach, such as optimizing the rank parameter or combining DMD with other
                techniques, could potentially improve prediction accuracy.
            </p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_report)
    
    logger.info(f"Summary report generated and saved to {output_dir}/report.html")


def main():
    """Main function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Analyze multiple stocks using DMD with Bloomberg data")
    parser.add_argument("--instruments", type=str, default="instruments.txt", help="Path to instruments file")
    parser.add_argument("--field", type=str, default="PX_LAST", help="Bloomberg field to analyze")
    parser.add_argument("--days", type=int, default=365*2, help="Number of days of historical data to use")
    parser.add_argument("--rank", type=int, default=10, help="Truncation rank for DMD")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--output-dir", type=str, default="dmd_analysis", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Check if instruments file exists, create sample if not
    if not os.path.exists(args.instruments):
        create_sample_instruments_file(args.instruments)
        print(f"Please edit {args.instruments} with your instruments and run the script again.")
        return
    
    # Read instruments
    instruments = read_instruments(args.instruments)
    
    if not instruments:
        logger.error("No instruments found. Exiting.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=args.days)
    
    # Analyze each instrument
    for instrument in instruments:
        try:
            analyze_stock_with_dmd(
                security=instrument,
                field=args.field,
                start_date=start_date,
                end_date=end_date,
                train_ratio=args.train_ratio,
                rank=args.rank,
                output_dir=args.output_dir
            )
        except Exception as e:
            logger.error(f"Error analyzing {instrument}: {e}")
    
    # Generate summary report
    generate_summary_report(args.output_dir)
    
    print(f"\nAnalysis completed for {len(instruments)} instruments")
    print(f"Summary report generated in {args.output_dir}/report.html")


if __name__ == "__main__":
    main()
