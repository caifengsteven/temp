#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMD Stock Analysis with Bloomberg Data
This script implements Dynamic Mode Decomposition (DMD) for stock analysis
using Bloomberg data as the source.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import blpapi
import datetime
import os
import logging
from typing import List, Dict, Tuple, Optional
from scipy.linalg import svd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bloomberg API constants
REFDATA_SVC = "//blp/refdata"


class DMD:
    """Dynamic Mode Decomposition for time series analysis"""
    
    def __init__(self, rank: Optional[int] = None):
        """Initialize DMD
        
        Args:
            rank: Truncation rank for SVD (if None, no truncation)
        """
        self.rank = rank
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        self.omega = None
        self.dt = 1.0  # Time step between snapshots
        
    def fit(self, X: np.ndarray) -> None:
        """Fit DMD model to data
        
        Args:
            X: Data matrix (features x time)
        """
        # Split data into two time-shifted matrices
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # SVD of X1
        U, Sigma, Vh = svd(X1, full_matrices=False)
        
        # Truncate SVD if rank is specified
        if self.rank is not None:
            r = min(self.rank, len(Sigma))
            U = U[:, :r]
            Sigma = Sigma[:r]
            Vh = Vh[:r, :]
        else:
            r = len(Sigma)
        
        # Compute reduced DMD matrix
        Sinv = np.diag(1.0 / Sigma)
        Atilde = U.T @ X2 @ Vh.T @ Sinv
        
        # Eigendecomposition of Atilde
        eigenvalues, eigenvectors = np.linalg.eig(Atilde)
        
        # Sort eigenvalues by magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute DMD modes
        self.modes = X2 @ Vh.T @ Sinv @ eigenvectors
        
        # Compute continuous-time eigenvalues
        self.omega = np.log(eigenvalues) / self.dt
        
        # Compute amplitudes
        self.eigenvalues = eigenvalues
        x1 = X1[:, 0]
        self.amplitudes = np.linalg.lstsq(self.modes, x1, rcond=None)[0]
        
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.eigenvalues is None:
            raise ValueError("Model must be fit before prediction")
        
        # Compute future time dynamics
        future_time_dynamics = np.zeros((steps, len(self.eigenvalues)), dtype=complex)
        
        for i in range(steps):
            future_time_dynamics[i, :] = self.eigenvalues ** (i + 1)
        
        # Predict future values
        forecast_data = self.modes @ np.diag(self.amplitudes) @ future_time_dynamics.T
        
        return np.real(forecast_data)
    
    def get_dominant_modes(self, n_modes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Get dominant DMD modes and frequencies
        
        Args:
            n_modes: Number of dominant modes to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Dominant modes and frequencies
        """
        if self.modes is None:
            raise ValueError("Model must be fit before getting dominant modes")
        
        # Sort modes by amplitude
        idx = np.argsort(np.abs(self.amplitudes))[::-1]
        dominant_modes = self.modes[:, idx[:n_modes]]
        dominant_freqs = np.abs(self.omega[idx[:n_modes]]) / (2 * np.pi)
        
        return dominant_modes, dominant_freqs


class BloombergDataFetcher:
    """Class to fetch stock data from Bloomberg"""
    
    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg Data Fetcher
        
        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None
        
    def start_session(self) -> bool:
        """Start a Bloomberg API session
        
        Returns:
            bool: True if session started successfully, False otherwise
        """
        logger.info("Starting Bloomberg API session...")
        
        # Initialize session options
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        
        # Create a session
        self.session = blpapi.Session(session_options)
        
        # Start the session
        if not self.session.start():
            logger.error("Failed to start session.")
            return False
        
        logger.info("Session started successfully.")
        
        # Open the reference data service
        if not self.session.openService(REFDATA_SVC):
            logger.error("Failed to open reference data service.")
            return False
        
        self.refdata_service = self.session.getService(REFDATA_SVC)
        logger.info("Reference data service opened successfully.")
        
        return True
    
    def stop_session(self) -> None:
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")
    
    def get_historical_data(self, security: str, field: str, 
                           start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """Get historical data for a security
        
        Args:
            security: Bloomberg security identifier
            field: Field to retrieve (e.g., "PX_LAST")
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        logger.info(f"Retrieving historical data for {security}...")
        
        # Create a request for historical data
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        # Add security
        request.append("securities", security)
        
        # Add field
        request.append("fields", field)
        
        # Set date range
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        data_points = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    
                    if security_data.hasElement("fieldData"):
                        field_data = security_data.getElement("fieldData")
                        
                        # Extract data points
                        for i in range(field_data.numValues()):
                            point = field_data.getValue(i)
                            data_point = {"date": point.getElementAsDatetime("date")}
                            
                            if point.hasElement(field):
                                data_point[field] = point.getElementAsFloat(field)
                            else:
                                data_point[field] = np.nan
                            
                            data_points.append(data_point)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        if data_points:
            df = pd.DataFrame(data_points)
            df.set_index("date", inplace=True)
            logger.info(f"Retrieved {len(df)} data points for {security}")
            return df
        else:
            logger.warning(f"No data retrieved for {security}")
            return pd.DataFrame()


def analyze_stock_with_dmd(security: str, field: str = "PX_LAST", 
                          start_date: Optional[datetime.datetime] = None, 
                          end_date: Optional[datetime.datetime] = None,
                          train_ratio: float = 0.8,
                          rank: Optional[int] = 10,
                          output_dir: str = "dmd_analysis") -> None:
    """Analyze a stock using DMD
    
    Args:
        security: Bloomberg security identifier
        field: Field to analyze (e.g., "PX_LAST")
        start_date: Start date for analysis (default: 2 years ago)
        end_date: End date for analysis (default: today)
        train_ratio: Ratio of data to use for training
        rank: Truncation rank for DMD
        output_dir: Directory to save results
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.datetime.now()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=365*2)  # 2 years of data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Bloomberg data fetcher
    fetcher = BloombergDataFetcher()
    
    try:
        # Start Bloomberg session
        if not fetcher.start_session():
            logger.error("Failed to start Bloomberg session. Exiting.")
            return
        
        # Get historical data
        df = fetcher.get_historical_data(security, field, start_date, end_date)
        
        if df.empty:
            logger.error(f"No data available for {security}. Exiting.")
            return
        
        # Save raw data
        df.to_csv(os.path.join(output_dir, f"{security.replace(' ', '_')}_raw_data.csv"))
        
        # Prepare data for DMD
        prices = df[field].dropna().values
        
        # Split into training and testing sets
        train_size = int(len(prices) * train_ratio)
        train_data = prices[:train_size]
        test_data = prices[train_size:]
        
        # Reshape for DMD (features x time)
        X = train_data.reshape(1, -1)
        
        # Fit DMD model
        dmd = DMD(rank=rank)
        dmd.fit(X)
        
        # Get dominant modes and frequencies
        _, dominant_freqs = dmd.get_dominant_modes(n_modes=5)
        
        # Calculate periods (in days)
        periods = 1.0 / dominant_freqs
        
        # Predict future values
        predicted = dmd.predict(len(test_data)).flatten()
        
        # Calculate error metrics
        mse = np.mean((test_data - predicted) ** 2)
        mae = np.mean(np.abs(test_data - predicted))
        
        # Save results
        results = {
            "security": security,
            "field": field,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "train_size": train_size,
            "test_size": len(test_data),
            "rank": rank,
            "mse": mse,
            "mae": mae,
            "dominant_frequencies": dominant_freqs.tolist(),
            "dominant_periods": periods.tolist()
        }
        
        # Save results to JSON
        import json
        with open(os.path.join(output_dir, f"{security.replace(' ', '_')}_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot training data
        plt.subplot(2, 1, 1)
        plt.plot(range(len(train_data)), train_data, label="Training Data")
        plt.title(f"DMD Analysis for {security} - {field}")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        # Plot test data vs. prediction
        plt.subplot(2, 1, 2)
        plt.plot(range(len(test_data)), test_data, label="Actual")
        plt.plot(range(len(predicted)), predicted, label="DMD Prediction")
        plt.title(f"DMD Prediction vs. Actual (MSE: {mse:.2f}, MAE: {mae:.2f})")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{security.replace(' ', '_')}_prediction.png"))
        plt.close()
        
        # Plot dominant frequencies
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(dominant_freqs)), dominant_freqs)
        plt.title(f"Dominant Frequencies for {security}")
        plt.xlabel("Mode Index")
        plt.ylabel("Frequency (cycles/day)")
        plt.xticks(range(len(dominant_freqs)))
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{security.replace(' ', '_')}_frequencies.png"))
        plt.close()
        
        # Plot dominant periods
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(periods)), periods)
        plt.title(f"Dominant Periods for {security}")
        plt.xlabel("Mode Index")
        plt.ylabel("Period (days)")
        plt.xticks(range(len(periods)))
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{security.replace(' ', '_')}_periods.png"))
        plt.close()
        
        logger.info(f"Analysis completed for {security}")
        logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Dominant frequencies (cycles/day): {dominant_freqs}")
        logger.info(f"Dominant periods (days): {periods}")
        
        print(f"\nAnalysis completed for {security}")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"Dominant periods (days): {periods}")
        print(f"Results saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop Bloomberg session
        fetcher.stop_session()


def main():
    """Main function"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Analyze stocks using DMD with Bloomberg data")
    parser.add_argument("--security", type=str, default="AAPL US Equity", help="Bloomberg security identifier")
    parser.add_argument("--field", type=str, default="PX_LAST", help="Bloomberg field to analyze")
    parser.add_argument("--days", type=int, default=365*2, help="Number of days of historical data to use")
    parser.add_argument("--rank", type=int, default=10, help="Truncation rank for DMD")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--output-dir", type=str, default="dmd_analysis", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Set dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=args.days)
    
    # Analyze stock
    analyze_stock_with_dmd(
        security=args.security,
        field=args.field,
        start_date=start_date,
        end_date=end_date,
        train_ratio=args.train_ratio,
        rank=args.rank,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
