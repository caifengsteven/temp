#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMD Stock Prediction and Backtesting
This script implements Dynamic Mode Decomposition (DMD) for stock prediction,
uses Bloomberg data for backtesting, and analyzes NASDAQ 100 stocks.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.linalg import svd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bloomberg API constants
REFDATA_SVC = "//blp/refdata"


class DMDStockPredictor:
    """Class for stock prediction using Dynamic Mode Decomposition (DMD)"""
    
    def __init__(self, rank: Optional[int] = None, dt: float = 1.0):
        """Initialize the DMD Stock Predictor
        
        Args:
            rank: Truncation rank for SVD (if None, no truncation)
            dt: Time step between snapshots
        """
        self.rank = rank
        self.dt = dt
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        self.omega = None
        self.Atilde = None
        self.X1 = None
        self.X2 = None
        self.time_dynamics = None
        self.reconstructed_data = None
        self.forecast_data = None
        
    def fit(self, X: np.ndarray) -> None:
        """Fit DMD model to data
        
        Args:
            X: Data matrix (features x time)
        """
        # Split data into two time-shifted matrices
        self.X1 = X[:, :-1]
        self.X2 = X[:, 1:]
        
        # SVD of X1
        U, Sigma, Vh = svd(self.X1, full_matrices=False)
        
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
        self.Atilde = U.T @ self.X2 @ Vh.T @ Sinv
        
        # Eigendecomposition of Atilde
        eigenvalues, eigenvectors = np.linalg.eig(self.Atilde)
        
        # Sort eigenvalues by magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute DMD modes
        self.modes = self.X2 @ Vh.T @ Sinv @ eigenvectors
        
        # Compute continuous-time eigenvalues
        self.omega = np.log(eigenvalues) / self.dt
        
        # Compute amplitudes
        self.eigenvalues = eigenvalues
        x1 = self.X1[:, 0]
        self.amplitudes = np.linalg.lstsq(self.modes, x1, rcond=None)[0]
        
        # Compute time dynamics
        t = np.arange(0, self.X1.shape[1]) * self.dt
        self.time_dynamics = np.vander(eigenvalues, len(t), increasing=True).T
        
        # Reconstruct data
        self.reconstructed_data = self.modes @ np.diag(self.amplitudes) @ self.time_dynamics
        
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
        t_pred = np.arange(0, steps) * self.dt
        future_time_dynamics = np.zeros((len(t_pred), len(self.eigenvalues)), dtype=complex)
        
        for i, t in enumerate(t_pred):
            future_time_dynamics[i, :] = self.eigenvalues ** (self.X1.shape[1] + i)
        
        # Predict future values
        self.forecast_data = self.modes @ np.diag(self.amplitudes) @ future_time_dynamics.T
        
        return np.real(self.forecast_data)
    
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
    
    def get_nasdaq100_constituents(self) -> List[str]:
        """Get list of NASDAQ 100 constituents
        
        Returns:
            List[str]: List of NASDAQ 100 constituents
        """
        logger.info("Retrieving NASDAQ 100 constituents...")
        
        # Create a request for index members
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        request.append("securities", "NDX Index")
        request.append("fields", "INDX_MEMBERS")
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        constituents = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data = msg.getElement("securityData")
                    
                    if security_data.hasElement("fieldData"):
                        field_data = security_data.getElement("fieldData")
                        
                        if field_data.hasElement("INDX_MEMBERS"):
                            members = field_data.getElement("INDX_MEMBERS")
                            
                            for i in range(members.numValues()):
                                member = members.getValue(i)
                                ticker = member.getElementAsString("Member Ticker and Exchange Code")
                                constituents.append(f"{ticker} Equity")
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        logger.info(f"Retrieved {len(constituents)} NASDAQ 100 constituents")
        return constituents
    
    def get_historical_data(self, securities: List[str], fields: List[str], 
                           start_date: datetime.datetime, end_date: datetime.datetime) -> Dict[str, pd.DataFrame]:
        """Get historical data for securities
        
        Args:
            securities: List of Bloomberg security identifiers
            fields: List of fields to retrieve
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping securities to their historical data
        """
        logger.info(f"Retrieving historical data for {len(securities)} securities...")
        
        # Process securities in batches to avoid request size limitations
        batch_size = 10
        all_data = {}
        
        for i in range(0, len(securities), batch_size):
            batch = securities[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(securities) + batch_size - 1) // batch_size}...")
            
            # Create a request for historical data
            request = self.refdata_service.createRequest("HistoricalDataRequest")
            
            # Add securities
            for security in batch:
                request.append("securities", security)
            
            # Add fields
            for field in fields:
                request.append("fields", field)
            
            # Set date range
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            
            # Send the request
            self.session.sendRequest(request)
            
            # Process the response
            batch_data = {security: pd.DataFrame() for security in batch}
            
            while True:
                event = self.session.nextEvent(500)  # Timeout in milliseconds
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        security = security_data.getElementAsString("security")
                        
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            
                            # Extract data points
                            data_points = []
                            for i in range(field_data.numValues()):
                                point = field_data.getValue(i)
                                data_point = {"date": point.getElementAsDatetime("date")}
                                
                                for field in fields:
                                    if point.hasElement(field):
                                        data_point[field] = point.getElementAsFloat(field)
                                    else:
                                        data_point[field] = np.nan
                                
                                data_points.append(data_point)
                            
                            # Convert to DataFrame
                            if data_points:
                                df = pd.DataFrame(data_points)
                                df.set_index("date", inplace=True)
                                batch_data[security] = df
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Add batch data to all data
            all_data.update(batch_data)
        
        logger.info(f"Retrieved historical data for {len(all_data)} securities")
        return all_data


class DMDBacktester:
    """Class for backtesting DMD stock prediction"""
    
    def __init__(self, train_window: int = 252, predict_window: int = 21, rank: Optional[int] = None):
        """Initialize the DMD Backtester
        
        Args:
            train_window: Number of days to use for training
            predict_window: Number of days to predict
            rank: Truncation rank for DMD (if None, no truncation)
        """
        self.train_window = train_window
        self.predict_window = predict_window
        self.rank = rank
        self.results = {}
        
    def backtest(self, data: Dict[str, pd.DataFrame], field: str = "PX_LAST") -> Dict[str, Dict]:
        """Backtest DMD prediction on historical data
        
        Args:
            data: Dictionary mapping securities to their historical data
            field: Field to use for prediction
            
        Returns:
            Dict[str, Dict]: Dictionary mapping securities to their backtest results
        """
        logger.info(f"Backtesting DMD prediction for {len(data)} securities...")
        
        self.results = {}
        
        for security, df in data.items():
            if df.empty or field not in df.columns:
                logger.warning(f"No data for {security}, skipping...")
                continue
            
            # Get price data
            prices = df[field].dropna()
            
            if len(prices) < self.train_window + self.predict_window:
                logger.warning(f"Insufficient data for {security}, skipping...")
                continue
            
            # Initialize results
            self.results[security] = {
                "actual": [],
                "predicted": [],
                "mse": [],
                "mae": [],
                "r2": [],
                "dominant_freqs": []
            }
            
            # Walk-forward testing
            for i in range(len(prices) - self.train_window - self.predict_window + 1):
                # Get training data
                train_data = prices.iloc[i:i+self.train_window].values.reshape(1, -1)
                
                # Get test data
                test_data = prices.iloc[i+self.train_window:i+self.train_window+self.predict_window].values
                
                # Fit DMD model
                dmd = DMDStockPredictor(rank=self.rank)
                dmd.fit(train_data)
                
                # Predict
                predicted = dmd.predict(self.predict_window).flatten()
                
                # Get dominant frequencies
                _, dominant_freqs = dmd.get_dominant_modes(n_modes=3)
                
                # Calculate metrics
                mse = mean_squared_error(test_data, predicted)
                mae = mean_absolute_error(test_data, predicted)
                r2 = r2_score(test_data, predicted)
                
                # Store results
                self.results[security]["actual"].append(test_data)
                self.results[security]["predicted"].append(predicted)
                self.results[security]["mse"].append(mse)
                self.results[security]["mae"].append(mae)
                self.results[security]["r2"].append(r2)
                self.results[security]["dominant_freqs"].append(dominant_freqs)
            
            # Convert lists to arrays
            self.results[security]["actual"] = np.array(self.results[security]["actual"])
            self.results[security]["predicted"] = np.array(self.results[security]["predicted"])
            self.results[security]["mse"] = np.array(self.results[security]["mse"])
            self.results[security]["mae"] = np.array(self.results[security]["mae"])
            self.results[security]["r2"] = np.array(self.results[security]["r2"])
            self.results[security]["dominant_freqs"] = np.array(self.results[security]["dominant_freqs"])
            
            logger.info(f"Completed backtest for {security}")
        
        logger.info(f"Completed backtesting for {len(self.results)} securities")
        return self.results
    
    def generate_report(self, output_dir: str = "dmd_report") -> None:
        """Generate a report of backtest results
        
        Args:
            output_dir: Directory to save the report
        """
        if not self.results:
            logger.error("No backtest results to report")
            return
        
        logger.info("Generating backtest report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate overall metrics
        overall_metrics = {
            "security": [],
            "mean_mse": [],
            "mean_mae": [],
            "mean_r2": [],
            "mean_dominant_freq": []
        }
        
        for security, result in self.results.items():
            overall_metrics["security"].append(security)
            overall_metrics["mean_mse"].append(np.mean(result["mse"]))
            overall_metrics["mean_mae"].append(np.mean(result["mae"]))
            overall_metrics["mean_r2"].append(np.mean(result["r2"]))
            overall_metrics["mean_dominant_freq"].append(np.mean(result["dominant_freqs"][:, 0]))
        
        # Convert to DataFrame
        overall_df = pd.DataFrame(overall_metrics)
        overall_df.sort_values("mean_r2", ascending=False, inplace=True)
        
        # Save overall metrics
        overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"), index=False)
        
        # Generate summary plots
        plt.figure(figsize=(12, 8))
        
        # Plot MSE distribution
        plt.subplot(2, 2, 1)
        sns.histplot(overall_df["mean_mse"], kde=True)
        plt.title("Distribution of Mean MSE")
        plt.xlabel("Mean MSE")
        
        # Plot MAE distribution
        plt.subplot(2, 2, 2)
        sns.histplot(overall_df["mean_mae"], kde=True)
        plt.title("Distribution of Mean MAE")
        plt.xlabel("Mean MAE")
        
        # Plot R² distribution
        plt.subplot(2, 2, 3)
        sns.histplot(overall_df["mean_r2"], kde=True)
        plt.title("Distribution of Mean R²")
        plt.xlabel("Mean R²")
        
        # Plot dominant frequency distribution
        plt.subplot(2, 2, 4)
        sns.histplot(overall_df["mean_dominant_freq"], kde=True)
        plt.title("Distribution of Mean Dominant Frequency")
        plt.xlabel("Frequency (cycles/day)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_metrics_distribution.png"))
        plt.close()
        
        # Generate top 10 stocks plot
        top10 = overall_df.head(10)
        
        plt.figure(figsize=(12, 8))
        plt.barh(top10["security"], top10["mean_r2"])
        plt.title("Top 10 Stocks by R² Score")
        plt.xlabel("Mean R²")
        plt.ylabel("Security")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top10_stocks.png"))
        plt.close()
        
        # Generate example prediction plots for top 5 stocks
        for security in top10["security"].head(5):
            result = self.results[security]
            
            # Plot last prediction
            plt.figure(figsize=(12, 6))
            plt.plot(result["actual"][-1], label="Actual")
            plt.plot(result["predicted"][-1], label="Predicted")
            plt.title(f"DMD Prediction for {security}")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{security.replace(' ', '_')}_prediction.png"))
            plt.close()
        
        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DMD Stock Prediction Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
                .metrics {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; width: 23%; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>DMD Stock Prediction Backtest Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>This report presents the results of backtesting Dynamic Mode Decomposition (DMD) for stock price prediction on NASDAQ 100 stocks.</p>
                    <p>Training window: {self.train_window} days</p>
                    <p>Prediction window: {self.predict_window} days</p>
                    <p>DMD rank: {self.rank if self.rank is not None else "Full"}</p>
                    <p>Number of stocks analyzed: {len(self.results)}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Average MSE</h3>
                        <p>{overall_df["mean_mse"].mean():.4f}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Average MAE</h3>
                        <p>{overall_df["mean_mae"].mean():.4f}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Average R²</h3>
                        <p>{overall_df["mean_r2"].mean():.4f}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Average Dominant Frequency</h3>
                        <p>{overall_df["mean_dominant_freq"].mean():.4f} cycles/day</p>
                    </div>
                </div>
                
                <h2>Overall Metrics Distribution</h2>
                <img src="overall_metrics_distribution.png" alt="Overall Metrics Distribution">
                
                <h2>Top 10 Stocks by R² Score</h2>
                <img src="top10_stocks.png" alt="Top 10 Stocks">
                
                <h2>Example Predictions</h2>
        """
        
        for security in top10["security"].head(5):
            security_file = security.replace(' ', '_')
            html_report += f"""
                <h3>{security}</h3>
                <img src="{security_file}_prediction.png" alt="{security} Prediction">
            """
        
        html_report += f"""
                <h2>Top 20 Stocks by Performance</h2>
                <table>
                    <tr>
                        <th>Security</th>
                        <th>Mean MSE</th>
                        <th>Mean MAE</th>
                        <th>Mean R²</th>
                        <th>Mean Dominant Frequency</th>
                    </tr>
        """
        
        for _, row in overall_df.head(20).iterrows():
            html_report += f"""
                    <tr>
                        <td>{row["security"]}</td>
                        <td>{row["mean_mse"]:.4f}</td>
                        <td>{row["mean_mae"]:.4f}</td>
                        <td>{row["mean_r2"]:.4f}</td>
                        <td>{row["mean_dominant_freq"]:.4f}</td>
                    </tr>
            """
        
        html_report += f"""
                </table>
                
                <h2>Conclusion</h2>
                <p>
                    Dynamic Mode Decomposition (DMD) shows varying effectiveness for stock price prediction across the NASDAQ 100 stocks.
                    The top-performing stocks demonstrate that DMD can capture underlying dynamics in some cases, with R² scores
                    reaching {overall_df["mean_r2"].max():.4f} for the best stock.
                </p>
                <p>
                    The dominant frequencies identified by DMD suggest that most stocks have cyclical patterns with periods
                    ranging from {1/overall_df["mean_dominant_freq"].max():.1f} to {1/overall_df["mean_dominant_freq"].min():.1f} days.
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
        
        logger.info(f"Report generated and saved to {output_dir}")


def main():
    """Main function to run DMD stock prediction and backtesting"""
    # Create output directory
    output_dir = "dmd_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Bloomberg data fetcher
    fetcher = BloombergDataFetcher()
    
    try:
        # Start Bloomberg session
        if not fetcher.start_session():
            logger.error("Failed to start Bloomberg session. Exiting.")
            return
        
        # Get NASDAQ 100 constituents
        nasdaq100 = fetcher.get_nasdaq100_constituents()
        
        # Define date range for backtesting
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365*2)  # 2 years of data
        
        # Get historical data
        historical_data = fetcher.get_historical_data(
            securities=nasdaq100,
            fields=["PX_LAST", "OPEN", "HIGH", "LOW", "VOLUME"],
            start_date=start_date,
            end_date=end_date
        )
        
        # Initialize backtester
        backtester = DMDBacktester(
            train_window=252,  # 1 year of trading days
            predict_window=21,  # 1 month of trading days
            rank=10  # Truncation rank for DMD
        )
        
        # Run backtest
        results = backtester.backtest(historical_data)
        
        # Generate report
        backtester.generate_report(output_dir)
        
        print(f"\nBacktesting completed. Report generated in {output_dir}/report.html")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop Bloomberg session
        fetcher.stop_session()


if __name__ == "__main__":
    main()
