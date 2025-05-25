import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime as dt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# For Bloomberg data access
try:
    import pdblp
    from pdblp import BCon
    HAS_BLOOMBERG = True
    print("Bloomberg API available.")
except ImportError:
    HAS_BLOOMBERG = False
    print("Bloomberg API not available. Will use sample data for demonstration.")


class SimilarityMeasures:
    """Similarity measures implementation for financial time series"""
    
    @staticmethod
    def exLift(series_i, series_j, time_zone_i=None, time_zone_j=None):
        """
        Calculate extended lift measure between two time series
        
        Parameters:
        -----------
        series_i : pd.Series
            First time series of returns
        series_j : pd.Series
            Second time series of returns
        time_zone_i : int, optional
            Time zone of first series (1-4)
        time_zone_j : int, optional
            Time zone of second series (1-4)
            
        Returns:
        --------
        dict
            Dictionary with four exLift values for all combinations of up/down movements
        """
        # Initialize counters
        uu, ud, du, dd = 0, 0, 0, 0
        
        # Calculate total up/down magnitudes for normalization
        ups_i = np.sum(np.abs(series_i[series_i > 0]))
        downs_i = np.sum(np.abs(series_i[series_i < 0]))
        ups_j = np.sum(np.abs(series_j[series_j > 0]))
        downs_j = np.sum(np.abs(series_j[series_j < 0]))
        
        # If time zones are provided, adjust the index for day shift
        if time_zone_i is not None and time_zone_j is not None:
            if time_zone_i <= time_zone_j:
                # i is before or same time zone as j, use next day for j
                for i in range(len(series_i) - 1):
                    if i + 1 >= len(series_j):
                        break
                    if series_i.iloc[i] > 0 and series_j.iloc[i+1] > 0:
                        uu += abs(series_j.iloc[i+1])
                    elif series_i.iloc[i] > 0 and series_j.iloc[i+1] < 0:
                        ud += abs(series_j.iloc[i+1])
                    elif series_i.iloc[i] < 0 and series_j.iloc[i+1] > 0:
                        du += abs(series_j.iloc[i+1])
                    elif series_i.iloc[i] < 0 and series_j.iloc[i+1] < 0:
                        dd += abs(series_j.iloc[i+1])
            else:
                # i is after j in time zone, use same day
                for i in range(len(series_i)):
                    if i >= len(series_j):
                        break
                    if series_i.iloc[i] > 0 and series_j.iloc[i] > 0:
                        uu += abs(series_j.iloc[i])
                    elif series_i.iloc[i] > 0 and series_j.iloc[i] < 0:
                        ud += abs(series_j.iloc[i])
                    elif series_i.iloc[i] < 0 and series_j.iloc[i] > 0:
                        du += abs(series_j.iloc[i])
                    elif series_i.iloc[i] < 0 and series_j.iloc[i] < 0:
                        dd += abs(series_j.iloc[i])
        else:
            # If no time zones, assume same day comparison
            for i in range(len(series_i)):
                if i >= len(series_j):
                    break
                if series_i.iloc[i] > 0 and series_j.iloc[i] > 0:
                    uu += abs(series_j.iloc[i])
                elif series_i.iloc[i] > 0 and series_j.iloc[i] < 0:
                    ud += abs(series_j.iloc[i])
                elif series_i.iloc[i] < 0 and series_j.iloc[i] > 0:
                    du += abs(series_j.iloc[i])
                elif series_i.iloc[i] < 0 and series_j.iloc[i] < 0:
                    dd += abs(series_j.iloc[i])
        
        # Calculate exLift values with safeguards for division by zero
        exlift_uu = uu / (ups_i * ups_j) if ups_i * ups_j > 0 else 0
        exlift_ud = ud / (ups_i * downs_j) if ups_i * downs_j > 0 else 0
        exlift_du = du / (downs_i * ups_j) if downs_i * ups_j > 0 else 0
        exlift_dd = dd / (downs_i * downs_j) if downs_i * downs_j > 0 else 0
        
        return {
            'uu': exlift_uu,
            'ud': exlift_ud,
            'du': exlift_du,
            'dd': exlift_dd
        }


class ExLiftNetwork:
    """Implementation of exLift network structure for financial time series"""
    
    def __init__(self):
        self.adjacency_matrix = None
        self.series_names = None
        self.time_zones = None
    
    def build_network(self, returns_df, time_zones=None):
        """
        Build exLift network from return series
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with return series as columns
        time_zones : dict, optional
            Dictionary mapping series names to time zones (1-4)
            
        Returns:
        --------
        np.ndarray
            Adjacency matrix of the network
        """
        self.series_names = returns_df.columns
        self.time_zones = time_zones
        n = len(returns_df.columns)
        
        # Initialize adjacency matrix blocks
        UU = np.zeros((n, n))
        UD = np.zeros((n, n))
        DU = np.zeros((n, n))
        DD = np.zeros((n, n))
        
        # Calculate exLift for each pair of series
        for i in tqdm(range(n), desc="Building exLift network"):
            series_i = returns_df.iloc[:, i]
            tz_i = time_zones[returns_df.columns[i]] if time_zones and returns_df.columns[i] in time_zones else None
            
            for j in range(n):
                series_j = returns_df.iloc[:, j]
                tz_j = time_zones[returns_df.columns[j]] if time_zones and returns_df.columns[j] in time_zones else None
                
                exlift = SimilarityMeasures.exLift(series_i, series_j, tz_i, tz_j)
                
                UU[i, j] = exlift['uu']
                UD[i, j] = exlift['ud']
                DU[i, j] = exlift['du']
                DD[i, j] = exlift['dd']
        
        # Combine blocks into full adjacency matrix
        self.adjacency_matrix = np.block([
            [UU, UD],
            [DU, DD]
        ])
        
        return self.adjacency_matrix


class DiMexRank:
    """PageRank-based Direction of Movement prediction with extended Lift Network"""
    
    def __init__(self, damping_factor=0.85, max_iter=100, tol=1e-6):
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        self.network = None
        self.adjacency_matrix = None
        self.series_names = None
        self.n_series = None
        
    def train(self, returns_df, time_zones=None):
        """
        Train DiMexRank by building exLift network
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with return series as columns
        time_zones : dict, optional
            Dictionary mapping series names to time zones (1-4)
        """
        self.network = ExLiftNetwork()
        self.adjacency_matrix = self.network.build_network(returns_df, time_zones)
        self.series_names = returns_df.columns
        self.n_series = len(self.series_names)
        
    def _modified_pagerank(self, init_vector):
        """
        Compute modified PageRank with personalization vector
        
        Parameters:
        -----------
        init_vector : np.ndarray
            Personalization vector for PageRank
            
        Returns:
        --------
        np.ndarray
            PageRank vector
        """
        n = len(self.adjacency_matrix)
        
        # Create stochastic matrix
        M = self.adjacency_matrix.copy()
        
        # Normalize rows to create transition probabilities
        row_sums = M.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        M = M / row_sums[:, np.newaxis]
        
        # Initialize PageRank vector
        pr = init_vector.copy()
        
        # Power iteration method
        for _ in range(self.max_iter):
            pr_new = self.damping_factor * (M.T @ pr) + (1 - self.damping_factor) / n * init_vector
            
            # Check convergence
            if np.linalg.norm(pr_new - pr, 1) < self.tol:
                return pr_new
            
            pr = pr_new
            
        return pr
    
    def predict(self, known_directions, known_indices):
        """
        Predict direction of movement using DiMexRank
        
        Parameters:
        -----------
        known_directions : np.ndarray
            Binary array with known directions (1 for up, 0 for down)
        known_indices : list
            Indices of series with known directions
            
        Returns:
        --------
        np.ndarray
            Predicted directions for all series (1 for up, 0 for down)
        """
        n = self.n_series
        
        # Prepare initialization vector
        init = np.zeros(2*n)
        
        # Set known directions
        for i, idx in enumerate(known_indices):
            if idx < n and i < len(known_directions):
                if known_directions[i] > 0:  # Up
                    init[idx] = 1
                else:  # Down
                    init[n + idx] = 1
                
        # Run modified PageRank
        pr = self._modified_pagerank(init)
        
        # Get predicted directions
        up_ranks = pr[:n]
        down_ranks = pr[n:]
        
        # Determine direction by comparing up and down ranks
        predicted_directions = np.zeros(n)
        for i in range(n):
            if up_ranks[i] > down_ranks[i]:
                predicted_directions[i] = 1
                
        return predicted_directions


class MixDiMex:
    """Mixture of experts using DiMexRank and supervised learning"""
    
    def __init__(self, supervised_model=None, damping_factor=0.85):
        if supervised_model is None:
            self.supervised_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.supervised_model = supervised_model
            
        self.dimexrank = DiMexRank(damping_factor=damping_factor)
        self.candidates = []
        self.series_names = None
        self.include_candidates = True
        self.supervised_models = {}
        
    def train(self, train_df, val_df, time_zones=None, lags=10):
        """
        Train MixDiMex model using training and validation data
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data with return series as columns
        val_df : pd.DataFrame
            Validation data with return series as columns
        time_zones : dict, optional
            Dictionary mapping series names to time zones (1-4)
        lags : int, optional
            Number of lags to use for supervised model
        """
        # Ensure both DataFrames have the same columns
        common_columns = train_df.columns.intersection(val_df.columns)
        if len(common_columns) < len(train_df.columns):
            print(f"Warning: {len(train_df.columns) - len(common_columns)} columns dropped due to missing data in validation set")
            train_df = train_df[common_columns]
            val_df = val_df[common_columns]
        
        self.series_names = train_df.columns.tolist()
        n_series = len(self.series_names)
        
        # Phase 1: Compare supervised vs DiMexRank
        print("Phase 1: Comparing supervised vs DiMexRank on validation set")
        
        # Train DiMexRank
        self.dimexrank.train(train_df, time_zones)
        
        # Prepare supervised models
        for series in self.series_names:
            self.supervised_models[series] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train supervised models
        for i, series in enumerate(self.series_names):
            # Create lagged features for supervised model
            X_train, y_train = self._create_features(train_df[series], lags)
            if len(X_train) > 0:
                # Train supervised model for this series
                self.supervised_models[series].fit(X_train, y_train)
        
        # Evaluate both models on validation set
        dimexrank_perf = np.zeros(n_series)
        supervised_perf = np.zeros(n_series)
        
        # First, evaluate DiMexRank
        # Assume American time zone series (zone 1) are known
        american_indices = [i for i, s in enumerate(self.series_names) 
                          if s in time_zones and time_zones[s] == 1]
        
        if not american_indices:
            print("Warning: No markets found in American time zone. Using first market as known.")
            k = 1
            american_indices = [0]
        
        for day in range(lags, len(val_df)):
            # Get known directions for this day
            known_directions = np.array([1 if val_df.iloc[day, i] > 0 else 0 for i in american_indices])
            
            # Predict directions using DiMexRank
            predicted = self.dimexrank.predict(known_directions, american_indices)
            
            # Compare with actual directions for non-American series
            non_american_indices = [i for i in range(n_series) if i not in american_indices]
            for i in non_american_indices:
                actual = 1 if val_df.iloc[day, i] > 0 else 0
                if predicted[i] == actual:
                    dimexrank_perf[i] += 1
        
        # Normalize to get accuracy (only for non-American markets)
        valid_days = len(val_df) - lags
        if valid_days > 0:
            for i in non_american_indices:
                dimexrank_perf[i] = dimexrank_perf[i] / valid_days
        
        # Now evaluate supervised models
        for i, series in enumerate(self.series_names):
            if i not in american_indices:  # Only evaluate non-American markets
                X_val, y_val = self._create_features(val_df[series], lags)
                if len(X_val) > 0 and len(y_val) > 0 and series in self.supervised_models:
                    try:
                        y_pred = self.supervised_models[series].predict(X_val)
                        supervised_perf[i] = accuracy_score(y_val, y_pred)
                    except Exception as e:
                        print(f"Error evaluating supervised model for {series}: {e}")
                        supervised_perf[i] = 0
        
        # Find candidates (series better predicted by supervised model)
        self.candidates = []
        for i, series in enumerate(self.series_names):
            if i not in american_indices:  # Only consider non-American markets
                if supervised_perf[i] > dimexrank_perf[i]:
                    self.candidates.append(series)
        
        print(f"Found {len(self.candidates)} candidate series better predicted by supervised model:")
        print(self.candidates)
        
        # Phase 2: Check if candidates are helpful in network
        if self.candidates:
            print("\nPhase 2: Checking if candidates are helpful in network")
            
            # Exclude candidates from network
            train_filtered = train_df.drop(columns=self.candidates)
            filtered_series_names = train_filtered.columns.tolist()
            time_zones_filtered = {s: tz for s, tz in time_zones.items() if s not in self.candidates}
            
            # Build network without candidates
            dimexrank_filtered = DiMexRank(damping_factor=self.dimexrank.damping_factor)
            dimexrank_filtered.train(train_filtered, time_zones_filtered)
            
            # Evaluate on validation set
            dimexrank_filtered_perf = np.zeros(len(filtered_series_names))
            val_filtered = val_df[filtered_series_names]
            
            # Adjust american indices for filtered dataset
            american_indices_filtered = [i for i, s in enumerate(filtered_series_names) 
                                       if s in time_zones_filtered and time_zones_filtered[s] == 1]
            
            if not american_indices_filtered:
                print("Warning: No American markets in filtered dataset. Using first market.")
                american_indices_filtered = [0]
            
            for day in range(lags, len(val_filtered)):
                # Get known directions for this day
                known_directions = np.array([1 if val_filtered.iloc[day, i] > 0 else 0 
                                           for i in american_indices_filtered])
                
                # Predict directions using DiMexRank
                predicted = dimexrank_filtered.predict(known_directions, american_indices_filtered)
                
                # Compare with actual directions for non-American series
                non_american_indices_filtered = [i for i in range(len(filtered_series_names)) 
                                               if i not in american_indices_filtered]
                
                for i in non_american_indices_filtered:
                    actual = 1 if val_filtered.iloc[day, i] > 0 else 0
                    if predicted[i] == actual:
                        dimexrank_filtered_perf[i] += 1
            
            # Normalize to get accuracy (only for non-American markets)
            valid_days = len(val_filtered) - lags
            if valid_days > 0:
                for i in non_american_indices_filtered:
                    dimexrank_filtered_perf[i] = dimexrank_filtered_perf[i] / valid_days
            
            # Map filtered indices back to original series
            filtered_to_orig = {s: i for i, s in enumerate(self.series_names)}
            
            # Compare with original DiMexRank performance
            non_candidate_series = [s for s in filtered_series_names 
                                  if s not in [self.series_names[i] for i in american_indices]]
            
            if non_candidate_series:
                # Calculate average performance for non-American, non-candidate series
                avg_perf_original = np.mean([dimexrank_perf[filtered_to_orig[s]] for s in non_candidate_series])
                
                # Calculate average performance for filtered model
                filtered_non_american_indices = [i for i, s in enumerate(filtered_series_names) 
                                              if s in non_candidate_series]
                avg_perf_filtered = np.mean([dimexrank_filtered_perf[i] for i in filtered_non_american_indices])
                
                print(f"Average DiMexRank performance with candidates: {avg_perf_original:.4f}")
                print(f"Average DiMexRank performance without candidates: {avg_perf_filtered:.4f}")
                
                # Decide whether to include candidates in network
                self.include_candidates = avg_perf_original >= avg_perf_filtered
            else:
                self.include_candidates = True
            
            print(f"Decision: {'Include' if self.include_candidates else 'Exclude'} candidates in network")
        
        # Final training on combined train+val data
        print("\nFinal training on combined dataset")
        combined_df = pd.concat([train_df, val_df])
        
        if not self.include_candidates and self.candidates:
            # Train DiMexRank without candidates
            combined_filtered = combined_df.drop(columns=self.candidates)
            time_zones_filtered = {s: tz for s, tz in time_zones.items() if s in combined_filtered.columns}
            self.dimexrank.train(combined_filtered, time_zones_filtered)
        else:
            # Train DiMexRank with all series
            self.dimexrank.train(combined_df, time_zones)
        
        # Train supervised models for candidates
        for series in self.candidates:
            X, y = self._create_features(combined_df[series], lags)
            if len(X) > 0 and len(y) > 0:
                self.supervised_models[series].fit(X, y)
        
    def _create_features(self, series, lags):
        """Create lagged features for supervised models"""
        data = []
        labels = []
        
        if len(series) <= lags:
            return np.array(data), np.array(labels)
        
        for i in range(lags, len(series)):
            data.append(series.iloc[i-lags:i].values)
            labels.append(1 if series.iloc[i] > 0 else 0)
            
        return np.array(data), np.array(labels)
    
    def predict(self, test_df, known_indices, lags=10):
        """
        Predict directions using MixDiMex
        
        Parameters:
        -----------
        test_df : pd.DataFrame
            Test data with return series as columns
        known_indices : list
            Indices of series with known directions (typically American markets)
        lags : int, optional
            Number of lags to use for supervised model
            
        Returns:
        --------
        np.ndarray
            Predicted directions for all series (1 for up, 0 for down)
        float
            Accuracy of predictions
        dict
            Dictionary with accuracy for each series
        """
        # Ensure test_df has the same columns as our trained model
        missing_columns = set(self.series_names) - set(test_df.columns)
        if missing_columns:
            print(f"Warning: {len(missing_columns)} columns missing in test data: {missing_columns}")
            # Create a DataFrame with only available columns
            available_series = [s for s in self.series_names if s in test_df.columns]
            test_df = test_df[available_series]
        
        n_series = len(self.series_names)
        
        # Initialize arrays for predictions and actuals
        predictions = np.zeros((len(test_df) - lags, n_series))
        actuals = np.zeros((len(test_df) - lags, n_series))
        
        # For each day in test set
        for day in range(lags, len(test_df)):
            day_idx = day - lags
            
            # Get actual directions for this day (for evaluation)
            for i, series in enumerate(self.series_names):
                if series in test_df.columns:
                    actuals[day_idx, i] = 1 if test_df.loc[test_df.index[day], series] > 0 else 0
            
            # Get known directions for this day
            valid_known_indices = [idx for idx in known_indices if idx < len(self.series_names)]
            known_series = [self.series_names[idx] for idx in valid_known_indices if self.series_names[idx] in test_df.columns]
            known_indices_adjusted = [list(test_df.columns).index(s) for s in known_series]
            
            if not known_indices_adjusted:
                print("Warning: No known indices found in test data. Using first column.")
                known_indices_adjusted = [0]
                
            known_directions = np.array([
                1 if test_df.iloc[day, i] > 0 else 0 
                for i in known_indices_adjusted
            ])
            
            # If not using candidates in network
            if not self.include_candidates and self.candidates:
                # Get filtered dataframe and series
                available_candidates = [c for c in self.candidates if c in test_df.columns]
                test_filtered = test_df.drop(columns=available_candidates)
                series_filtered = test_filtered.columns.tolist()
                
                # Map known_indices to filtered indices
                known_indices_filtered = []
                for idx in known_indices:
                    if idx < len(self.series_names):
                        series_name = self.series_names[idx]
                        if series_name in series_filtered:
                            known_indices_filtered.append(series_filtered.index(series_name))
                
                if not known_indices_filtered:
                    known_indices_filtered = [0]  # Use first series if no known indices
                
                # Adjust known_directions for filtered dataset
                known_directions_filtered = np.array([
                    1 if test_filtered.iloc[day, i] > 0 else 0 
                    for i in known_indices_filtered
                ])
                
                # Predict with DiMexRank for non-candidate series
                try:
                    dimex_predictions = self.dimexrank.predict(known_directions_filtered, known_indices_filtered)
                    
                    # Map predictions back to original indices
                    for i, series in enumerate(series_filtered):
                        if i < len(dimex_predictions) and series in self.series_names:
                            orig_idx = self.series_names.index(series)
                            predictions[day_idx, orig_idx] = dimex_predictions[i]
                except Exception as e:
                    print(f"Error predicting with DiMexRank on day {day}: {e}")
            else:
                # Predict with DiMexRank for all series
                try:
                    dimex_predictions = self.dimexrank.predict(known_directions, known_indices_adjusted)
                    
                    # Use DiMexRank predictions for non-candidate series
                    for i, series in enumerate(self.series_names):
                        if series not in self.candidates and i < len(dimex_predictions):
                            predictions[day_idx, i] = dimex_predictions[i]
                except Exception as e:
                    print(f"Error predicting with DiMexRank on day {day}: {e}")
            
            # Use supervised model for candidate series
            for series in self.candidates:
                if series in test_df.columns and series in self.supervised_models:
                    i = self.series_names.index(series)
                    X = test_df[series].iloc[day-lags:day].values.reshape(1, -1)
                    try:
                        predictions[day_idx, i] = self.supervised_models[series].predict(X)[0]
                    except Exception as e:
                        print(f"Error predicting with supervised model for {series} on day {day}: {e}")
                        # If prediction fails, use default (previous day's value)
                        if day > 0:
                            predictions[day_idx, i] = 1 if test_df.iloc[day-1, test_df.columns.get_loc(series)] > 0 else 0
        
        # Calculate accuracy
        correct = (predictions == actuals).sum()
        total = predictions.size
        accuracy = correct / total if total > 0 else 0
        
        # Calculate accuracy per series
        series_accuracy = {}
        for i, series in enumerate(self.series_names):
            correct_series = (predictions[:, i] == actuals[:, i]).sum()
            total_series = len(predictions)
            series_accuracy[series] = correct_series / total_series if total_series > 0 else 0
        
        return predictions, accuracy, series_accuracy


def load_data_from_bloomberg(symbols, start_date, end_date, retries=3, chunk_years=5):
    """
    Load market data from Bloomberg
    
    Parameters:
    -----------
    symbols : list
        List of Bloomberg ticker symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    retries : int, optional
        Number of retries for failed requests
    chunk_years : int, optional
        Maximum years per request to avoid timeouts
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with closing prices for each symbol
    """
    if not HAS_BLOOMBERG:
        raise ImportError("Bloomberg API not available")
    
    print("Connecting to Bloomberg...")
    con = BCon(timeout=60000)
    con.start()
    print("Connected to Bloomberg successfully")
    
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Split date range into chunks to avoid timeouts
    date_chunks = []
    current_start = start_dt
    
    while current_start < end_dt:
        chunk_end = min(current_start + pd.DateOffset(years=chunk_years), end_dt)
        date_chunks.append((current_start, chunk_end))
        current_start = chunk_end + pd.DateOffset(days=1)
    
    # Initialize dictionary to store results
    all_data = {}
    
    for symbol in tqdm(symbols, desc="Loading data from Bloomberg"):
        symbol_data = []
        
        for chunk_start, chunk_end in date_chunks:
            for attempt in range(retries):
                try:
                    print(f"Fetching {symbol} from {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    
                    # Request data from Bloomberg
                    data = con.bdh(
                        tickers=symbol,
                        flds=['PX_LAST'],
                        start_date=chunk_start.strftime('%Y%m%d'),
                        end_date=chunk_end.strftime('%Y%m%d')
                    )
                    
                    if not data.empty:
                        symbol_data.append(data)
                        break
                    else:
                        print(f"No data returned for {symbol} on attempt {attempt+1}/{retries}")
                        if attempt < retries - 1:
                            time.sleep(2)  # Wait before retry
                            
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait before retry
        
        if symbol_data:
            # Combine all chunks for this symbol
            combined_data = pd.concat(symbol_data)
            
            # Handle multi-index if present
            if isinstance(combined_data.columns, pd.MultiIndex):
                price_cols = [col for col in combined_data.columns if 'PX_LAST' in col[1]]
                if price_cols:
                    all_data[symbol] = combined_data[price_cols[0]]
                else:
                    all_data[symbol] = combined_data.iloc[:, 0]
            else:
                all_data[symbol] = combined_data.iloc[:, 0]
            
            print(f"Loaded {len(all_data[symbol])} data points for {symbol}")
    
    # Close connection
    con.stop()
    print("Bloomberg connection closed")
    
    # Create a DataFrame from all symbols that have data
    if all_data:
        # We'll use the index as a basis for alignment
        all_indices = sorted(set().union(*[series.index for series in all_data.values()]))
        
        # Create a DataFrame with all dates
        df = pd.DataFrame(index=all_indices)
        
        # Add each series to the DataFrame
        for symbol, series in all_data.items():
            df[symbol] = series
    
        return df
    
    return pd.DataFrame()


def calculate_returns(prices_df):
    """Calculate log returns from price series"""
    returns_df = np.log(prices_df / prices_df.shift(1))
    # Replace infinite values with NaN
    returns_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return returns_df


def evaluate_strategy(predictions, actuals, test_df, lags=10, stop_loss_pct=0.02, take_profit_pct=0.04):
    """
    Evaluate trading strategy based on direction predictions
    
    Parameters:
    -----------
    predictions : np.ndarray
        Array of predicted directions (1 for up, 0 for down)
    actuals : np.ndarray
        Array of actual directions (1 for up, 0 for down)
    test_df : pd.DataFrame
        DataFrame with return data
    lags : int, optional
        Number of lags used in predictions
    stop_loss_pct : float, optional
        Stop loss percentage
    take_profit_pct : float, optional
        Take profit percentage
        
    Returns:
    --------
    dict
        Dictionary with strategy performance metrics
    """
    n_series = predictions.shape[1]
    n_days = predictions.shape[0]
    series_names = test_df.columns[:n_series]
    
    # Initialize performance metrics
    performance = {
        'accuracy': {},
        'cumulative_returns': {},
        'win_rate': {},
        'avg_win': {},
        'avg_loss': {},
        'sharpe_ratio': {},
        'max_drawdown': {},
        'returns': {}
    }
    
    # Get the dates for the test period (accounting for lags)
    test_dates = test_df.index[lags:lags+n_days]
    
    # Evaluate for each series
    for i in range(min(n_series, len(series_names))):
        series_name = series_names[i]
        
        # Calculate accuracy
        correct = (predictions[:, i] == actuals[:, i]).sum()
        performance['accuracy'][series_name] = correct / n_days if n_days > 0 else 0
        
        # Initialize trading simulation
        capital = 10000.0
        position = 0
        entry_price = 0
        daily_returns = []
        equity_curve = [capital]
        trades = []
        
        # Simulate trading
        for day in range(n_days):
            if day >= len(test_dates):
                break
                
            # Get returns for this day
            current_return = test_df.loc[test_dates[day], series_name]
            
            # Get prediction (0 = down, 1 = up)
            prediction = predictions[day, i]
            
            # Execute trading strategy
            # Simple strategy: Go long on up prediction, short on down prediction
            trade_return = 0
            
            if prediction == 1:  # Predicted up
                trade_return = current_return  # Long position
            else:  # Predicted down
                trade_return = -current_return  # Short position
            
            # Record daily return
            daily_returns.append(trade_return)
            
            # Update equity
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
        
        # Convert returns to numpy array
        daily_returns = np.array(daily_returns)
        performance['returns'][series_name] = daily_returns
        
        # Calculate performance metrics
        if len(daily_returns) > 0:
            # Cumulative return
            performance['cumulative_returns'][series_name] = (equity_curve[-1] / equity_curve[0]) - 1
            
            # Win rate
            wins = sum(daily_returns > 0)
            performance['win_rate'][series_name] = wins / len(daily_returns) if len(daily_returns) > 0 else 0
            
            # Average win/loss
            winning_returns = daily_returns[daily_returns > 0]
            losing_returns = daily_returns[daily_returns < 0]
            
            performance['avg_win'][series_name] = np.mean(winning_returns) if len(winning_returns) > 0 else 0
            performance['avg_loss'][series_name] = np.mean(losing_returns) if len(losing_returns) > 0 else 0
            
            # Sharpe ratio (annualized, assuming 252 trading days)
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            if std_return > 0:
                performance['sharpe_ratio'][series_name] = (mean_return / std_return) * np.sqrt(252)
            else:
                performance['sharpe_ratio'][series_name] = 0
            
            # Maximum drawdown
            cum_returns = np.cumprod(1 + daily_returns) - 1
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / (1 + running_max)
            performance['max_drawdown'][series_name] = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            performance['cumulative_returns'][series_name] = 0
            performance['win_rate'][series_name] = 0
            performance['avg_win'][series_name] = 0
            performance['avg_loss'][series_name] = 0
            performance['sharpe_ratio'][series_name] = 0
            performance['max_drawdown'][series_name] = 0
    
    # Calculate aggregated metrics
    if performance['accuracy']:
        performance['overall_accuracy'] = np.mean(list(performance['accuracy'].values()))
        performance['overall_cumulative_return'] = np.mean(list(performance['cumulative_returns'].values()))
        performance['overall_win_rate'] = np.mean(list(performance['win_rate'].values()))
        performance['overall_sharpe_ratio'] = np.mean(list(performance['sharpe_ratio'].values()))
        performance['overall_max_drawdown'] = np.mean(list(performance['max_drawdown'].values()))
    else:
        performance['overall_accuracy'] = 0
        performance['overall_cumulative_return'] = 0
        performance['overall_win_rate'] = 0
        performance['overall_sharpe_ratio'] = 0
        performance['overall_max_drawdown'] = 0
    
    return performance


def plot_equity_curves(performance, series_names=None):
    """Plot equity curves for selected series"""
    if not performance['returns'] or all(len(r) == 0 for r in performance['returns'].values()):
        print("No return data available for plotting")
        return
        
    if series_names is None:
        # Select top 5 performing series by Sharpe ratio
        sharpe_ratios = performance['sharpe_ratio']
        if not sharpe_ratios:
            print("No Sharpe ratio data available for plotting")
            return
        series_names = sorted(sharpe_ratios.keys(), key=lambda x: sharpe_ratios[x], reverse=True)[:5]
    
    plt.figure(figsize=(12, 8))
    
    for series in series_names:
        if series not in performance['returns']:
            continue
            
        # Get returns and calculate equity curve
        returns = performance['returns'][series]
        if len(returns) == 0:
            continue
            
        initial_capital = 10000
        equity_curve = initial_capital * np.cumprod(1 + returns)
        
        # Plot equity curve
        plt.plot(equity_curve, label=f"{series} (Sharpe: {performance['sharpe_ratio'][series]:.2f})")
    
    plt.title("Equity Curves for Top Performing Series")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the strategy"""
    # Define market symbols and time zones
    market_symbols = {
        # America (Zone 1)
        'SPX Index': 1,      # S&P 500
        'INDU Index': 1,     # Dow Jones Industrial Average
        'RTY Index': 1,      # Russell 2000
        'CCMP Index': 1,     # Nasdaq Composite
        
        # Europe & Africa (Zone 2)
        'UKX Index': 2,      # FTSE 100
        'DAX Index': 2,      # German DAX
        'CAC Index': 2,      # CAC 40
        'CO1 Comdty': 2,     # Brent Crude Oil
        'XAU Curncy': 2,     # Gold
        
        # Asia (Zone 3 & 4)
        'NKY Index': 4,      # Nikkei 225
        'HSI Index': 4,      # Hang Seng
        'KOSPI Index': 4,    # Korea KOSPI
        'AS51 Index': 4      # Australia ASX 200
    }
    
    # Define time period
    start_date = '2015-01-01'
    end_date = '2022-01-01'
    
    # Try to load data from Bloomberg
    try:
        if HAS_BLOOMBERG:
            print("Attempting to load data from Bloomberg...")
            prices_df = load_data_from_bloomberg(list(market_symbols.keys()), start_date, end_date)
            if prices_df.empty:
                raise ValueError("No data returned from Bloomberg")
        else:
            raise ImportError("Bloomberg API not available")
    except Exception as e:
        print(f"Error loading Bloomberg data: {e}")
        print("Using sample data for demonstration")
        
        # Create sample data for demonstration
        print("Creating sample data for demonstration")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(42)
        
        # Generate synthetic price data
        prices_data = {}
        for symbol in market_symbols.keys():
            # Generate random walk with drift
            returns = np.random.normal(0.0002, 0.01, size=len(dates))
            prices = 100 * np.cumprod(1 + returns)
            prices_data[symbol] = prices
        
        prices_df = pd.DataFrame(prices_data, index=dates)
    
    # Check which symbols actually have data
    available_symbols = [col for col in prices_df.columns if not prices_df[col].isna().all()]
    if len(available_symbols) < len(market_symbols):
        print(f"Warning: Only {len(available_symbols)}/{len(market_symbols)} symbols have data")
        print(f"Missing: {set(market_symbols.keys()) - set(available_symbols)}")
    
    # Filter market_symbols to only include available symbols
    market_symbols = {symbol: zone for symbol, zone in market_symbols.items() if symbol in available_symbols}
    
    # Filter prices_df to only include columns in market_symbols
    prices_df = prices_df[list(market_symbols.keys())]
    
    # Calculate returns
    returns_df = calculate_returns(prices_df).dropna()
    
    print(f"Price data shape: {prices_df.shape}")
    print(f"Return data shape: {returns_df.shape}")
    
    # Align prices_df with returns_df for later use
    prices_df = prices_df.loc[returns_df.index]
    
    # Check for any remaining NaN values
    nan_count = returns_df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in return data. Filling with 0.")
        returns_df.fillna(0, inplace=True)
    
    # Split data into train, validation, and test sets
    train_size = int(len(returns_df) * 0.7)
    val_size = int(len(returns_df) * 0.2)
    
    train_df = returns_df.iloc[:train_size]
    val_df = returns_df.iloc[train_size:train_size+val_size]
    test_df = returns_df.iloc[train_size+val_size:]
    
    print(f"Train set: {len(train_df)} days ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Validation set: {len(val_df)} days ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"Test set: {len(test_df)} days ({test_df.index[0]} to {test_df.index[-1]})")
    
    # Train MixDiMex model
    model = MixDiMex()
    model.train(train_df, val_df, time_zones=market_symbols, lags=10)
    
    # Identify indices of American markets for testing
    known_indices = [i for i, (symbol, zone) in enumerate(market_symbols.items()) 
                    if zone == 1 and symbol in test_df.columns]
    
    # Predict on test set
    predictions, accuracy, series_accuracy = model.predict(test_df, known_indices, lags=10)
    
    # Print accuracy results
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nAccuracy by series:")
    for series, acc in sorted(series_accuracy.items(), key=lambda x: x[1], reverse=True):
        print(f"{series}: {acc:.4f}")
    
    # Get actual directions for the test set
    actuals = np.zeros((len(test_df) - 10, len(model.series_names)))
    for day in range(10, len(test_df)):
        day_idx = day - 10
        for i, series in enumerate(model.series_names):
            if i < actuals.shape[1] and series in test_df.columns:
                actuals[day_idx, i] = 1 if test_df.iloc[day][series] > 0 else 0
    
    # Evaluate trading strategy directly on returns data
    performance = evaluate_strategy(predictions, actuals, test_df.iloc[10:])
    
    # Print performance metrics
    print("\nTrading Strategy Performance:")
    print(f"Overall Accuracy: {performance['overall_accuracy']:.4f}")
    print(f"Overall Cumulative Return: {performance['overall_cumulative_return']*100:.2f}%")
    print(f"Overall Win Rate: {performance['overall_win_rate']*100:.2f}%")
    print(f"Overall Sharpe Ratio: {performance['overall_sharpe_ratio']:.4f}")
    print(f"Overall Max Drawdown: {performance['overall_max_drawdown']*100:.2f}%")
    
    # Print top 3 performing markets by Sharpe ratio
    top_markets = sorted(performance['sharpe_ratio'].items(), key=lambda x: x[1], reverse=True)[:3]
    print("\nTop 3 Markets by Sharpe Ratio:")
    for market, sharpe in top_markets:
        print(f"{market}: {sharpe:.4f} (Return: {performance['cumulative_returns'][market]*100:.2f}%, "
              f"Win Rate: {performance['win_rate'][market]*100:.2f}%)")
    
    # Plot equity curves for top markets
    plot_equity_curves(performance)


if __name__ == "__main__":
    main()