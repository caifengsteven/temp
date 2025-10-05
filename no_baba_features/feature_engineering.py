"""
Feature Engineering Module - NO BABA FEATURES VERSION
Creates features for BABA premium/discount prediction using ONLY:
- Historical premium/discount
- HK (9988.HK) features
- Macro features
- Peer features (PDD)

EXCLUDED BABA-specific features:
- BABA volume ratio
- BABA last 30min price change
- BABA first 30min volume
- BABA option volume
- BABA realized volatility
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: pandas_ta not installed. Technical indicators will use manual calculation.")

import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for BABA premium/discount prediction - NO BABA FEATURES"""
    
    def __init__(self, adr_ratio: int = None):
        self.adr_ratio = adr_ratio or config.ADR_CONVERSION_RATIO
    
    def calculate_target_variable(self, 
                                  baba_us: pd.DataFrame,
                                  baba_hk: pd.DataFrame,
                                  usdhkd: pd.DataFrame,
                                  common_days: pd.DataFrame) -> pd.DataFrame:
        """Calculate BABA premium/discount target variable"""
        logger.info("Calculating target variable: BABA_prem_discount")
        
        baba_us = baba_us.copy()
        baba_us['date'] = pd.to_datetime(baba_us['date'])
        baba_us = baba_us.sort_values('date')
        
        baba_hk = baba_hk.copy()
        baba_hk['date'] = pd.to_datetime(baba_hk['date'])
        baba_hk = baba_hk.sort_values('date')
        
        usdhkd = usdhkd.copy()
        usdhkd['date'] = pd.to_datetime(usdhkd['date'])
        
        common_days['date'] = pd.to_datetime(common_days['date'])
        
        df = common_days.copy()
        
        if 'close' not in baba_us.columns:
            logger.error(f"Available BABA US columns: {list(baba_us.columns)}")
            raise ValueError("Expected 'close' column in BABA US data")
        
        if 'open' not in baba_hk.columns:
            logger.error(f"Available BABA HK columns: {list(baba_hk.columns)}")
            raise ValueError("Expected 'open' column in BABA HK data")
        
        df = df.merge(baba_us[['date', 'close']], on='date', how='left')
        df.rename(columns={'close': 'baba_close'}, inplace=True)
        df = df.merge(usdhkd[['date', 'usdhkd_rate']], on='date', how='left')
        
        baba_hk_next = baba_hk[['date', 'open']].copy()
        baba_hk_next['date'] = baba_hk_next['date'] - pd.Timedelta(days=1)
        baba_hk_next.rename(columns={'open': 'hk_open_next'}, inplace=True)
        
        df = df.merge(baba_hk_next, on='date', how='left')
        
        df['baba_close_hkd'] = df['baba_close'] * df['usdhkd_rate'] / self.adr_ratio
        df['baba_prem_discount'] = (df['baba_close_hkd'] / df['hk_open_next']) - 1
        
        df = df.dropna(subset=['baba_prem_discount'])
        
        logger.info(f"Calculated target for {len(df)} days")
        
        return df[['date', 'baba_prem_discount']]
    
    def create_historical_premium_features(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """Create historical premium/discount features (past 5 days)"""
        logger.info("Creating historical premium/discount features")
        
        df = target_df.copy()
        df = df.sort_values('date')
        
        for i in range(1, config.PREM_DISCOUNT_LOOKBACK + 1):
            df[f'prem_discount_lag_{i}'] = df['baba_prem_discount'].shift(i)
        
        lag_cols = ['date'] + [f'prem_discount_lag_{i}' for i in range(1, config.PREM_DISCOUNT_LOOKBACK + 1)]
        return df[lag_cols]
    
    def create_price_momentum_features(self, baba_hk: pd.DataFrame) -> pd.DataFrame:
        """Create 9988.HK price momentum features"""
        logger.info("Creating HK price momentum features")
        
        df = baba_hk.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if 'close' not in df.columns:
            logger.error(f"Available HK columns: {list(df.columns)}")
            raise ValueError("Expected 'close' column in HK data")
        
        df['hk_pct_change_t'] = df['close'].pct_change() * 100
        
        for i in range(1, config.PRICE_CHANGE_LOOKBACK + 1):
            df[f'hk_pct_change_lag_{i}'] = df['hk_pct_change_t'].shift(i)
        
        cols = ['date', 'hk_pct_change_t'] + [f'hk_pct_change_lag_{i}' for i in range(1, config.PRICE_CHANGE_LOOKBACK + 1)]
        
        return df[cols]
    
    def create_volume_features(self, baba_hk: pd.DataFrame) -> pd.DataFrame:
        """Create HK volume features ONLY (no BABA volume)"""
        logger.info("Creating HK volume features (BABA volume excluded)")
        
        baba_hk = baba_hk.copy()
        baba_hk['date'] = pd.to_datetime(baba_hk['date'])
        baba_hk = baba_hk.sort_values('date')
        baba_hk['hk_volume_avg_30d'] = baba_hk['volume'].rolling(window=config.VOLUME_AVG_WINDOW).mean()
        baba_hk['hk_volume_ratio'] = baba_hk['volume'] / baba_hk['hk_volume_avg_30d']
        
        return baba_hk[['date', 'hk_volume_ratio']]
    
    def create_technical_indicators(self, baba_hk: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators (RSI) for HK"""
        logger.info("Creating technical indicators")
        
        df = baba_hk.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if ta is not None:
            df[f'hk_rsi_{config.RSI_PERIOD}'] = ta.rsi(df['close'], length=config.RSI_PERIOD)
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_PERIOD).mean()
            rs = gain / loss
            df[f'hk_rsi_{config.RSI_PERIOD}'] = 100 - (100 / (1 + rs))
        
        return df[['date', f'hk_rsi_{config.RSI_PERIOD}']]

    def create_macro_features(self,
                             vix: pd.DataFrame,
                             treasury: pd.DataFrame,
                             usdhkd: pd.DataFrame) -> pd.DataFrame:
        """Create macro economic features"""
        logger.info("Creating macro features")

        vix = vix.copy()
        vix['date'] = pd.to_datetime(vix['date'])

        treasury = treasury.copy()
        treasury['date'] = pd.to_datetime(treasury['date'])

        usdhkd = usdhkd.copy()
        usdhkd['date'] = pd.to_datetime(usdhkd['date'])
        usdhkd = usdhkd.sort_values('date')
        usdhkd['usdhkd_pct_change'] = usdhkd['usdhkd_rate'].pct_change() * 100

        df = vix[['date', 'vix_level']].merge(
            treasury[['date', 'us_10y_yield', 'us_3m_yield']],
            on='date',
            how='outer'
        )
        df = df.merge(
            usdhkd[['date', 'usdhkd_pct_change']],
            on='date',
            how='outer'
        )

        return df

    def create_peer_features(self, pdd: pd.DataFrame) -> pd.DataFrame:
        """Create peer stock features (PDD)"""
        logger.info("Creating peer features")

        df = pdd.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        df['pdd_pct_change'] = df['close'].pct_change() * 100

        return df[['date', 'pdd_pct_change']]

    def merge_all_features(self,
                          target_df: pd.DataFrame,
                          historical_prem: pd.DataFrame,
                          price_momentum: pd.DataFrame,
                          volume_features: pd.DataFrame,
                          technical_indicators: pd.DataFrame,
                          macro_features: pd.DataFrame,
                          peer_features: pd.DataFrame) -> pd.DataFrame:
        """Merge all features (NO BABA-specific features)"""
        logger.info("Merging all features (NO BABA features)")
        
        df = target_df.copy()
        
        df = df.merge(historical_prem, on='date', how='left')
        df = df.merge(price_momentum, on='date', how='left')
        df = df.merge(volume_features, on='date', how='left')
        df = df.merge(technical_indicators, on='date', how='left')
        df = df.merge(macro_features, on='date', how='left')
        df = df.merge(peer_features, on='date', how='left')
        
        df = df.sort_values('date')
        
        logger.info(f"Created feature matrix with shape {df.shape}")
        logger.info(f"Features: {list(df.columns)}")
        
        return df

    def create_all_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create all features from raw data (NO BABA features)"""
        logger.info("Starting feature engineering pipeline (NO BABA FEATURES)")
        
        target_df = self.calculate_target_variable(
            data['baba_us'],
            data['baba_hk'],
            data['usdhkd'],
            data['common_days']
        )
        
        historical_prem = self.create_historical_premium_features(target_df)
        price_momentum = self.create_price_momentum_features(data['baba_hk'])
        volume_features = self.create_volume_features(data['baba_hk'])
        technical_indicators = self.create_technical_indicators(data['baba_hk'])
        macro_features = self.create_macro_features(data['vix'], data['treasury'], data['usdhkd'])
        peer_features = self.create_peer_features(data['pdd'])
        
        df = self.merge_all_features(
            target_df,
            historical_prem,
            price_momentum,
            volume_features,
            technical_indicators,
            macro_features,
            peer_features
        )
        
        logger.info("Feature engineering completed (NO BABA FEATURES)")
        logger.info(f"Excluded features: BABA volume, BABA intraday, BABA volatility, BABA options")
        
        return df

