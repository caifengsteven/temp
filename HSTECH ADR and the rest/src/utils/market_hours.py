"""
Market Hours Utilities for HSTECH Estimation System

This module provides utilities for handling market hours and time zones.
"""

from datetime import datetime, timezone, time
from typing import Dict, Tuple
import pytz

from ..models import MarketHours


class MarketHoursChecker:
    """
    Utility class for checking market hours and determining when to run estimations.
    """
    
    def __init__(self, market_hours_config: Dict[str, MarketHours]):
        self.market_hours = market_hours_config
        
        # Time zones
        self.hk_tz = pytz.timezone('Asia/Hong_Kong')
        self.us_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
    
    def is_hk_market_open(self, dt: datetime = None) -> bool:
        """Check if Hong Kong market is currently open."""
        if dt is None:
            dt = datetime.now(self.utc_tz)
        
        # Convert to HK time
        hk_time = dt.astimezone(self.hk_tz)
        
        # Check if it's a weekday
        if hk_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Get market hours
        if "hong_kong" not in self.market_hours:
            return False
        
        hk_hours = self.market_hours["hong_kong"]
        open_time = time.fromisoformat(hk_hours.open)
        close_time = time.fromisoformat(hk_hours.close)
        
        current_time = hk_time.time()
        
        return open_time <= current_time <= close_time
    
    def is_us_market_open(self, dt: datetime = None) -> bool:
        """Check if US market is currently open."""
        if dt is None:
            dt = datetime.now(self.utc_tz)
        
        # Convert to US Eastern time
        us_time = dt.astimezone(self.us_tz)
        
        # Check if it's a weekday
        if us_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Get market hours
        if "us" not in self.market_hours:
            return False
        
        us_hours = self.market_hours["us"]
        open_time = time.fromisoformat(us_hours.open)
        close_time = time.fromisoformat(us_hours.close)
        
        current_time = us_time.time()
        
        return open_time <= current_time <= close_time
    
    def should_run_estimation(self, dt: datetime = None) -> Tuple[bool, str]:
        """
        Determine if HSTECH estimation should be run at the given time.
        
        Returns:
            Tuple of (should_run, reason)
        """
        if dt is None:
            dt = datetime.now(self.utc_tz)
        
        hk_open = self.is_hk_market_open(dt)
        us_open = self.is_us_market_open(dt)
        
        if hk_open:
            return False, "Hong Kong market is open - use real-time HSTECH data"
        
        if not us_open:
            return False, "US market is closed - no new data available"
        
        return True, "Hong Kong market closed, US market open - estimation needed"
    
    def get_next_estimation_time(self, dt: datetime = None) -> datetime:
        """Get the next time when estimation should be run."""
        if dt is None:
            dt = datetime.now(self.utc_tz)
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated scheduling
        
        # If US market is about to open, schedule for then
        us_time = dt.astimezone(self.us_tz)
        
        if "us" in self.market_hours:
            us_hours = self.market_hours["us"]
            open_time = time.fromisoformat(us_hours.open)
            
            # Schedule for next US market open
            next_open = us_time.replace(
                hour=open_time.hour,
                minute=open_time.minute,
                second=0,
                microsecond=0
            )
            
            # If already past today's open, schedule for tomorrow
            if us_time.time() > open_time:
                next_open = next_open.replace(day=next_open.day + 1)
            
            # Skip weekends
            while next_open.weekday() >= 5:
                next_open = next_open.replace(day=next_open.day + 1)
            
            return next_open.astimezone(self.utc_tz)
        
        # Default: next hour
        return dt.replace(minute=0, second=0, microsecond=0).replace(hour=dt.hour + 1)
    
    def get_market_status_summary(self, dt: datetime = None) -> Dict[str, any]:
        """Get comprehensive market status summary."""
        if dt is None:
            dt = datetime.now(self.utc_tz)
        
        hk_open = self.is_hk_market_open(dt)
        us_open = self.is_us_market_open(dt)
        should_estimate, reason = self.should_run_estimation(dt)
        
        return {
            "timestamp_utc": dt.isoformat(),
            "hong_kong_market_open": hk_open,
            "us_market_open": us_open,
            "should_run_estimation": should_estimate,
            "estimation_reason": reason,
            "next_estimation_time": self.get_next_estimation_time(dt).isoformat(),
            "local_times": {
                "hong_kong": dt.astimezone(self.hk_tz).isoformat(),
                "us_eastern": dt.astimezone(self.us_tz).isoformat(),
                "utc": dt.isoformat()
            }
        }


def create_market_hours_checker(market_hours_config: Dict[str, MarketHours]) -> MarketHoursChecker:
    """Create and return a MarketHoursChecker instance."""
    return MarketHoursChecker(market_hours_config)
