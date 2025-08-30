"""
Market Regime Detection Module using Hidden Markov Models

This module implements a sophisticated market regime detector that classifies market conditions
into different states based on price action, volatility, and volume patterns.

Author: AI Hedge Fund System
Date: 2024
"""

from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
import math
import statistics
import os
from dotenv import load_dotenv

# Graceful import handling with fallback flags
HAS_NUMPY = False
HAS_PANDAS = False
HAS_SCIPY = False
HAS_HMMLEARN = False

try:
    import numpy as np
    HAS_NUMPY = True
    logging.info("NumPy available - using optimized numerical computations")
except ImportError:
    logging.warning("NumPy not available - using pure Python fallbacks")

try:
    import pandas as pd
    HAS_PANDAS = True
    logging.info("Pandas available - using DataFrame operations")
except ImportError:
    logging.warning("Pandas not available - using basic data structures")

try:
    import scipy.stats as stats
    HAS_SCIPY = True
    logging.info("SciPy available - using advanced statistical functions")
except ImportError:
    logging.warning("SciPy not available - using basic statistical calculations")

try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
    logging.info("HMMLearn available - using Hidden Markov Models for regime detection")
except ImportError:
    logging.warning("HMMLearn not available - using heuristic regime detection")
    warnings.warn("HMMLearn not available. Using fallback heuristic regime detection. "
                 "Install with: pip install hmmlearn")

from src.tools.api import get_prices, prices_to_df
from src.data.cache import get_cache
from pathlib import Path
import json


@dataclass
class MarketRegime:
    """Data class to hold market regime classification results."""
    market_state: str  # "bull", "bear", "neutral"
    volatility: str    # "high", "low"
    structure: str     # "trending", "mean_reverting"
    confidence: float  # 0.0-1.0
    timestamp: datetime
    raw_probabilities: Optional[Dict[str, float]] = None


class HeuristicRegimeDetector:
    """
    Pure Python fallback regime detector when scientific libraries are unavailable.
    Uses simple but effective heuristics based on moving averages and price patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".HeuristicDetector")
        self.logger.info("Using pure Python heuristic regime detection")
    
    def detect_regime(self, prices: List[Dict], date: str) -> MarketRegime:
        """
        Detect market regime using pure Python heuristics.
        
        Args:
            prices: List of price dictionaries with 'close', 'volume', 'time' keys
            date: Date string for regime detection
            
        Returns:
            MarketRegime object
        """
        try:
            if not prices or len(prices) < 50:
                return self._get_default_regime(date)
            
            # Extract close prices and volumes
            close_prices = [float(p.get('close', 0)) for p in prices if p.get('close')]
            volumes = [float(p.get('volume', 0)) for p in prices if p.get('volume')]
            
            if len(close_prices) < 20:
                return self._get_default_regime(date)
            
            # Market State Detection (Bull/Bear/Neutral)
            market_state, market_confidence = self._detect_market_state(close_prices)
            
            # Volatility Detection (High/Low)
            volatility, vol_confidence = self._detect_volatility_regime(close_prices)
            
            # Structure Detection (Trending/Mean-Reverting)
            structure, struct_confidence = self._detect_market_structure(close_prices)
            
            # Combined confidence
            overall_confidence = (market_confidence + vol_confidence + struct_confidence) / 3
            
            return MarketRegime(
                market_state=market_state,
                volatility=volatility,
                structure=structure,
                confidence=overall_confidence,
                timestamp=datetime.strptime(date, '%Y-%m-%d'),
                raw_probabilities={
                    'method': 'heuristic',
                    'market_confidence': market_confidence,
                    'volatility_confidence': vol_confidence,
                    'structure_confidence': struct_confidence
                }
            )
            
        except Exception as e:
            self.logger.error(f"Heuristic regime detection failed: {e}")
            return self._get_default_regime(date)
    
    def _detect_market_state(self, prices: List[float]) -> Tuple[str, float]:
        """Detect bull/bear/neutral market using moving averages."""
        if len(prices) < 20:
            return "neutral", 0.3
        
        # Calculate simple moving averages
        ma_short = self._simple_moving_average(prices[-20:], 10)  # 10-day MA
        ma_long = self._simple_moving_average(prices[-50:], 50)   # 50-day MA
        
        current_price = prices[-1]
        
        # Calculate percentage above/below MAs
        price_vs_short = (current_price - ma_short) / ma_short if ma_short > 0 else 0
        price_vs_long = (current_price - ma_long) / ma_long if ma_long > 0 else 0
        
        # Calculate recent trend strength
        recent_prices = prices[-20:]
        trend_strength = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        
        # Decision logic
        if price_vs_short > 0.05 and price_vs_long > 0.02 and trend_strength > 0.05:
            confidence = min(0.8, abs(trend_strength) * 10)
            return "bull", confidence
        elif price_vs_short < -0.05 and price_vs_long < -0.02 and trend_strength < -0.05:
            confidence = min(0.8, abs(trend_strength) * 10)
            return "bear", confidence
        else:
            return "neutral", 0.6
    
    def _detect_volatility_regime(self, prices: List[float]) -> Tuple[str, float]:
        """Detect high/low volatility using standard deviation."""
        if len(prices) < 20:
            return "low", 0.3
        
        # Calculate returns (simple percentage changes)
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return "low", 0.3
        
        # Calculate volatility (annualized standard deviation approximation)
        volatility = self._standard_deviation(returns) * math.sqrt(252)  # Approximate annualization
        
        # Historical volatility comparison (simple percentile approach)
        recent_vol = self._standard_deviation(returns[-20:]) * math.sqrt(252) if len(returns) >= 20 else volatility
        historical_vol = volatility
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio > 1.3:
            confidence = min(0.8, (vol_ratio - 1.0) * 2)
            return "high", confidence
        else:
            confidence = min(0.8, (2.0 - vol_ratio))
            return "low", confidence
    
    def _detect_market_structure(self, prices: List[float]) -> Tuple[str, float]:
        """Detect trending vs mean-reverting structure."""
        if len(prices) < 30:
            return "mean_reverting", 0.3
        
        # Simple trend consistency check
        short_ma = self._simple_moving_average(prices[-10:], 10)
        medium_ma = self._simple_moving_average(prices[-20:], 20)
        long_ma = self._simple_moving_average(prices[-30:], 30)
        
        # Check for consistent trend direction
        trend_consistency = 0
        if short_ma > medium_ma > long_ma:  # Consistent uptrend
            trend_consistency = 1
        elif short_ma < medium_ma < long_ma:  # Consistent downtrend  
            trend_consistency = 1
        else:
            trend_consistency = 0
        
        # Calculate trend strength using linear regression approximation
        trend_strength = self._approximate_trend_strength(prices[-20:])
        
        # Decision logic
        if trend_consistency > 0.5 and abs(trend_strength) > 0.02:
            confidence = min(0.8, abs(trend_strength) * 20)
            return "trending", confidence
        else:
            confidence = 0.7
            return "mean_reverting", confidence
    
    def _simple_moving_average(self, values: List[float], period: int) -> float:
        """Calculate simple moving average."""
        if not values or period <= 0:
            return 0.0
        
        period = min(period, len(values))
        return sum(values[-period:]) / period
    
    def _standard_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation using pure Python."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        variance = sum(squared_diffs) / len(values)
        return math.sqrt(variance)
    
    def _approximate_trend_strength(self, prices: List[float]) -> float:
        """Approximate linear regression slope for trend strength."""
        if len(prices) < 3:
            return 0.0
        
        n = len(prices)
        x_values = list(range(n))
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(prices) / n
        
        # Calculate slope using least squares approximation
        numerator = sum((x_values[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize slope relative to price level
        return slope / y_mean if y_mean > 0 else 0.0
    
    def _get_default_regime(self, date: str) -> MarketRegime:
        """Return conservative default regime when detection fails."""
        return MarketRegime(
            market_state="neutral",
            volatility="low",
            structure="mean_reverting",
            confidence=0.3,
            timestamp=datetime.strptime(date, '%Y-%m-%d'),
            raw_probabilities={'method': 'default', 'reason': 'insufficient_data'}
        )


class RegimeDetector:
    """
    Advanced market regime detector using Hidden Markov Models for market state classification.
    
    Features:
    - Market State Detection: Bull/Bear/Neutral regimes
    - Volatility Regime: High/Low volatility environments
    - Market Structure: Trending vs Mean-Reverting patterns
    - Uses SPY as benchmark for regime classification
    - Caches results for performance
    """
    
    def __init__(self, benchmark_ticker: str = "SPY", cache_days: int = 30):
        """
        Initialize the regime detector.
        
        Args:
            benchmark_ticker: Ticker to use for market regime detection (default: SPY)
            cache_days: Number of days to cache regime results
        """
        # Load existing .env file
        load_dotenv()
        self.api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
        
        self.benchmark_ticker = benchmark_ticker
        self.cache_days = cache_days
        self.cache = get_cache()
        self.logger = logging.getLogger(__name__)
        
        # Initialize market data cache directory
        self.market_cache_dir = Path("data/market_cache")
        self.market_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize heuristic detector as fallback
        self.heuristic_detector = HeuristicRegimeDetector()
        
        # Log API key status
        if not self.api_key:
            self.logger.warning("FINANCIAL_DATASETS_API_KEY not found in .env file - may have limited API access")
        else:
            self.logger.info(f"Financial Datasets API key loaded successfully: {self.api_key[:8]}..." if len(self.api_key) >= 8 else "API key loaded")
        
        # Available tickers for fallback (from API error message)
        self.available_tickers = ['AAPL', 'BRK.B', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
        
        # Model parameters
        self.lookback_days = 252  # 1 year of data for training
        self.min_data_points = 60  # Minimum data points needed
        
        # Cached models (only used if HMM is available)
        self._market_state_model = None
        self._volatility_model = None
        self._structure_model = None
        self._last_training_date = None
        self._cached_regimes = {}
        
        # Feature computation parameters
        self.feature_windows = {
            'returns_short': 5,
            'returns_medium': 20,
            'returns_long': 60,
            'volatility': 20,
            'volume': 20
        }
        
        # Log which detection method will be used
        if HAS_HMMLEARN and HAS_SCIPY and HAS_NUMPY and HAS_PANDAS:
            self.logger.info("RegimeDetector initialized with HMM-based detection")
        else:
            missing_libs = []
            if not HAS_HMMLEARN: missing_libs.append("hmmlearn")
            if not HAS_SCIPY: missing_libs.append("scipy")
            if not HAS_NUMPY: missing_libs.append("numpy")
            if not HAS_PANDAS: missing_libs.append("pandas")
            self.logger.warning(f"RegimeDetector using fallback heuristic detection. Missing: {', '.join(missing_libs)}")
        
    def get_current_regime(self, date: str, api_key: Optional[str] = None) -> MarketRegime:
        """
        Get the current market regime for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            api_key: Optional API key for data fetching (defaults to .env key)
            
        Returns:
            MarketRegime object with current regime classification
        """
        # Check cache first
        cache_key = f"regime_{date}"
        if cache_key in self._cached_regimes:
            cached_result = self._cached_regimes[cache_key]
            if (datetime.now() - cached_result.timestamp).days < self.cache_days:
                return cached_result
        
        # Use provided API key or fall back to instance API key
        effective_api_key = api_key or self.api_key
        
        try:
            # Get market data for regime detection with SPY fallback
            prices = self._get_market_proxy_data(date, effective_api_key)
            
            if not prices or len(prices) < self.min_data_points:
                self.logger.warning(f"Insufficient data for regime detection on {date}")
                return self._get_fallback_regime(date)
            
            # Choose detection method based on available libraries
            if HAS_HMMLEARN and HAS_SCIPY and HAS_NUMPY and HAS_PANDAS:
                # Use HMM-based detection with full scientific stack
                df = prices_to_df(prices)
                features = self._compute_features(df)
                
                # Validate feature dimensions for HMM compatibility
                if features.shape[1] != 2:
                    self.logger.error(f"Feature shape validation failed: expected (n, 2), got {features.shape}")
                    raise ValueError(f"HMM requires exactly 2 features, got {features.shape[1]}")
                
                regime = self._detect_regime_hmm(features, date)
            else:
                # Use heuristic-based fallback detection
                regime = self.heuristic_detector.detect_regime(prices, date)
            
            # Cache the result
            self._cached_regimes[cache_key] = regime
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in regime detection for {date}: {str(e)}")
            return self._get_fallback_regime(date)
    
    def _ensure_dict_format(self, data_item) -> Dict:
        """
        Ensure data is in dict format, handling both Pydantic models and dicts.
        
        Args:
            data_item: Either a Pydantic model or dict
            
        Returns:
            Dictionary representation of the data
        """
        # First check: if it's already a dict, return it immediately
        if isinstance(data_item, dict):
            return data_item
        
        # Second check: if it's a list, handle it appropriately
        if isinstance(data_item, list):
            if not data_item:
                return {}
            return data_item[0] if isinstance(data_item[0], dict) else {}
        
        # Third check: if it has model_dump and is NOT a dict (double-check)
        if hasattr(data_item, 'model_dump') and not isinstance(data_item, dict):
            try:
                return data_item.model_dump()
            except Exception as e:
                self.logger.warning(f"Failed to call model_dump on {type(data_item)}: {e}")
        
        # Fourth check: if it has dict method (Pydantic v1)
        if hasattr(data_item, 'dict') and not isinstance(data_item, dict):
            try:
                return data_item.dict()
            except Exception as e:
                self.logger.warning(f"Failed to call dict on {type(data_item)}: {e}")
        
        # Last resort: convert object attributes to dict
        try:
            price_dict = {}
            for attr in ['time', 'date', 'open', 'high', 'low', 'close', 'volume']:
                if hasattr(data_item, attr):
                    price_dict[attr] = getattr(data_item, attr)
            if price_dict:  # Only return if we found some attributes
                return price_dict
        except Exception as e:
            self.logger.warning(f"Failed to convert object attributes: {e}")
        
        # Ultimate fallback: return empty dict
        self.logger.error(f"Could not convert {type(data_item)} to dict format: {data_item}")
        return {}
    
    def _load_from_market_cache(self, ticker: str, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """Load cached market data if available and fresh."""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        cache_file = self.market_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                # Check if cache is still fresh (within cache_days)
                file_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if file_age <= self.cache_days:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    self.logger.info(f"Using cached market data for {ticker} ({start_date} to {end_date})")
                    return cached_data
                else:
                    self.logger.debug(f"Cache for {ticker} is {file_age} days old, fetching fresh data")
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        return None
    
    def _save_to_market_cache(self, ticker: str, start_date: str, end_date: str, data: List[Dict]):
        """Save market data to cache."""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        cache_file = self.market_cache_dir / f"{cache_key}.json"
        
        try:
            # Ensure all data is in dict format
            cached_data = [self._ensure_dict_format(item) for item in data]
            
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
            self.logger.debug(f"Cached {len(cached_data)} data points for {ticker} ({start_date} to {end_date})")
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {ticker}: {e}")
    
    def _get_market_proxy_data(self, date: str, api_key: Optional[str]) -> List[Dict]:
        """
        Handle SPY access with fallback to available tickers.
        
        Args:
            date: Date for regime detection
            api_key: API key for data fetching
            
        Returns:
            List of price data dictionaries
        """
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=self.lookback_days)
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Check cache first
        cached_data = self._load_from_market_cache(self.benchmark_ticker, start_date_str, date)
        if cached_data and len(cached_data) >= self.min_data_points:
            return cached_data
        
        try:
            # Try SPY first with API key
            self.logger.debug(f"Fetching fresh {self.benchmark_ticker} data with API key: {'Yes' if api_key else 'No'}")
            
            prices = get_prices(
                ticker=self.benchmark_ticker,
                start_date=start_date_str,
                end_date=date,
                api_key=api_key
            )
            
            if prices and len(prices) >= self.min_data_points:
                self.logger.info(f"Successfully fetched {len(prices)} data points for {self.benchmark_ticker}")
                # Convert to dict format using utility function
                price_dicts = [self._ensure_dict_format(price) for price in prices]
                
                # Cache the results for future use
                self._save_to_market_cache(self.benchmark_ticker, start_date_str, date, price_dicts)
                
                return price_dicts
            
        except Exception as e:
            if '401' in str(e) or 'SPY' in str(e) or 'Missing API key' in str(e):
                self.logger.warning(f"{self.benchmark_ticker} not accessible: {str(e)}")
                self.logger.info(f"Using market proxy from available tickers: {self.available_tickers}")
                
                # Try fallback using available tickers to create market proxy
                return self._create_market_index_from_available_tickers(date, api_key)
            else:
                # Re-raise non-authentication errors
                raise
        
        # If SPY didn't return enough data, try fallback
        self.logger.warning(f"Insufficient {self.benchmark_ticker} data, using market proxy")
        return self._create_market_index_from_available_tickers(date, api_key)
    
    def _create_market_index_from_available_tickers(self, date: str, api_key: Optional[str]) -> List[Dict]:
        """
        Create a market index proxy using available tickers.
        
        Args:
            date: Date for regime detection
            api_key: API key for data fetching
            
        Returns:
            List of price data dictionaries representing market proxy
        """
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=self.lookback_days)
        
        all_ticker_data = {}
        successful_tickers = []
        
        # Fetch data for available tickers with caching
        for ticker in self.available_tickers:
            try:
                # Check cache first for individual ticker
                cached_ticker_data = self._load_from_market_cache(ticker, start_date.strftime('%Y-%m-%d'), date)
                
                if cached_ticker_data and len(cached_ticker_data) >= self.min_data_points:
                    # Use cached data
                    all_ticker_data[ticker] = {price.get('time') or price.get('date', ''): price for price in cached_ticker_data}
                    successful_tickers.append(ticker)
                    self.logger.debug(f"Using cached data for {ticker} ({len(cached_ticker_data)} data points)")
                    continue
                
                # Fetch fresh data if not in cache
                self.logger.debug(f"Fetching fresh {ticker} data for market proxy")
                prices = get_prices(
                    ticker=ticker,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=date,
                    api_key=api_key
                )
                
                if prices and len(prices) >= self.min_data_points:
                    # Convert to dict format using utility function
                    ticker_prices = [self._ensure_dict_format(price) for price in prices]
                    
                    # Cache the data
                    self._save_to_market_cache(ticker, start_date.strftime('%Y-%m-%d'), date, ticker_prices)
                    
                    all_ticker_data[ticker] = {price.get('time') or price.get('date', ''): price for price in ticker_prices}
                    successful_tickers.append(ticker)
                    self.logger.debug(f"Successfully fetched {len(prices)} data points for {ticker}")
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch {ticker}: {str(e)}")
                continue
        
        if not successful_tickers:
            raise Exception("No ticker data available for market proxy creation")
        
        self.logger.info(f"Creating market proxy from {len(successful_tickers)} tickers: {successful_tickers}")
        
        # Create equal-weighted market index
        return self._compute_equal_weighted_index(all_ticker_data, successful_tickers)
    
    def _compute_equal_weighted_index(self, ticker_data: Dict, tickers: List[str]) -> List[Dict]:
        """
        Compute equal-weighted market index from multiple tickers.
        
        Args:
            ticker_data: Dictionary of ticker data
            tickers: List of ticker symbols
            
        Returns:
            List of market index price dictionaries
        """
        # Find common dates across all tickers
        all_dates = set()
        for ticker in tickers:
            all_dates.update(ticker_data[ticker].keys())
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        
        market_index = []
        
        for date_str in sorted_dates:
            # Get prices for this date from available tickers
            daily_prices = []
            daily_volumes = []
            
            for ticker in tickers:
                if date_str in ticker_data[ticker]:
                    price_data = ticker_data[ticker][date_str]
                    close_price = float(price_data.get('close', 0))
                    volume = float(price_data.get('volume', 0))
                    
                    if close_price > 0:
                        daily_prices.append(close_price)
                        daily_volumes.append(volume)
            
            if daily_prices:
                # Calculate equal-weighted average (normalize by first price to create index)
                if not market_index:
                    # First data point - use as base (index = 100)
                    base_price = sum(daily_prices) / len(daily_prices)
                    index_value = 100.0
                    previous_avg = base_price
                else:
                    # Calculate current average and index value
                    current_avg = sum(daily_prices) / len(daily_prices)
                    index_value = market_index[-1]['close'] * (current_avg / previous_avg)
                    previous_avg = current_avg
                
                # Create market index data point
                market_index.append({
                    'time': date_str,
                    'date': date_str,
                    'open': index_value,  # Simplified - using same value
                    'high': index_value * 1.005,  # Add small variation
                    'low': index_value * 0.995,
                    'close': index_value,
                    'volume': sum(daily_volumes) if daily_volumes else 1000000  # Combined volume
                })
        
        self.logger.info(f"Created market proxy index with {len(market_index)} data points")
        return market_index
    
    def verify_api_authentication(self) -> bool:
        """
        Verify the API key from .env is working.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.api_key:
            self.logger.error("✗ FINANCIAL_DATASETS_API_KEY not found in .env file")
            return False
        
        self.logger.debug(f"API key loaded: {self.api_key[:8]}..." if len(self.api_key) >= 8 else "API key loaded")
        
        # Test with a simple request to AAPL (known to be available)
        try:
            from datetime import date
            test_date = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            today = date.today().strftime('%Y-%m-%d')
            
            test_prices = get_prices('AAPL', test_date, today, self.api_key)
            
            if test_prices:
                self.logger.info("✓ API authentication successful")
                return True
            else:
                self.logger.error("✗ API authentication failed - no data returned")
                return False
                
        except Exception as e:
            self.logger.error(f"✗ API authentication failed: {str(e)}")
            return False
    
    def clear_market_cache(self, ticker: Optional[str] = None, older_than_days: int = 30):
        """
        Clear old cache files to manage disk space.
        
        Args:
            ticker: Specific ticker to clear (if None, clears all)
            older_than_days: Remove cache files older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        if ticker:
            cache_pattern = f"{ticker}_*.json"
        else:
            cache_pattern = "*.json"
        
        cache_files = self.market_cache_dir.glob(cache_pattern)
        removed_count = 0
        
        for cache_file in cache_files:
            try:
                file_age = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < cutoff_date:
                    cache_file.unlink()
                    removed_count += 1
                    self.logger.debug(f"Removed old cache: {cache_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} old cache files")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Report cache usage statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.market_cache_dir.glob("*.json"))
            total_files = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Group by ticker
            ticker_counts = {}
            for cache_file in cache_files:
                ticker = cache_file.stem.split('_')[0]
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            
            return {
                "total_cached_periods": total_files,
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_location": str(self.market_cache_dir),
                "ticker_breakdown": ticker_counts,
                "oldest_cache": min((datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files), default=None),
                "newest_cache": max((datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files), default=None)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {"error": str(e), "cache_location": str(self.market_cache_dir)}
    
    def _df_to_dict_records(self, prices: List[Dict]) -> List[Dict]:
        """Convert price list to dict records format when pandas is not available."""
        return prices
    
    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute feature matrix for HMM training/inference.
        HMM uses exactly 2 features: daily returns and realized volatility.
        
        Args:
            df: Price DataFrame with OHLCV data
            
        Returns:
            numpy array with shape (n_samples, 2) for HMM compatibility
        """
        # 1. Daily returns
        returns = df['close'].pct_change()
        
        # 2. Realized volatility (20-day rolling, annualized)
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Combine only the 2 features needed for HMM
        feature_df = pd.DataFrame({
            'returns': returns,
            'volatility': realized_vol
        })
        
        # Drop NaN rows
        feature_df = feature_df.dropna()
        
        if len(feature_df) < self.min_data_points:
            raise ValueError(f"Insufficient valid data points: {len(feature_df)}")
        
        # Normalize features to help HMM convergence
        normalized_features = self._normalize_features(feature_df.values)
        
        # Ensure we have exactly 2 features
        assert normalized_features.shape[1] == 2, f"Expected 2 features, got {normalized_features.shape[1]}"
        
        return normalized_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to improve HMM performance."""
        if not HAS_NUMPY:
            raise ValueError("NumPy required for feature normalization in HMM mode")
        
        normalized = np.zeros_like(features)
        for i in range(features.shape[1]):
            col = features[:, i]
            # Remove extreme outliers (using numpy percentile)
            col = np.clip(col, np.percentile(col, 1), np.percentile(col, 99))
            # Standardize using median and std (more robust than mean)
            mean_val = np.median(col)
            std_val = np.std(col)
            if std_val > 0:
                normalized[:, i] = (col - mean_val) / std_val
            else:
                normalized[:, i] = col - mean_val
                
        return normalized
    
    def _detect_regime_hmm(self, features: np.ndarray, date: str) -> MarketRegime:
        """
        Detect regime using Hidden Markov Models.
        
        Args:
            features: Normalized feature matrix
            date: Date for regime detection
            
        Returns:
            MarketRegime object
        """
        try:
            # Train or update HMM models if needed
            self._train_hmm_models(features)
            
            # Get the latest observation for classification
            latest_obs = features[-1:, :]
            
            # Market State Detection (3 states: Bull/Bear/Neutral)
            market_probs = self._market_state_model.predict_proba(latest_obs)[0]
            market_state_idx = np.argmax(market_probs)
            market_states = ['bear', 'neutral', 'bull']
            market_state = market_states[market_state_idx]
            market_confidence = market_probs[market_state_idx]
            
            # Volatility Regime Detection (2 states: High/Low)
            latest_vol = latest_obs[:, [1]]  # volatility feature only
            vol_probs = self._volatility_model.predict_proba(latest_vol)[0]
            vol_idx = np.argmax(vol_probs)
            vol_states = ['low', 'high']
            volatility = vol_states[vol_idx]
            vol_confidence = vol_probs[vol_idx]
            
            # Structure Detection (2 states: Trending/Mean-Reverting)
            latest_returns = latest_obs[:, [0]]  # returns feature only
            struct_probs = self._structure_model.predict_proba(latest_returns)[0]
            struct_idx = np.argmax(struct_probs)
            struct_states = ['mean_reverting', 'trending']
            structure = struct_states[struct_idx]
            struct_confidence = struct_probs[struct_idx]
            
            # Combined confidence (weighted average)
            overall_confidence = (
                0.5 * market_confidence + 
                0.25 * vol_confidence + 
                0.25 * struct_confidence
            )
            
            return MarketRegime(
                market_state=market_state,
                volatility=volatility,
                structure=structure,
                confidence=float(overall_confidence),
                timestamp=datetime.strptime(date, '%Y-%m-%d'),
                raw_probabilities={
                    'market': {state: float(prob) for state, prob in zip(market_states, market_probs)},
                    'volatility': {state: float(prob) for state, prob in zip(vol_states, vol_probs)},
                    'structure': {state: float(prob) for state, prob in zip(struct_states, struct_probs)}
                }
            )
            
        except Exception as e:
            self.logger.error(f"HMM regime detection failed: {str(e)}")
            return self._detect_regime_heuristic(features, date)
    
    def _train_hmm_models(self, features: np.ndarray) -> None:
        """Train HMM models for regime detection with 2-feature input."""
        if len(features) < self.min_data_points:
            raise ValueError("Insufficient data for HMM training")
        
        # Validate feature shape
        if features.shape[1] != 2:
            raise ValueError(f"Expected 2 features for HMM, got {features.shape[1]}")
        
        try:
            # Market State Model (3 hidden states) - uses both features
            self._market_state_model = hmm.GaussianHMM(
                n_components=3, 
                covariance_type="diag",
                n_iter=200,  # Increased from 100
                tol=0.01,    # Relaxed tolerance for convergence
                verbose=False,  # Suppress warnings
                init_params="random",  # Better initialization
                random_state=42
            )
            self._market_state_model.fit(features)
            
            # Volatility Model (2 hidden states) - uses volatility feature only
            vol_features = features[:, [1]]  # volatility feature (column 1)
            self._volatility_model = hmm.GaussianHMM(
                n_components=2,
                covariance_type="diag", 
                n_iter=200,  # Increased from 100
                tol=0.01,    # Relaxed tolerance for convergence
                verbose=False,  # Suppress warnings
                init_params="random",  # Better initialization
                random_state=42
            )
            self._volatility_model.fit(vol_features)
            
            # Structure Model (2 hidden states) - uses returns feature only  
            struct_features = features[:, [0]]  # returns feature (column 0)
            self._structure_model = hmm.GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=200,  # Increased from 100
                tol=0.01,    # Relaxed tolerance for convergence
                verbose=False,  # Suppress warnings
                init_params="random",  # Better initialization
                random_state=42
            )
            self._structure_model.fit(struct_features)
            
            self._last_training_date = datetime.now()
            
        except Exception as e:
            self.logger.error(f"HMM training failed: {str(e)}")
            raise
    
    def _detect_regime_heuristic(self, features: np.ndarray, date: str) -> MarketRegime:
        """
        Fallback heuristic-based regime detection when HMM is not available.
        Uses only 2 features: returns (index 0) and volatility (index 1).
        
        Args:
            features: Feature matrix with shape (n_samples, 2)
            date: Date string
            
        Returns:
            MarketRegime object
        """
        try:
            # Validate feature shape
            if features.shape[1] != 2:
                raise ValueError(f"Expected 2 features for heuristic detection, got {features.shape[1]}")
            
            # Get latest feature values
            latest = features[-1, :]
            recent = features[-20:, :] if len(features) >= 20 else features
            
            # Market State Detection (based on recent returns patterns)
            recent_returns = recent[:, 0]  # returns column
            avg_return = np.mean(recent_returns)
            return_trend = np.polyfit(range(len(recent_returns)), recent_returns, 1)[0]  # trend slope
            
            if avg_return > 0.001 and return_trend > 0:
                market_state = "bull"
                market_confidence = min(0.8, abs(avg_return) * 100)
            elif avg_return < -0.001 and return_trend < 0:
                market_state = "bear"  
                market_confidence = min(0.8, abs(avg_return) * 100)
            else:
                market_state = "neutral"
                market_confidence = 0.6
            
            # Volatility Detection (based on realized vol percentile)
            recent_vol = recent[:, 1]  # volatility column
            vol_percentile = np.percentile(recent_vol, 70)  # 70th percentile of recent volatility
            current_vol = latest[1]  # current volatility
            
            if current_vol > vol_percentile:
                volatility = "high"
                vol_confidence = min(0.8, (current_vol - vol_percentile) / (vol_percentile + 1e-8))
            else:
                volatility = "low"
                vol_confidence = min(0.8, (vol_percentile - current_vol) / (vol_percentile + 1e-8))
            
            # Structure Detection (based on return consistency)
            return_std = np.std(recent_returns)
            return_abs_mean = np.mean(np.abs(recent_returns))
            
            if return_std > 0.01 and return_abs_mean > 0.005:
                structure = "trending"
                struct_confidence = min(0.8, return_std * 50)
            else:
                structure = "mean_reverting"
                struct_confidence = min(0.8, 0.8 - return_std * 50)
            
            # Overall confidence
            overall_confidence = (market_confidence + vol_confidence + struct_confidence) / 3
            
            return MarketRegime(
                market_state=market_state,
                volatility=volatility,
                structure=structure,
                confidence=float(overall_confidence),
                timestamp=datetime.strptime(date, '%Y-%m-%d')
            )
            
        except Exception as e:
            self.logger.error(f"Heuristic regime detection failed: {str(e)}")
            return self._get_fallback_regime(date)
    
    def _get_fallback_regime(self, date: str) -> MarketRegime:
        """Return a conservative default regime when all detection methods fail."""
        return MarketRegime(
            market_state="neutral",
            volatility="low", 
            structure="mean_reverting",
            confidence=0.3,
            timestamp=datetime.strptime(date, '%Y-%m-%d')
        )
    
    def get_regime_history(self, start_date: str, end_date: str, api_key: Optional[str] = None) -> List[MarketRegime]:
        """
        Get regime history for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            api_key: Optional API key for data fetching
            
        Returns:
            List of MarketRegime objects
        """
        regimes = []
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get regimes for each business day
        current_date = start_dt
        while current_date <= end_dt:
            # Skip weekends (simple check)
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                regime = self.get_current_regime(current_date.strftime('%Y-%m-%d'), api_key)
                regimes.append(regime)
            
            current_date += timedelta(days=1)
        
        return regimes
    
    def get_regime_statistics(self, regimes: List[MarketRegime]) -> Dict:
        """
        Calculate statistics about regime periods.
        
        Args:
            regimes: List of MarketRegime objects
            
        Returns:
            Dictionary with regime statistics
        """
        if not regimes:
            return {}
        
        # Count regimes
        market_counts = {}
        vol_counts = {}
        struct_counts = {}
        
        for regime in regimes:
            market_counts[regime.market_state] = market_counts.get(regime.market_state, 0) + 1
            vol_counts[regime.volatility] = vol_counts.get(regime.volatility, 0) + 1
            struct_counts[regime.structure] = struct_counts.get(regime.structure, 0) + 1
        
        total_days = len(regimes)
        
        return {
            'total_days': total_days,
            'market_distribution': {k: v/total_days for k, v in market_counts.items()},
            'volatility_distribution': {k: v/total_days for k, v in vol_counts.items()},
            'structure_distribution': {k: v/total_days for k, v in struct_counts.items()},
            'average_confidence': np.mean([r.confidence for r in regimes])
        }


# Singleton instance for global use
_regime_detector = None


def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector


def get_current_market_regime(date: str, api_key: Optional[str] = None) -> MarketRegime:
    """
    Convenience function to get current market regime.
    
    Args:
        date: Date in YYYY-MM-DD format
        api_key: Optional API key for data fetching
        
    Returns:
        MarketRegime object
    """
    detector = get_regime_detector()
    return detector.get_current_regime(date, api_key)


# Adaptive parameter adjustment utilities
class AdaptiveThresholds:
    """
    Utility class for adjusting trading parameters based on market regimes.
    """
    
    # Base thresholds for different parameters
    BASE_THRESHOLDS = {
        # Warren Buffett parameters
        'roe_threshold': 0.15,
        'debt_to_equity_threshold': 0.5,
        'operating_margin_threshold': 0.15,
        'current_ratio_threshold': 1.5,
        'pe_ratio_threshold': 25,
        'discount_rate': 0.10,
        'margin_of_safety': 0.25,
        
        # Technical parameters
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bollinger_period': 20,
        'bollinger_std': 2.0,
        'ema_short': 8,
        'ema_medium': 21,
        'ema_long': 55,
        'adx_period': 14,
        'atr_period': 14
    }
    
    # Regime-based adjustment factors
    REGIME_ADJUSTMENTS = {
        'bull': {
            'pe_ratio_threshold': 1.3,      # Allow higher P/E ratios
            'discount_rate': 0.95,          # Lower discount rate
            'margin_of_safety': 0.8,        # Reduce margin of safety
            'rsi_oversold': 0.9,            # Lower oversold (27 instead of 30)
            'rsi_overbought': 1.1,          # Higher overbought (77 instead of 70)
            'bollinger_std': 1.1            # Wider bands
        },
        'bear': {
            'pe_ratio_threshold': 0.7,      # Stricter P/E requirements
            'discount_rate': 1.2,           # Higher discount rate
            'margin_of_safety': 1.4,        # Higher margin of safety (35%)
            'roe_threshold': 1.2,           # Higher ROE requirement (18%)
            'rsi_oversold': 1.1,            # Higher oversold (33 instead of 30)
            'rsi_overbought': 0.9,          # Lower overbought (63 instead of 70)
            'bollinger_std': 0.8            # Tighter bands
        },
        'neutral': {
            # Use base thresholds
        }
    }
    
    VOLATILITY_ADJUSTMENTS = {
        'high': {
            'margin_of_safety': 1.2,        # Increase margin of safety
            'bollinger_std': 1.3,           # Wider Bollinger bands
            'atr_period': 1.5,              # Longer ATR period (21 instead of 14)
            'rsi_period': 1.2               # Longer RSI period (17 instead of 14)
        },
        'low': {
            'bollinger_std': 0.8,           # Tighter Bollinger bands
            'atr_period': 0.8,              # Shorter ATR period (11 instead of 14)
            'rsi_period': 0.8               # Shorter RSI period (11 instead of 14)
        }
    }
    
    STRUCTURE_ADJUSTMENTS = {
        'trending': {
            'ema_weight_increase': 0.2,     # Emphasize trend indicators
            'rsi_weight_decrease': 0.2,     # De-emphasize mean reversion
            'adx_period': 0.8               # Shorter ADX period for responsiveness
        },
        'mean_reverting': {
            'rsi_weight_increase': 0.2,     # Emphasize mean reversion
            'ema_weight_decrease': 0.1,     # De-emphasize trend following
            'bollinger_period': 0.8         # Shorter Bollinger period (16 instead of 20)
        }
    }
    
    @classmethod
    def get_adaptive_threshold(cls, parameter: str, regime: MarketRegime, sector: Optional[str] = None) -> float:
        """
        Get an adaptive threshold based on market regime and optionally sector.
        
        Args:
            parameter: Parameter name (e.g., 'roe_threshold', 'discount_rate')
            regime: Current market regime
            sector: Optional sector for sector-specific adjustments
            
        Returns:
            Adjusted threshold value
        """
        if parameter not in cls.BASE_THRESHOLDS:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        base_value = cls.BASE_THRESHOLDS[parameter]
        adjusted_value = base_value
        
        # Apply market state adjustments
        market_adj = cls.REGIME_ADJUSTMENTS.get(regime.market_state, {})
        if parameter in market_adj:
            adjusted_value *= market_adj[parameter]
        
        # Apply volatility adjustments
        vol_adj = cls.VOLATILITY_ADJUSTMENTS.get(regime.volatility, {})
        if parameter in vol_adj:
            adjusted_value *= vol_adj[parameter]
        
        # Apply structure adjustments
        struct_adj = cls.STRUCTURE_ADJUSTMENTS.get(regime.structure, {})
        if parameter in struct_adj:
            adjusted_value *= struct_adj[parameter]
        
        # Apply sector-specific adjustments if provided
        if sector:
            sector_multiplier = cls._get_sector_multiplier(parameter, sector)
            adjusted_value *= sector_multiplier
        
        return adjusted_value
    
    @classmethod
    def _get_sector_multiplier(cls, parameter: str, sector: str) -> float:
        """Get sector-specific multiplier for a parameter."""
        # Simplified sector adjustments - in practice, these would be more sophisticated
        sector_adjustments = {
            'technology': {
                'pe_ratio_threshold': 1.5,   # Tech can have higher P/E
                'roe_threshold': 1.1,        # Expect higher ROE
                'operating_margin_threshold': 1.2
            },
            'utilities': {
                'pe_ratio_threshold': 0.8,   # Lower P/E expected
                'roe_threshold': 0.7,        # Lower ROE acceptable
                'debt_to_equity_threshold': 2.0  # Higher debt acceptable
            },
            'healthcare': {
                'pe_ratio_threshold': 1.2,
                'roe_threshold': 1.05,
                'margin_of_safety': 1.1
            },
            'financials': {
                'debt_to_equity_threshold': 3.0,  # Banks naturally have higher leverage
                'current_ratio_threshold': 0.6    # Different liquidity metrics
            }
        }
        
        sector_adj = sector_adjustments.get(sector.lower(), {})
        return sector_adj.get(parameter, 1.0)


def get_adaptive_thresholds(regime: MarketRegime, sector: Optional[str] = None) -> Dict[str, float]:
    """
    Get all adaptive thresholds for a given regime and sector.
    
    Args:
        regime: Current market regime
        sector: Optional sector specification
        
    Returns:
        Dictionary of adjusted thresholds
    """
    thresholds = {}
    
    for parameter in AdaptiveThresholds.BASE_THRESHOLDS:
        thresholds[parameter] = AdaptiveThresholds.get_adaptive_threshold(
            parameter, regime, sector
        )
    
    return thresholds