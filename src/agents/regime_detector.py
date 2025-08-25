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
        self.benchmark_ticker = benchmark_ticker
        self.cache_days = cache_days
        self.cache = get_cache()
        self.logger = logging.getLogger(__name__)
        
        # Initialize heuristic detector as fallback
        self.heuristic_detector = HeuristicRegimeDetector()
        
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
            api_key: Optional API key for data fetching
            
        Returns:
            MarketRegime object with current regime classification
        """
        # Check cache first
        cache_key = f"regime_{date}"
        if cache_key in self._cached_regimes:
            cached_result = self._cached_regimes[cache_key]
            if (datetime.now() - cached_result.timestamp).days < self.cache_days:
                return cached_result
        
        try:
            # Get market data for regime detection
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Fetch price data
            prices = get_prices(
                ticker=self.benchmark_ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=date,
                api_key=api_key
            )
            
            if not prices or len(prices) < self.min_data_points:
                self.logger.warning(f"Insufficient data for regime detection on {date}")
                return self._get_fallback_regime(date)
            
            # Choose detection method based on available libraries
            if HAS_HMMLEARN and HAS_SCIPY and HAS_NUMPY and HAS_PANDAS:
                # Use HMM-based detection with full scientific stack
                df = prices_to_df(prices)
                features = self._compute_features(df)
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
    
    def _df_to_dict_records(self, prices: List[Dict]) -> List[Dict]:
        """Convert price list to dict records format when pandas is not available."""
        return prices
    
    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute feature matrix for HMM training/inference.
        
        Args:
            df: Price DataFrame with OHLCV data
            
        Returns:
            numpy array with computed features
        """
        features = []
        
        # 1. Returns at different horizons
        returns = df['close'].pct_change()
        
        # Short-term returns (1-day, 5-day)
        returns_1d = returns
        returns_5d = df['close'].pct_change(periods=5)
        returns_20d = df['close'].pct_change(periods=20)
        
        # 2. Realized volatility (20-day rolling)
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # 3. Volume ratio (current volume / 20-day average)
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_ma
        
        # 4. Price momentum indicators
        momentum_20 = returns.rolling(window=20).sum()
        momentum_60 = returns.rolling(window=60).sum()
        
        # 5. Volatility of volatility (regime change indicator)
        vol_of_vol = realized_vol.rolling(window=10).std()
        
        # 6. Trend strength (using moving average crossover)
        ma_5 = df['close'].rolling(5).mean()
        ma_20 = df['close'].rolling(20).mean()
        trend_strength = (ma_5 - ma_20) / ma_20
        
        # 7. Price location within recent range
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        price_position = (df['close'] - low_20) / (high_20 - low_20)
        
        # Combine features
        feature_df = pd.DataFrame({
            'returns_1d': returns_1d,
            'returns_5d': returns_5d,
            'returns_20d': returns_20d,
            'realized_vol': realized_vol,
            'volume_ratio': volume_ratio.fillna(1.0),
            'momentum_20': momentum_20,
            'momentum_60': momentum_60,
            'vol_of_vol': vol_of_vol,
            'trend_strength': trend_strength,
            'price_position': price_position.fillna(0.5)
        })
        
        # Drop NaN rows and normalize features
        feature_df = feature_df.dropna()
        
        if len(feature_df) < self.min_data_points:
            raise ValueError(f"Insufficient valid data points: {len(feature_df)}")
        
        # Normalize features to help HMM convergence
        normalized_features = self._normalize_features(feature_df.values)
        
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
            vol_probs = self._volatility_model.predict_proba(latest_obs)[0]
            vol_idx = np.argmax(vol_probs)
            vol_states = ['low', 'high']
            volatility = vol_states[vol_idx]
            vol_confidence = vol_probs[vol_idx]
            
            # Structure Detection (2 states: Trending/Mean-Reverting)
            struct_probs = self._structure_model.predict_proba(latest_obs)[0]
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
        """Train HMM models for regime detection."""
        if len(features) < self.min_data_points:
            raise ValueError("Insufficient data for HMM training")
        
        try:
            # Market State Model (3 hidden states)
            self._market_state_model = hmm.GaussianHMM(
                n_components=3, 
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            self._market_state_model.fit(features)
            
            # Volatility Model (2 hidden states)
            vol_features = features[:, [3, 7]]  # volatility and vol_of_vol features
            self._volatility_model = hmm.GaussianHMM(
                n_components=2,
                covariance_type="diag", 
                n_iter=100,
                random_state=42
            )
            self._volatility_model.fit(vol_features)
            
            # Structure Model (2 hidden states)
            struct_features = features[:, [5, 6, 8]]  # momentum and trend features
            self._structure_model = hmm.GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=100,
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
        
        Args:
            features: Feature matrix
            date: Date string
            
        Returns:
            MarketRegime object
        """
        try:
            # Get latest feature values
            latest = features[-1, :]
            recent = features[-20:, :] if len(features) >= 20 else features
            
            # Market State Detection (based on momentum and price position)
            momentum_score = np.mean([latest[5], latest[6]])  # momentum_20, momentum_60
            price_position = latest[9]  # price position in range
            
            if momentum_score > 0.02 and price_position > 0.7:
                market_state = "bull"
                market_confidence = min(0.8, abs(momentum_score) * 10)
            elif momentum_score < -0.02 and price_position < 0.3:
                market_state = "bear"
                market_confidence = min(0.8, abs(momentum_score) * 10)
            else:
                market_state = "neutral"
                market_confidence = 0.6
            
            # Volatility Detection (based on realized vol percentile)
            vol_percentile = np.percentile(recent[:, 3], 70)  # 70th percentile of recent volatility
            current_vol = latest[3]  # realized_vol
            
            if current_vol > vol_percentile:
                volatility = "high"
                vol_confidence = min(0.8, (current_vol - vol_percentile) / vol_percentile)
            else:
                volatility = "low"
                vol_confidence = min(0.8, (vol_percentile - current_vol) / vol_percentile)
            
            # Structure Detection (based on trend consistency)
            trend_values = recent[:, 8]  # trend_strength
            trend_consistency = np.std(trend_values)
            avg_trend = np.mean(trend_values)
            
            if trend_consistency < 0.5 and abs(avg_trend) > 0.01:
                structure = "trending"
                struct_confidence = min(0.8, abs(avg_trend) * 20)
            else:
                structure = "mean_reverting"
                struct_confidence = min(0.8, trend_consistency * 2)
            
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