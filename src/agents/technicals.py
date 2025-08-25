import math

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.api_key import get_api_key_from_state
from src.agents.regime_detector import get_current_market_regime, AdaptiveThresholds
import json
import pandas as pd
import numpy as np
import logging

from src.tools.api import get_prices, prices_to_df
from src.utils.progress import progress

logger = logging.getLogger(__name__)


def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling NaN cases
    
    Args:
        value: The value to convert (can be pandas scalar, numpy value, etc.)
        default: Default value to return if the input is NaN or invalid
    
    Returns:
        float: The converted value or default if NaN/invalid
    """
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


##### Adaptive Technical Analyst #####
def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """
    Regime-aware technical analysis system that adapts strategies and parameters based on market conditions:
    1. Trend Following (emphasized in trending regimes)
    2. Mean Reversion (emphasized in mean-reverting regimes) 
    3. Momentum (adapted for volatility)
    4. Volatility Analysis (regime-aware thresholds)
    5. Statistical Arbitrage Signals (regime-specific parameters)
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    # Get current market regime for adaptive parameters
    try:
        current_regime = get_current_market_regime(end_date, api_key)
        logger.info(f"Technical analysis using regime: {current_regime.market_state}/{current_regime.volatility}/{current_regime.structure} (confidence: {current_regime.confidence:.2f})")
    except Exception as e:
        logger.warning(f"Failed to detect market regime: {e}. Using fallback neutral regime.")
        from src.agents.regime_detector import MarketRegime
        from datetime import datetime
        current_regime = MarketRegime(
            market_state="neutral",
            volatility="low", 
            structure="mean_reverting",
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, f"Analyzing price data (regime: {current_regime.market_state}/{current_regime.structure})")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: No price data found")
            continue

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        progress.update_status(agent_id, ticker, "Calculating regime-adaptive trend signals")
        trend_signals = calculate_trend_signals_adaptive(prices_df, current_regime)

        progress.update_status(agent_id, ticker, "Calculating adaptive mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals_adaptive(prices_df, current_regime)

        progress.update_status(agent_id, ticker, "Calculating regime-aware momentum")
        momentum_signals = calculate_momentum_signals_adaptive(prices_df, current_regime)

        progress.update_status(agent_id, ticker, "Analyzing volatility with regime context")
        volatility_signals = calculate_volatility_signals_adaptive(prices_df, current_regime)

        progress.update_status(agent_id, ticker, "Statistical analysis with adaptive parameters")
        stat_arb_signals = calculate_stat_arb_signals_adaptive(prices_df, current_regime)

        # Adaptive strategy weights based on regime
        strategy_weights = get_adaptive_strategy_weights(current_regime)

        progress.update_status(agent_id, ticker, "Combining signals with regime-aware weighting")
        combined_signal = weighted_signal_combination_adaptive(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
            current_regime,
        )

        # Generate detailed analysis report for this ticker with regime context
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "reasoning": {
                "market_regime": {
                    "state": current_regime.market_state,
                    "volatility": current_regime.volatility,
                    "structure": current_regime.structure,
                    "confidence": round(current_regime.confidence * 100),
                    "adaptive_weights": strategy_weights
                },
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                    "regime_adjustments": trend_signals.get("regime_adjustments", "")
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                    "regime_adjustments": mean_reversion_signals.get("regime_adjustments", "")
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                    "regime_adjustments": momentum_signals.get("regime_adjustments", "")
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                    "regime_adjustments": volatility_signals.get("regime_adjustments", "")
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                    "regime_adjustments": stat_arb_signals.get("regime_adjustments", "")
                },
            },
        }
        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(technical_analysis, indent=4))

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst (Adaptive)")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = technical_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def get_adaptive_strategy_weights(regime):
    """
    Get adaptive strategy weights based on market regime.
    
    Args:
        regime: MarketRegime object
        
    Returns:
        Dictionary of strategy weights
    """
    # Base weights
    base_weights = {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "momentum": 0.25,
        "volatility": 0.15,
        "stat_arb": 0.15,
    }
    
    # Regime-based adjustments
    if regime.structure == "trending":
        # Emphasize trend-following and momentum in trending markets
        base_weights["trend"] = 0.35
        base_weights["momentum"] = 0.30
        base_weights["mean_reversion"] = 0.15
        base_weights["volatility"] = 0.10
        base_weights["stat_arb"] = 0.10
    elif regime.structure == "mean_reverting":
        # Emphasize mean reversion and statistical arbitrage
        base_weights["mean_reversion"] = 0.35
        base_weights["stat_arb"] = 0.25
        base_weights["trend"] = 0.15
        base_weights["momentum"] = 0.15
        base_weights["volatility"] = 0.10
    
    # High volatility adjustments - increase volatility strategy weight
    if regime.volatility == "high":
        base_weights["volatility"] = min(0.25, base_weights["volatility"] * 1.5)
        # Normalize other weights
        remaining_weight = 1.0 - base_weights["volatility"]
        for key in ["trend", "mean_reversion", "momentum", "stat_arb"]:
            base_weights[key] = base_weights[key] * (remaining_weight / sum(base_weights[k] for k in ["trend", "mean_reversion", "momentum", "stat_arb"]))
    
    # Ensure weights sum to 1.0
    total_weight = sum(base_weights.values())
    for key in base_weights:
        base_weights[key] /= total_weight
    
    return base_weights


def calculate_trend_signals_adaptive(prices_df, regime):
    """
    Adaptive trend following strategy with regime-aware parameters
    """
    # Get adaptive parameters
    ema_short = int(AdaptiveThresholds.get_adaptive_threshold("ema_short", regime))
    ema_medium = int(AdaptiveThresholds.get_adaptive_threshold("ema_medium", regime))
    ema_long = int(AdaptiveThresholds.get_adaptive_threshold("ema_long", regime))
    adx_period = int(AdaptiveThresholds.get_adaptive_threshold("adx_period", regime))
    
    # Calculate EMAs with adaptive periods
    ema_short_series = calculate_ema(prices_df, ema_short)
    ema_medium_series = calculate_ema(prices_df, ema_medium)
    ema_long_series = calculate_ema(prices_df, ema_long)

    # Calculate ADX for trend strength with adaptive period
    adx = calculate_adx(prices_df, adx_period)

    # Determine trend direction and strength
    short_trend = ema_short_series > ema_medium_series
    medium_trend = ema_medium_series > ema_long_series

    # Adaptive trend strength thresholds based on regime
    min_adx_threshold = 20 if regime.volatility == "low" else 25  # Higher threshold in volatile markets
    strong_adx_threshold = 40 if regime.market_state == "bull" else 35  # Different expectations by regime
    
    trend_strength = adx["adx"].iloc[-1] / 100.0
    
    # Enhanced confidence calculation based on regime
    base_confidence = min(trend_strength, 1.0)
    
    # Regime-specific confidence adjustments
    if regime.structure == "trending":
        confidence_multiplier = 1.2  # Higher confidence in trending regimes
    else:
        confidence_multiplier = 0.8   # Lower confidence in mean-reverting regimes
    
    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = min(base_confidence * confidence_multiplier, 1.0)
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish" 
        confidence = min(base_confidence * confidence_multiplier, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    # Apply ADX-based confidence filter
    current_adx = adx["adx"].iloc[-1]
    if current_adx < min_adx_threshold:
        confidence *= 0.5  # Reduce confidence in weak trend environments
    elif current_adx > strong_adx_threshold:
        confidence = min(confidence * 1.2, 1.0)  # Boost confidence in strong trends
    
    regime_adjustments = f"EMA periods: {ema_short}/{ema_medium}/{ema_long} (adapted for {regime.structure} regime), ADX threshold: {min_adx_threshold}"

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": safe_float(current_adx),
            "trend_strength": safe_float(trend_strength),
            "ema_short_period": ema_short,
            "ema_medium_period": ema_medium,
            "ema_long_period": ema_long,
        },
        "regime_adjustments": regime_adjustments
    }


def calculate_mean_reversion_signals_adaptive(prices_df, regime):
    """
    Adaptive mean reversion strategy with regime-aware parameters
    """
    # Get adaptive parameters
    rsi_period = int(AdaptiveThresholds.get_adaptive_threshold("rsi_period", regime))
    rsi_oversold = AdaptiveThresholds.get_adaptive_threshold("rsi_oversold", regime)
    rsi_overbought = AdaptiveThresholds.get_adaptive_threshold("rsi_overbought", regime)
    bollinger_period = int(AdaptiveThresholds.get_adaptive_threshold("bollinger_period", regime))
    bollinger_std = AdaptiveThresholds.get_adaptive_threshold("bollinger_std", regime)
    
    # Calculate z-score of price relative to moving average (adaptive period)
    ma_period = max(20, bollinger_period)  # Use bollinger period or minimum 20
    ma_adaptive = prices_df["close"].rolling(window=ma_period).mean()
    std_adaptive = prices_df["close"].rolling(window=ma_period).std()
    z_score = (prices_df["close"] - ma_adaptive) / std_adaptive

    # Calculate adaptive Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands_adaptive(prices_df, bollinger_period, bollinger_std)

    # Calculate RSI with adaptive parameters
    rsi_adaptive = calculate_rsi(prices_df, rsi_period)

    # Mean reversion signals with adaptive thresholds
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    current_rsi = rsi_adaptive.iloc[-1]
    current_z = z_score.iloc[-1]
    
    # Adaptive signal generation based on regime
    if regime.structure == "mean_reverting":
        # More aggressive mean reversion signals
        z_threshold = 1.5  # Lower threshold for mean reversion
        bb_threshold_low = 0.25
        bb_threshold_high = 0.75
        confidence_multiplier = 1.3
    else:
        # Conservative mean reversion in trending markets
        z_threshold = 2.0  # Higher threshold
        bb_threshold_low = 0.15
        bb_threshold_high = 0.85
        confidence_multiplier = 0.8

    # Generate signals with adaptive parameters
    if (current_z < -z_threshold and price_vs_bb < bb_threshold_low) or current_rsi < rsi_oversold:
        signal = "bullish"
        confidence = min(abs(current_z) / 4 * confidence_multiplier, 1.0)
    elif (current_z > z_threshold and price_vs_bb > bb_threshold_high) or current_rsi > rsi_overbought:
        signal = "bearish"
        confidence = min(abs(current_z) / 4 * confidence_multiplier, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    regime_adjustments = f"RSI: {rsi_period}d/{rsi_oversold:.0f}/{rsi_overbought:.0f}, BB: {bollinger_period}d/{bollinger_std:.1f}σ (adapted for {regime.structure} regime)"

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": safe_float(current_z),
            "price_vs_bb": safe_float(price_vs_bb),
            "rsi": safe_float(current_rsi),
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "bollinger_std": bollinger_std
        },
        "regime_adjustments": regime_adjustments
    }


def calculate_momentum_signals_adaptive(prices_df, regime):
    """
    Adaptive momentum strategy with regime-aware parameters
    """
    # Price momentum with adaptive lookback periods
    returns = prices_df["close"].pct_change()
    
    # Adaptive momentum periods based on volatility regime
    if regime.volatility == "high":
        # Shorter periods in high volatility
        mom_short = returns.rolling(10).sum()
        mom_medium = returns.rolling(30).sum()
        mom_long = returns.rolling(60).sum()
        weights = [0.5, 0.3, 0.2]  # Emphasize shorter-term momentum
    else:
        # Standard periods in low volatility
        mom_short = returns.rolling(21).sum()
        mom_medium = returns.rolling(63).sum()
        mom_long = returns.rolling(126).sum()
        weights = [0.4, 0.3, 0.3]  # More balanced

    # Volume momentum with adaptive period
    volume_period = 15 if regime.volatility == "high" else 21
    volume_ma = prices_df["volume"].rolling(volume_period).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Calculate adaptive momentum score
    momentum_score = (weights[0] * mom_short + weights[1] * mom_medium + weights[2] * mom_long).iloc[-1]

    # Volume confirmation with adaptive threshold
    volume_threshold = 1.2 if regime.volatility == "high" else 1.0
    volume_confirmation = volume_momentum.iloc[-1] > volume_threshold

    # Adaptive signal thresholds based on regime
    if regime.market_state == "bull":
        bullish_threshold = 0.03    # Lower threshold in bull markets
        bearish_threshold = -0.08   # Higher threshold for bearish signals
    elif regime.market_state == "bear":
        bullish_threshold = 0.08    # Higher threshold in bear markets
        bearish_threshold = -0.03   # Lower threshold for bearish signals
    else:
        bullish_threshold = 0.05
        bearish_threshold = -0.05

    # Generate signals
    if momentum_score > bullish_threshold and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 8, 1.0)
    elif momentum_score < bearish_threshold and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 8, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    # Regime-based confidence adjustment
    if regime.structure == "trending":
        confidence = min(confidence * 1.2, 1.0)  # Higher confidence in trending markets
    else:
        confidence *= 0.9  # Slightly lower confidence in mean-reverting markets

    regime_adjustments = f"Momentum periods: {weights}, Volume threshold: {volume_threshold}, Signal thresholds: {bullish_threshold:.2f}/{bearish_threshold:.2f} (adapted for {regime.market_state}/{regime.volatility})"

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_score": safe_float(momentum_score),
            "volume_momentum": safe_float(volume_momentum.iloc[-1]),
            "momentum_short": safe_float(mom_short.iloc[-1]),
            "momentum_medium": safe_float(mom_medium.iloc[-1]),
            "momentum_long": safe_float(mom_long.iloc[-1]),
            "volume_threshold": volume_threshold,
        },
        "regime_adjustments": regime_adjustments
    }


def calculate_volatility_signals_adaptive(prices_df, regime):
    """
    Adaptive volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Adaptive volatility calculation period based on regime
    vol_period = 15 if regime.volatility == "high" else 21
    hist_vol = returns.rolling(vol_period).std() * math.sqrt(252)

    # Volatility regime detection with adaptive periods
    vol_ma_period = 45 if regime.volatility == "high" else 63
    vol_ma = hist_vol.rolling(vol_ma_period).mean()
    vol_regime_ratio = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(vol_ma_period).std()

    # ATR with adaptive period
    atr_period = int(AdaptiveThresholds.get_adaptive_threshold("atr_period", regime))
    atr = calculate_atr(prices_df, atr_period)
    atr_ratio = atr / prices_df["close"]

    # Generate signals based on regime-adaptive volatility thresholds
    current_vol_regime = vol_regime_ratio.iloc[-1]
    vol_z = vol_z_score.iloc[-1]
    
    # Adaptive thresholds based on market regime
    if regime.market_state == "bull":
        low_vol_threshold = 0.75   # More sensitive to vol expansion in bull markets
        high_vol_threshold = 1.3
    elif regime.market_state == "bear":
        low_vol_threshold = 0.85   # Less sensitive in bear markets
        high_vol_threshold = 1.4
    else:
        low_vol_threshold = 0.8
        high_vol_threshold = 1.2

    # Signal generation with adaptive logic
    if current_vol_regime < low_vol_threshold and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > high_vol_threshold and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    # Regime-based confidence adjustment
    if regime.volatility == current_vol_regime_classification(current_vol_regime):
        confidence = min(confidence * 1.1, 1.0)  # Higher confidence when regime matches current state
    else:
        confidence *= 0.9  # Lower confidence during regime transitions

    regime_adjustments = f"Vol period: {vol_period}d, Vol MA: {vol_ma_period}d, Thresholds: {low_vol_threshold:.2f}/{high_vol_threshold:.2f} (adapted for {regime.market_state})"

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": safe_float(hist_vol.iloc[-1]),
            "volatility_regime": safe_float(current_vol_regime),
            "volatility_z_score": safe_float(vol_z),
            "atr_ratio": safe_float(atr_ratio.iloc[-1]),
            "vol_period": vol_period,
            "atr_period": atr_period,
        },
        "regime_adjustments": regime_adjustments
    }


def calculate_stat_arb_signals_adaptive(prices_df, regime):
    """
    Adaptive statistical arbitrage signals with regime-aware parameters
    """
    # Calculate price distribution statistics with adaptive periods
    returns = prices_df["close"].pct_change()
    
    # Adaptive lookback period based on volatility
    lookback_period = 45 if regime.volatility == "high" else 63

    # Skewness and kurtosis with adaptive window
    skew = returns.rolling(lookback_period).skew()
    kurt = returns.rolling(lookback_period).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent_adaptive(prices_df["close"], regime)

    # Adaptive signal generation based on regime
    current_skew = skew.iloc[-1] if not pd.isna(skew.iloc[-1]) else 0
    current_kurt = kurt.iloc[-1] if not pd.isna(kurt.iloc[-1]) else 0
    
    # Regime-specific thresholds
    if regime.structure == "mean_reverting":
        hurst_threshold = 0.45  # More lenient threshold for mean reversion
        skew_threshold = 0.8    # Lower threshold for skewness
        confidence_multiplier = 1.2
    else:  # trending regime
        hurst_threshold = 0.35  # Stricter threshold
        skew_threshold = 1.2    # Higher threshold for skewness
        confidence_multiplier = 0.8

    # Generate signal based on adaptive statistical properties
    if hurst < hurst_threshold:
        if current_skew > skew_threshold:
            signal = "bullish"
            confidence = (hurst_threshold - hurst) * 2 * confidence_multiplier
        elif current_skew < -skew_threshold:
            signal = "bearish"
            confidence = (hurst_threshold - hurst) * 2 * confidence_multiplier
        else:
            signal = "neutral"
            confidence = 0.5
    else:
        signal = "neutral"
        confidence = 0.4  # Lower confidence when not mean-reverting

    # Clip confidence to valid range
    confidence = max(0.0, min(confidence, 1.0))

    regime_adjustments = f"Lookback: {lookback_period}d, Hurst threshold: {hurst_threshold:.2f}, Skew threshold: ±{skew_threshold:.1f} (adapted for {regime.structure})"

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": safe_float(hurst),
            "skewness": safe_float(current_skew),
            "kurtosis": safe_float(current_kurt),
            "lookback_period": lookback_period,
            "hurst_threshold": hurst_threshold,
            "skew_threshold": skew_threshold,
        },
        "regime_adjustments": regime_adjustments
    }


def weighted_signal_combination_adaptive(signals, weights, regime):
    """
    Adaptive signal combination with regime-aware weighting
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0
    regime_bonus = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

        # Add regime consistency bonus
        if is_signal_consistent_with_regime(signal["signal"], strategy, regime):
            regime_bonus += 0.1 * weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Apply regime bonus
    final_score += regime_bonus

    # Adaptive signal thresholds based on regime confidence
    threshold_multiplier = regime.confidence
    bullish_threshold = 0.2 * threshold_multiplier
    bearish_threshold = -0.2 * threshold_multiplier

    # Convert back to signal
    if final_score > bullish_threshold:
        signal = "bullish"
    elif final_score < bearish_threshold:
        signal = "bearish"
    else:
        signal = "neutral"

    # Final confidence with regime adjustment
    confidence = min(abs(final_score) * (1 + regime_bonus), 1.0)

    return {"signal": signal, "confidence": confidence}


def is_signal_consistent_with_regime(signal, strategy, regime):
    """
    Check if a signal is consistent with the current market regime
    """
    # Trend-following strategies should be more reliable in trending regimes
    if strategy in ["trend", "momentum"] and regime.structure == "trending":
        return True
    
    # Mean reversion strategies should be more reliable in mean-reverting regimes  
    if strategy in ["mean_reversion", "stat_arb"] and regime.structure == "mean_reverting":
        return True
    
    # Volatility strategies are generally regime-agnostic
    if strategy == "volatility":
        return True
        
    return False


def current_vol_regime_classification(vol_ratio):
    """Classify current volatility regime"""
    return "high" if vol_ratio > 1.1 else "low"


def calculate_hurst_exponent_adaptive(price_series: pd.Series, regime, max_lag: int = None) -> float:
    """
    Calculate adaptive Hurst Exponent with regime-aware parameters
    """
    if max_lag is None:
        # Adaptive max_lag based on volatility regime
        max_lag = 15 if regime.volatility == "high" else 20
    
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5


def calculate_bollinger_bands_adaptive(prices_df: pd.DataFrame, window: int, std_dev: float) -> tuple[pd.Series, pd.Series]:
    """Calculate adaptive Bollinger Bands with regime-aware parameters"""
    sma = prices_df["close"].rolling(window).mean()
    std = prices_df["close"].rolling(window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


# Preserve original functions that don't need major modifications
def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5