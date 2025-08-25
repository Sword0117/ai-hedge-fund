# AI Hedge Fund Adaptive System - Phase 1 Implementation Guide

This guide explains the new adaptive, regime-aware trading system that dynamically adjusts parameters based on market conditions.

## 🎯 What Changed

### Phase 1 Improvements: Market Regime Detection & Adaptive Parameters

The AI hedge fund system now includes:

1. **Market Regime Detection** - Hidden Markov Model-based classification of market conditions
2. **Adaptive Warren Buffett Agent** - Dynamic valuation thresholds based on market regime
3. **Adaptive Technical Analysis** - Regime-aware technical indicators and strategy weighting
4. **Unified Parameter System** - Centralized adaptive threshold management

## 🏗️ Architecture Overview

```
Market Data → Regime Detector → Adaptive Agents → Portfolio Manager
     ↓              ↓                ↓               ↓
  Price/Vol/    Bull/Bear/      Adjusted         Final Trading
  News Data    High/Low Vol   Thresholds        Decisions
```

## 📊 Regime Detection System

### Market Regimes Detected:
- **Market State**: Bull / Bear / Neutral 
- **Volatility**: High / Low
- **Structure**: Trending / Mean-Reverting

### Data Sources:
- SPY benchmark for regime classification
- Features: Returns (1d, 5d, 20d), Realized volatility, Volume ratios, Momentum indicators

### Implementation:
```python
from src.agents.regime_detector import get_current_market_regime

# Get current regime
regime = get_current_market_regime("2024-03-15", api_key)
print(f"Market: {regime.market_state}, Vol: {regime.volatility}, Structure: {regime.structure}")
print(f"Confidence: {regime.confidence:.2f}")
```

## 🎯 Adaptive Warren Buffett Agent

### Key Improvements:

#### 1. Dynamic Valuation Thresholds
```python
# Static (OLD):    ROE > 15%, P/E < 25, Discount Rate = 10%
# Adaptive (NEW):  Thresholds adjust by regime

# Bull Market:     ROE > 15%, P/E < 32.5, Discount Rate = 9.5%  
# Bear Market:     ROE > 18%, P/E < 17.5, Discount Rate = 12%
# High Volatility: Margin of Safety increases by 20%
```

#### 2. Regime-Aware DCF Modeling
- **Growth Assumptions**: Conservative in bear markets, slightly optimistic in bull markets
- **Discount Rates**: Dynamic based on volatility (8.5%-12% range)
- **Margin of Safety**: 10% (bull) to 30% (bear + high vol)
- **Projection Periods**: Shorter in bear markets (4 vs 5 years)

#### 3. Enhanced Circle of Competence
The LLM now considers how market regimes affect business model sustainability:
- Bull markets: Slightly more flexible on growth premiums
- Bear markets: Emphasize defensive characteristics and pricing power
- High volatility: Focus on predictable earnings and balance sheet strength

### Example Adaptive Analysis Output:
```json
{
  "AAPL": {
    "signal": "bullish",
    "confidence": 82,
    "reasoning": "In this Bull/Low Volatility/Trending regime, Apple meets our adapted criteria with ROE of 28.8% (threshold: 16.5%), conservative debt levels, and strong pricing power. The regime-adjusted discount rate of 9.2% yields an intrinsic value with 10% margin of safety appropriate for current bull market conditions..."
  }
}
```

## 🔧 Adaptive Technical Analysis 

### Key Improvements:

#### 1. Regime-Aware Strategy Weighting
```python
# Trending Regime:     Trend (35%) + Momentum (30%) + Mean Rev (15%) + Vol (10%) + Stat Arb (10%)
# Mean-Reverting:      Mean Rev (35%) + Stat Arb (25%) + Trend (15%) + Momentum (15%) + Vol (10%)
# High Volatility:     Vol strategy weight increased by 50%
```

#### 2. Dynamic Technical Indicators
```python
# RSI Periods:         Bear market = 17 days, Bull market = 11 days  
# Bollinger Bands:     High vol = 2.6σ, Low vol = 1.6σ
# EMA Periods:         Adaptive based on regime structure
# ADX Thresholds:      Higher in volatile markets (25 vs 20)
```

#### 3. Regime-Consistent Signal Boosting
- Trend/momentum signals get +20% confidence in trending regimes
- Mean reversion signals get +30% confidence in mean-reverting regimes
- Signals inconsistent with regime are penalized (-20% confidence)

### Example Adaptive Technical Output:
```json
{
  "MSFT": {
    "signal": "bullish", 
    "confidence": 78,
    "reasoning": {
      "market_regime": {
        "state": "bull",
        "volatility": "low", 
        "structure": "trending",
        "adaptive_weights": {"trend": 0.35, "momentum": 0.30, ...}
      },
      "trend_following": {
        "signal": "bullish",
        "confidence": 85,
        "regime_adjustments": "EMA periods: 7/19/48 (adapted for trending regime), ADX threshold: 20"
      }
    }
  }
}
```

## ⚙️ Configuration & Setup

### 1. Install New Dependencies
```bash
# Add to poetry dependencies
poetry add hmmlearn scipy

# Or if using pip
pip install hmmlearn scipy
```

### 2. Environment Variables
No new environment variables needed - the system uses existing API keys for data access.

### 3. Regime Detection Cache
The system automatically caches regime detection results for 30 days to improve performance.

## 🚀 Usage Examples

### Basic Usage (No Changes)
The adaptive system works transparently with existing commands:

```bash
# CLI usage - regime detection happens automatically
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Backtesting - uses adaptive parameters automatically  
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01
```

### Advanced Usage - Regime Information

#### Check Current Market Regime
```python
from src.agents.regime_detector import get_current_market_regime

regime = get_current_market_regime("2024-03-15")
print(f"Current regime: {regime.market_state}/{regime.volatility}/{regime.structure}")
print(f"Confidence: {regime.confidence:.2f}")
```

#### Get Adaptive Thresholds
```python
from src.agents.regime_detector import AdaptiveThresholds

# Get adaptive ROE threshold for current regime
roe_threshold = AdaptiveThresholds.get_adaptive_threshold("roe_threshold", regime, "technology")
print(f"ROE threshold for tech in {regime.market_state} market: {roe_threshold:.1%}")

# Get all adaptive thresholds
all_thresholds = get_adaptive_thresholds(regime, "technology")
```

#### Regime History Analysis
```python
from src.agents.regime_detector import get_regime_detector

detector = get_regime_detector()
history = detector.get_regime_history("2024-01-01", "2024-03-15")

# Calculate regime statistics
stats = detector.get_regime_statistics(history)
print(f"Bull market days: {stats['market_distribution']['bull']:.1%}")
```

## 📈 Expected Performance Improvements

### Backtesting Enhancements:
- **Better Regime Adaptation**: Stricter requirements in bear markets should reduce drawdowns
- **Improved Timing**: Dynamic technical parameters should capture trends better
- **Risk Management**: Adaptive margins of safety provide better downside protection

### Key Performance Indicators to Monitor:
1. **Sharpe Ratio**: Expected improvement from better risk-adjusted returns
2. **Maximum Drawdown**: Should be lower due to adaptive risk management
3. **Regime Transition Performance**: Better handling of market regime changes
4. **Win Rate**: Improved signal quality should increase win rate

## 🔍 Testing the Adaptive System

### 1. Regime Detection Accuracy
```bash
# Test regime detection with known historical periods
poetry run python -c "
from src.agents.regime_detector import get_current_market_regime
import datetime

# Test on known bear market period (COVID crash)
regime = get_current_market_regime('2020-03-20')  
print('COVID crash regime:', regime.market_state, regime.volatility)

# Test on known bull market period  
regime = get_current_market_regime('2021-06-15')
print('Bull market regime:', regime.market_state, regime.volatility)
"
```

### 2. Threshold Adaptation Verification
```bash
# Run with show-reasoning to see adaptive thresholds
poetry run python src/main.py --ticker AAPL --show-reasoning --end-date 2020-03-20
poetry run python src/main.py --ticker AAPL --show-reasoning --end-date 2021-06-15
```

### 3. Backtesting Comparison
```bash
# Backtest through different market regimes
poetry run python src/backtester.py --ticker AAPL,MSFT --start-date 2020-01-01 --end-date 2024-01-01

# Compare performance in different regime periods
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. HMM Library Not Found
```bash
# If hmmlearn installation fails
pip install --upgrade pip
pip install hmmlearn

# Or use conda
conda install -c conda-forge hmmlearn
```

#### 2. Regime Detection Fallback
If regime detection fails, the system automatically falls back to:
- Market State: "neutral"  
- Volatility: "low"
- Structure: "mean_reverting"
- Confidence: 0.3

#### 3. Performance Issues
- Regime detection is cached for 30 days
- If experiencing slowdowns, clear cache: Delete files in data cache directory
- For backtesting, regime is calculated once per day and reused

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)

# This will show regime detection and adaptive threshold information
```

## 📚 Technical Implementation Details

### 1. Regime Detector Architecture
- **HMM Model**: 3-state Gaussian HMM for market classification
- **Feature Engineering**: 10 normalized features including returns, volatility, volume
- **Fallback System**: Heuristic-based detection if HMM fails
- **Caching**: File-based caching with 30-day expiration

### 2. Adaptive Threshold System
- **Base Parameters**: Stored in `AdaptiveThresholds.BASE_THRESHOLDS`
- **Regime Multipliers**: Applied based on market state, volatility, and structure
- **Sector Adjustments**: Optional sector-specific parameter modifications
- **Bounds Checking**: All thresholds validated to stay within reasonable ranges

### 3. Integration Points
- **Warren Buffett Agent**: Calls regime detector once per run, applies to all analysis functions
- **Technical Agent**: Uses regime-aware strategy weighting and parameter adaptation
- **Backward Compatibility**: System works without any changes to existing code

## 🎯 Next Steps (Future Phases)

### Phase 2: Enhanced Signal Fusion (Planned)
- Replace LLM portfolio manager with ML ensemble
- Dynamic agent weighting based on historical performance
- Real-time performance tracking and adaptation

### Phase 3: Multi-Asset Expansion (Planned)  
- Extend regime detection to ES/NQ futures
- Asset-specific parameter adaptation
- Cross-asset correlation analysis

### Phase 4: Advanced Risk Management (Planned)
- Multi-factor risk models
- Dynamic position sizing via Kelly criterion
- Tail risk management integration

## 📞 Support

For issues with the adaptive system:
1. Check logs for regime detection errors
2. Verify all dependencies are installed correctly
3. Test with fallback regime if needed
4. Review threshold adaptations in debug mode

The adaptive system maintains full backward compatibility - existing scripts will work unchanged with the new regime-aware intelligence operating transparently in the background.