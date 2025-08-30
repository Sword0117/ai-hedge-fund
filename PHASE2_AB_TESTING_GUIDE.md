# Phase 2 A/B Testing Guide - ML Ensemble vs LLM Portfolio Manager

This guide explains how to run and compare the traditional LLM portfolio manager against the new ML ensemble system for signal fusion and decision making.

## 🎯 Overview

Phase 2 introduces an Enhanced Signal Fusion system that:

- **ML Ensemble Models**: Uses trained RandomForest/GradientBoosting models instead of LLMs
- **Dynamic Agent Weighting**: Adjusts agent importance based on historical performance  
- **Performance Tracking**: Monitors and logs all agent predictions and outcomes
- **Regime-Aware Decisions**: Adapts model selection based on market conditions

## 🏗️ Architecture Comparison

### Traditional System (LLM Mode)
```
Agent Signals → LLM Portfolio Manager → Trading Decisions
     ↓                    ↓                    ↓
Warren Buffett      Prompt-based          Buy/Sell/Hold
Technical Analysis   Reasoning            with Quantities
Sentiment Analysis   Human-like           & Confidence
                    Decision Making
```

### Enhanced System (ML Ensemble Mode)
```
Agent Signals → Performance Tracker → Signal Ensemble → Trading Decisions
     ↓               ↓                     ↓                ↓
Warren Buffett   Dynamic Weights    ML Feature         Buy/Sell/Hold
Technical Analysis  (Accuracy-based)  Engineering       with Quantities
Sentiment Analysis  Calibration       Regime Models     & Confidence
```

## 🚀 Quick Start A/B Testing

### Step 1: Install Dependencies

```bash
# Install ML dependencies
poetry add scikit-learn joblib

# Or with pip
pip install scikit-learn joblib
```

### Step 2: Generate Training Data

First, run backtests to generate training data:

```bash
# Run historical backtests (LLM mode) to generate training data
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2023-01-01 --end-date 2024-01-01 --output data/training_backtest.json

# Additional training data
poetry run python src/backtester.py --ticker TSLA,GOOGL,META --start-date 2023-01-01 --end-date 2024-01-01 --output data/training_backtest_2.json
```

### Step 3: Train ML Ensemble Models

```bash
# Train ensemble models on historical data
poetry run python scripts/train_signal_ensemble.py --backtest-data data/training_backtest.json --start-date 2023-01-01 --end-date 2023-12-01

# Check training results
ls data/models/  # Should show bull_model.joblib, bear_model.joblib, neutral_model.joblib, etc.
```

### Step 4: Configure A/B Testing

Create or update `config/portfolio_config.yaml`:

```yaml
# Enable enhanced portfolio manager
portfolio_manager:
  use_enhanced_manager: true
  use_ml_ensemble: true  # Set to false for LLM mode
  
# Enable performance tracking
performance_tracking:
  enabled: true
  
# Enable A/B testing logging
ab_testing:
  enabled: true
  decision_logging: true
```

### Step 5: Run A/B Comparison

```bash
# Run LLM mode backtest
poetry run python src/backtester.py --ticker AAPL,MSFT --start-date 2024-01-01 --config-file config/llm_config.yaml --output results/llm_backtest.json

# Run ML Ensemble mode backtest  
poetry run python src/backtester.py --ticker AAPL,MSFT --start-date 2024-01-01 --config-file config/ml_config.yaml --output results/ml_backtest.json

# Compare results
poetry run python scripts/compare_ab_results.py --llm-results results/llm_backtest.json --ml-results results/ml_backtest.json
```

## ⚙️ Configuration Options

### LLM Mode Configuration (`config/llm_config.yaml`)
```yaml
portfolio_manager:
  use_enhanced_manager: true
  use_ml_ensemble: false  # Use LLM decisions

performance_tracking:
  enabled: true  # Still track for comparison

ab_testing:
  enabled: true
  decision_logging: true
```

### ML Ensemble Mode Configuration (`config/ml_config.yaml`)
```yaml
portfolio_manager:
  use_enhanced_manager: true
  use_ml_ensemble: true  # Use ML ensemble decisions
  
  ensemble_config:
    confidence_threshold: 0.65
    models_dir: "data/models"

performance_tracking:
  enabled: true
  history_window: 60
  
ab_testing:
  enabled: true
  decision_logging: true
```

## 📊 Performance Metrics Comparison

### Key Metrics to Track

| Metric | Description | LLM Expected | ML Expected |
|--------|-------------|--------------|-------------|
| **Sharpe Ratio** | Risk-adjusted returns | Higher reasoning quality | Better pattern recognition |
| **Accuracy** | Correct signal prediction rate | Variable by reasoning | Consistent based on training |
| **Confidence Calibration** | How well confidence predicts success | Often overconfident | Better calibrated |
| **Max Drawdown** | Largest peak-to-trough loss | Dependent on market narrative | Driven by historical patterns |
| **Decision Speed** | Time to generate decisions | Slower (API calls) | Faster (local inference) |

### Running Performance Analysis

```bash
# Generate comprehensive performance report
poetry run python scripts/analyze_performance.py --llm-log logs/llm_decisions.json --ml-log logs/ml_decisions.json --output reports/ab_comparison.html

# View agent-level performance
poetry run python -c "
from src.agents.performance_tracker import AgentPerformanceTracker
tracker = AgentPerformanceTracker()
report = tracker.get_performance_report()
print(json.dumps(report, indent=2))
"
```

## 🔍 Detailed Analysis Workflow

### 1. Pre-Test Validation

```bash
# Test both systems work correctly
poetry run python test_adaptive_system.py  # Test ML components
poetry run python src/main.py --ticker AAPL --show-reasoning  # Test LLM mode

# Verify model loading
poetry run python -c "
from src.agents.signal_fusion import create_signal_ensemble
ensemble = create_signal_ensemble()
print(f'Models loaded: {ensemble.is_trained}')
print(f'Model info: {ensemble.get_model_info()}')
"
```

### 2. Controlled A/B Test

```python
# Example A/B test script
import json
from datetime import datetime
from src.agents.portfolio_manager import get_enhanced_portfolio_manager

# Configure both managers
llm_config = {"use_ml_ensemble": False}  
ml_config = {"use_ml_ensemble": True}

llm_manager = get_enhanced_portfolio_manager(llm_config)
ml_manager = get_enhanced_portfolio_manager(ml_config)

# Test with same inputs
test_signals = {
    'warren_buffett_agent': {'signal': 'bullish', 'confidence': 75.0},
    'technical_analyst_agent': {'signal': 'bearish', 'confidence': 60.0}
}

# Compare decisions (implement in actual script)
```

### 3. Live Performance Monitoring

```bash
# Monitor live performance during backtesting
tail -f logs/portfolio_manager.log | grep -E "(LLM Decision|ML Ensemble)"

# Track performance updates
tail -f logs/portfolio_manager.log | grep "Performance updated"
```

## 📈 Expected Results & Interpretation

### Scenarios Where ML Ensemble Should Excel

1. **Consistent Market Patterns**: When historical patterns repeat
2. **High-Frequency Decisions**: Faster inference than LLM calls
3. **Confidence Calibration**: Better-calibrated confidence scores
4. **Agent Performance Tracking**: Dynamic weighting based on actual performance

### Scenarios Where LLM May Excel

1. **Novel Market Conditions**: Events not seen in training data
2. **Complex Reasoning**: Multi-step logical analysis
3. **Narrative Understanding**: Incorporating qualitative information
4. **Market Regime Shifts**: Adapting to fundamentally new conditions

### Statistical Significance Testing

```python
# Example statistical comparison
import scipy.stats as stats

# Compare Sharpe ratios
llm_returns = [...]  # Daily returns from LLM backtest
ml_returns = [...]   # Daily returns from ML backtest

# Calculate Sharpe ratios
llm_sharpe = np.mean(llm_returns) / np.std(llm_returns) * np.sqrt(252)
ml_sharpe = np.mean(ml_returns) / np.std(ml_returns) * np.sqrt(252)

# Statistical test for difference in means
t_stat, p_value = stats.ttest_ind(llm_returns, ml_returns)
print(f"Sharpe - LLM: {llm_sharpe:.3f}, ML: {ml_sharpe:.3f}")
print(f"Difference significant: {p_value < 0.05} (p={p_value:.4f})")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Models Not Loading**
   ```bash
   # Check model files exist
   ls -la data/models/
   
   # Retrain if missing
   poetry run python scripts/train_signal_ensemble.py --backtest-data data/training_backtest.json
   ```

2. **Performance Tracking Errors**
   ```bash
   # Check storage directory
   ls -la data/agent_performance/
   
   # Clear corrupted data
   rm -rf data/agent_performance/*.json.gz
   ```

3. **Configuration Not Loading**
   ```bash
   # Verify config file syntax
   python -c "import yaml; yaml.safe_load(open('config/portfolio_config.yaml'))"
   
   # Check default values
   python -c "from src.agents.portfolio_manager import get_enhanced_portfolio_manager; print(get_enhanced_portfolio_manager().config)"
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from src.agents.signal_fusion import create_signal_ensemble
from src.agents.performance_tracker import AgentPerformanceTracker

ensemble = create_signal_ensemble()
tracker = AgentPerformanceTracker()

print("Ensemble info:", ensemble.get_model_info())
print("Tracker weights:", tracker.calculate_agent_weights("AAPL", datetime.now()))
```

## 📋 A/B Testing Checklist

### Pre-Testing
- [ ] Historical training data generated (>=1000 samples)
- [ ] ML models trained and saved successfully  
- [ ] Both LLM and ML modes tested individually
- [ ] Configuration files validated
- [ ] Performance tracking initialized

### During Testing
- [ ] Both systems running on identical input data
- [ ] Decision logs being captured correctly
- [ ] Performance metrics updating in real-time
- [ ] No system errors or crashes
- [ ] Resource usage within acceptable limits

### Post-Testing Analysis  
- [ ] Statistical significance tests completed
- [ ] Performance metrics compared across multiple dimensions
- [ ] Agent-level performance analyzed
- [ ] Market regime impact assessed
- [ ] Recommendations documented for production usage

## 🎯 Next Steps After A/B Testing

Based on results, you can:

1. **ML Ensemble Wins**: Deploy ML ensemble as primary system
2. **LLM Wins**: Continue using LLM with performance tracking
3. **Mixed Results**: Use hybrid approach - ML for certain regimes, LLM for others
4. **Inconclusive**: Extend testing period or refine models

The enhanced system provides full backward compatibility, so you can switch between modes dynamically based on market conditions or performance criteria.

---

## 📞 Support

For issues with the A/B testing system:

1. Check logs in `logs/portfolio_manager.log`
2. Verify configuration files are valid YAML
3. Ensure all dependencies are installed: `poetry install`
4. Test individual components before running full backtests
5. Use debug logging for detailed troubleshooting