# Training Guide

Complete guide to training and optimizing the ML ensemble models for market regime detection and signal fusion.

## Overview

The AI Hedge Fund uses a sophisticated ML ensemble system that combines:
- **Hidden Markov Models (HMM)** for market regime detection
- **Gradient Boosting Classifiers** for bull/bear markets  
- **Random Forest Classifiers** for neutral markets
- **Feature engineering** from agent confidence scores and market indicators

## Training Pipeline Architecture

```
Market Data → Agent Analysis → Feature Engineering → Regime Detection → Model Training → Performance Validation
```

### Key Components

1. **Data Generation**: Automated backtesting to create training samples
2. **Regime Classification**: HMM-based market state detection (bull/bear/neutral)
3. **Feature Engineering**: Agent confidence scores, agreement metrics, market features
4. **Model Training**: Regime-specific classifiers with hyperparameter optimization
5. **Cross-Validation**: Stratified k-fold with class imbalance handling

## Quick Start Training

### Basic Training Command

```bash
# Train models with default parameters (recommended for first time)
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-05-01 \
  --end-date 2025-08-26 \
  --tickers AAPL,MSFT,NVDA
```

### Training Process Steps

1. **Data Generation** (5-10 minutes)
   - Runs backtester across date range
   - Generates 300+ balanced training samples
   - Captures agent decisions and confidence scores

2. **Feature Engineering** (1-2 minutes)
   - Extracts agent signals and metadata
   - Calculates agreement metrics and consensus
   - Adds market regime and temporal features

3. **Regime Detection Training** (1-2 minutes)
   - Trains HMM on market volatility and structure
   - Classifies historical periods into bull/bear/neutral
   - Validates regime probability distributions

4. **Model Training** (2-3 minutes)
   - Trains separate models for each regime
   - Performs hyperparameter optimization
   - Validates with stratified cross-validation

5. **Performance Evaluation** (30 seconds)
   - Calculates accuracy metrics per regime
   - Saves model metadata and feature importance
   - Generates performance report

## Advanced Training Options

### Extended Training Period

```bash
# Use longer period for more robust models (recommended for production)
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --force
```

### Custom Ticker Selection

```bash
# Train with specific stocks
poetry run python scripts/train_signal_ensemble.py \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA,BRK.B \
  --regenerate
```

### Force Retraining

```bash
# Overwrite existing models and data
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --force \
  --regenerate
```

### Sample Data Generation

```bash
# Generate synthetic training data for testing
poetry run python scripts/train_signal_ensemble.py --generate-sample
```

## Understanding Training Output

### Successful Training Example

```
INFO: Starting ML signal ensemble training...
INFO: Using date range: 2024-05-01 to 2025-08-26
INFO: Training with tickers: ['AAPL', 'MSFT', 'NVDA']

=== DATA GENERATION ===
INFO: Generating enhanced training data...
INFO: Running backtest for sample generation...
INFO: Generated 312 total samples
INFO: Regime distribution: bull=104, bear=102, neutral=106

=== FEATURE ENGINEERING ===
INFO: Processing agent signals and features...
INFO: Feature columns: 21 features extracted
INFO: Added temporal and regime-based features

=== REGIME DETECTION ===
INFO: Training market regime detector...
INFO: HMM converged in 45 iterations
INFO: Regime probabilities: bull=33.2%, bear=32.8%, neutral=34.0%

=== MODEL TRAINING ===
INFO: Training bull market model...
SUCCESS: Bull market model - Accuracy: 68.0% (±10.6%)
INFO: Training bear market model...
SUCCESS: Bear market model - Accuracy: 80.5% (±9.8%)
INFO: Training neutral market model...
SUCCESS: Neutral market model - Accuracy: 80.6% (±6.7%)

=== PERFORMANCE SUMMARY ===
Overall ML Ensemble Performance: 80.3% (+28.6% vs baseline)
Models saved to: data/models/
Training completed in: 8m 32s
```

### Performance Metrics Explained

- **Accuracy**: Percentage of correct buy/sell/hold predictions
- **Standard Deviation**: Cross-validation stability measure
- **Baseline Comparison**: Improvement over random agent decisions
- **Regime Distribution**: Balance of bull/bear/neutral samples

## Feature Engineering Details

### Core Agent Features

Generated from each trading agent's analysis:

```python
# Warren Buffett Agent
warren_buffett_bullish     # Bullish signal strength (0-1)
warren_buffett_bearish     # Bearish signal strength (0-1)
warren_buffett_confidence  # Overall confidence (0-1)

# Technical Analysis Agent  
technical_analyst_bullish     # Technical bullish signals
technical_analyst_bearish     # Technical bearish signals
technical_analyst_confidence  # Technical analysis confidence

# Sentiment Agent
sentiment_bullish     # News sentiment bullishness
sentiment_bearish     # News sentiment bearishness  
sentiment_confidence  # Sentiment analysis confidence
```

### Consensus Features

Derived from agent agreement and consensus:

```python
agent_agreement      # How well agents agree (0-1)
confidence_variance  # Variance in agent confidence scores
bullish_consensus    # Overall bullish consensus (0-1)
```

### Market Regime Features

From HMM regime detection:

```python
regime_bull              # Bull market probability
regime_bear              # Bear market probability
regime_volatility_high   # High volatility indicator
regime_structure_trending # Trending market structure
```

### Temporal Features

Time-based market patterns:

```python
day_of_week  # Day of week effects (0-6)
month        # Monthly seasonality (1-12)
```

### Agent Weighting Features

Dynamic agent importance (future expansion):

```python
warren_buffett_weight      # Dynamic weight for fundamental analysis
technical_analyst_weight   # Dynamic weight for technical analysis
sentiment_weight           # Dynamic weight for sentiment analysis
```

## Model Architecture

### Regime-Specific Models

Each market regime uses optimized algorithms:

**Bull Markets**: Gradient Boosting Classifier
- Best for capturing non-linear growth patterns
- Hyperparameters: learning_rate=0.1, max_depth=5, n_estimators=100

**Bear Markets**: Gradient Boosting Classifier  
- Optimized for downtrend detection
- Hyperparameters: learning_rate=0.2, max_depth=5, n_estimators=100

**Neutral Markets**: Random Forest Classifier
- Better for sideways market complexity
- Hyperparameters: max_depth=5, max_features='sqrt', n_estimators=200

### Hyperparameter Optimization

The system uses GridSearchCV with stratified k-fold cross-validation:

```python
# Bull/Bear markets - Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Neutral markets - Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'max_features': ['sqrt', 'log2']
}
```

## Data Generation Process

### Automated Backtesting

The training pipeline automatically generates data by:

1. **Running Backtester**: Executes trading decisions across date range
2. **Capturing Decisions**: Records all agent buy/sell/hold decisions  
3. **Extracting Features**: Saves confidence scores and reasoning
4. **Regime Labeling**: Classifies each sample's market regime
5. **Balancing Classes**: Ensures equal representation across regimes

### Sample Generation Strategy

```python
# Target distribution for balanced training
MIN_PER_REGIME = 100  # Minimum samples per regime
TARGET_TOTAL = 300    # Target total samples

# Realistic decision distributions per regime
BULL_DECISIONS = {'buy': 0.6, 'hold': 0.3, 'sell': 0.1}
BEAR_DECISIONS = {'buy': 0.1, 'hold': 0.3, 'sell': 0.6}
NEUTRAL_DECISIONS = {'buy': 0.2, 'hold': 0.6, 'sell': 0.2}
```

### Fallback Data Sources

If backtester fails or insufficient data:

1. **Historical Cache**: Uses previously generated samples
2. **Sample Generator**: Creates synthetic balanced dataset
3. **Reduced Complexity**: Falls back to simpler feature sets

## Training Data Management

### Data Storage Structure

```
data/
├── models/
│   ├── regime_detector_hmm.pkl      # HMM regime detection model
│   ├── signal_ensemble_bull.pkl     # Bull market classifier
│   ├── signal_ensemble_bear.pkl     # Bear market classifier  
│   ├── signal_ensemble_neutral.pkl  # Neutral market classifier
│   ├── feature_scaler.pkl           # Feature normalization
│   └── model_metadata.json          # Training metadata
├── training/
│   ├── backtest_decisions_cache.json # Generated training samples
│   └── sample_training_data.json    # Synthetic fallback data
└── cache/
    └── market_data/                 # Cached price data
```

### Model Versioning

Each training run creates versioned models:

```json
{
  "version": "2.0",
  "training_date": "2025-08-28T17:42:34.582196",
  "training_samples": 300,
  "regimes_trained": ["bull", "bear", "neutral"],
  "performance": {
    "bull": {"accuracy": 0.680, "cv_score": 0.680},
    "bear": {"accuracy": 0.805, "cv_score": 0.805}, 
    "neutral": {"accuracy": 0.806, "cv_score": 0.806}
  }
}
```

## Performance Optimization

### Training Speed Optimization

**Faster Training** (2-3 minutes):
```bash
# Use shorter date range
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-08-01 \
  --end-date 2024-12-31 \
  --tickers AAPL,MSFT
```

**Production Training** (15-20 minutes):
```bash  
# Use extended period for best accuracy
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA,BRK.B
```

### Memory Optimization

For systems with limited RAM:

1. **Reduce ticker count**: Use 2-3 tickers instead of 6
2. **Shorter date range**: Use 3-6 months instead of 12+
3. **Batch processing**: Train one regime at a time
4. **Clear cache**: Remove old training data before starting

### Accuracy Optimization  

**Best Practices**:
- Use 12+ months of training data
- Include diverse market conditions (bull, bear, sideways)
- Ensure balanced class distribution (±10% per regime)
- Validate on out-of-sample data
- Monitor feature importance for stability

## Troubleshooting Training Issues

### Common Training Problems

**1. Insufficient Training Data**
```
ERROR: Only generated 45 samples, need minimum 100 per regime
```

**Solution**:
```bash
# Use longer date range
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26

# Or generate sample data
poetry run python scripts/train_signal_ensemble.py --generate-sample
```

**2. Single-Class Training Data**
```
ERROR: Bull regime has only 1 unique class, cannot train classifier
```

**Solution**:
```bash
# Force regeneration with broader date range
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --regenerate --force
```

**3. HMM Convergence Warnings**
```
WARNING: HMM did not converge. Consider increasing n_iter or tol
```

**Solution**: The system automatically handles this with relaxed parameters. Models will still train successfully.

**4. API Rate Limiting**
```
ERROR: Rate limit exceeded for market data API
```

**Solution**:
```bash
# Wait 1-2 minutes and retry, or use cached data
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26
```

**5. Memory Issues**
```
ERROR: MemoryError during model training
```

**Solution**:
```bash
# Reduce training data size
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-06-01 \
  --end-date 2024-12-31 \
  --tickers AAPL,MSFT
```

## Model Evaluation and Validation

### Cross-Validation Process

The system uses stratified 5-fold cross-validation:

1. **Stratified Splits**: Maintains class balance in each fold
2. **Multiple Metrics**: Accuracy, precision, recall, F1-score
3. **Stability Testing**: Standard deviation across folds
4. **Regime-Specific**: Separate validation for each market regime

### Feature Importance Analysis

After training, review which features matter most:

```bash
# Check model metadata for feature importance
cat data/models/model_metadata.json | jq '.feature_importance'
```

**Key Features** (typically most important):
- `bullish_consensus`: Overall agent agreement
- `confidence_variance`: Agent confidence spread
- `month`: Seasonal market patterns
- `technical_analyst_confidence`: Technical signal strength

### Performance Benchmarking

**Baseline Comparison**:
- Random decisions: ~50% accuracy
- Single agent: ~51-55% accuracy  
- ML ensemble: ~80% accuracy

**Production Targets**:
- Bull markets: >65% accuracy
- Bear markets: >75% accuracy
- Neutral markets: >75% accuracy
- Overall ensemble: >75% accuracy

## Deployment and Production

### Model Update Schedule

**Recommended Schedule**:
- **Daily**: Use existing models for trading decisions
- **Weekly**: Check performance metrics and feature drift
- **Monthly**: Retrain with latest market data
- **Quarterly**: Full model architecture review

### Production Training Command

```bash
# Production-ready training with full validation
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date $(date +%Y-%m-%d) \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA,BRK.B \
  --force \
  --regenerate
```

### Model Monitoring

Track these metrics in production:

1. **Accuracy Degradation**: Monitor live trading accuracy vs training
2. **Feature Drift**: Watch for changes in feature distributions  
3. **Regime Shifts**: Ensure HMM detects new market conditions
4. **Class Imbalance**: Monitor buy/sell/hold decision ratios

## Next Steps

After successful training:

1. **Run Backtests**: Test models on historical data
2. **Paper Trading**: Simulate live trading with current models
3. **Performance Analysis**: Compare ML vs baseline performance
4. **Model Updates**: Schedule regular retraining
5. **Feature Engineering**: Experiment with additional features

For operational guidance, see the [Operations Guide](OPERATIONS_GUIDE.md).