# Operations Guide

Daily operations, monitoring, and best practices for running the AI Hedge Fund system in production.

## Overview

This guide covers the operational aspects of running the AI Hedge Fund system, including daily workflows, monitoring, maintenance, and troubleshooting procedures.

## Daily Operations Workflow

### Morning Routine (Market Open)

**1. System Health Check** (2-3 minutes)
```bash
# Check model status
poetry run python -c "
import os
from pathlib import Path
model_dir = Path('data/models')
if model_dir.exists():
    models = list(model_dir.glob('*.pkl'))
    print(f'✓ Found {len(models)} trained models')
    metadata = model_dir / 'model_metadata.json'
    if metadata.exists():
        import json
        with open(metadata) as f:
            data = json.load(f)
        print(f'✓ Model version: {data[\"version\"]}')
        print(f'✓ Training date: {data[\"training_date\"]}')
        print(f'✓ Training samples: {data[\"training_samples\"]}')
    else:
        print('⚠ Model metadata missing')
else:
    print('✗ No trained models found - run training first')
"

# Check API connectivity
poetry run python -c "
from src.llm.models import detect_llm_provider
model, provider = detect_llm_provider()
if model:
    print(f'✓ LLM Provider: {provider} ({model})')
else:
    print('⚠ No LLM provider available - using technical analysis only')
"
```

**2. Market Data Validation** (1-2 minutes)
```bash
# Test market data access
poetry run python -c "
from src.data.financial_datasets import FinancialDatasetsService
import os
from datetime import datetime, timedelta

service = FinancialDatasetsService()
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

try:
    data = service.get_price_data(['AAPL'], start_date, end_date)
    if not data.empty:
        print(f'✓ Market data access OK - {len(data)} records retrieved')
        print(f'✓ Latest data: {data.index[-1].strftime(\"%Y-%m-%d\")}')
    else:
        print('⚠ No market data returned')
except Exception as e:
    print(f'✗ Market data error: {e}')
"
```

### Live Trading Analysis

**Single Stock Analysis**:
```bash
# Quick analysis for one stock
poetry run python src/main.py \
  --tickers AAPL \
  --show-reasoning
```

**Portfolio Analysis**:
```bash
# Full portfolio analysis  
poetry run python src/main.py \
  --tickers AAPL,MSFT,NVDA \
  --show-reasoning
```

**Expected Output Format**:
```json
{
  "decisions": {
    "AAPL": {
      "action": "buy",
      "confidence": 0.85,
      "reasoning": "Strong technical indicators with ML ensemble confidence 85%"
    }
  },
  "market_regime": "bull",
  "regime_confidence": 0.73
}
```

### End-of-Day Review

**1. Performance Tracking** (5 minutes)
```bash
# Run backtest for today's decisions
poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA \
  --start-date $(date -d "1 day ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --analysts technicals

# Review decision quality
grep -E "(BUY|SELL|HOLD)" logs/backtest_$(date +%Y%m%d).log || echo "No trading log found"
```

**2. Model Performance Check**
```bash
# Check if retraining is needed
poetry run python -c "
import json
from datetime import datetime, timedelta
from pathlib import Path

metadata_file = Path('data/models/model_metadata.json')
if metadata_file.exists():
    with open(metadata_file) as f:
        data = json.load(f)
    
    training_date = datetime.fromisoformat(data['training_date'].replace('Z', '+00:00'))
    days_old = (datetime.now() - training_date.replace(tzinfo=None)).days
    
    print(f'Model age: {days_old} days')
    if days_old > 30:
        print('⚠ Consider retraining - model is >30 days old')
    elif days_old > 14:
        print('ℹ Model is >14 days old - monitor performance closely')
    else:
        print('✓ Model is current')
else:
    print('✗ No model metadata found')
"
```

## Weekly Operations

### Model Performance Review

**Monday Morning Review** (15 minutes):
```bash
# Run extended backtest for previous week
START_DATE=$(date -d "1 week ago" +%Y-%m-%d)
END_DATE=$(date -d "1 day ago" +%Y-%m-%d)

poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --analysts technicals > weekly_performance_$(date +%Y%m%d).txt

# Review results
tail -20 weekly_performance_$(date +%Y%m%d).txt
```

### System Maintenance

**Data Cleanup**:
```bash
# Clean old cache files (>30 days)
find data/cache -name "*.json" -mtime +30 -delete 2>/dev/null || echo "No old cache files found"

# Clean old log files (>90 days) 
find logs -name "*.log" -mtime +90 -delete 2>/dev/null || echo "No old log files found"

# Optimize data storage
du -sh data/ && echo "Current data directory size"
```

**Dependency Updates**:
```bash
# Check for dependency updates (monthly)
poetry show --outdated

# Update non-breaking dependencies
poetry update --dry-run
```

## Monthly Operations

### Model Retraining

**Full Model Retrain** (20-30 minutes):
```bash
# Get current date for training range
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "12 months ago" +%Y-%m-%d)

echo "Retraining models with data from $START_DATE to $END_DATE"

# Backup current models
mkdir -p data/models/backup/$(date +%Y%m%d)
cp data/models/*.pkl data/models/backup/$(date +%Y%m%d)/ 2>/dev/null || echo "No existing models to backup"

# Retrain with latest data
poetry run python scripts/train_signal_ensemble.py \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA,BRK.B \
  --force \
  --regenerate

echo "Model retraining completed. Check performance metrics above."
```

### Performance Analysis

**Monthly Performance Report**:
```bash
# Generate comprehensive backtest report
START_DATE=$(date -d "1 month ago" +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)

poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --analysts technicals > monthly_report_$(date +%Y%m).txt

# Extract key metrics
echo "=== MONTHLY PERFORMANCE SUMMARY ===" >> monthly_report_$(date +%Y%m).txt
grep -E "(Total Return|Sharpe Ratio|Max Drawdown|Win Rate)" monthly_report_$(date +%Y%m).txt || echo "Standard metrics not found"
```

## Monitoring and Alerts

### Key Performance Indicators (KPIs)

**Daily Monitoring**:
- Model prediction accuracy (target: >75%)
- API response times (target: <2 seconds)
- Data freshness (target: <24 hours old)
- System uptime (target: >99.9%)

**Weekly Monitoring**:
- Trading decision distribution (buy/hold/sell ratios)
- Regime detection accuracy 
- Feature importance stability
- Memory and CPU usage trends

**Monthly Monitoring**:
- Overall portfolio performance vs benchmarks
- Model accuracy degradation over time
- Feature drift detection
- API cost analysis

### Automated Health Checks

**Create monitoring script** (`scripts/health_check.py`):
```python
#!/usr/bin/env python3
"""Daily system health check"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

def check_models():
    """Check model availability and freshness"""
    model_dir = Path('data/models')
    if not model_dir.exists():
        return False, "Model directory not found"
    
    models = list(model_dir.glob('*.pkl'))
    if len(models) < 4:  # Expecting 4 core models
        return False, f"Only {len(models)} models found, expected 4+"
        
    metadata_file = model_dir / 'model_metadata.json'
    if not metadata_file.exists():
        return False, "Model metadata missing"
        
    with open(metadata_file) as f:
        data = json.load(f)
    
    training_date = datetime.fromisoformat(data['training_date'])
    days_old = (datetime.now() - training_date.replace(tzinfo=None)).days
    
    if days_old > 60:
        return False, f"Models are {days_old} days old - retraining recommended"
    elif days_old > 30:
        return True, f"Models are {days_old} days old - consider retraining"
    
    return True, f"Models are current ({days_old} days old)"

def check_api_keys():
    """Check API key availability"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ['FINANCIAL_DATASETS_API_KEY']
    optional_keys = ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    
    missing_required = [key for key in required_keys if not os.getenv(key)]
    available_optional = [key for key in optional_keys if os.getenv(key)]
    
    if missing_required:
        return False, f"Missing required API keys: {missing_required}"
    
    if not available_optional:
        return True, "No LLM API keys found - will use technical analysis only"
    
    return True, f"API keys OK - LLM providers: {len(available_optional)}"

if __name__ == "__main__":
    print("=== AI Hedge Fund Health Check ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check models
    model_ok, model_msg = check_models()
    print(f"Models: {'✓' if model_ok else '✗'} {model_msg}")
    
    # Check API keys
    api_ok, api_msg = check_api_keys()
    print(f"API Keys: {'✓' if api_ok else '✗'} {api_msg}")
    
    # Overall status
    if model_ok and api_ok:
        print("✓ System healthy")
        sys.exit(0)
    else:
        print("⚠ System issues detected")
        sys.exit(1)
```

**Add to daily cron job** (Linux/Mac):
```bash
# Run health check every day at 8 AM
0 8 * * * cd /path/to/ai-hedge-fund && poetry run python scripts/health_check.py >> logs/health_check.log 2>&1
```

## Troubleshooting Common Issues

### Performance Issues

**1. Slow Trading Analysis**
```bash
# Check system resources
top -p $(pgrep -f "python.*main.py")

# Profile memory usage
poetry run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Solution: Reduce ticker count or restart system
```

**2. API Rate Limiting**
```bash
# Check API usage
poetry run python -c "
import os
from datetime import datetime
print(f'Current time: {datetime.now()}')
print('If getting rate limit errors:')
print('- Wait 60 seconds between requests')
print('- Use fewer tickers per analysis') 
print('- Check API provider status page')
"
```

**3. Outdated Market Data**
```bash
# Force refresh market data cache
rm -rf data/cache/market_data/
poetry run python src/main.py --tickers AAPL  # This will rebuild cache
```

### Model Issues

**1. Poor Prediction Accuracy**
```bash
# Check model training date
poetry run python -c "
import json
with open('data/models/model_metadata.json') as f:
    data = json.load(f)
print('Model performance:')
for regime, perf in data['performance'].items():
    print(f'  {regime}: {perf[\"accuracy\"]:.1%}')
"

# If accuracy <60%, retrain models
poetry run python scripts/train_signal_ensemble.py \
  --start-date $(date -d "12 months ago" +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --force
```

**2. Missing Model Files**
```bash
# Check for required model files
ls -la data/models/
echo "Required files:"
echo "- signal_ensemble_bull.pkl"
echo "- signal_ensemble_bear.pkl" 
echo "- signal_ensemble_neutral.pkl"
echo "- regime_detector_hmm.pkl"
echo "- feature_scaler.pkl"
echo "- model_metadata.json"

# If missing, retrain from scratch
poetry run python scripts/train_signal_ensemble.py --generate-sample
```

### Data Issues

**1. Network Connectivity**
```bash
# Test internet connection
curl -I https://api.financialdatasets.ai/v1/health 2>/dev/null && echo "✓ API reachable" || echo "✗ API unreachable"

# Test DNS resolution
nslookup api.financialdatasets.ai || echo "DNS resolution failed"
```

**2. Corrupted Cache**
```bash
# Clear all cached data
rm -rf data/cache/
mkdir -p data/cache/market_data

# Rebuild cache with test data
poetry run python -c "
from src.data.financial_datasets import FinancialDatasetsService
from datetime import datetime, timedelta
service = FinancialDatasetsService()
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
data = service.get_price_data(['AAPL'], start_date, end_date)
print(f'Cache rebuilt: {len(data)} records')
"
```

## Best Practices

### Security

**API Key Management**:
- Never commit `.env` file to version control
- Use environment variables in production
- Rotate API keys quarterly  
- Monitor API usage for unusual patterns

**Data Protection**:
- Backup models before retraining
- Version control model metadata
- Encrypt sensitive configuration files
- Regular security updates

### Performance Optimization

**Resource Management**:
- Monitor memory usage during training
- Close unused Python processes
- Use SSD storage for model files
- Implement data caching strategies

**Scalability**:
- Limit concurrent API requests
- Batch similar operations
- Use async processing for I/O operations
- Implement circuit breaker patterns for APIs

### Operational Excellence

**Documentation**:
- Log all trading decisions with reasoning
- Document model configuration changes
- Maintain operational runbooks
- Track performance metrics over time

**Change Management**:
- Test model changes in staging first
- Implement gradual rollout procedures
- Maintain rollback capabilities
- Document all production changes

### Risk Management

**Monitoring Thresholds**:
- Model accuracy below 65%: Investigate immediately
- API errors >10%: Check connectivity and keys
- Memory usage >80%: Optimize or scale resources
- Training failures: Validate data and dependencies

**Contingency Plans**:
- **API Failure**: Fall back to technical analysis only
- **Model Failure**: Use cached predictions from last successful run  
- **Data Issues**: Use alternative data sources or cached data
- **System Failure**: Implement manual override procedures

## Reporting and Analytics

### Daily Reports

**Automated Daily Summary**:
```bash
# Create daily report template
cat > daily_report_template.txt << EOF
=== AI HEDGE FUND DAILY REPORT ===
Date: $(date +%Y-%m-%d)
Time: $(date +%H:%M:%S)

SYSTEM STATUS:
- Models: [CHECK_MODELS]
- APIs: [CHECK_APIS] 
- Data: [CHECK_DATA]

TRADING DECISIONS:
[TRADING_SUMMARY]

PERFORMANCE:
[PERFORMANCE_METRICS]

NOTES:
[MANUAL_NOTES]
EOF
```

### Weekly Analytics

**Performance Dashboard**:
- Win/loss ratios by stock and regime
- Model confidence vs actual accuracy  
- Feature importance trends
- API usage and cost analysis

### Monthly Business Review

**Executive Summary Metrics**:
- Total portfolio performance vs benchmarks
- Model accuracy trends by regime
- System reliability and uptime
- Operational cost analysis
- Improvement recommendations

## Integration with External Systems

### Portfolio Management Systems

**Export Format**:
```json
{
  "timestamp": "2025-08-30T10:00:00Z",
  "decisions": {
    "AAPL": {"action": "buy", "confidence": 0.85, "shares": 100},
    "MSFT": {"action": "hold", "confidence": 0.67, "shares": 0}
  },
  "market_regime": "bull",
  "risk_level": "moderate"
}
```

### Risk Management Systems

**Risk Metrics Export**:
- Position sizing recommendations
- Maximum drawdown estimates  
- Correlation analysis
- Volatility predictions

### Compliance and Audit

**Audit Trail Requirements**:
- Log all input data sources
- Record model versions used
- Document decision rationale
- Maintain data lineage

## Getting Help

### Support Channels

**Technical Issues**:
1. Check this operations guide
2. Review error logs in `logs/` directory
3. Search GitHub issues: https://github.com/virattt/ai-hedge-fund/issues
4. Create new issue with system information

**Performance Questions**:
1. Run diagnostic scripts provided
2. Compare with baseline performance metrics
3. Review training and operational logs
4. Engage community support

### Escalation Procedures

**Critical Issues** (System Down):
1. Execute immediate rollback procedures
2. Switch to manual trading mode
3. Document incident timeline
4. Engage technical support

**Performance Degradation**:
1. Run full system health check
2. Review recent changes
3. Compare with historical baselines
4. Plan corrective actions

Remember: This system is for educational and research purposes only. Always consult with financial advisors for actual trading decisions.