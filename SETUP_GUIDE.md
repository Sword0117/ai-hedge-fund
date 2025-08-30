# Setup Guide

Complete step-by-step installation guide for the AI Hedge Fund system.

## Prerequisites

Before starting, ensure you have these installed on your system:

- **Python 3.12+** - Required for the trading system
- **Poetry** - For dependency management
- **Git** - For cloning the repository
- **Node.js 18+** (Optional) - Only if using the web application

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: Minimum 8GB, recommended 16GB for ML training
- **Storage**: At least 2GB free space for dependencies and model data
- **Internet Connection**: Required for market data and API access

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Install Python Dependencies

Using Poetry (recommended):
```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

Alternative using pip:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required for market data
FINANCIAL_DATASETS_API_KEY=your_key_here

# Choose ONE LLM provider (recommended: Google Gemini)
GOOGLE_API_KEY=your_gemini_key_here

# Alternative providers (choose one):
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_claude_key_here
# GROQ_API_KEY=your_groq_key_here
```

#### Getting API Keys

**Financial Datasets API** (Required):
1. Visit [Financial Datasets](https://financialdatasets.ai)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Free tier includes: AAPL, MSFT, NVDA, BRK.B, GOOGL, TSLA

**Google Gemini API** (Recommended):
1. Visit [Google AI Studio](https://aistudio.google.com)
2. Create a new project or select existing
3. Generate API key in the API keys section
4. Free tier: 15 requests per minute

**Alternative LLM Providers**:
- **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Visit [Anthropic Console](https://console.anthropic.com)
- **Groq**: Visit [Groq Console](https://console.groq.com)

### 4. Verify Installation

Test the basic setup:

```bash
# Check Python environment
poetry run python --version

# Verify dependencies
poetry run python -c "import pandas, numpy, scikit_learn; print('Core dependencies OK')"

# Test API connectivity
poetry run python -c "from src.llm.models import detect_llm_provider; print(detect_llm_provider())"
```

### 5. Initial Model Training

Before running backtests, train the ML ensemble models:

```bash
# Generate training data and train models (this may take 10-15 minutes)
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-05-01 \
  --end-date 2025-08-26 \
  --tickers AAPL,MSFT,NVDA
```

Expected output:
```
INFO: Starting ML signal ensemble training...
INFO: Generating enhanced training data...
INFO: Generated 300+ training samples across regimes
INFO: Training regime-specific models...
SUCCESS: Bull market model accuracy: 68.0%
SUCCESS: Bear market model accuracy: 80.5%
SUCCESS: Neutral market model accuracy: 80.6%
SUCCESS: Models saved to data/models/
```

## Verification Tests

### Test 1: Quick Backtest

Run a simple backtest to verify everything works:

```bash
poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --analysts technicals
```

### Test 2: Single Day Trading Decision

Test the main trading system:

```bash
poetry run python src/main.py --tickers AAPL,MSFT,NVDA
```

You should see:
1. Analyst selection menu
2. LLM model selection menu
3. Trading analysis output with buy/sell/hold decisions

### Test 3: Web Application (Optional)

If you want to use the web interface:

```bash
# Navigate to app directory
cd app

# On Windows:
run.bat

# On macOS/Linux:
chmod +x run.sh
./run.sh
```

## Troubleshooting

### Common Issues

**1. Poetry not found**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
# Add to PATH (restart terminal after)
export PATH="$HOME/.local/bin:$PATH"
```

**2. Python version issues**
```bash
# Check Python version
python --version
# Should be 3.12 or higher

# If using pyenv:
pyenv install 3.12.0
pyenv local 3.12.0
```

**3. API key errors**
```bash
# Verify .env file exists and has correct format
cat .env
# Check API key validity
poetry run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('FINANCIAL_DATASETS_API_KEY:', bool(os.getenv('FINANCIAL_DATASETS_API_KEY')))"
```

**4. Training fails with insufficient data**
```bash
# Generate sample data for testing
poetry run python scripts/train_signal_ensemble.py --generate-sample

# Or use extended date range
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --force
```

**5. Windows-specific issues**
```bash
# If you see Unicode errors, set environment variable:
set PYTHONIOENCODING=utf-8

# Or use PowerShell instead of Command Prompt
```

**6. Memory issues during training**
```bash
# Reduce training data size
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-06-01 \
  --end-date 2024-12-31 \
  --tickers AAPL,MSFT  # Use fewer tickers
```

### Performance Optimization

**For faster training:**
- Use SSD storage for the project directory
- Close unnecessary applications during training
- Use fewer tickers for initial testing

**For better accuracy:**
- Use extended date ranges (12+ months)
- Include more diverse tickers
- Ensure API rate limits aren't hit during data generation

### Getting Help

If you encounter issues:

1. **Check logs**: Most errors include detailed logging
2. **Verify dependencies**: Run `poetry install` again
3. **Check API quotas**: Ensure you haven't exceeded rate limits
4. **Update dependencies**: Run `poetry update`
5. **File issues**: Report bugs at [GitHub Issues](https://github.com/virattt/ai-hedge-fund/issues)

## Next Steps

Once installation is complete:

1. **Review the [Training Guide](TRAINING_GUIDE.md)** for ML ensemble workflow
2. **Read the [Operations Guide](OPERATIONS_GUIDE.md)** for daily usage
3. **Run your first backtest** with historical data
4. **Experiment with different analysts** and time periods

## System Architecture Overview

After installation, your system includes:

- **Core Agents**: Warren Buffett, Technical Analysis, Sentiment Analysis
- **ML Ensemble**: Gradient boosting and random forest models per regime
- **Regime Detection**: Hidden Markov Models for market state classification
- **Risk Management**: Dynamic position sizing and risk controls
- **Data Pipeline**: Automated training data generation and model updates

Ready to start trading analysis!