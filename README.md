# AI Hedge Fund with Adaptive Market Intelligence

An advanced multi-agent trading system featuring:
- **Adaptive Regime Detection**: HMM-based market state classification (bull/bear/neutral)
- **ML Ensemble Signal Fusion**: Machine learning models that combine agent signals
- **Dynamic Parameter Optimization**: Market regime-aware thresholds
- **Multi-Provider LLM Support**: Google Gemini, OpenAI, Anthropic, or non-LLM mode

## Key Improvements (v2.0)
- ✅ **Phase 1**: Adaptive parameter system with market regime detection
- ✅ **Phase 2**: ML ensemble signal fusion with performance tracking
- 🚧 **Phase 3**: Advanced risk management (coming soon)
- 🚧 **Phase 4**: Multi-asset expansion (futures support planned)

## System Architecture

The system combines traditional investment analysis with cutting-edge machine learning:

### Core Agents
- **Warren Buffett Agent** - Value-oriented fundamental analysis
- **Technical Analysis Agent** - Chart patterns and technical indicators
- **Sentiment Agent** - Market sentiment and news analysis
- **Risk Manager** - Position sizing and risk controls
- **Portfolio Manager** - Final trading decisions and order execution

### Intelligence Layer
- **Regime Detector** - Hidden Markov Models classify market conditions
- **ML Ensemble** - Gradient boosting and random forest models per regime
- **Feature Engineering** - Agent confidence scores, agreement metrics, market features
- **Performance Tracking** - Real-time model accuracy and signal quality monitoring

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Prerequisites

- Python 3.12+
- Poetry for dependency management
- At least one API key (optional but recommended):
  - **Financial Datasets API** (required for market data)
  - **Google Gemini API** (GOOGLE_API_KEY) - recommended
  - **OpenAI API** (OPENAI_API_KEY)
  - **Anthropic API** (ANTHROPIC_API_KEY)

### Free Tier Tickers (no API credits required)
- AAPL, MSFT, NVDA (default)
- BRK.B, GOOGL, TSLA

## Quick Start

### 1. Installation
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
```

### 2. API Configuration
Create a `.env` file:
```bash
# Required for market data
FINANCIAL_DATASETS_API_KEY=your_key_here

# Optional LLM provider (choose one)
GOOGLE_API_KEY=your_gemini_key_here
# OR
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_claude_key_here
```

### 3. Train ML Models
```bash
# Generate training data and train models
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-05-01 \
  --end-date 2025-08-26 \
  --tickers AAPL,MSFT,NVDA
```

### 4. Run Backtest
```bash
# Run backtest with ML ensemble
poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --analysts technicals
```

## Table of Contents
- [Setup Guide](SETUP_GUIDE.md) - Detailed installation steps
- [Training Guide](TRAINING_GUIDE.md) - ML model training workflow
- [Operations Guide](OPERATIONS_GUIDE.md) - Daily operations and monitoring
- [Contributing](#contributing)
- [License](#license)

## Performance Results

### ML Ensemble vs Baseline Performance
Based on recent backtesting with 300+ training samples:

| Metric | Baseline | ML Ensemble | Improvement |
|--------|----------|-------------|-------------|
| **Bull Market Accuracy** | ~50% | **68.0%** | +18.0% |
| **Bear Market Accuracy** | ~45% | **80.5%** | +35.5% |
| **Neutral Market Accuracy** | ~52% | **80.6%** | +28.6% |
| **Overall Performance** | 51.7% | **80.3%** | **+28.6%** |

### Key Features

#### ✅ Adaptive Regime Detection
- Hidden Markov Models classify market conditions (bull/bear/neutral)
- Dynamic parameter adjustment based on market state
- Real-time regime probability tracking

#### ✅ ML Ensemble Signal Fusion
- Separate models trained for each market regime
- Feature engineering from agent confidence scores
- Cross-validation with stratified splits for robustness

#### ✅ Multi-Provider LLM Support
- **Google Gemini** (gemini-2.0-flash-exp) - Recommended
- **OpenAI GPT** (gpt-4)
- **Anthropic Claude** (claude-3-sonnet)
- **Non-LLM mode** (technical analysis only)

#### ✅ Enhanced Data Generation
- 300+ balanced training samples
- Realistic buy/sell/hold distributions per regime
- Automatic fallback when backtester fails

## Advanced Usage

### Model Training Options
```bash
# Extended training period (recommended)
poetry run python scripts/train_signal_ensemble.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-26 \
  --force

# Using specific tickers
poetry run python scripts/train_signal_ensemble.py \
  --tickers AAPL,MSFT,NVDA,GOOGL,TSLA \
  --regenerate

# Generate sample data for testing
poetry run python scripts/train_signal_ensemble.py --generate-sample
```

### Backtesting Options
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA main
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

You can also specify a `--show-reasoning` flag to print the reasoning of each agent to the console.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --show-reasoning main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --show-reasoning main
```

You can optionally specify the start and end dates to make decisions for a specific time period.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

#### Running the Backtester (with Poetry)
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

#### Running the Backtester (with Docker)
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA backtest
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


You can optionally specify the start and end dates to backtest over a specific time period.

```bash
# With Poetry:
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest
```

You can also specify a `--ollama` flag to run the backtester using local LLMs.
```bash
# With Poetry:
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --ollama

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --ollama backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --ollama backtest
```

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. **This is recommended for most users, especially those who prefer visual interfaces over command line tools.**

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />

#### For Mac/Linux:
```bash
cd app && ./run.sh
```

If you get a "permission denied" error, run this first:
```bash
cd app && chmod +x run.sh && ./run.sh
```

#### For Windows:
```bash
# Go to /app directory
cd app

# Run the app
\.run.bat
```

**That's it!** These scripts will:
1. Check for required dependencies (Node.js, Python, Poetry)
2. Install all dependencies automatically  
3. Start both frontend and backend services
4. **Automatically open your web browser** to the application


#### Detailed Setup Instructions

For detailed setup instructions, troubleshooting, and advanced configuration options, see:
- [Full-Stack App Documentation](./app/README.md)
- [Frontend Documentation](./app/frontend/README.md)  
- [Backend Documentation](./app/backend/README.md)


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
