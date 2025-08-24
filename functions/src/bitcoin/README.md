# Trade Prediction Model

Welcome to the Trade Prediction Model, a machine learning-driven trading system built with TensorFlow.js to predict trading decisions (Buy, Sell, Hold) for Bitcoin (BTC) cryptocurrency. This system combines a deep learning model with rule-based strategies to analyze historical price and volume data, technical indicators, and market trends, making it suitable for short-term cryptocurrency trading.

## Table of Contents

- [Overview](#overview)
- [Trading Type and Use Cases](#trading-type-and-use-cases)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Training the Model](#training-the-model)
- [Backtesting](#backtesting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Trade Prediction Model is designed to assist traders by providing data-driven trading signals for Bitcoin (BTC), leveraging a hybrid approach that integrates a convolutional neural network (CNN) and long short-term memory (LSTM) layers with rule-based trading strategies (Trend Following, Mean Reversion, Breakout). It processes sequential market data, including BTC prices and volumes, to generate actionable insights. The system supports training, real-time predictions, and historical backtesting, with model weights stored in Firebase Storage.

## Trading Type and Use Cases

### Type of Trading

This system is optimized for **short-term cryptocurrency trading**, specifically **day trading** and **swing trading** on Bitcoin (BTC). It focuses on capturing price movements over horizons of a few hours to several days, using a 7-day lookahead for labeling training data (configurable in `DataProcessor.ts`). The model combines machine learning predictions with technical analysis to identify entry and exit points based on momentum, reversals, and breakouts.

### When to Use

- **Volatile Markets**: The model excels in high-volatility environments typical of cryptocurrencies, where rapid price swings create opportunities.
- **Trend Identification**: Use it when seeking to capitalize on emerging trends (upward or downward) confirmed by technical indicators.
- **Reversion Plays**: Ideal for spotting overbought/oversold conditions (e.g., RSI < 20 or > 75) for mean-reversion trades.
- **Breakout Scenarios**: Deploy during periods of consolidation followed by high volume, signaling potential breakouts.
- **Avoid in Low Volatility**: Less effective in flat, low-volume markets where signals may lack conviction (e.g., ADX proxy < 0.025).
- **Complementary Tool**: Best used alongside human judgment or other market analysis, not as a standalone automated system.

### Example Scenarios

- **Buy**: RSI < 20, momentum > 0.5, and an uptrend (SMA7 > SMA50).
- **Sell**: RSI > 75, negative momentum, and a downtrend (SMA7 < SMA50).
- **Hold**: Neutral conditions (e.g., RSI 25-75, weak trend strength).

## Features

- **Deep Learning Model**: CNN + LSTM architecture for time-series prediction.
- **Technical Indicators**: RSI, MACD, Bollinger Bands, VWAP, ATR, Fibonacci levels, and pattern detection (e.g., Double Top, Head and Shoulders).
- **Trading Strategies**: Combines ML predictions with Trend Following, Mean Reversion, and Breakout logic.
- **Data Processing**: Feature extraction, normalization, and dataset balancing.
- **Backtesting**: Simulates trading performance with metrics like Sharpe Ratio and Max Drawdown.
- **Model Persistence**: Stores and retrieves weights via Firebase Storage.
- **Custom Metrics**: Focal loss, precision, recall, and F1 score for imbalanced data.

## Prerequisites

- **Node.js**: v16.x or higher
- **npm**: v8.x or higher
- **Firebase Account**: For storage and authentication
- **API Access**: Historical price data source (e.g., CoinGecko, Binance API)
- **Dependencies**: Listed in `package.json`

## Installation

1. **Clone the Repository**:

```bash
 git clone https://github.com/yourusername/trade-prediction-model.git
 cd trade-prediction-model
```

2. **Install Dependencies**:

```bash
npm install
```

3. **Set Up Environment Variables: Create a .env file in the root directory**:

```plaintext
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY=your-private-key
FIREBASE_CLIENT_EMAIL=your-client-email
API_KEY=your-api-key-for-price-data
```

Replace placeholders with your Firebase credentials and API key for historical data.

4. **Initialize Firebase**:

Verify Firebase Admin SDK setup in `src/api/FirebaseService.ts`.

## Usage

### Training the Model

```bash
npm run train:btc
```

Trains the model with historical BTC data, saving weights to Firebase Storage. Requires sufficient data (default: 450 days).

### Backtesting

```bash
npm run train:btc:backtest
```

Runs backtesting on historical data to evaluate model performance and trading strategy effectiveness.

### Project Structure

```
bitcoin/
├── callbacks/           # Training callbacks and monitoring
├── DataProcessor.ts     # Data preprocessing and feature engineering
├── FeatureCalculator.ts # Technical indicator calculations
├── FeatureSequenceGenerator.ts # Sequence generation for ML models
├── Metrics.ts          # Performance metrics and evaluation
├── TradeExecutor.ts    # Trade execution logic
├── TradeModelBacktester.ts # Backtesting engine
├── TradeModelFactory.ts # Model architecture and creation
├── TradeModelPredictor.ts # Prediction engine
├── TradeModelTrainer.ts # Training orchestration
├── TradeModelWeightManager.ts # Weight management and persistence
└── TradingStrategy.ts  # Rule-based trading strategies
```

## Key Components

### 1. DataProcessor

Handles data preprocessing, feature extraction, and dataset preparation.

**Key Features:**

- Historical data loading and validation
- Feature engineering and normalization
- Dataset balancing for imbalanced classes
- Training/validation split management

### 2. FeatureCalculator

Computes technical indicators and market patterns.

**Indicators:**

- RSI, MACD, Bollinger Bands
- VWAP, ATR, Fibonacci levels
- Pattern detection (Double Top, Head and Shoulders)
- Volume analysis and momentum indicators

### 3. FeatureSequenceGenerator

Creates time-series sequences for LSTM/RNN models.

**Features:**

- Fixed-length sequence generation
- Batch processing for training
- Feature vector composition (62 BTC features)
- Memory-efficient processing

### 4. TradeModelFactory

Builds and configures the neural network architecture.

**Architecture:**

- CNN layers for feature extraction
- LSTM layers for temporal modeling
- Attention mechanisms for focus
- Dense layers for final predictions

### 5. TradingStrategy

Implements rule-based trading logic.

**Strategies:**

- Trend Following
- Mean Reversion
- Breakout Detection
- Risk management and position sizing

## Training the Model

### Training Process

1. **Data Preparation**: Load historical BTC price and volume data
2. **Feature Engineering**: Compute technical indicators and patterns
3. **Sequence Generation**: Create time-series sequences for training
4. **Model Training**: Train the neural network with validation
5. **Weight Persistence**: Save trained weights to Firebase Storage

### Training Configuration

```typescript
// Model architecture parameters
const MODEL_CONFIG = {
  TIMESTEPS: 30,
  // BTC feature count determined dynamically by FeatureDetector.getFeatureCount()
  LSTM_UNITS_1: 48,
  LSTM_UNITS_2: 24,
  DROPOUT_RATE: 0.35,
  // ... more configuration
};
```

### Training Metrics

- **Loss**: Focal loss for imbalanced data
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Per-class performance
- **F1 Score**: Balanced performance metric

## Backtesting

### Backtesting Engine

The `TradeModelBacktester` simulates historical trading performance:

- **Trade Simulation**: Executes trades based on model predictions
- **Performance Metrics**: Calculates returns, Sharpe ratio, max drawdown
- **Risk Analysis**: Position sizing and risk management
- **Strategy Evaluation**: Compares different trading strategies

### Key Metrics

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Example Backtest Results

```typescript
{
  totalReturn: 0.45,        // 45% total return
  sharpeRatio: 1.2,         // Good risk-adjusted returns
  maxDrawdown: -0.15,       // 15% maximum decline
  winRate: 0.62,            // 62% winning trades
  profitFactor: 1.8,        // 1.8x profit to loss ratio
  totalTrades: 156          // Number of executed trades
}
```

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and test thoroughly**
4. **Submit a pull request**

### Code Standards

- **TypeScript**: Use strict typing and interfaces
- **Documentation**: Update docs for any API changes
- **Testing**: Add unit tests for new features
- **Performance**: Optimize for large datasets

### Testing

```bash
# Run unit tests
npm test

# Run backtesting
npm run backtest

# Run training with validation
npm run train:btc
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This trading system is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## Support

For questions, issues, or contributions:

1. **Check existing issues** on GitHub
2. **Create a new issue** with detailed information
3. **Join discussions** in the project community
4. **Review documentation** for common solutions

---

**Note**: This system is designed for Bitcoin (BTC) trading and has been optimized for short-term cryptocurrency markets. The model architecture and feature engineering are specifically tailored for BTC price movements and market dynamics.
