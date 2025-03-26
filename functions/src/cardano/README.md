# Trade Prediction Model

Welcome to the Trade Prediction Model, a machine learning-driven trading system built with TensorFlow.js to predict trading decisions (Buy, Sell, Hold) for Cardano (ADA) cryptocurrency. This system combines a deep learning model with rule-based strategies to analyze historical price and volume data, technical indicators, and market trends, making it suitable for short-term cryptocurrency trading.

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

The Trade Prediction Model is designed to assist traders by providing data-driven trading signals for Cardano (ADA), leveraging a hybrid approach that integrates a convolutional neural network (CNN) and long short-term memory (LSTM) layers with rule-based trading strategies (Trend Following, Mean Reversion, Breakout). It processes sequential market data, including ADA and Bitcoin (BTC) prices and volumes, to generate actionable insights. The system supports training, real-time predictions, and historical backtesting, with model weights stored in Firebase Storage.

## Trading Type and Use Cases

### Type of Trading

This system is optimized for **short-term cryptocurrency trading**, specifically **day trading** and **swing trading** on Cardano (ADA). It focuses on capturing price movements over horizons of a few hours to several days, using a 5-day lookahead for labeling training data (configurable in `DataProcessor.ts`). The model combines machine learning predictions with technical analysis to identify entry and exit points based on momentum, reversals, and breakouts.

### When to Use

- **Volatile Markets**: The model excels in high-volatility environments typical of cryptocurrencies, where rapid price swings create opportunities.
- **Trend Identification**: Use it when seeking to capitalize on emerging trends (upward or downward) confirmed by technical indicators and BTC correlation.
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
npm run train:ada
```

Trains the model with historical ADA and BTC data, saving weights to Firebase Storage. Requires sufficient data (default: 450 days).

### Backtesting

```bash
npm run train:ada:backtest
```

## Project Structure

```plaintext
cardano/
├── src/
│   ├── api/                # API services (Firebase, price data)
│   ├── callbacks/          # Custom TensorFlow.js callbacks
│   ├── constants.ts          # Constants (e.g., periods, file paths)
│   ├── types.ts              # TypeScript type definitions
│   ├── TradeModelTrainer.ts # Model training logic
│   ├── TradeModelPredictor.ts # Prediction logic
│   ├── TradeModelBacktester.ts # Backtesting logic
│   ├── DataProcessor.ts    # Data preparation and feature extraction
│   ├── FeatureCalculator.ts # Technical indicator calculations
│   ├── FeatureSequenceGenerator.ts # Sequence generation for model input
│   ├── ModelWeightManager.ts # Model weight handling
│   ├── Metrics.ts          # Custom metrics (focal loss, F1, etc.)
│   └── TradeModelFactory.ts # Neural network architecture
├── .env                    # Environment variables
├── package.json            # Dependencies and scripts
└── README.md               # This file
```

## Key Components

### TradeModelTrainer

Trains the CNN-LSTM model with callbacks like Cyclic Learning Rate and Best Weights.

### TradeModelPredictor

Predicts trading decisions using ML and rule-based strategies, refining outputs with market conditions.

### TradeModelBacktester

Simulates trades with risk management (e.g., stop-loss, take-profit) and calculates performance metrics.

### DataProcessor

Prepares and balances training data, labeling based on future price changes.

### FeatureCalculator

Computes technical indicators and patterns for feature generation.

### Metrics

Implements custom metrics optimized for trading signal evaluation.

### TradeModelFactory

Defines the CNN-LSTM architecture with regularization and dropout.

## Contributing

1. Fork the repository.
2. Create a branch (git checkout -b feature/your-feature).
3. Commit changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## License

MIT License. See for details.
