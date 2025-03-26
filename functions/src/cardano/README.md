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

This system is optimized for **short-term cryptocurrency trading**, specifically **day trading** and **swing trading** on Cardano (ADA). It focuses on capturing price movements over horizons of a few hours to several days, using a 7-day lookahead for labeling training data (configurable in `DataProcessor.ts`). The model combines machine learning predictions with technical analysis to identify entry and exit points based on momentum, reversals, and breakouts.

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

### Feature Calculations and Meanings

The FeatureCalculator class generates a set of technical indicators and features used by the model. Below is a detailed list of the calculations and their meanings:

- **RSI (Relative Strength Index)**:

  - **Calculation**: 100 - (100 / (1 + RS)), where RS = (Average Gain over 14 periods) / (Average Loss over 14 periods). Gains and losses are based on price changes over the period.
  - **Meaning**: Measures momentum and overbought/oversold conditions. RSI < 20 suggests oversold (potential buy), RSI > 80 suggests overbought (potential sell).

- **Previous RSI**:

  - **Calculation**: Same as RSI but calculated on the previous period’s data (shifted back by one day).
  - **Meaning**: Provides context for RSI trend direction.

- **SMA7 (7-day Simple Moving Average)**:

  - **Calculation**: (Sum of prices over 7 days) / 7.
  - **Meaning**: Smooths short-term price fluctuations to identify trends. Compared with longer SMAs to detect crossovers.

- **SMA21 (21-day Simple Moving Average)**:

  - **Calculation**: (Sum of prices over 21 days) / 21.
  - **Meaning**: Indicates medium-term trend direction.

- **Previous SMA7 and SMA21**:

  - **Calculation**: SMA7 and SMA21 calculated on data shifted back by one day.
  - **Meaning**: Used to detect SMA crossovers (e.g., SMA7 crossing above SMA21 signals a bullish trend).

- **MACD Line**:

  - **Calculation**: EMA12 - EMA26, where EMA12 is the 12-day Exponential Moving Average and EMA26 is the 26-day EMA. EMA = Price _ k + Previous EMA _ (1 - k), with k = 2 / (period + 1).
  - **Meaning**: Measures momentum and trend direction. Positive values indicate bullish momentum.

- **Signal Line**:

  - **Calculation**: 9-day EMA of the MACD Line.
  - **Meaning**: Smooths MACD to generate crossover signals. MACD crossing above Signal Line is bullish.

- **Current Price**:

  - **Calculation**: The price at the current day index.
  - **Meaning**: The raw price input for comparison with indicators.

- **Upper Bollinger Band**:

  - **Calculation**: SMA20 + 2 \_ StdDev20, where SMA20 is the 20-day SMA and StdDev20 is the standard deviation over 20 days.
  - **Meaning**: Indicates potential overbought levels or resistance.

- **Lower Bollinger Band**:

  - **Calculation**: SMA20 - 2 \_ StdDev20.
  - **Meaning**: Indicates potential oversold levels or support.

- **OBV (On-Balance Volume)**:

  - **Calculation**: Cumulative sum where volume is added if price increases, subtracted if price decreases, or unchanged if price is flat.
  - **Meaning**: Tracks volume flow to confirm price trends. Rising OBV with price suggests bullishness.

- **ATR (Average True Range)**:

  - **Calculation**: Average of the true range (absolute price change) over 14 days.
  - **Meaning**: Measures volatility. Higher ATR indicates greater price movement.

- **ATR Baseline**:

  - **Calculation**: ATR calculated up to the previous day.
  - **Meaning**: Provides a reference for current volatility trends.

- **Z-Score**:

  - **Calculation**: (Current Price - SMA20) / StdDev20.
  - **Meaning**: Indicates how far the price deviates from the mean in standard deviations. Useful for mean-reversion strategies.

- **VWAP (Volume-Weighted Average Price)**:

  - **Calculation**: (Sum of (Price \_ Volume) over 7 days) / (Sum of Volume over 7 days).
  - **Meaning**: Represents the average price weighted by volume, often a benchmark for fair value.

- **Stochastic RSI**:

  - **Calculation**: ((Current RSI - Lowest RSI over 14 periods) / (Highest RSI over 14 periods - Lowest RSI)) \_ 100.
  - **Meaning**: Combines RSI with stochastic oscillator to identify overbought/oversold conditions with momentum.

- **Previous Stochastic RSI**:

  - **Calculation**: Stochastic RSI calculated on the previous day’s data.
  - **Meaning**: Tracks changes in Stochastic RSI for trend analysis.

- **Fibonacci 61.8% Level**:

  - **Calculation**: Low + (High - Low) \_ 0.618, where High and Low are the max and min prices over 30 days.
  - **Meaning**: A key retracement level often acting as support or resistance.

- **Previous Day’s Price**:

  - **Calculation**: Price from the previous day.
  - **Meaning**: Used to calculate daily changes and momentum.

- **Volume Oscillator**:

  - **Calculation**: ((SMA of Volume over 5 days - SMA over 14 days) / SMA over 14 days) \_ 100.
  - **Meaning**: Measures volume trend strength. Positive values indicate increasing volume momentum.

- **Previous Volume Oscillator**:

  - **Calculation**: Volume Oscillator calculated on the previous day’s data.
  - **Meaning**: Tracks volume momentum changes.

- **Double Top (Binary)**:

  - **Calculation**: Detects two peaks at similar levels with a trough between, followed by a price drop below the trough, with volume confirmation.
  - **Meaning**: Bearish reversal pattern indicating potential sell.

- **Head and Shoulders (Binary)**:

  - **Calculation**: Identifies a peak (head) between two lower peaks (shoulders) with a neckline break, confirmed by volume.
  - **Meaning**: Strong bearish reversal signal.

- **Previous MACD Line**:

  - **Calculation**: MACD Line calculated on the previous day’s data.
  - **Meaning**: Used to detect MACD crossovers.

- **Triple Top (Binary)**:

  - **Calculation**: Detects three peaks at similar levels with troughs, followed by a price drop below support, with volume confirmation.
  - **Meaning**: Strong bearish reversal pattern.

- **Volume Spike (Binary)**:

  - **Calculation**: Current volume > 2 \_ SMA of volume over 5 days.
  - **Meaning**: Indicates significant buying/selling pressure, often confirming breakouts or reversals.

- **Momentum**:

  - **Calculation**: Current Price - Price from 10 days ago.
  - **Meaning**: Measures price velocity. Positive momentum supports buys, negative supports sells.

- **Price Change Percentage**:

  - **Calculation**: ((Current Price - Previous Price) / Previous Price) \_ 100.
  - **Meaning**: Daily price change as a percentage, useful for volatility assessment.

- **Volume-Adjusted Momentum**:

  - **Calculation**: (Current Price - Price from 10 days ago) / ATR.
  - **Meaning**: Normalizes momentum by volatility for more robust signals.

- **Triple Bottom (Binary, ADA only)**:

  - **Calculation**: Detects three similar troughs with decreasing volume, followed by a breakout above resistance.
  - **Meaning**: Bullish reversal pattern indicating potential buy.

- **ADX Proxy (ADA only)**:

  - **Calculation**: |SMA7 - SMA50| / SMA50.
  - **Meaning**: Approximates trend strength. Higher values indicate stronger trends.

- **ADA/BTC Price Ratio (ADA only)**:

  - **Calculation**: Current ADA Price / Current BTC Price.
  - **Meaning**: Measures ADA’s performance relative to BTC, capturing correlation effects.

These features are computed for both ADA (32 features) and BTC (29 features, excluding Triple Bottom, ADX Proxy, and ADA/BTC Ratio), forming a 61-feature input vector per timestep for the model.

## Contributing

1. Fork the repository.
2. Create a branch (git checkout -b feature/your-feature).
3. Commit changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## License

MIT License. See for details.
