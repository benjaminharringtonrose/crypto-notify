# FeatureCalculator Class Documentation

## Overview

The `FeatureCalculator` class is a comprehensive technical analysis engine that computes over 30+ technical indicators and pattern recognition features for cryptocurrency trading. It serves as the foundation for feature engineering in the Cardano trading system, providing both basic technical indicators and advanced pattern detection algorithms.

## Architecture

The FeatureCalculator follows a modular architecture with three main categories of functionality:

```
Technical Indicators → Pattern Detection → Feature Computation
     ↓                      ↓                    ↓
  RSI, MACD, SMA    Chart Patterns    Combined Feature Vector
  ATR, VWAP, OBV    Triple Top/Bottom  (ADA: 32 features, BTC: 29 features)
```

### Key Responsibilities

- **Technical Indicators**: Calculate standard and custom technical indicators
- **Pattern Recognition**: Detect chart patterns like head & shoulders, triple tops/bottoms
- **Feature Engineering**: Combine indicators into normalized feature vectors
- **Data Processing**: Handle price and volume data with proper validation
- **Performance Optimization**: Efficient calculations with minimal memory overhead

## Class Structure

```typescript
export default class FeatureCalculator {
  // Public methods for indicator calculation
  public calculateRSI(prices: number[], period?: number): number;
  public calculateMACD(prices: number[]): {
    macdLine: number;
    signalLine: number;
  };
  public calculateATR(prices: number[], period?: number): number;

  // Pattern detection methods
  public detectHeadAndShoulders(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean;
  public detectTripleTop(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean;
  public detectTripleBottom(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean;

  // Main computation methods
  public calculateIndicators(params: CalculateIndicatorsParams): Indicators;
  public compute(params: ComputeParams): number[];
}
```

## Core Interfaces

### ComputeParams

```typescript
interface ComputeParams {
  prices: number[]; // Historical price array
  volumes: number[]; // Historical volume array
  dayIndex: number; // Current time index
  currentPrice: number; // Current price
  isBTC: boolean; // Whether processing BTC data
  btcPrice?: number; // BTC price for ADA calculations
}
```

### CalculateIndicatorsParams

```typescript
interface CalculateIndicatorsParams {
  prices: number[]; // Historical price array
  volumes: number[]; // Historical volume array
  dayIndex: number; // Current time index
  currentPrice: number; // Current price
}
```

## Technical Indicators

### 1. Trend Indicators

#### Simple Moving Average (SMA)

```typescript
public calculateSMA(prices: number[]): number
```

- **Purpose**: Identifies trend direction and support/resistance levels
- **Calculation**: Arithmetic mean of prices over specified period
- **Usage**: Trend confirmation, dynamic support/resistance

#### Exponential Moving Average (EMA)

```typescript
public calculateEMA(prices: number[], period: number): number
```

- **Purpose**: Smooth trend indicator with more weight on recent prices
- **Calculation**: Weighted average with exponential decay
- **Usage**: MACD calculation, trend momentum

#### MACD (Moving Average Convergence Divergence)

- **Components**: MACD Line, Signal Line
- **Purpose**: Trend momentum and potential reversal signals
- **Calculation**: Difference between 12-period and 26-period EMAs

### 2. Momentum Indicators

#### Relative Strength Index (RSI)

```typescript
public calculateRSI(prices: number[], period?: number): number
```

- **Purpose**: Identifies overbought/oversold conditions
- **Range**: 0-100 (0 = oversold, 100 = overbought)
- **Thresholds**: < 30 (oversold), > 70 (overbought)
- **Calculation**: Average gains vs average losses over period

#### Stochastic RSI

```typescript
public calculateStochRSI(prices: number[], rsiPeriod?: number, stochPeriod?: number, smoothPeriod?: number): { stochRsi: number; signal: number }
```

- **Purpose**: RSI momentum oscillator
- **Components**: %K (stochRsi), %D (signal)
- **Usage**: Divergence detection, momentum confirmation

### 3. Volatility Indicators

#### Average True Range (ATR)

```typescript
public calculateATR(prices: number[], period?: number): number
```

- **Purpose**: Measures market volatility
- **Calculation**: Average of true ranges over period
- **Usage**: Stop-loss placement, volatility-based position sizing

#### Bollinger Bands

- **Components**: Upper Band, Lower Band, Middle Band (SMA)
- **Purpose**: Volatility and potential reversal signals
- **Calculation**: SMA ± (2 × Standard Deviation)

### 4. Volume Indicators

#### Volume Weighted Average Price (VWAP)

```typescript
public calculateVWAP(prices: number[], volumes: number[], period?: number): number
```

- **Purpose**: Fair value price based on volume
- **Calculation**: Σ(Price × Volume) / Σ(Volume)
- **Usage**: Intraday trading, institutional activity

#### On-Balance Volume (OBV)

- **Purpose**: Volume-based trend confirmation
- **Calculation**: Cumulative volume with price direction
- **Usage**: Divergence detection, trend strength

#### Volume Oscillator

- **Purpose**: Volume trend analysis
- **Calculation**: (Short-term Volume SMA - Long-term Volume SMA) / Long-term Volume SMA
- **Usage**: Volume confirmation, trend strength

### 5. Support/Resistance Indicators

#### Fibonacci Retracement Levels

```typescript
public calculateFibonacciLevels(prices: number[], period?: number): { levels: number[]; high: number; low: number }
```

- **Levels**: 23.6%, 38.2%, 50%, 61.8%, 100%
- **Purpose**: Potential support/resistance levels
- **Usage**: Entry/exit points, risk management

## Pattern Detection

### 1. Reversal Patterns

#### Head and Shoulders

```typescript
public detectHeadAndShoulders(prices: number[], volumes: number[], currentPrice: number): boolean
```

- **Pattern**: Three peaks with middle peak highest
- **Signal**: Bearish reversal
- **Confirmation**: Volume analysis, neckline break

#### Double Top

```typescript
public detectDoubleTop(prices: number[], volumes: number[], currentPrice: number): boolean
```

- **Pattern**: Two similar peaks at resistance level
- **Signal**: Bearish reversal
- **Confirmation**: Volume spike at second peak

#### Triple Top

```typescript
public detectTripleTop(prices: number[], volumes: number[], currentPrice: number): boolean
```

- **Pattern**: Three peaks at resistance level
- **Signal**: Strong bearish reversal
- **Confirmation**: Volume analysis, support break

### 2. Continuation Patterns

#### Triple Bottom

```typescript
public detectTripleBottom(prices: number[], volumes: number[], currentPrice: number): boolean
```

- **Pattern**: Three similar troughs at support level
- **Signal**: Bullish reversal
- **Confirmation**: Volume spike at breakout

## Feature Computation

### Main Compute Method

```typescript
public compute(params: ComputeParams): number[]
```

The `compute` method orchestrates the entire feature engineering pipeline:

1. **Data Validation**: Ensures input data integrity
2. **Indicator Calculation**: Computes all technical indicators
3. **Feature Assembly**: Combines indicators into feature vector
4. **Normalization**: Applies scaling and normalization

### Feature Vector Structure

#### ADA Features (32 total)

```typescript
[
  // Momentum Indicators
  rsi,
  prevRsi,
  stochRsi,
  prevStochRsi,

  // Trend Indicators
  sma7,
  sma21,
  prevSma7,
  prevSma21,
  macdLine,
  signalLine,
  prevMacdLine,

  // Price Information
  currentPrice,
  prevPrice,
  vwap,

  // Volatility Indicators
  atr,
  atrBaseline,
  upperBand,
  zScore,

  // Volume Indicators
  obv,
  volumeOscillator,
  prevVolumeOscillator,
  isVolumeSpike,

  // Pattern Detection
  isDoubleTop,
  isHeadAndShoulders,
  isTripleTop,
  isTripleBottom,

  // Advanced Features
  momentum,
  priceChangePct,
  volAdjustedMomentum,
  adxProxy,
  trendRegime,
  btcRatio,
];
```

#### BTC Features (29 total)

```typescript
[
  // Same as ADA minus:
  // - isTripleBottom
  // - adxProxy
  // - trendRegime
  // - btcRatio
];
```

## Usage Examples

### Basic Indicator Calculation

```typescript
import FeatureCalculator from "./FeatureCalculator";

const calculator = new FeatureCalculator();

// Calculate RSI
const rsi = calculator.calculateRSI(prices, 14);

// Calculate MACD
const macd =
  calculator.calculateEMA(prices, 12) - calculator.calculateEMA(prices, 26);

// Calculate ATR for volatility
const atr = calculator.calculateATR(prices, 14);
```

### Pattern Detection

```typescript
// Detect chart patterns
const isHeadAndShoulders = calculator.detectHeadAndShoulders(
  prices,
  volumes,
  currentPrice
);
const isTripleTop = calculator.detectTripleTop(prices, volumes, currentPrice);
const isTripleBottom = calculator.detectTripleBottom(
  prices,
  volumes,
  currentPrice
);
```

### Complete Feature Computation

```typescript
// For ADA
const adaFeatures = calculator.compute({
  prices: adaPrices,
  volumes: adaVolumes,
  dayIndex: currentIndex,
  currentPrice: currentAdaPrice,
  isBTC: false,
  btcPrice: currentBtcPrice,
});

// For BTC
const btcFeatures = calculator.compute({
  prices: btcPrices,
  volumes: btcVolumes,
  dayIndex: currentIndex,
  currentPrice: currentBtcPrice,
  isBTC: true,
});
```

### Advanced Indicator Analysis

```typescript
// Get comprehensive indicators
const indicators = calculator.calculateIndicators({
  prices: prices,
  volumes: volumes,
  dayIndex: dayIndex,
  currentPrice: currentPrice,
});

// Access specific indicators
const { rsi, macdLine, signalLine, atr, vwap } = indicators;
```

## Performance Considerations

### Optimization Strategies

- **Efficient Looping**: Single-pass calculations where possible
- **Memory Management**: Minimal array copying and temporary storage
- **Early Returns**: Validation checks prevent unnecessary computations
- **Period Validation**: Ensures sufficient data before calculation

### Computational Complexity

- **Basic Indicators**: O(n) where n is the period length
- **Pattern Detection**: O(n) for single-pass algorithms
- **Feature Computation**: O(n) for complete feature vector

## Error Handling

### Data Validation

- **Array Length Checks**: Ensures sufficient data for calculations
- **Index Bounds**: Validates dayIndex within array bounds
- **Null Safety**: Graceful handling of missing or invalid data
- **Default Values**: Returns sensible defaults for edge cases

### Fallback Mechanisms

- **Insufficient Data**: Returns last known value or default
- **Division by Zero**: Handles edge cases in ratio calculations
- **Invalid Periods**: Uses default periods when specified periods are invalid

## Dependencies

### External Dependencies

- **Types**: `Indicators` interface from types.ts
- **Constants**: `MODEL_CONFIG`, `PERIODS` from constants.ts

### Internal Dependencies

- **Mathematical Functions**: Built-in Math functions for calculations
- **Array Methods**: Native JavaScript array methods for data manipulation

## Testing

### Unit Testing Strategy

- **Individual Indicators**: Test each technical indicator separately
- **Pattern Detection**: Validate pattern recognition algorithms
- **Edge Cases**: Test with minimal data, extreme values
- **Performance Tests**: Large dataset processing capabilities

### Test Data Requirements

- **Historical Data**: Real cryptocurrency price/volume data
- **Edge Cases**: Minimal periods, extreme price movements
- **Pattern Examples**: Known chart patterns for validation

## Future Enhancements

### Potential Improvements

- **Additional Indicators**: Williams %R, Commodity Channel Index (CCI)
- **Machine Learning**: Dynamic feature selection and importance ranking
- **Real-time Processing**: Streaming data support for live trading
- **Custom Periods**: User-defined calculation periods
- **Performance Metrics**: Calculation time and memory usage tracking

### Advanced Features

- **Multi-timeframe Analysis**: Support for different time intervals
- **Custom Pattern Recognition**: User-defined pattern detection rules
- **Statistical Analysis**: Advanced statistical measures and distributions
- **Backtesting Integration**: Historical performance analysis tools
