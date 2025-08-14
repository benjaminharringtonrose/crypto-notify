# FeatureCalculator Class Documentation

## Overview

The `FeatureCalculator` class is a comprehensive technical analysis engine that computes over 60+ technical indicators and pattern recognition features for cryptocurrency trading. It serves as the foundation for feature engineering in the Bitcoin trading system, providing both basic technical indicators and advanced pattern detection algorithms.

## Architecture

The FeatureCalculator follows a modular architecture with three main categories of functionality:

```
Technical Indicators → Pattern Detection → Feature Computation
     ↓                      ↓                    ↓
  RSI, MACD, SMA    Chart Patterns    Combined Feature Vector
  ATR, VWAP, OBV    Triple Top/Bottom  (BTC: 62 features)
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

- **Purpose**: Smooths price data with exponential weighting
- **Calculation**: EMA = Price × k + Previous EMA × (1 - k), where k = 2/(period + 1)
- **Usage**: MACD calculation, trend identification

#### Moving Average Convergence Divergence (MACD)

```typescript
public calculateMACD(prices: number[]): { macdLine: number; signalLine: number }
```

- **Purpose**: Momentum and trend-following indicator
- **Calculation**: MACD Line = EMA12 - EMA26, Signal Line = EMA9 of MACD
- **Usage**: Signal generation, trend confirmation

### 2. Momentum Indicators

#### Relative Strength Index (RSI)

```typescript
public calculateRSI(prices: number[], period?: number): number
```

- **Purpose**: Measures speed and magnitude of price changes
- **Calculation**: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
- **Usage**: Overbought/oversold conditions, divergence signals

#### Stochastic RSI

```typescript
public calculateStochRSI(rsiValues: number[], period?: number): number
```

- **Purpose**: Combines RSI with stochastic oscillator
- **Calculation**: (Current RSI - Lowest RSI) / (Highest RSI - Lowest RSI) × 100
- **Usage**: Momentum confirmation, entry/exit timing

### 3. Volatility Indicators

#### Average True Range (ATR)

```typescript
public calculateATR(prices: number[], period?: number): number
```

- **Purpose**: Measures market volatility
- **Calculation**: Average of true range over specified period
- **Usage**: Stop-loss placement, volatility assessment

#### Bollinger Bands

```typescript
public calculateBollingerBands(prices: number[], period?: number): {
  upper: number;
  middle: number;
  lower: number;
}
```

- **Purpose**: Identifies overbought/oversold conditions
- **Calculation**: Middle = SMA, Upper/Lower = Middle ± (2 × Standard Deviation)
- **Usage**: Support/resistance, volatility expansion/contraction

### 4. Volume Indicators

#### On-Balance Volume (OBV)

```typescript
public calculateOBV(prices: number[], volumes: number[]): number
```

- **Purpose**: Tracks volume flow to confirm price trends
- **Calculation**: Cumulative sum based on price direction
- **Usage**: Trend confirmation, volume analysis

#### Volume-Weighted Average Price (VWAP)

```typescript
public calculateVWAP(prices: number[], volumes: number[], period?: number): number
```

- **Purpose**: Represents average price weighted by volume
- **Calculation**: Σ(Price × Volume) / Σ(Volume)
- **Usage**: Fair value assessment, institutional activity

#### Volume Oscillator

```typescript
public calculateVolumeOscillator(volumes: number[]): number
```

- **Purpose**: Measures volume trend strength
- **Calculation**: ((SMA5 - SMA14) / SMA14) × 100
- **Usage**: Volume momentum, trend confirmation

### 5. Support/Resistance Indicators

#### Fibonacci Retracement

```typescript
public calculateFibonacciLevels(prices: number[], period?: number): {
  fib61_8: number;
  fib50: number;
  fib38_2: number;
}
```

- **Purpose**: Identifies potential support/resistance levels
- **Calculation**: Based on high/low range over specified period
- **Usage**: Entry/exit points, trend continuation

## Pattern Recognition

### 1. Reversal Patterns

#### Head and Shoulders

```typescript
public detectHeadAndShoulders(
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean
```

- **Pattern**: Three peaks with middle peak (head) higher than side peaks (shoulders)
- **Signal**: Bearish reversal pattern
- **Confirmation**: Volume confirmation on neckline break

#### Double Top

```typescript
public detectDoubleTop(
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean
```

- **Pattern**: Two peaks at similar levels with trough between
- **Signal**: Bearish reversal pattern
- **Confirmation**: Price break below trough level

#### Triple Top

```typescript
public detectTripleTop(
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean
```

- **Pattern**: Three peaks at similar levels with troughs between
- **Signal**: Strong bearish reversal pattern
- **Confirmation**: Volume confirmation on breakdown

#### Triple Bottom

```typescript
public detectTripleBottom(
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean
```

- **Pattern**: Three troughs at similar levels with peaks between
- **Signal**: Bullish reversal pattern
- **Confirmation**: Volume confirmation on breakout

### 2. Continuation Patterns

#### Volume Spike Detection

```typescript
public detectVolumeSpike(volumes: number[], threshold?: number): boolean
```

- **Pattern**: Unusual volume increase
- **Signal**: Potential breakout or reversal
- **Usage**: Confirmation of price movements

## Feature Computation

### Main Compute Method

```typescript
public compute(params: ComputeParams): number[]
```

The main computation method combines all indicators into a normalized feature vector:

#### BTC Features (62 total)

```typescript
const baseFeatures = [
  // Core technical indicators
  indicators.rsi,
  indicators.prevRsi,
  indicators.sma7,
  indicators.sma21,
  indicators.prevSma7,
  indicators.prevSma21,
  indicators.macdLine,
  indicators.signalLine,
  indicators.currentPrice,
  indicators.upperBand,
  indicators.lowerBand,
  indicators.obvValues[indicators.obvValues.length - 1] / 1e6,
  indicators.obv,
  indicators.atr,
  indicators.atrBaseline,
  indicators.zScore,
  indicators.vwap,
  indicators.stochRsi,
  indicators.stochRsiSignal,
  indicators.prevStochRsi,
  indicators.fib61_8,
  dayIndex > 0 ? prices[dayIndex - 1] : prices[0],
  indicators.volumeOscillator,
  indicators.prevVolumeOscillator,

  // Pattern detection (binary)
  indicators.isDoubleTop ? 1 : 0,
  indicators.isHeadAndShoulders ? 1 : 0,
  indicators.prevMacdLine,
  indicators.isTripleTop ? 1 : 0,
  indicators.isTripleBottom ? 1 : 0,
  indicators.isVolumeSpike ? 1 : 0,

  // Momentum and trend features
  indicators.momentum,
  indicators.priceChangePct,
  indicators.sma20,
  indicators.volAdjustedMomentum,
  indicators.trendRegime,
  indicators.adxProxy,

  // Market regime features
  indicators.volatilityRegimeScore,
  indicators.trendRegimeScore,
  indicators.momentumRegimeScore,
  indicators.realizedVolatility,
  indicators.regimeScore,

  // Additional technical indicators
  indicators.sma50 || indicators.sma20,
  indicators.sma200 || indicators.sma20,

  // Price-based features
  currentPrice / (indicators.sma20 || currentPrice),
  currentPrice / (indicators.sma50 || currentPrice),
  currentPrice / (indicators.sma200 || currentPrice),

  // Volume-based features
  volumes[dayIndex] /
    (volumes
      .slice(Math.max(0, dayIndex - 20), dayIndex + 1)
      .reduce((a, b) => a + b, 0) /
      Math.min(21, dayIndex + 1)),
  volumes[dayIndex] /
    (volumes
      .slice(Math.max(0, dayIndex - 50), dayIndex + 1)
      .reduce((a, b) => a + b, 0) /
      Math.min(51, dayIndex + 1)),

  // Momentum features
  indicators.momentum / (indicators.atr || 1),
  indicators.priceChangePct / (indicators.atr || 1),

  // RSI-based features
  indicators.rsi / 100,
  indicators.prevRsi / 100,

  // MACD-based features
  indicators.macdLine / (indicators.currentPrice || 1),
  indicators.signalLine / (indicators.currentPrice || 1),

  // Bollinger Bands features
  (indicators.currentPrice - indicators.upperBand) /
    (indicators.upperBand - indicators.lowerBand || 1),
  (indicators.currentPrice - indicators.lowerBand) /
    (indicators.upperBand - indicators.lowerBand || 1),

  // Stochastic features
  indicators.stochRsi / 100,
  indicators.stochRsiSignal / 100,

  // Fibonacci features
  indicators.fib61_8 / (indicators.currentPrice || 1),

  // ATR-based features
  indicators.atr / (indicators.currentPrice || 1),
  indicators.atrBaseline / (indicators.currentPrice || 1),

  // VWAP features
  indicators.vwap / (indicators.currentPrice || 1),
];
```

## Usage Examples

### Basic Indicator Calculation

```typescript
import FeatureCalculator from "./FeatureCalculator";

const calculator = new FeatureCalculator();

// Calculate RSI
const rsi = calculator.calculateRSI(prices, 14);
console.log(`RSI: ${rsi}`);

// Calculate MACD
const macd = calculator.calculateMACD(prices);
console.log(`MACD Line: ${macd.macdLine}, Signal: ${macd.signalLine}`);
```

### Pattern Detection

```typescript
// Detect head and shoulders pattern
const isHeadAndShoulders = calculator.detectHeadAndShoulders(
  prices,
  volumes,
  currentPrice
);

if (isHeadAndShoulders) {
  console.log(
    "Head and Shoulders pattern detected - potential bearish reversal"
  );
}
```

### Complete Feature Computation

```typescript
// Compute all features for BTC
const btcFeatures = calculator.compute({
  prices: btcPrices,
  volumes: btcVolumes,
  dayIndex: currentIndex,
  currentPrice: btcPrices[currentIndex],
  isBTC: true,
});

console.log(`Generated ${btcFeatures.length} features`);
// Output: Generated 62 features
```

## Performance Considerations

### Optimization Strategies

- **Efficient Calculations**: Minimize redundant computations
- **Memory Management**: Reuse arrays and objects where possible
- **Caching**: Cache frequently used indicator values
- **Batch Processing**: Process multiple time periods efficiently

### Computational Complexity

- **Single Indicator**: O(n) where n is the lookback period
- **Pattern Detection**: O(m) where m is the pattern window
- **Feature Computation**: O(k) where k is the number of indicators

## Error Handling

### Data Validation

- **Length Checks**: Ensure sufficient data for calculations
- **Null/Undefined**: Handle missing or invalid data gracefully
- **Boundary Conditions**: Manage edge cases in calculations

### Fallback Mechanisms

- **Default Values**: Provide sensible defaults for missing data
- **Error Recovery**: Continue processing with available data
- **Logging**: Record calculation errors for debugging

## Dependencies

### External Dependencies

- **TypeScript**: Type safety and interfaces
- **Array Methods**: Native JavaScript array operations
- **Mathematical Functions**: Math library for calculations

### Internal Dependencies

- **Constants**: Configuration values from constants.ts
- **Types**: TypeScript interfaces and types
- **Utilities**: Helper functions for calculations

## Testing

### Unit Testing Strategy

- **Individual Indicators**: Test each indicator calculation
- **Pattern Detection**: Validate pattern recognition logic
- **Feature Integration**: Test complete feature computation
- **Edge Cases**: Test boundary conditions and error scenarios

### Test Data Requirements

- **Historical Data**: Real cryptocurrency price/volume data
- **Known Patterns**: Data with confirmed chart patterns
- **Edge Cases**: Minimal data, extreme values, missing data

## Future Enhancements

### Potential Improvements

- **Additional Indicators**: Implement more technical indicators
- **Machine Learning**: Add ML-based pattern recognition
- **Real-time Processing**: Optimize for live data streams
- **Custom Indicators**: Allow user-defined indicator formulas

### Advanced Features

- **Multi-timeframe Analysis**: Support different time intervals
- **Cross-asset Correlation**: Analyze relationships between assets
- **Sentiment Integration**: Incorporate market sentiment data
- **Adaptive Parameters**: Dynamic indicator parameter adjustment
