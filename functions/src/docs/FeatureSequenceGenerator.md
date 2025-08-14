# FeatureSequenceGenerator Class Documentation

## Overview

The `FeatureSequenceGenerator` class is a specialized component in the Bitcoin trading system that creates time-series feature sequences for machine learning models. It transforms raw cryptocurrency price and volume data into structured, multi-dimensional feature sequences that can be used for training LSTM/RNN models or other time-series prediction algorithms.

## Architecture

The FeatureSequenceGenerator follows a sequence-oriented architecture:

```
Raw Data Input → Feature Computation → Sequence Assembly → Batch Generation → Output
     ↓                ↓                    ↓                ↓
BTC Prices         FeatureCalculator    Timestep Arrays   Multiple Sequences
BTC Volumes        Technical Indicators  Feature Vectors   Training Dataset
```

### Key Responsibilities

- **Sequence Generation**: Create fixed-length feature sequences from historical data
- **Feature Integration**: Process BTC features into unified feature vectors
- **Batch Processing**: Generate multiple sequences for training datasets
- **Data Validation**: Ensure sequence integrity and proper dimensions
- **Memory Management**: Efficient handling of large datasets

## Class Structure

```typescript
export class FeatureSequenceGenerator {
  private timesteps: number;

  constructor(timesteps: number);

  public generateSequence(
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): number[][];

  public generateBatchSequences(
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number,
    stepDays: number
  ): number[][][];
}
```

### Constructor Parameters

- `timesteps: number` - Number of time steps in each feature sequence

## Methods

### Public Methods

#### `generateSequence(btcPrices, btcVolumes, startIndex, endIndex): number[][]`

Generates a single feature sequence from the specified data range.

**Parameters:**

- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `startIndex: number` - Starting index for sequence generation
- `endIndex: number` - Ending index for sequence generation

**Returns:**

- `number[][]` - 2D array representing the feature sequence (timesteps × features)

**Process:**

1. **Index Validation**: Ensures safe index boundaries
2. **Feature Computation**: Uses FeatureCalculator for each time point
3. **Feature Integration**: Processes BTC features
4. **Sequence Padding**: Handles insufficient data scenarios
5. **Length Enforcement**: Ensures exact timestep length

**Sequence Structure:**

```typescript
[
  [btcFeatures_0], // Timestep 1
  [btcFeatures_1], // Timestep 2
  [btcFeatures_2], // Timestep 3
  // ... more timesteps
];
```

#### `generateBatchSequences(btcPrices, btcVolumes, startIndex, endIndex, stepDays): number[][][]`

Generates multiple feature sequences for batch training.

**Parameters:**

- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `startIndex: number` - Starting index for batch generation
- `endIndex: number` - Ending index for batch generation
- `stepDays: number` - Step size between sequences (in days)

**Returns:**

- `number[][][]` - 3D array of feature sequences (samples × timesteps × features)

**Process:**

1. **Data Validation**: Checks minimum data requirements
2. **Iterative Generation**: Creates sequences at specified intervals
3. **Quality Control**: Validates sequence length and integrity
4. **Batch Assembly**: Collects all valid sequences
5. **Progress Logging**: Reports generation statistics

## Data Flow

### 1. Input Data Preparation

```typescript
const btcPrices = [50000, 50100, 50200, ...];  // Historical BTC prices
const btcVolumes = [500, 550, 600, ...];  // Historical BTC volumes
```

### 2. Feature Computation

```typescript
const featureCalculator = new FeatureCalculator();

const btcFeatures = featureCalculator.compute({
  prices: btcPrices,
  volumes: btcVolumes,
  dayIndex: i,
  currentPrice: btcPrices[i],
  isBTC: true,
});
```

### 3. Feature Integration

```typescript
const features = btcFeatures;
// BTC: 62 features total
```

### 4. Sequence Assembly

```typescript
sequence.push(features);
// Each timestep contains 62 features
```

### 5. Sequence Padding

```typescript
while (sequence.length < this.timesteps) {
  sequence.unshift(sequence[0] || Array(MODEL_CONFIG.FEATURE_COUNT).fill(0));
}
```

## Feature Vector Composition

### Total Feature Count

- **BTC Features**: 62 technical indicators and patterns

### Feature Categories

```typescript
[
  // BTC Features (62)
  rsi,
  prevRsi,
  sma7,
  sma21,
  prevSma7,
  prevSma21,
  macdLine,
  signalLine,
  currentPrice,
  upperBand,
  lowerBand,
  obvValues[obvValues.length - 1] / 1e6,
  obv,
  atr,
  atrBaseline,
  zScore,
  vwap,
  stochRsi,
  stochRsiSignal,
  prevStochRsi,
  fib61_8,
  prevPrice,
  volumeOscillator,
  prevVolumeOscillator,
  isDoubleTop ? 1 : 0,
  isHeadAndShoulders ? 1 : 0,
  prevMacdLine,
  isTripleTop ? 1 : 0,
  isTripleBottom ? 1 : 0,
  isVolumeSpike ? 1 : 0,
  momentum,
  priceChangePct,
  sma20,
  volAdjustedMomentum,
  trendRegime,
  adxProxy,
  // Market Regime Features
  volatilityRegimeScore,
  trendRegimeScore,
  momentumRegimeScore,
  realizedVolatility,
  regimeScore,
  // Additional technical indicators
  sma50 || sma20,
  sma200 || sma20,
  // Price-based features
  currentPrice / (sma20 || currentPrice),
  currentPrice / (sma50 || currentPrice),
  currentPrice / (sma200 || currentPrice),
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
  momentum / (atr || 1),
  priceChangePct / (atr || 1),
  // RSI-based features
  rsi / 100,
  prevRsi / 100,
  // MACD-based features
  macdLine / (currentPrice || 1),
  signalLine / (currentPrice || 1),
  // Bollinger Bands features
  (currentPrice - upperBand) / (upperBand - lowerBand || 1),
  (currentPrice - lowerBand) / (upperBand - lowerBand || 1),
  // Stochastic features
  stochRsi / 100,
  stochRsiSignal / 100,
  // Fibonacci features
  fib61_8 / (currentPrice || 1),
  // ATR-based features
  atr / (currentPrice || 1),
  atrBaseline / (currentPrice || 1),
  // VWAP features
  vwap / (currentPrice || 1),
];
```

## Usage Examples

### Basic Sequence Generation

```typescript
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";

const generator = new FeatureSequenceGenerator(20); // 20 timesteps

const sequence = generator.generateSequence(
  btcPrices,
  btcVolumes,
  100, // startIndex
  119 // endIndex
);

console.log(`Sequence shape: ${sequence.length} × ${sequence[0].length}`);
// Output: Sequence shape: 20 × 62
```

### Batch Sequence Generation

```typescript
// Generate sequences every 5 days
const batchSequences = generator.generateBatchSequences(
  btcPrices,
  btcVolumes,
  100, // startIndex
  500, // endIndex
  5 // stepDays
);

console.log(`Generated ${batchSequences.length} sequences`);
console.log(
  `Each sequence: ${batchSequences[0].length} × ${batchSequences[0][0].length}`
);
// Output: Generated 80 sequences
// Output: Each sequence: 20 × 62
```

### Training Data Preparation

```typescript
// Prepare training dataset
const trainingSequences = generator.generateBatchSequences(
  btcPrices,
  btcVolumes,
  0, // startIndex
  1000, // endIndex
  1 // stepDays (daily sequences)
);

// Convert to training format
const X = trainingSequences; // Features
const y = generateLabels(trainingSequences); // Labels

console.log(
  `Training data shape: ${X.length} × ${X[0].length} × ${X[0][0].length}`
);
// Output: Training data shape: 1000 × 20 × 62
```

## Configuration

### Timestep Configuration

- **Short-term**: 10-20 timesteps for momentum trading
- **Medium-term**: 20-50 timesteps for trend following
- **Long-term**: 50+ timesteps for position trading

### Data Requirements

- **Minimum Length**: `timesteps + 1` data points
- **Recommended Length**: `timesteps * 10` for robust training
- **Data Quality**: Clean, non-missing price and volume data

## Performance Considerations

### Optimization Strategies

- **Efficient Looping**: Single-pass sequence generation
- **Memory Management**: Minimal temporary array creation
- **Feature Calculator Reuse**: Creates new instance per sequence (consider optimization)
- **Batch Processing**: Generates multiple sequences efficiently

### Computational Complexity

- **Single Sequence**: O(timesteps × feature_calculation)
- **Batch Generation**: O(sequences × timesteps × feature_calculation)
- **Memory Usage**: O(sequences × timesteps × features)

### Scalability

- **Small Datasets**: < 1000 sequences - Real-time generation
- **Medium Datasets**: 1000-10000 sequences - Batch processing
- **Large Datasets**: > 10000 sequences - Consider streaming or chunking

## Error Handling

### Data Validation

- **Length Checks**: Ensures sufficient data for sequence generation
- **Index Bounds**: Prevents array out-of-bounds access
- **Sequence Validation**: Confirms proper sequence dimensions

### Fallback Mechanisms

- **Insufficient Data**: Returns empty sequence array
- **Invalid Indices**: Uses safe index boundaries
- **Missing Features**: Fills with zero values for padding

### Error Logging

- **Data Length Errors**: Logs insufficient data warnings
- **Sequence Validation**: Reports invalid sequence lengths
- **Generation Statistics**: Provides batch generation summaries

## Dependencies

### External Dependencies

- **FeatureCalculator**: Core technical analysis engine
- **MODEL_CONFIG**: Configuration constants for feature counts

### Internal Dependencies

- **Array Methods**: Native JavaScript array manipulation
- **Mathematical Operations**: Basic arithmetic for index calculations

## Testing

### Unit Testing Strategy

- **Sequence Generation**: Test individual sequence creation
- **Batch Processing**: Validate batch generation logic
- **Edge Cases**: Test with minimal data and boundary conditions
- **Feature Integration**: Verify BTC feature processing

### Test Data Requirements

- **Historical Data**: Real cryptocurrency price/volume data
- **Edge Cases**: Minimal periods, extreme values
- **Sequence Validation**: Known sequence patterns for verification

### Performance Testing

- **Memory Usage**: Monitor memory consumption during batch generation
- **Execution Time**: Measure sequence generation performance
- **Scalability**: Test with large datasets

## Future Enhancements

### Potential Improvements

- **Feature Calculator Caching**: Reuse calculator instances for performance
- **Parallel Processing**: Multi-threaded sequence generation
- **Memory Optimization**: Streaming sequence generation for large datasets
- **Custom Feature Selection**: Dynamic feature subset generation

### Advanced Features

- **Multi-timeframe Sequences**: Support for different time intervals
- **Feature Normalization**: Built-in scaling and normalization
- **Sequence Augmentation**: Data augmentation techniques
- **Real-time Generation**: Streaming sequence generation for live trading

### Integration Enhancements

- **Database Integration**: Direct database sequence generation
- **API Integration**: RESTful sequence generation endpoints
- **Caching Layer**: Redis-based sequence caching
- **Monitoring**: Real-time performance and quality metrics
