# FeatureSequenceGenerator Class Documentation

## Overview

The `FeatureSequenceGenerator` class is a specialized component in the Cardano trading system that creates time-series feature sequences for machine learning models. It transforms raw cryptocurrency price and volume data into structured, multi-dimensional feature sequences that can be used for training LSTM/RNN models or other time-series prediction algorithms.

## Architecture

The FeatureSequenceGenerator follows a sequence-oriented architecture:

```
Raw Data Input → Feature Computation → Sequence Assembly → Batch Generation → Output
     ↓                ↓                    ↓                ↓
ADA/BTC Prices    FeatureCalculator    Timestep Arrays   Multiple Sequences
ADA/BTC Volumes   Technical Indicators  Feature Vectors   Training Dataset
```

### Key Responsibilities

- **Sequence Generation**: Create fixed-length feature sequences from historical data
- **Feature Integration**: Combine ADA and BTC features into unified feature vectors
- **Batch Processing**: Generate multiple sequences for training datasets
- **Data Validation**: Ensure sequence integrity and proper dimensions
- **Memory Management**: Efficient handling of large datasets

## Class Structure

```typescript
export class FeatureSequenceGenerator {
  private timesteps: number;

  constructor(timesteps: number);

  public generateSequence(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): number[][];

  public generateBatchSequences(
    adaPrices: number[],
    adaVolumes: number[],
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

#### `generateSequence(adaPrices, adaVolumes, btcPrices, btcVolumes, startIndex, endIndex): number[][]`

Generates a single feature sequence from the specified data range.

**Parameters:**

- `adaPrices: number[]` - Historical ADA price data
- `adaVolumes: number[]` - Historical ADA volume data
- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `startIndex: number` - Starting index for sequence generation
- `endIndex: number` - Ending index for sequence generation

**Returns:**

- `number[][]` - 2D array representing the feature sequence (timesteps × features)

**Process:**

1. **Index Validation**: Ensures safe index boundaries
2. **Feature Computation**: Uses FeatureCalculator for each time point
3. **Feature Integration**: Combines ADA and BTC features
4. **Sequence Padding**: Handles insufficient data scenarios
5. **Length Enforcement**: Ensures exact timestep length

**Sequence Structure:**

```typescript
[
  [adaFeatures_0 + btcFeatures_0], // Timestep 1
  [adaFeatures_1 + btcFeatures_1], // Timestep 2
  [adaFeatures_2 + btcFeatures_2], // Timestep 3
  // ... more timesteps
];
```

#### `generateBatchSequences(adaPrices, adaVolumes, btcPrices, btcVolumes, startIndex, endIndex, stepDays): number[][][]`

Generates multiple feature sequences for batch training.

**Parameters:**

- `adaPrices: number[]` - Historical ADA price data
- `adaVolumes: number[]` - Historical ADA volume data
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
const adaPrices = [100, 101, 102, ...];  // Historical ADA prices
const adaVolumes = [1000, 1100, 1200, ...];  // Historical ADA volumes
const btcPrices = [50000, 50100, 50200, ...];  // Historical BTC prices
const btcVolumes = [500, 550, 600, ...];  // Historical BTC volumes
```

### 2. Feature Computation

```typescript
const featureCalculator = new FeatureCalculator();

const adaFeatures = featureCalculator.compute({
  prices: adaPrices,
  volumes: adaVolumes,
  dayIndex: i,
  currentPrice: adaPrices[i],
  isBTC: false,
});

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
const combinedFeatures = [...adaFeatures, ...btcFeatures];
// ADA: 32 features + BTC: 29 features = 61 total features
```

### 4. Sequence Assembly

```typescript
sequence.push(combinedFeatures);
// Each timestep contains 61 features
```

### 5. Sequence Padding

```typescript
while (sequence.length < this.timesteps) {
  sequence.unshift(sequence[0] || Array(MODEL_CONFIG.FEATURE_COUNT).fill(0));
}
```

## Feature Vector Composition

### Total Feature Count

- **ADA Features**: 32 technical indicators and patterns
- **BTC Features**: 29 technical indicators and patterns
- **Combined Features**: 61 total features per timestep

### Feature Categories

```typescript
[
  // ADA Features (32)
  rsi,
  prevRsi,
  sma7,
  sma21,
  macdLine,
  signalLine,
  currentPrice,
  upperBand,
  obv,
  atr,
  atrBaseline,
  zScore,
  vwap,
  stochRsi,
  prevStochRsi,
  fib61_8,
  volumeOscillator,
  prevVolumeOscillator,
  isDoubleTop,
  isHeadAndShoulders,
  isTripleTop,
  isTripleBottom,
  momentum,
  priceChangePct,
  volAdjustedMomentum,
  adxProxy,
  trendRegime,
  btcRatio,

  // BTC Features (29)
  rsi,
  prevRsi,
  sma7,
  sma21,
  macdLine,
  signalLine,
  currentPrice,
  upperBand,
  obv,
  atr,
  atrBaseline,
  zScore,
  vwap,
  stochRsi,
  prevStochRsi,
  fib61_8,
  volumeOscillator,
  prevVolumeOscillator,
  isDoubleTop,
  isHeadAndShoulders,
  isTripleTop,
  momentum,
  priceChangePct,
  volAdjustedMomentum,
];
```

## Usage Examples

### Basic Sequence Generation

```typescript
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";

const generator = new FeatureSequenceGenerator(20); // 20 timesteps

const sequence = generator.generateSequence(
  adaPrices,
  adaVolumes,
  btcPrices,
  btcVolumes,
  100, // startIndex
  119 // endIndex
);

console.log(`Sequence shape: ${sequence.length} × ${sequence[0].length}`);
// Output: Sequence shape: 20 × 61
```

### Batch Sequence Generation

```typescript
// Generate sequences every 5 days
const batchSequences = generator.generateBatchSequences(
  adaPrices,
  adaVolumes,
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
// Output: Each sequence: 20 × 61
```

### Training Data Preparation

```typescript
// Prepare training dataset
const trainingSequences = generator.generateBatchSequences(
  adaPrices,
  adaVolumes,
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
// Output: Training data shape: 1000 × 20 × 61
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
- **Feature Integration**: Verify ADA/BTC feature combination

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
