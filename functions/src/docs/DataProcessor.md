# DataProcessor Class Documentation

## Overview

The `DataProcessor` class is a core component of the Bitcoin trading system that handles data preparation, feature engineering, and dataset balancing for machine learning models. It processes historical cryptocurrency data (BTC) and transforms it into training-ready sequences with proper labels.

## Architecture

The DataProcessor follows a comprehensive data processing pipeline:

```
Data Fetching → Feature Engineering → Sequence Building → Label Generation → Dataset Balancing
     ↓                ↓                    ↓                ↓                ↓
BTC Historical    Technical Indicators  Time-Series      Future Price     Balanced Training
Price/Volume      Pattern Detection     Sequences        Movement Labels   Dataset
```

### Key Responsibilities

- **Data Fetching**: Retrieves historical price and volume data for BTC
- **Feature Engineering**: Computes technical indicators and patterns
- **Sequence Generation**: Creates time-series sequences for ML models
- **Label Generation**: Creates trading labels based on future price movements
- **Dataset Balancing**: Handles class imbalance in trading datasets
- **Data Validation**: Ensures data quality and consistency

## Class Structure

```typescript
export class DataProcessor {
  private readonly config: ModelConfig;
  private readonly startDaysAgo: number;
  private difficultyLevel: number;

  constructor(config: ModelConfig, startDaysAgo: number);

  // Curriculum Learning Methods
  public setDifficultyLevel(level: number): void;
  public getDifficultyLevel(): number;

  // Core Processing Methods
  private buildSequence(
    btcData: HistoricalData,
    index: number
  ): number[][] | null;
  private computeFeatures(btcData: HistoricalData, index: number): number[];
  private validateFeatures(btcFeatures: number[]): boolean;
  private labelData(params: LabelParams): number;
  private balanceDataset(
    X: number[][][],
    y: number[]
  ): { X: number[][][]; y: number[] };

  // Main Interface
  public async prepareData(): Promise<{ X: number[][][]; y: number[] }>;
  public computeFeatureStats(features: number[][]): {
    mean: number[];
    std: number[];
  };
}
```

## Core Interfaces

### HistoricalData

```typescript
interface HistoricalData {
  prices: number[]; // Historical price data
  volumes: number[]; // Historical volume data
  timestamps: number[]; // Unix timestamps
}
```

### LabelParams

```typescript
interface LabelParams {
  prices: number[]; // Price array for labeling
  dayIndex: number; // Current day index
  threshold?: number; // Price change threshold (default: 0.001)
  horizon?: number; // Days to look ahead (default: 7)
}
```

### ModelConfig

```typescript
interface ModelConfig {
  timesteps: number; // Number of timesteps in sequences
  // Number of BTC features determined dynamically by FeatureDetector.getFeatureCount()
}
```

### PrepareDataResult

```typescript
interface PrepareDataResult {
  X: number[][][]; // 3D array: [samples, timesteps, features]
  y: number[]; // Binary labels: 0 (sell), 1 (buy)
}
```

## Methods

### Public Methods

#### `setDifficultyLevel(level: number): void`

Sets the curriculum learning difficulty level for data filtering.

**Parameters:**

- `level: number` - Difficulty level between 0.1 and 1.0 (1.0 = use all data)

**Usage:** Controls which samples are included based on complexity scoring.

#### `getDifficultyLevel(): number`

Returns the current curriculum learning difficulty level.

**Returns:**

- `number` - Current difficulty level (0.1-1.0)

#### `prepareData(): Promise<{ X: number[][][]; y: number[] }>`

Prepares complete training dataset with sequences and labels.

**Returns:**

- `Promise<{ X: number[][][]; y: number[] }>` - Training sequences and labels

**Process:**

1. **Data Fetching**: Retrieves historical BTC data from CryptoCompareService
2. **Sequence Generation**: Creates feature sequences with noise augmentation
3. **Label Generation**: Creates trading labels with 0.001 threshold and 7-day horizon
4. **Dataset Balancing**: Ensures balanced buy/sell distribution
5. **Curriculum Filtering**: Applies difficulty-based sample filtering

#### `computeFeatureStats(features: number[][]): { mean: number[]; std: number[] }`

Computes normalization statistics for feature arrays.

**Parameters:**

- `features: number[][]` - 2D array of feature vectors

**Returns:**

- `{ mean: number[]; std: number[] }` - Mean and standard deviation for each feature

**Features:**

- **Robust Statistics**: Uses clipping to prevent outlier dominance
- **Minimum STD**: Ensures minimum 0.01 standard deviation to prevent division by zero

### Private Methods

#### `buildSequence(btcData: HistoricalData, index: number): number[][] | null`

Builds a feature sequence for a specific time index with data augmentation.

**Parameters:**

- `btcData: HistoricalData` - BTC historical data
- `index: number` - Time index for sequence generation

**Returns:**

- `number[][] | null` - Feature sequence or null if invalid

**Features:**

- **Data Augmentation**: Adds volatility-based noise and temporal shifts
- **ATR-based Scaling**: Uses Average True Range for volatility-aware augmentation
- **Sequence Padding**: Ensures correct timestep length

#### `computeFeatures(btcData: HistoricalData, index: number): number[]`

Computes feature vector for a specific time index.

**Parameters:**

- `btcData: HistoricalData` - BTC historical data
- `index: number` - Time index for feature computation

**Returns:**

- `number[]` - Feature vector (36 BTC features)

#### `labelData(params: LabelParams): number`

Generates trading labels based on future price movements.

**Parameters:**

- `params: LabelParams` - Labeling parameters

**Returns:**

- `number` - Binary label (0: sell, 1: buy)

**Configuration:**

- **Threshold**: 0.001 (0.1% price change)
- **Horizon**: 7 days lookahead
- **Strategy**: Simple threshold-based labeling

## Data Flow

### 1. Historical Data Retrieval

```typescript
// Fetch historical BTC data
const { btcData } = await this.fetchHistoricalData();

console.log(`Retrieved ${btcData.prices.length} days of BTC data`);
console.log(`Latest price: $${btcData.prices[btcData.prices.length - 1]}`);
console.log(`Latest volume: ${btcData.volumes[btcData.volumes.length - 1]}`);
```

### 2. Feature Computation

```typescript
// Compute features for each time point
for (
  let i = this.timesteps;
  i < btcData.prices.length - this.lookaheadDays;
  i++
) {
  const btcFeatures = this.computeFeatures(btcData, i);

  if (this.validateFeatures(btcFeatures)) {
    // Features are valid, proceed with sequence building
  }
}
```

### 3. Sequence Building

```typescript
// Build feature sequence
const sequence = this.buildSequence(btcData, i);

if (sequence) {
  // Sequence is valid, add to training data
  sequences.push(sequence);

  // Generate corresponding label
  const label = this.labelData({
    prices: btcData.prices,
    dayIndex: i,
  });
  labels.push(label);
}
```

### 4. Dataset Preparation

```typescript
// Prepare complete training dataset
const trainingData = await this.prepareTrainingData();

console.log("Training Dataset Summary:");
console.log(`Total samples: ${trainingData.metadata.totalSamples}`);
console.log(`Buy samples: ${trainingData.metadata.buySamples}`);
console.log(`Hold samples: ${trainingData.metadata.holdSamples}`);
console.log(`Feature count: ${trainingData.metadata.featureCount}`);
```

## Labeling Strategy

### Price Movement Analysis

The labeling strategy analyzes future price movements to determine optimal trading actions:

```typescript
const labelData = (params: LabelParams): number => {
  const { prices, dayIndex, lookaheadDays = 7, threshold = 0.02 } = params;

  // Calculate future price change
  const currentPrice = prices[dayIndex];
  const futurePrice = prices[dayIndex + lookaheadDays];
  const priceChange = (futurePrice - currentPrice) / currentPrice;

  // Assign label based on threshold
  return priceChange > threshold ? 1 : 0; // 1: buy, 0: hold/sell
};
```

### Label Distribution

Typical label distribution in cryptocurrency trading:

- **Buy Labels (1)**: 20-30% of dataset
- **Hold/Sell Labels (0)**: 70-80% of dataset

This imbalance reflects the nature of trading where:

- Most time periods are not optimal for buying
- Buy opportunities occur during specific market conditions
- Hold/sell decisions are more common than buy decisions

## Dataset Balancing

### Class Imbalance Handling

The DataProcessor implements several strategies to handle class imbalance:

1. **Oversampling**: Duplicates minority class samples
2. **Undersampling**: Reduces majority class samples
3. **Synthetic Generation**: Creates synthetic minority samples
4. **Weighted Loss**: Applies class weights in loss function

### Balancing Strategies

```typescript
// Example balancing approach
const balanceDataset = (sequences: number[][][], labels: number[]) => {
  const buyIndices = labels
    .map((label, index) => (label === 1 ? index : -1))
    .filter((i) => i !== -1);
  const holdIndices = labels
    .map((label, index) => (label === 0 ? index : -1))
    .filter((i) => i !== -1);

  // Balance by oversampling minority class
  const targetCount = Math.max(buyIndices.length, holdIndices.length);

  // Oversample buy samples
  const balancedBuyIndices = oversample(buyIndices, targetCount);
  const balancedHoldIndices = oversample(holdIndices, targetCount);

  return {
    sequences: [...balancedBuyIndices, ...balancedHoldIndices].map(
      (i) => sequences[i]
    ),
    labels: [...balancedBuyIndices, ...balancedHoldIndices].map(
      (i) => labels[i]
    ),
  };
};
```

## Usage Examples

### Basic Data Processing

```typescript
import { DataProcessor } from "./DataProcessor";

const processor = new DataProcessor();

// Fetch historical data
const { btcData } = await processor.fetchHistoricalData();

// Compute features for a specific day
const features = processor.computeFeatures(btcData, 100);

console.log(`Generated ${features.length} features`);
// Output: Generated 36 features
```

### Complete Training Data Preparation

```typescript
// Prepare complete training dataset
const trainingData = await processor.prepareTrainingData();

console.log("Dataset Statistics:");
console.log(`Sequences: ${trainingData.sequences.length}`);
console.log(`Labels: ${trainingData.labels.length}`);
console.log(
  `Buy ratio: ${
    trainingData.metadata.buySamples / trainingData.metadata.totalSamples
  }`
);
```

### Custom Labeling

```typescript
// Custom labeling with different parameters
const customLabel = processor.labelData({
  prices: btcData.prices,
  dayIndex: 100,
  lookaheadDays: 14, // 2 weeks lookahead
  threshold: 0.05, // 5% threshold
});

console.log(`Custom label: ${customLabel}`);
```

## Configuration

### Feature Configuration

```typescript
const FEATURE_CONFIG = {
  // Number of BTC features obtained dynamically via FeatureDetector.getFeatureCount()
  TIMESTEPS: 30, // Sequence length
  LOOKAHEAD_DAYS: 7, // Labeling lookahead
  PRICE_CHANGE_THRESHOLD: 0.02, // 2% price change threshold
};
```

### Data Quality Settings

```typescript
const QUALITY_CONFIG = {
  MIN_DATA_LENGTH: 450, // Minimum days of data
  MAX_MISSING_VALUES: 0.01, // Maximum 1% missing values
  FEATURE_VALIDATION: true, // Enable feature validation
  SEQUENCE_VALIDATION: true, // Enable sequence validation
};
```

## Performance Considerations

### Optimization Strategies

- **Batch Processing**: Process multiple sequences efficiently
- **Memory Management**: Minimize memory allocation
- **Caching**: Cache computed features and indicators
- **Parallel Processing**: Process features in parallel

### Computational Complexity

- **Feature Computation**: O(n × features) per time point
- **Sequence Generation**: O(timesteps × features) per sequence
- **Dataset Preparation**: O(samples × timesteps × features)

## Error Handling

### Data Validation

- **API Failures**: Graceful handling of data source failures
- **Missing Data**: Interpolation or exclusion of missing values
- **Invalid Features**: Detection and handling of NaN/Infinity values
- **Sequence Integrity**: Validation of sequence dimensions and content

### Fallback Mechanisms

- **Default Values**: Sensible defaults for missing data
- **Error Recovery**: Continue processing with available data
- **Logging**: Comprehensive error logging for debugging

## Dependencies

### External Dependencies

- **CryptoCompareService**: Historical data retrieval
- **FeatureCalculator**: Technical indicator computation
- **FeatureSequenceGenerator**: Sequence generation

### Internal Dependencies

- **Array Methods**: Native JavaScript array operations
- **Mathematical Functions**: Math library for calculations
- **Date/Time**: Date manipulation for timestamps

## Testing

### Unit Testing Strategy

- **Data Fetching**: Test API integration and error handling
- **Feature Computation**: Validate technical indicator calculations
- **Sequence Generation**: Test sequence building logic
- **Label Generation**: Verify labeling strategy
- **Dataset Balancing**: Test class imbalance handling

### Test Data Requirements

- **Historical Data**: Real cryptocurrency price/volume data
- **Edge Cases**: Minimal data, extreme values, missing data
- **Known Patterns**: Data with predictable outcomes

## Future Enhancements

### Potential Improvements

- **Multi-Asset Support**: Extend beyond BTC to other cryptocurrencies
- **Advanced Labeling**: More sophisticated labeling strategies
- **Real-time Processing**: Streaming data processing capabilities
- **Custom Features**: User-defined feature engineering

### Advanced Features

- **Multi-timeframe Analysis**: Support different time intervals
- **Sentiment Integration**: Incorporate market sentiment data
- **Regime Detection**: Market condition-aware processing
- **Adaptive Thresholds**: Dynamic threshold adjustment
