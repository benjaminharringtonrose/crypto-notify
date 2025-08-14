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
  private cryptoCompare: CryptoCompareService;
  private featureCalculator: FeatureCalculator;
  private sequenceGenerator: FeatureSequenceGenerator;

  constructor();

  public async fetchHistoricalData(): Promise<{ btcData: HistoricalData }>;
  public buildSequence(
    btcData: HistoricalData,
    index: number
  ): number[][] | null;
  public computeFeatures(btcData: HistoricalData, index: number): number[];
  public validateFeatures(btcFeatures: number[]): boolean;
  public labelData(params: LabelParams): number;
  public async prepareTrainingData(): Promise<TrainingData>;
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
  lookaheadDays?: number; // Days to look ahead (default: 7)
  threshold?: number; // Price change threshold (default: 0.02)
}
```

### TrainingData

```typescript
interface TrainingData {
  sequences: number[][][]; // 3D array: [samples, timesteps, features]
  labels: number[]; // Binary labels: 0 (hold/sell), 1 (buy)
  metadata: {
    totalSamples: number;
    buySamples: number;
    holdSamples: number;
    featureCount: number;
  };
}
```

## Methods

### Public Methods

#### `fetchHistoricalData(): Promise<{ btcData: HistoricalData }>`

Fetches historical cryptocurrency data from external APIs.

**Returns:**

- `Promise<{ btcData: HistoricalData }>` - Historical BTC data

**Process:**

1. **API Integration**: Uses CryptoCompareService for data retrieval
2. **Data Validation**: Ensures data quality and completeness
3. **Error Handling**: Manages API failures and data inconsistencies
4. **Data Formatting**: Standardizes data format for processing

#### `buildSequence(btcData: HistoricalData, index: number): number[][] | null`

Builds a feature sequence for a specific time index.

**Parameters:**

- `btcData: HistoricalData` - BTC historical data
- `index: number` - Time index for sequence generation

**Returns:**

- `number[][] | null` - Feature sequence or null if invalid

**Process:**

1. **Index Validation**: Ensures valid time index
2. **Feature Computation**: Calculates technical indicators
3. **Sequence Assembly**: Combines BTC features
4. **Quality Check**: Validates sequence integrity

#### `computeFeatures(btcData: HistoricalData, index: number): number[]`

Computes feature vector for a specific time index.

**Parameters:**

- `btcData: HistoricalData` - BTC historical data
- `index: number` - Time index for feature computation

**Returns:**

- `number[]` - Feature vector (62 BTC features)

**Process:**

1. **Technical Analysis**: Computes all technical indicators
2. **Pattern Detection**: Identifies chart patterns
3. **Feature Integration**: Combines all features into vector
4. **Normalization**: Applies scaling and normalization

#### `validateFeatures(btcFeatures: number[]): boolean`

Validates feature vector integrity and quality.

**Parameters:**

- `btcFeatures: number[]` - BTC feature vector

**Returns:**

- `boolean` - True if features are valid

**Validation Checks:**

- **Length**: Ensures correct feature count
- **Range**: Validates feature value ranges
- **NaN/Infinity**: Checks for invalid numerical values
- **Consistency**: Verifies feature relationships

#### `labelData(params: LabelParams): number`

Generates trading labels based on future price movements.

**Parameters:**

- `params: LabelParams` - Labeling parameters

**Returns:**

- `number` - Binary label (0: hold/sell, 1: buy)

**Labeling Logic:**

1. **Future Price Analysis**: Examines price movement over lookahead period
2. **Threshold Comparison**: Compares price change to threshold
3. **Label Assignment**: Assigns buy (1) or hold/sell (0) label
4. **Edge Case Handling**: Manages boundary conditions

#### `prepareTrainingData(): Promise<TrainingData>`

Prepares complete training dataset with sequences and labels.

**Returns:**

- `Promise<TrainingData>` - Complete training dataset

**Process:**

1. **Data Fetching**: Retrieves historical BTC data
2. **Sequence Generation**: Creates feature sequences
3. **Label Generation**: Creates trading labels
4. **Dataset Balancing**: Handles class imbalance
5. **Quality Assurance**: Validates final dataset

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
// Output: Generated 62 features
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
  BTC_FEATURE_COUNT: 62, // Number of BTC features
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
