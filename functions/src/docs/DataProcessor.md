# DataProcessor Class Documentation

## Overview

The `DataProcessor` class is a core component of the Cardano trading system that handles data preparation, feature engineering, and dataset balancing for machine learning models. It processes historical cryptocurrency data (ADA and BTC) and transforms it into training-ready sequences with proper labels.

## Architecture

The DataProcessor follows a pipeline-based architecture:

```
Historical Data Fetch → Feature Computation → Sequence Building → Data Labeling → Dataset Balancing → Output
```

### Key Responsibilities

- **Data Fetching**: Retrieves historical price and volume data for ADA and BTC
- **Feature Engineering**: Computes technical indicators and features using the FeatureCalculator
- **Sequence Generation**: Creates time-series sequences for LSTM/RNN models
- **Data Labeling**: Generates binary labels (buy/sell) based on future price movements
- **Dataset Balancing**: Ensures equal representation of buy/sell samples
- **Noise Injection**: Adds controlled randomness to improve model robustness

## Class Structure

```typescript
export class DataProcessor {
  private readonly config: ModelConfig;
  private readonly startDaysAgo: number;

  constructor(config: ModelConfig, startDaysAgo: number);
  // ... methods
}
```

### Constructor Parameters

- `config: ModelConfig` - Configuration object containing model parameters
- `startDaysAgo: number` - Number of days to look back for historical data

## Methods

### Public Methods

#### `prepareData(): Promise<{ X: number[][][]; y: number[] }>`

Main method that orchestrates the entire data preparation pipeline.

**Returns:**

- `X: number[][][]` - 3D array of feature sequences (samples × timesteps × features)
- `y: number[]` - Binary labels (0 = sell, 1 = buy)

**Process:**

1. Fetches historical data for ADA and BTC
2. Builds feature sequences for each time point
3. Labels data based on future price movements
4. Balances the dataset for equal buy/sell representation

#### `computeFeatureStats(features: number[][]): { mean: number[]; std: number[] }`

Computes statistical measures for feature normalization.

**Parameters:**

- `features: number[][]` - 2D array of feature vectors

**Returns:**

- `mean: number[]` - Mean values for each feature
- `std: number[]` - Standard deviation values for each feature

### Private Methods

#### `fetchHistoricalData(): Promise<{ adaData: HistoricalData; btcData: HistoricalData }>`

Retrieves historical price and volume data from CryptoCompare API.

**Returns:**

- `adaData: HistoricalData` - ADA cryptocurrency data
- `btcData: HistoricalData` - BTC cryptocurrency data

#### `buildSequence(adaData: HistoricalData, btcData: HistoricalData, index: number): number[][] | null`

Constructs a feature sequence for a specific time point.

**Parameters:**

- `adaData: HistoricalData` - ADA historical data
- `btcData: HistoricalData` - BTC historical data
- `index: number` - Current time index

**Returns:**

- `number[][] | null` - Feature sequence or null if validation fails

**Features:**

- Combines ADA and BTC features
- Adds volatility-based noise using ATR (Average True Range)
- Ensures proper sequence length

#### `computeFeaturePair(adaData: HistoricalData, btcData: HistoricalData, index: number): [number[], number[]]`

Computes features for both cryptocurrencies at a specific time point.

**Returns:**

- `[number[], number[]]` - Tuple of ADA and BTC feature vectors

#### `validateFeatures(adaFeatures: number[], btcFeatures: number[]): boolean`

Validates that feature vectors meet expected dimensions.

**Returns:**

- `boolean` - True if features are valid, false otherwise

#### `addNoise(features: number[], scale: number): number[]`

Injects controlled noise into feature vectors.

**Parameters:**

- `features: number[]` - Input feature vector
- `scale: number` - Noise scaling factor based on volatility

**Returns:**

- `number[]` - Features with added noise

#### `adjustSequenceLength(sequence: number[][]): number[][]`

Ensures sequence meets the required timestep length.

**Parameters:**

- `sequence: number[][]` - Input sequence

**Returns:**

- `number[][]` - Sequence with adjusted length

#### `balanceDataset(X: number[][][], y: number[]): { X: number[][][]; y: number[] }`

Balances the dataset to have equal buy/sell samples.

**Parameters:**

- `X: number[][][]` - Feature sequences
- `y: number[]` - Labels

**Returns:**

- `{ X: number[][][]; y: number[] }` - Balanced dataset

#### `labelData({ prices, dayIndex, threshold, horizon }): number`

Generates binary labels based on future price movements.

**Parameters:**

- `prices: number[]` - Price array
- `dayIndex: number` - Current time index
- `threshold: number` - Price change threshold (default: 0.07 = 7%)
- `horizon: number` - Future time horizon (default: 7 days)

**Returns:**

- `number` - Binary label (0 = sell, 1 = buy)

**Labeling Logic:**

- If future average price > current price + threshold → Buy (1)
- Otherwise → Sell (0)

## Data Flow

### 1. Data Fetching

```typescript
const { adaData, btcData } = await this.fetchHistoricalData();
```

### 2. Feature Computation

```typescript
const [adaFeatures, btcFeatures] = this.computeFeaturePair(adaData, btcData, i);
```

### 3. Sequence Building

```typescript
const sequence = this.buildSequence(adaData, btcData, i);
```

### 4. Data Labeling

```typescript
const label = this.labelData({ prices: adaData.prices, dayIndex: i });
```

### 5. Dataset Balancing

```typescript
const balancedData = this.balanceDataset(X, y);
```

## Configuration

The DataProcessor relies on several configuration constants:

- `MODEL_CONFIG.ADA_FEATURE_COUNT` - Number of features for ADA
- `MODEL_CONFIG.BTC_FEATURE_COUNT` - Number of features for BTC
- `PERIODS.ATR` - Period for ATR calculation
- `config.timesteps` - Number of time steps in sequences

## Usage Examples

### Basic Usage

```typescript
import { DataProcessor } from "./DataProcessor";
import { MODEL_CONFIG } from "../constants";

const processor = new DataProcessor(MODEL_CONFIG, 365);
const { X, y } = await processor.prepareData();
```

### With Custom Configuration

```typescript
const customConfig = {
  ...MODEL_CONFIG,
  timesteps: 20,
  // other custom parameters
};

const processor = new DataProcessor(customConfig, 180);
const trainingData = await processor.prepareData();
```

## Error Handling

The DataProcessor includes several validation mechanisms:

- **Feature Validation**: Ensures feature vectors meet expected dimensions
- **Sequence Validation**: Returns null for invalid sequences
- **Data Length Validation**: Checks that sufficient historical data exists
- **API Error Handling**: Gracefully handles CryptoCompare API failures

## Performance Considerations

- **Parallel Data Fetching**: Uses `Promise.all` for concurrent API calls
- **Efficient Looping**: Processes data in single pass where possible
- **Memory Management**: Avoids unnecessary array copies
- **Logging**: Provides progress feedback for long-running operations

## Dependencies

- `CryptoCompareService` - For fetching historical cryptocurrency data
- `FeatureCalculator` - For computing technical indicators
- `MODEL_CONFIG` - Model configuration constants
- `PERIODS` - Technical indicator periods
- `HistoricalData`, `ModelConfig` - TypeScript interfaces

## Testing

The DataProcessor can be tested with:

- **Unit Tests**: Individual method functionality
- **Integration Tests**: End-to-end data preparation pipeline
- **Mock Data**: Historical data fixtures for consistent testing
- **Performance Tests**: Large dataset processing capabilities

## Future Enhancements

Potential improvements include:

- **Caching**: Store processed data to avoid reprocessing
- **Streaming**: Process data in chunks for very large datasets
- **Feature Selection**: Dynamic feature importance ranking
- **Data Augmentation**: Additional noise injection strategies
- **Validation Split**: Automatic train/validation/test splitting
