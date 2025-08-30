# Bitcoin Ensemble Prediction System

This module provides a complete TypeScript implementation of a Bitcoin price prediction system using ensemble machine learning methods. It recreates the Python architecture from the `functions/src/python/` directory with equivalent functionality.

## Architecture Overview

The system consists of four main components:

### 1. Data Collection (`DataCollector.ts`)

- Fetches historical Bitcoin price data from CoinGecko API
- Generates realistic sample data when API is unavailable
- Validates data quality and integrity
- Supports various time periods and intervals

### 2. Feature Engineering (`FeatureEngineer.ts`)

- Creates comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Implements moving averages, volatility measures, and price patterns
- Generates target variables for 3-day price prediction
- Provides feature scaling and selection capabilities

### 3. Machine Learning Models (`Models.ts`)

- **Random Forest**: Ensemble of decision trees with bootstrap sampling
- **Logistic Regression**: Linear model with sigmoid activation
- **Neural Network**: Multi-layer perceptron with backpropagation
- **Ensemble Methods**: Weighted voting and probability averaging
- Comprehensive model evaluation and metrics calculation

### 4. Prediction Engine (`PredictionEngine.ts`)

- Orchestrates the entire prediction pipeline
- Handles model training, loading, and persistence
- Provides real-time prediction capabilities
- Includes prediction validation and accuracy tracking

## Key Features

### Technical Indicators

- **Price-based**: Price changes, volatility measures
- **Moving Averages**: SMA, EMA with various periods
- **Momentum**: RSI, MACD, Stochastic Oscillator, Williams %R
- **Volatility**: Bollinger Bands, ATR, CCI
- **Volume**: OBV, MFI, Volume ratios
- **Support/Resistance**: Dynamic levels based on recent price action
- **Time Features**: Cyclical encoding of day/week/month patterns

### Machine Learning Models

- **Random Forest**: 100 trees with configurable depth and split criteria
- **Logistic Regression**: Gradient descent optimization
- **Neural Network**: 3 hidden layers (100, 50, 25 neurons) with ReLU activation
- **Ensemble**: Weighted voting based on individual model performance

### Prediction Capabilities

- **3-day price direction prediction** (UP/DOWN)
- **Confidence scoring** for prediction reliability
- **Model contribution analysis** showing which models influence decisions
- **Historical accuracy tracking** and validation
- **Real-time updates** with fresh data

## Usage Examples

### Basic Usage

```typescript
import { EnsemblePredictor } from "./ensemble";

// Initialize the prediction engine
const engine = new EnsemblePredictor();

// Load or train models
await engine.loadOrTrainModels();

// Make a prediction
const prediction = await engine.makePrediction();
console.log(
  `Prediction: ${prediction.direction} with ${prediction.confidence}% confidence`
);

// Get prediction summary
const summary = engine.getPredictionSummary();
console.log(`Total predictions: ${summary.predictionCount}`);
```

### Advanced Usage

```typescript
import {
  BitcoinDataCollector,
  FeatureEngineer,
  BitcoinPredictor,
} from "./ensemble";

// Custom data collection
const collector = new BitcoinDataCollector({ dataDir: "custom_data" });
const data = await collector.fetchHistoricalData("2y");

// Feature engineering
const engineer = new FeatureEngineer();
const dataWithFeatures = engineer.createTechnicalIndicators(data);
const dataWithTarget = engineer.createTargetVariable(dataWithFeatures, 5); // 5-day prediction

// Model training
const { X, y } = engineer.selectFeatures(dataWithTarget);
const predictor = new BitcoinPredictor();
const results = await predictor.trainModels(X, y);

// Ensemble prediction
const ensembleResult = await predictor.ensemblePredict(X.slice(-1));
console.log(`Ensemble prediction: ${ensembleResult.predictions[0]}`);
```

### Testing Individual Components

```typescript
import {
  testDataCollection,
  testFeatureEngineering,
  testModels,
  testPredictionEngine,
} from "./ensemble";

// Test data collection
await testDataCollection();

// Test feature engineering
await testFeatureEngineering();

// Test model training
await testModels();

// Test complete pipeline
await testPredictionEngine();
```

## API Reference

### BitcoinDataCollector

```typescript
class BitcoinDataCollector {
  constructor(config?: BitcoinDataCollectorConfig);

  async fetchHistoricalData(
    period?: string,
    interval?: string
  ): Promise<BitcoinDataPoint[]>;
  async getLatestData(days?: number): Promise<BitcoinDataPoint[]>;
  validateData(data: BitcoinDataPoint[]): boolean;
  async saveData(data: BitcoinDataPoint[], filename?: string): Promise<string>;
  async loadData(filename: string): Promise<BitcoinDataPoint[]>;
}
```

### FeatureEngineer

```typescript
class FeatureEngineer {
  createTechnicalIndicators(data: BitcoinDataPoint[]): FeatureDataPoint[];
  createTargetVariable(
    data: FeatureDataPoint[],
    predictionDays?: number
  ): FeatureDataPoint[];
  selectFeatures(
    data: FeatureDataPoint[],
    targetCol?: string,
    excludeCols?: string[]
  ): FeatureEngineeringResult;
  scaleFeatures(
    X: number[][],
    method?: "standard" | "minmax"
  ): { scaledX: number[][]; scaler: any };
  createLagFeatures(
    data: FeatureDataPoint[],
    columns: string[],
    lags?: number[]
  ): FeatureDataPoint[];
  createRollingFeatures(
    data: FeatureDataPoint[],
    columns: string[],
    windows?: number[]
  ): FeatureDataPoint[];
}
```

### BitcoinPredictor

```typescript
class BitcoinPredictor {
  constructor(config?: ModelConfig);

  async trainModels(
    X: number[][],
    y: number[],
    skipHyperparameterTuning?: boolean
  ): Promise<TrainingResult>;
  async ensemblePredict(
    X: number[][]
  ): Promise<{
    predictions: number[];
    probabilities: number[];
    confidenceScores: number[];
  }>;
  getEnsembleWeights(): EnsembleWeights;
  getFeatureColumns(): string[];
  printTrainingSummary(results: TrainingResult): void;
}
```

### BitcoinPredictionEngine

```typescript
class BitcoinPredictionEngine {
  constructor(config?: PredictionEngineConfig);

  async loadOrTrainModels(
    forceRetrain?: boolean,
    dataPeriod?: string
  ): Promise<boolean>;
  async makePrediction(
    useEnsemble?: boolean,
    confidenceThreshold?: number
  ): Promise<PredictionResult>;
  getPredictionSummary(): PredictionSummary;
  async validatePrediction(daysAhead?: number): Promise<ValidationResult>;
  getModelPerformance(): ModelPerformance;
  async updatePrediction(): Promise<PredictionResult>;
  async getLatestPrice(): Promise<number>;
  async getPriceHistory(
    days?: number
  ): Promise<Array<{ date: string; price: number }>>;
  async getPredictionAccuracy(): Promise<{
    correct: number;
    total: number;
    accuracy: number;
  }>;
}
```

## Data Structures

### BitcoinDataPoint

```typescript
interface BitcoinDataPoint {
  Date: Date;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Dividends?: number;
  Stock_Splits?: number;
}
```

### PredictionResult

```typescript
interface PredictionResult {
  timestamp: string;
  predictionDate: string;
  currentPrice: number;
  prediction: number; // 1 for up, 0 for down
  probability: number;
  confidence: number;
  direction: "UP" | "DOWN";
  modelsUsed: string[];
  predictionWindow: string;
  isConfident: boolean;
}
```

## Configuration

### Data Collector Config

```typescript
interface BitcoinDataCollectorConfig {
  dataDir?: string; // Directory for storing data files
  symbol?: string; // Trading symbol (default: 'BTC-USD')
}
```

### Model Config

```typescript
interface ModelConfig {
  modelsDir?: string; // Directory for storing model files
}
```

### Prediction Engine Config

```typescript
interface PredictionEngineConfig {
  modelsDir?: string; // Directory for storing model files
  dataDir?: string; // Directory for storing data files
}
```

## Performance Considerations

### Model Training

- **Random Forest**: ~2-5 seconds for 100 trees
- **Logistic Regression**: ~1-2 seconds for 1000 iterations
- **Neural Network**: ~5-10 seconds for 500 iterations
- **Total Training**: ~10-20 seconds for complete ensemble

### Prediction Speed

- **Single Prediction**: ~100-200ms
- **Batch Predictions**: ~50ms per prediction
- **Real-time Updates**: Suitable for 1-minute intervals

### Memory Usage

- **Training**: ~50-100MB for 1000 data points
- **Prediction**: ~10-20MB for loaded models
- **Data Storage**: ~1MB per 1000 data points

## Limitations and Considerations

### Current Limitations

1. **Model Persistence**: Models are not currently saved to disk (in-memory only)
2. **API Dependencies**: Relies on CoinGecko API with fallback to generated data
3. **Simplified ML**: Uses basic implementations rather than optimized libraries
4. **No GPU Acceleration**: All computations are CPU-based

### Future Enhancements

1. **Model Persistence**: Save/load trained models to/from disk
2. **Advanced ML Libraries**: Integrate with TensorFlow.js or ML5.js
3. **Real-time Data**: WebSocket connections for live price feeds
4. **More Models**: Add XGBoost, LightGBM, and other advanced algorithms
5. **Hyperparameter Tuning**: Automated optimization of model parameters
6. **Backtesting**: Comprehensive historical performance analysis

## Error Handling

The system includes comprehensive error handling:

```typescript
try {
  const prediction = await engine.makePrediction();
  console.log("Prediction successful:", prediction);
} catch (error) {
  console.error("Prediction failed:", error.message);
  // Handle specific error types
  if (error.message.includes("No recent data")) {
    // Handle data availability issues
  } else if (error.message.includes("models may not be trained")) {
    // Handle model training issues
  }
}
```

## Testing

Run individual component tests:

```bash
# Test data collection
npm run test:data-collection

# Test feature engineering
npm run test:feature-engineering

# Test models
npm run test:models

# Test complete pipeline
npm run test:prediction-engine
```

## Contributing

When contributing to this ensemble system:

1. **Add New Technical Indicators**: Extend `FeatureEngineer.createTechnicalIndicators()`
2. **Add New Models**: Implement the `BaseModel` interface
3. **Improve Data Sources**: Extend `BitcoinDataCollector` with new APIs
4. **Enhance Validation**: Add more comprehensive prediction validation methods
5. **Performance Optimization**: Profile and optimize slow operations

## License

This ensemble prediction system is part of the crypto-notify project and follows the same licensing terms.
