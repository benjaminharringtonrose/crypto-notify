# TradeModelPredictor Class Documentation

## Overview

The `TradeModelPredictor` class is the core prediction engine of the Bitcoin trading system that combines machine learning model inference with advanced technical analysis to generate comprehensive trading signals. It leverages a pre-trained deep learning model to predict trading decisions (Buy, Sell, Hold) based on historical BTC price and volume data, technical indicators, and market patterns.

## Architecture

The TradeModelPredictor follows a hybrid prediction architecture:

```
BTC Prices   Feature        Neural Network   Market Metrics   Trading Signals
     ↓         ↓                    ↓                ↓              ↓
Historical Data  Technical Analysis  ML Predictions  Rule-Based Logic  Final Decision
BTC Volumes   Pattern Detection   Confidence Scores  Risk Assessment  Buy/Sell/Hold
```

### Key Responsibilities

- **Model Inference**: Executes pre-trained neural network predictions
- **Feature Engineering**: Computes technical indicators and patterns
- **Signal Generation**: Combines ML predictions with rule-based logic
- **Risk Assessment**: Evaluates market conditions and volatility
- **Decision Refinement**: Applies market context to finalize trading decisions

## Class Structure

```typescript
export class TradeModelPredictor {
  private model: tf.LayersModel;
  private featureCalculator: FeatureCalculator;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps: number;

  constructor(model: tf.LayersModel);

  public async predict(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): Promise<PredictionResult>;

  public async predictBatch(
    btcPrices: number[],
    btcVolumes: number[],
    startIndex: number,
    endIndex: number
  ): Promise<PredictionResult[]>;

  private computeMarketMetrics(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): MarketMetrics;
}
```

### Constructor Parameters

- `model: tf.LayersModel` - Pre-trained TensorFlow.js model

## Core Interfaces

### PredictionResult

```typescript
interface PredictionResult {
  buyProb: number; // Probability of buy recommendation
  sellProb: number; // Probability of sell recommendation
  holdProb: number; // Probability of hold recommendation
  confidence: number; // Overall prediction confidence
  strategy: string; // Applied trading strategy
  marketConditions: MarketMetrics;
  timestamp: number; // Prediction timestamp
}
```

### MarketMetrics

```typescript
interface MarketMetrics {
  rsi: number; // Relative Strength Index
  momentum: number; // Price momentum
  volatility: number; // Market volatility (ATR)
  trendStrength: number; // Trend strength indicator
  volumeProfile: number; // Volume analysis
  supportResistance: {
    support: number;
    resistance: number;
  };
}
```

## Methods

### Public Methods

#### `predict(btcPrices, btcVolumes, currentIndex): Promise<PredictionResult>`

Generates a single trading prediction for the current market state.

**Parameters:**

- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `currentIndex: number` - Current time index

**Returns:**

- `Promise<PredictionResult>` - Comprehensive trading prediction

**Process:**

1. **Data Validation**: Ensures sufficient historical data
2. **Feature Generation**: Computes technical indicators and patterns
3. **Sequence Creation**: Builds input sequence for neural network
4. **Model Inference**: Executes neural network prediction
5. **Market Analysis**: Computes current market metrics
6. **Signal Refinement**: Applies rule-based logic to ML predictions
7. **Decision Finalization**: Generates final trading recommendation

#### `predictBatch(btcPrices, btcVolumes, startIndex, endIndex): Promise<PredictionResult[]>`

Generates multiple predictions for a range of time indices.

**Parameters:**

- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `startIndex: number` - Starting time index
- `endIndex: number` - Ending time index

**Returns:**

- `Promise<PredictionResult[]>` - Array of predictions

**Process:**

1. **Range Validation**: Ensures valid index range
2. **Batch Processing**: Processes multiple time points efficiently
3. **Consistency Check**: Validates prediction consistency
4. **Performance Optimization**: Minimizes redundant computations

### Private Methods

#### `computeMarketMetrics(btcPrices, btcVolumes, currentIndex): MarketMetrics`

Computes comprehensive market condition metrics.

**Parameters:**

- `btcPrices: number[]` - Historical BTC price data
- `btcVolumes: number[]` - Historical BTC volume data
- `currentIndex: number` - Current time index

**Returns:**

- `MarketMetrics` - Current market condition analysis

**Metrics Computed:**

- **RSI**: Overbought/oversold conditions
- **Momentum**: Price velocity and direction
- **Volatility**: Market volatility using ATR
- **Trend Strength**: Trend direction and strength
- **Volume Profile**: Volume analysis and patterns
- **Support/Resistance**: Key price levels

## Prediction Process

### 1. Data Preparation

```typescript
// Validate input data
if (btcPrices.length < this.timesteps || btcVolumes.length < this.timesteps) {
  throw new Error("Insufficient historical data for prediction");
}

// Extract relevant data window
const startIndex = Math.max(0, btcPrices.length - this.timesteps - 1);
const endIndex = btcPrices.length - 1;

const priceWindow = btcPrices.slice(startIndex, endIndex + 1);
const volumeWindow = btcVolumes.slice(startIndex, endIndex + 1);
```

### 2. Feature Engineering

```typescript
// Generate feature sequence
const sequence = this.sequenceGenerator.generateSequence(
  priceWindow,
  volumeWindow,
  startIndex,
  endIndex
);

// Validate sequence
if (!sequence || sequence.length !== this.timesteps) {
  throw new Error("Invalid feature sequence generated");
}
```

### 3. Model Inference

```typescript
// Prepare input tensor
const inputTensor = tf.tensor3d(
  [sequence],
  [1, this.timesteps, MODEL_CONFIG.FEATURE_COUNT]
);

// Execute prediction
const prediction = (await this.model.predict(inputTensor)) as tf.Tensor;

// Extract probabilities
const [buyProb, sellProb, holdProb] = await prediction.array();
```

### 4. Market Analysis

```typescript
// Compute market metrics
const marketMetrics = this.computeMarketMetrics(
  btcPrices,
  btcVolumes,
  currentIndex
);

// Analyze market conditions
const isVolatile = marketMetrics.volatility > VOLATILITY_THRESHOLD;
const isTrending = marketMetrics.trendStrength > TREND_THRESHOLD;
const isOverbought = marketMetrics.rsi > 70;
const isOversold = marketMetrics.rsi < 30;
```

### 5. Signal Refinement

```typescript
// Apply rule-based logic to ML predictions
let finalBuyProb = buyProb;
let finalSellProb = sellProb;
let finalHoldProb = holdProb;

// Adjust based on market conditions
if (isOverbought && buyProb > 0.5) {
  finalBuyProb *= 0.7; // Reduce buy probability
  finalHoldProb *= 1.2; // Increase hold probability
}

if (isOversold && sellProb > 0.5) {
  finalSellProb *= 0.7; // Reduce sell probability
  finalBuyProb *= 1.2; // Increase buy probability
}

// Normalize probabilities
const total = finalBuyProb + finalSellProb + finalHoldProb;
finalBuyProb /= total;
finalSellProb /= total;
finalHoldProb /= total;
```

### 6. Decision Finalization

```typescript
// Determine final decision
const maxProb = Math.max(finalBuyProb, finalSellProb, finalHoldProb);
let decision: string;
let confidence: number;

if (maxProb === finalBuyProb) {
  decision = "BUY";
  confidence = finalBuyProb;
} else if (maxProb === finalSellProb) {
  decision = "SELL";
  confidence = finalSellProb;
} else {
  decision = "HOLD";
  confidence = finalHoldProb;
}

// Apply confidence threshold
if (confidence < CONFIDENCE_THRESHOLD) {
  decision = "HOLD"; // Default to hold if confidence is low
}
```

## Market Analysis

### Technical Indicators

The predictor analyzes several key technical indicators:

#### RSI (Relative Strength Index)

```typescript
const rsi = this.featureCalculator.calculateRSI(btcPrices, 14);

// RSI interpretation
if (rsi > 70) {
  // Overbought condition - potential sell signal
} else if (rsi < 30) {
  // Oversold condition - potential buy signal
}
```

#### Momentum Analysis

```typescript
const momentumWindow = btcPrices.slice(-momentumWindowSize);
const shortMomentum = this.calculateMomentum(momentumWindow.slice(-3));
const longMomentum = this.calculateMomentum(momentumWindow);

// Momentum interpretation
if (shortMomentum > 0 && longMomentum > 0) {
  // Strong upward momentum
} else if (shortMomentum < 0 && longMomentum < 0) {
  // Strong downward momentum
}
```

#### Volatility Assessment

```typescript
const atr = this.featureCalculator.calculateATR(btcPrices, 14);
const volatilityRegime = this.classifyVolatility(atr);

// Volatility-based adjustments
if (volatilityRegime === "HIGH") {
  // Reduce position sizes, increase stop-loss
} else if (volatilityRegime === "LOW") {
  // Increase position sizes, tighter stop-loss
}
```

### Pattern Recognition

The predictor identifies key chart patterns:

#### Support and Resistance

```typescript
const supportResistance = this.identifySupportResistance(btcPrices);
const currentPrice = btcPrices[currentIndex];

if (currentPrice <= supportResistance.support * 1.02) {
  // Near support - potential buy opportunity
} else if (currentPrice >= supportResistance.resistance * 0.98) {
  // Near resistance - potential sell opportunity
}
```

#### Volume Analysis

```typescript
const volumeProfile = this.analyzeVolume(btcVolumes);
const isVolumeSpike =
  volumeProfile.currentVolume > volumeProfile.averageVolume * 2;

if (isVolumeSpike && buyProb > 0.6) {
  // High volume confirms buy signal
  confidence *= 1.2;
}
```

## Usage Examples

### Basic Prediction

```typescript
import { TradeModelPredictor } from "./TradeModelPredictor";

// Initialize predictor with trained model
const predictor = new TradeModelPredictor(trainedModel);

// Historical BTC data
const btcPrices = [
  /* 30+ days of BTC prices */
];
const btcVolumes = [
  /* 30+ days of BTC volumes */
];

// Generate prediction
const prediction = await predictor.predict(
  btcPrices,
  btcVolumes,
  btcPrices.length - 1
);

console.log("Trading Prediction:");
console.log(`Decision: ${prediction.strategy}`);
console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
console.log(`Buy Probability: ${(prediction.buyProb * 100).toFixed(2)}%`);
console.log(`Sell Probability: ${(prediction.sellProb * 100).toFixed(2)}%`);
console.log(`Hold Probability: ${(prediction.holdProb * 100).toFixed(2)}%`);
```

### Batch Predictions

```typescript
// Generate predictions for multiple days
const predictions = await predictor.predictBatch(
  btcPrices,
  btcVolumes,
  100, // startIndex
  120 // endIndex
);

// Analyze prediction trends
const buySignals = predictions.filter((p) => p.strategy === "BUY");
const sellSignals = predictions.filter((p) => p.strategy === "SELL");

console.log(`Buy signals: ${buySignals.length}`);
console.log(`Sell signals: ${sellSignals.length}`);
console.log(
  `Average confidence: ${
    predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
  }`
);
```

### Market Analysis Integration

```typescript
// Analyze market conditions
const marketMetrics = predictor.computeMarketMetrics(
  btcPrices,
  btcVolumes,
  currentIndex
);

console.log("Market Analysis:");
console.log(`RSI: ${marketMetrics.rsi.toFixed(2)}`);
console.log(`Momentum: ${marketMetrics.momentum.toFixed(4)}`);
console.log(`Volatility: ${marketMetrics.volatility.toFixed(2)}`);
console.log(`Trend Strength: ${marketMetrics.trendStrength.toFixed(4)}`);

// Use market metrics for decision making
if (marketMetrics.rsi < 30 && marketMetrics.momentum > 0) {
  console.log(
    "Oversold condition with positive momentum - potential buy opportunity"
  );
}
```

## Configuration

### Prediction Parameters

```typescript
const PREDICTION_CONFIG = {
  TIMESTEPS: 30, // Sequence length
  FEATURE_COUNT: 36, // Number of features
  CONFIDENCE_THRESHOLD: 0.6, // Minimum confidence for action
  VOLATILITY_THRESHOLD: 0.05, // High volatility threshold
  TREND_THRESHOLD: 0.02, // Strong trend threshold
  MOMENTUM_WINDOW: 10, // Momentum calculation window
};
```

### Market Analysis Settings

```typescript
const MARKET_CONFIG = {
  RSI_PERIOD: 14, // RSI calculation period
  ATR_PERIOD: 14, // ATR calculation period
  SUPPORT_RESISTANCE_PERIOD: 20, // Support/resistance period
  VOLUME_ANALYSIS_PERIOD: 10, // Volume analysis period
  OVERBOUGHT_THRESHOLD: 70, // RSI overbought level
  OVERSOLD_THRESHOLD: 30, // RSI oversold level
};
```

## Performance Considerations

### Optimization Strategies

- **Tensor Reuse**: Reuse tensors to minimize memory allocation
- **Batch Processing**: Process multiple predictions efficiently
- **Caching**: Cache computed features and indicators
- **Async Processing**: Use async/await for non-blocking operations

### Computational Complexity

- **Single Prediction**: O(timesteps × features × model_complexity)
- **Batch Prediction**: O(batch_size × timesteps × features × model_complexity)
- **Market Analysis**: O(analysis_window × indicators)

## Error Handling

### Data Validation

- **Length Checks**: Ensures sufficient historical data
- **Index Bounds**: Validates time index within data range
- **Feature Validation**: Checks feature sequence integrity
- **Model Validation**: Verifies model is loaded and ready

### Prediction Errors

- **Model Failures**: Handles model inference errors gracefully
- **Feature Errors**: Manages feature computation failures
- **Memory Errors**: Handles tensor memory issues
- **Timeout Errors**: Manages prediction timeouts

### Fallback Mechanisms

- **Default Predictions**: Provides sensible defaults on failure
- **Error Recovery**: Attempts to recover from transient errors
- **Logging**: Comprehensive error logging for debugging

## Dependencies

### External Dependencies

- **TensorFlow.js**: Neural network inference
- **FeatureCalculator**: Technical indicator computation
- **FeatureSequenceGenerator**: Sequence generation

### Internal Dependencies

- **Array Methods**: Native JavaScript array operations
- **Mathematical Functions**: Math library for calculations
- **Async/Await**: Promise-based asynchronous operations

## Testing

### Unit Testing Strategy

- **Model Inference**: Test neural network predictions
- **Feature Generation**: Validate technical indicator calculations
- **Market Analysis**: Test market condition analysis
- **Signal Refinement**: Verify rule-based logic
- **Error Handling**: Test error scenarios and recovery

### Test Data Requirements

- **Historical Data**: Real BTC price/volume data
- **Known Patterns**: Data with predictable outcomes
- **Edge Cases**: Minimal data, extreme values, missing data

### Performance Testing

- **Memory Usage**: Monitor tensor memory consumption
- **Execution Time**: Measure prediction performance
- **Scalability**: Test with large datasets

## Future Enhancements

### Potential Improvements

- **Multi-timeframe Analysis**: Support different time intervals
- **Ensemble Models**: Combine multiple model predictions
- **Real-time Processing**: Streaming prediction capabilities
- **Custom Indicators**: User-defined technical indicators

### Advanced Features

- **Sentiment Integration**: Incorporate market sentiment data
- **Regime Detection**: Market condition-aware predictions
- **Adaptive Parameters**: Dynamic parameter adjustment
- **Portfolio Optimization**: Multi-asset prediction support

### Integration Enhancements

- **API Integration**: RESTful prediction endpoints
- **Database Integration**: Store prediction results
- **Real-time Feeds**: Connect to live market data
- **Alert System**: Automated trading signal alerts
