# TradeModelPredictor Class Documentation

## Overview

The `TradeModelPredictor` class is the core prediction engine of the Cardano trading system that combines machine learning model inference with advanced technical analysis to generate comprehensive trading signals. It leverages a pre-trained deep learning model to predict buy/sell probabilities while computing additional market indicators for enhanced decision-making.

## Architecture

The TradeModelPredictor implements a sophisticated prediction pipeline:

```
Input Data → Feature Generation → Model Inference → Technical Analysis → Comprehensive Output
     ↓              ↓                ↓                ↓                ↓
[ADA/BTC Prices   Feature        Neural Network   Market Metrics   Trading Signals
 & Volumes]       Sequences      Predictions      & Indicators     & Confidence
```

### Key Design Principles

- **Hybrid Approach**: Combines ML predictions with traditional technical analysis
- **Real-time Processing**: Optimized for live market data analysis
- **Comprehensive Output**: Provides multiple indicators for informed decisions
- **Memory Management**: Efficient TensorFlow.js tensor handling
- **Error Handling**: Robust validation and error management

## Class Structure

```typescript
export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps: number;
  private model: tf.LayersModel;
  private isWeightsLoaded: boolean;

  constructor();

  private async loadWeightsAsync(): Promise<void>;
  public async predict(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[]
  ): Promise<PredictionResult>;
}
```

### Core Dependencies

- **ModelWeightManager**: Manages model weights and feature normalization
- **FeatureSequenceGenerator**: Creates feature sequences for model input
- **TradeModelFactory**: Creates the neural network architecture
- **FirebaseService**: Handles data persistence and retrieval
- **TensorFlow.js**: Core ML inference engine

## Prediction Process

### 1. Data Preparation

#### Input Validation

```typescript
// Validate input data arrays
if (
  sequence.length !== MODEL_CONFIG.TIMESTEPS ||
  sequence[0].length !== MODEL_CONFIG.FEATURE_COUNT
) {
  throw new Error(
    `Sequence shape mismatch: expected [${MODEL_CONFIG.TIMESTEPS}, ${
      MODEL_CONFIG.FEATURE_COUNT
    }], got [${sequence.length}, ${sequence[0]?.length || 0}]`
  );
}
```

#### Sequence Generation

```typescript
const startIndex = Math.max(0, adaPrices.length - this.timesteps - 1);
const endIndex = adaPrices.length - 1;
const sequence = this.sequenceGenerator
  .generateSequence(
    adaPrices,
    adaVolumes,
    btcPrices,
    btcVolumes,
    startIndex,
    endIndex
  )
  .slice(-this.timesteps);
```

### 2. Feature Normalization

#### Tensor Creation

```typescript
const features = tf.tensor3d(
  [sequence],
  [1, MODEL_CONFIG.TIMESTEPS, MODEL_CONFIG.FEATURE_COUNT]
);
```

#### Normalization Process

```typescript
const means = tf.tensor1d(this.weightManager.getFeatureMeans());
const stds = tf.tensor1d(this.weightManager.getFeatureStds());
const featuresNormalized = features.sub(means).div(stds.add(1e-6));
```

**Purpose**: Standardizes features using pre-computed mean and standard deviation values
**Benefits**: Ensures model receives properly scaled input data
**Implementation**: Uses epsilon (1e-6) to prevent division by zero

### 3. Model Inference

#### Prediction Execution

```typescript
const logits = this.model.predict(featuresNormalized) as tf.Tensor2D;
const logitsArray = await logits.data();
const [sellLogit, buyLogit] = [logitsArray[0], logitsArray[1]];
```

#### Probability Calculation

```typescript
const probs = tf.tensor2d([[sellLogit, buyLogit]]).softmax();
const probArray = await probs.data();
const [sellProb, buyProb] = [probArray[0], probArray[1]];
const confidence = Math.max(buyProb, sellProb);
```

**Process**: Raw logits → Softmax activation → Probability distribution
**Output**: Normalized probabilities for sell and buy decisions
**Confidence**: Maximum probability indicating prediction certainty

## Technical Indicators

### 1. ATR (Average True Range)

#### Dynamic Window Sizing

```typescript
const atr = sequence[sequence.length - 1][11];
const momentumWindowSize =
  atr > 0.05 ? 5 : atr > 0.02 ? 10 : atr > 0.01 ? 14 : 20;
```

**Logic**: Higher volatility = shorter momentum window

- **ATR > 0.05**: 5-day window (high volatility)
- **ATR > 0.02**: 10-day window (medium volatility)
- **ATR > 0.01**: 14-day window (low volatility)
- **ATR ≤ 0.01**: 20-day window (very low volatility)

### 2. Momentum Indicators

#### Long-term Momentum

```typescript
const momentumWindow = adaPrices.slice(-momentumWindowSize);
const momentum =
  momentumWindow.length >= 2
    ? (momentumWindow[momentumWindow.length - 1] - momentumWindow[0]) /
      momentumWindow[0]
    : 0;
```

**Purpose**: Measures price movement over dynamic time period
**Calculation**: Percentage change from start to end of window
**Adaptation**: Window size adjusts based on market volatility

#### Short-term Momentum

```typescript
const shortMomentumWindow = adaPrices.slice(-3);
const shortMomentum =
  shortMomentumWindow.length >= 2
    ? (shortMomentumWindow[shortMomentumWindow.length - 1] -
        shortMomentumWindow[0]) /
      shortMomentumWindow[0]
    : 0;
```

**Purpose**: Captures immediate price momentum
**Window**: Fixed 3-day period for consistent short-term analysis
**Use Case**: Detecting rapid market changes and reversals

### 3. Trend Analysis

#### Trend Slope

```typescript
const trendWindow = adaPrices.slice(-momentumWindowSize);
const trendSlope =
  trendWindow.length >= 2
    ? (trendWindow[trendWindow.length - 1] - trendWindow[0]) /
      (trendWindow.length - 1)
    : 0;
```

**Purpose**: Measures linear trend strength and direction
**Calculation**: Average daily price change over momentum window
**Interpretation**: Positive = uptrend, Negative = downtrend

### 4. Advanced Indicators

#### Momentum Divergence

```typescript
const momentumDivergence = shortMomentum - momentum;
```

**Purpose**: Detects potential trend reversals
**Logic**: Short-term vs. long-term momentum comparison
**Signals**:

- Positive divergence = potential reversal
- Negative divergence = trend continuation

#### Volatility-Adjusted Momentum

```typescript
const volatilityAdjustedMomentum = momentum / (atr || 0.01);
```

**Purpose**: Normalizes momentum by market volatility
**Benefits**: Provides context-aware momentum assessment
**Fallback**: Uses 0.01 as minimum ATR to prevent division issues

#### Trend Strength

```typescript
const trendStrength = trendSlope * volatilityAdjustedMomentum;
```

**Purpose**: Combines trend direction with momentum magnitude
**Interpretation**: Higher absolute values indicate stronger trends
**Use Case**: Trend confirmation and strength assessment

#### ATR Breakout

```typescript
const atrWindow = sequence.slice(-PERIODS.ATR_BREAKOUT).map((s) => s[11]);
const atrSma =
  atrWindow.length >= PERIODS.ATR_BREAKOUT
    ? atrWindow.reduce((sum, val) => sum + val, 0) / atrWindow.length
    : atr;
const atrBreakout = atr / atrSma;
```

**Purpose**: Detects volatility breakouts
**Calculation**: Current ATR relative to moving average
**Interpretation**: Values > 1 indicate above-average volatility

## Output Interface

### PredictionResult Structure

```typescript
interface PredictionResult {
  // Model Predictions
  buyLogit: number; // Raw buy signal strength
  sellLogit: number; // Raw sell signal strength
  buyProb: number; // Normalized buy probability
  sellProb: number; // Normalized sell probability
  confidence: number; // Prediction confidence level

  // Technical Indicators
  momentum: number; // Long-term momentum
  shortMomentum: number; // Short-term momentum
  trendSlope: number; // Trend direction and strength
  atr: number; // Current volatility measure

  // Advanced Metrics
  momentumDivergence: number; // Short vs. long momentum difference
  volatilityAdjustedMomentum: number; // Volatility-normalized momentum
  trendStrength: number; // Combined trend and momentum
  atrBreakout: number; // Volatility breakout indicator
}
```

### Output Interpretation

#### Trading Signals

- **buyProb > sellProb**: Buy recommendation
- **sellProb > buyProb**: Sell recommendation
- **confidence**: Reliability of the prediction

#### Market Conditions

- **momentum**: Overall price movement strength
- **trendSlope**: Market direction and intensity
- **atr**: Current market volatility level

#### Advanced Insights

- **momentumDivergence**: Potential reversal signals
- **trendStrength**: Confirmed trend strength
- **atrBreakout**: Volatility regime changes

## Usage Examples

### Basic Prediction

```typescript
import { TradeModelPredictor } from "./TradeModelPredictor";

// Initialize predictor
const predictor = new TradeModelPredictor();

// Prepare market data
const adaPrices = [
  /* 30+ days of ADA prices */
];
const adaVolumes = [
  /* 30+ days of ADA volumes */
];
const btcPrices = [
  /* 30+ days of BTC prices */
];
const btcVolumes = [
  /* 30+ days of BTC volumes */
];

// Get prediction
const prediction = await predictor.predict(
  adaPrices,
  adaVolumes,
  btcPrices,
  btcVolumes
);

console.log(`Buy Probability: ${(prediction.buyProb * 100).toFixed(2)}%`);
console.log(`Sell Probability: ${(prediction.sellProb * 100).toFixed(2)}%`);
console.log(`Confidence: ${(prediction.confidence * 100).toFixed(2)}%`);
console.log(`Trend Slope: ${prediction.trendSlope.toFixed(4)}`);
console.log(`ATR: ${prediction.atr.toFixed(4)}`);
```

### Trading Decision Logic

```typescript
// Simple trading strategy
if (prediction.buyProb > 0.6 && prediction.confidence > 0.7) {
  if (prediction.trendStrength > 0.01 && prediction.momentum > 0.02) {
    console.log("Strong Buy Signal");
    // Execute buy order
  } else {
    console.log("Weak Buy Signal - Wait for confirmation");
  }
} else if (prediction.sellProb > 0.6 && prediction.confidence > 0.7) {
  if (prediction.trendStrength < -0.01 || prediction.momentum < -0.02) {
    console.log("Strong Sell Signal");
    // Execute sell order
  } else {
    console.log("Weak Sell Signal - Monitor closely");
  }
} else {
  console.log("Hold Position - No clear signal");
}
```

### Risk Management

```typescript
// Volatility-based position sizing
const positionSize = prediction.confidence * (1 / prediction.atr);

// ATR breakout monitoring
if (prediction.atrBreakout > 1.5) {
  console.log("High volatility detected - reduce position size");
  positionSize *= 0.5;
}

// Trend confirmation
if (Math.abs(prediction.trendStrength) < 0.005) {
  console.log("Weak trend - use smaller position sizes");
  positionSize *= 0.7;
}
```

## Performance Considerations

### Memory Management

#### Tensor Disposal

```typescript
// Clean up tensors to prevent memory leaks
features.dispose();
featuresNormalized.dispose();
logits.dispose();
probs.dispose();
means.dispose();
stds.dispose();
```

**Critical**: Proper tensor cleanup prevents memory accumulation
**Pattern**: Dispose of all intermediate tensors after use
**Benefit**: Maintains stable memory usage during continuous predictions

### Computational Efficiency

#### Optimized Operations

- **Batch Processing**: Single prediction per call for real-time use
- **Efficient Normalization**: Pre-computed means/stds for fast scaling
- **Minimal Tensor Operations**: Streamlined inference pipeline

#### Performance Monitoring

```typescript
// Uncomment for performance analysis
// const startTime = performance.now();
// const endTime = performance.now();
// console.log(`Prediction executed in ${(endTime - startTime).toFixed(2)} ms`);
```

## Error Handling

### Input Validation

#### Data Requirements

- **Minimum Length**: Arrays must contain sufficient historical data
- **Shape Consistency**: Feature sequences must match expected dimensions
- **Data Quality**: Valid numeric values for all inputs

#### Error Types

```typescript
// Sequence shape mismatch
throw new Error(`Sequence shape mismatch: expected [${MODEL_CONFIG.TIMESTEPS}, ${MODEL_CONFIG.FEATURE_COUNT}], got [${sequence.length}, ${sequence[0]?.length || 0}]`);

// Weight loading failures
catch (error) {
  console.error("Failed to load weights:", error);
  throw error;
}
```

### Fallback Mechanisms

#### ATR Fallback

```typescript
const volatilityAdjustedMomentum = momentum / (atr || 0.01);
```

**Purpose**: Prevents division by zero
**Fallback Value**: 0.01 as minimum ATR threshold
**Benefit**: Ensures calculation stability

## Dependencies

### External Dependencies

- **@tensorflow/tfjs-node**: Core ML inference engine
- **TensorFlow.js**: Neural network operations and tensor management

### Internal Dependencies

- **ModelWeightManager**: Weight loading and feature normalization
- **FeatureSequenceGenerator**: Feature sequence creation
- **TradeModelFactory**: Neural network architecture
- **FirebaseService**: Data persistence and retrieval
- **MODEL_CONFIG**: Model configuration constants
- **PERIODS**: Technical analysis period constants

## Configuration

### Model Parameters

```typescript
// From MODEL_CONFIG
TIMESTEPS: number; // Number of historical time steps
FEATURE_COUNT: number; // Number of features per time step

// From PERIODS
ATR_BREAKOUT: number; // ATR calculation window size
```

### Feature Configuration

```typescript
// Feature indices in sequence
const atrIndex = 11; // ATR feature position
const priceIndex = 0; // Price feature position
const volumeIndex = 1; // Volume feature position
```

## Testing

### Unit Testing Strategy

- **Input Validation**: Test with various data array lengths and shapes
- **Feature Generation**: Verify correct sequence creation
- **Model Inference**: Test prediction accuracy and output format
- **Error Handling**: Test error conditions and edge cases

### Integration Testing

- **End-to-End Pipeline**: Test complete prediction workflow
- **Data Compatibility**: Verify with real market data
- **Performance Testing**: Measure prediction latency and memory usage
- **Output Validation**: Verify technical indicator calculations

### Test Data Requirements

- **Historical Data**: Real ADA/BTC price and volume data
- **Edge Cases**: Extreme market conditions and volatility
- **Data Quality**: Missing data, outliers, and noise scenarios
- **Performance Data**: Large datasets for scalability testing

## Monitoring and Logging

### Performance Metrics

- **Prediction Latency**: Time to generate predictions
- **Memory Usage**: Tensor memory consumption
- **Model Accuracy**: Prediction confidence and accuracy
- **Error Rates**: Failed predictions and error frequency

### Debug Information

```typescript
// Comprehensive logging (commented out in production)
console.log(`ATR: ${atr.toFixed(4)}, Momentum Window: ${momentumWindowSize}`);
console.log(
  `Logits: [Sell: ${sellLogit.toFixed(4)}, Buy: ${buyLogit.toFixed(4)}]`
);
console.log(
  `Probs: [Sell: ${sellProb.toFixed(4)}, Buy: ${buyProb.toFixed(4)}]`
);
console.log(
  `Confidence: ${confidence.toFixed(4)}, Momentum: ${momentum.toFixed(4)}`
);
```

## Future Enhancements

### Potential Improvements

- **Batch Predictions**: Process multiple time periods simultaneously
- **Ensemble Methods**: Combine multiple model predictions
- **Real-time Updates**: Continuous model retraining and updates
- **Advanced Indicators**: Additional technical analysis metrics

### Advanced Features

- **Confidence Intervals**: Uncertainty quantification for predictions
- **Market Regime Detection**: Automatic market condition classification
- **Adaptive Parameters**: Dynamic adjustment based on market conditions
- **Performance Analytics**: Detailed prediction performance tracking

### Integration Enhancements

- **API Endpoints**: RESTful prediction services
- **WebSocket Support**: Real-time prediction streaming
- **Dashboard Integration**: Real-time prediction visualization
- **Alert System**: Automated trading signal notifications
