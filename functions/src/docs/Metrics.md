# Metrics Class Documentation

## Overview

The `Metrics` class is a comprehensive evaluation and loss function library for the Cardano trading system's machine learning models. It provides specialized metrics for binary classification problems in cryptocurrency trading, including custom loss functions, precision/recall calculations, F1 scores, and comprehensive model evaluation capabilities.

## Architecture

The Metrics class follows a utility-based architecture with specialized trading metrics:

```
Loss Functions → Trading Metrics → Model Evaluation → Performance Analysis
     ↓              ↓                    ↓                ↓
Focal Loss      Precision/Recall    Confusion Matrix   Comprehensive Reports
Alpha Weighting  Buy/Sell Specific   Model Predictions  Performance Metrics
```

### Key Responsibilities

- **Custom Loss Functions**: Implements focal loss with alpha weighting for imbalanced datasets
- **Trading-Specific Metrics**: Provides buy/sell specific precision, recall, and F1 scores
- **Model Evaluation**: Comprehensive model performance assessment
- **Performance Analysis**: Detailed trading strategy performance metrics
- **TensorFlow Integration**: Seamless integration with TensorFlow.js models

## Class Structure

```typescript
export class Metrics {
  // Loss Functions
  static focalLoss(
    yTrue: tf.Tensor,
    yPred: tf.Tensor,
    gamma?: number,
    alphaArr?: [number, number]
  ): tf.Scalar;

  // Precision Metrics
  static precisionBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static precisionSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static customPrecision(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar };

  // Recall Metrics
  static recallBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static recallSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static customRecall(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar };

  // F1 Score Metrics
  static customF1Buy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static customF1Sell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
  static customF1(
    yTrue: tf.Tensor,
    yPred: tf.Tensor
  ): { buy: tf.Scalar; sell: tf.Scalar };

  // Model Evaluation
  static calculateMetrics(
    predictedLabels: number[],
    yArray: number[]
  ): MetricsResult;
  static async evaluateModel(
    model: tf.LayersModel,
    X: tf.Tensor,
    y: tf.Tensor
  ): Promise<void>;
}
```

## Core Interfaces

### MetricsResult

```typescript
interface MetricsResult {
  precisionBuy: number; // Precision for buy predictions
  precisionSell: number; // Precision for sell predictions
  recallBuy: number; // Recall for buy predictions
  recallSell: number; // Recall for sell predictions
  f1Buy: number; // F1 score for buy predictions
  f1Sell: number; // F1 score for sell predictions
}
```

## Loss Functions

### Focal Loss

```typescript
static focalLoss(
  yTrue: tf.Tensor,
  yPred: tf.Tensor,
  gamma: number = TRAINING_CONFIG.GAMMA,
  alphaArr: [number, number] = TRAINING_CONFIG.ALPHA
): tf.Scalar
```

**Purpose**: Addresses class imbalance and hard example mining in trading datasets.

**Parameters:**

- `yTrue: tf.Tensor` - Ground truth labels
- `yPred: tf.Tensor` - Model predictions
- `gamma: number` - Focusing parameter (default: from TRAINING_CONFIG)
- `alphaArr: [number, number]` - Class weights [sell_weight, buy_weight]

**Mathematical Formula**:

```
FL(pt) = -αt * (1 - pt)^γ * log(pt)
```

Where:

- `pt` = predicted probability for true class
- `γ` = focusing parameter (reduces loss for easy examples)
- `αt` = class weight for true class

**Benefits**:

- **Class Imbalance Handling**: Alpha weighting for underrepresented classes
- **Hard Example Mining**: Gamma parameter focuses on difficult predictions
- **Trading Optimization**: Specifically designed for buy/sell classification

## Trading-Specific Metrics

### Precision Metrics

#### Buy Precision

```typescript
static precisionBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Measures accuracy of buy predictions
- **Formula**: True Positive Buys / Total Predicted Buys
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: How reliable are buy signals?

#### Sell Precision

```typescript
static precisionSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Measures accuracy of sell predictions
- **Formula**: True Positive Sells / Total Predicted Sells
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: How reliable are sell signals?

### Recall Metrics

#### Buy Recall

```typescript
static recallBuy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Measures coverage of actual buy opportunities
- **Formula**: True Positive Buys / Total Actual Buys
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: How many profitable opportunities were captured?

#### Sell Recall

```typescript
static recallSell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Measures coverage of actual sell opportunities
- **Formula**: True Positive Sells / Total Actual Sells
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: How many profitable exits were captured?

### F1 Score Metrics

#### Buy F1 Score

```typescript
static customF1Buy(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Harmonic mean of buy precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: Balanced measure of buy signal quality

#### Sell F1 Score

```typescript
static customF1Sell(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar
```

- **Purpose**: Harmonic mean of sell precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 - 1.0 (higher is better)
- **Trading Context**: Balanced measure of sell signal quality

## Model Evaluation

### Comprehensive Metrics Calculation

```typescript
static calculateMetrics(predictedLabels: number[], yArray: number[]): MetricsResult
```

**Purpose**: Computes all trading metrics from raw predictions and labels.

**Parameters:**

- `predictedLabels: number[]` - Model predictions (0 = sell, 1 = buy)
- `yArray: number[]` - Ground truth labels (0 = sell, 1 = buy)

**Returns**: Complete metrics object with precision, recall, and F1 scores.

**Process**:

1. **Confusion Matrix Calculation**: True positives, false positives, etc.
2. **Metric Computation**: Precision, recall, and F1 for both classes
3. **Edge Case Handling**: Division by zero protection
4. **Result Assembly**: Structured metrics object

### Model Performance Evaluation

```typescript
static async evaluateModel(
  model: tf.LayersModel,
  X: tf.Tensor,
  y: tf.Tensor
): Promise<void>
```

**Purpose**: Comprehensive evaluation of trained trading models.

**Parameters:**

- `model: tf.LayersModel` - Trained TensorFlow.js model
- `X: tf.Tensor` - Input features for evaluation
- `y: tf.Tensor` - Ground truth labels

**Output**: Detailed console report including:

- Prediction distribution (buy/sell counts)
- Precision, recall, and F1 scores
- Confusion matrix
- Performance summary

## Usage Examples

### Custom Loss Function in Training

```typescript
import { Metrics } from "./Metrics";
import * as tf from "@tensorflow/tfjs-node";

// Define custom loss function
const customLoss = (yTrue: tf.Tensor, yPred: tf.Tensor) => {
  return Metrics.focalLoss(yTrue, yPred, 2.0, [0.3, 0.7]);
};

// Use in model compilation
model.compile({
  optimizer: "adam",
  loss: customLoss,
  metrics: ["accuracy"],
});
```

### Individual Metric Calculation

```typescript
// Calculate buy precision
const buyPrecision = Metrics.precisionBuy(yTrue, yPred);
console.log(`Buy Precision: ${await buyPrecision.data()}`);

// Calculate sell recall
const sellRecall = Metrics.recallSell(yTrue, yPred);
console.log(`Sell Recall: ${await sellRecall.data()}`);

// Get comprehensive F1 scores
const f1Scores = Metrics.customF1(yTrue, yPred);
console.log(`Buy F1: ${await f1Scores.buy.data()}`);
console.log(`Sell F1: ${await f1Scores.sell.data()}`);
```

### Model Evaluation

```typescript
// Evaluate model performance
await Metrics.evaluateModel(model, testX, testY);

// Expected output:
// Predicted Buy: 150, Sell: 350
// Precision Buy: 0.8234, Precision Sell: 0.9143
// Recall Buy: 0.7567, Recall Sell: 0.8921
// F1 Buy: 0.7889, F1 Sell: 0.9030
// Confusion Matrix: [[312, 38], [32, 118]]
```

### Batch Metrics Calculation

```typescript
// Calculate metrics from arrays
const predictions = [1, 0, 1, 1, 0, 0, 1, 0];
const actual = [1, 0, 1, 0, 0, 1, 1, 0];

const metrics = Metrics.calculateMetrics(predictions, actual);
console.log("Buy Precision:", metrics.precisionBuy);
console.log("Sell Recall:", metrics.recallSell);
console.log("Buy F1:", metrics.f1Buy);
```

## Performance Considerations

### TensorFlow.js Optimization

- **Memory Management**: Proper tensor disposal to prevent memory leaks
- **Batch Processing**: Efficient tensor operations for large datasets
- **GPU Acceleration**: Leverages TensorFlow.js GPU capabilities when available

### Computational Efficiency

- **Single Pass Calculations**: Metrics computed in single iteration
- **Vectorized Operations**: Tensor operations for parallel processing
- **Minimal Memory Allocation**: Reuses tensors where possible

## Error Handling

### Edge Case Protection

- **Division by Zero**: Adds small epsilon (1e-6) to prevent errors
- **Empty Predictions**: Handles cases with no predictions for a class
- **Invalid Inputs**: Validates tensor shapes and data types

### Robust Evaluation

- **Missing Data**: Graceful handling of incomplete datasets
- **Model Failures**: Error handling for prediction failures
- **Memory Issues**: Automatic tensor cleanup and disposal

## Dependencies

### External Dependencies

- **TensorFlow.js**: Core machine learning framework
- **@tensorflow/tfjs-node**: Node.js TensorFlow.js backend

### Internal Dependencies

- **TRAINING_CONFIG**: Configuration constants for loss function parameters
- **Tensor Operations**: TensorFlow.js tensor manipulation methods

## Configuration

### Training Configuration Constants

```typescript
// From constants.ts
TRAINING_CONFIG = {
  GAMMA: 2.0, // Focal loss focusing parameter
  ALPHA: [0.3, 0.7], // Class weights [sell, buy]
};
```

### Metric Thresholds

- **Precision Threshold**: 0.7+ for reliable trading signals
- **Recall Threshold**: 0.6+ for capturing opportunities
- **F1 Threshold**: 0.65+ for balanced performance

## Testing

### Unit Testing Strategy

- **Individual Metrics**: Test each metric calculation separately
- **Edge Cases**: Test with extreme values and edge cases
- **Tensor Operations**: Validate TensorFlow.js integration
- **Performance Tests**: Measure computation time for large datasets

### Test Data Requirements

- **Balanced Datasets**: Equal buy/sell representation
- **Imbalanced Datasets**: Realistic trading class distributions
- **Edge Cases**: Zero predictions, single class predictions
- **Large Datasets**: Performance testing with realistic data sizes

## Future Enhancements

### Potential Improvements

- **Additional Metrics**: Sharpe ratio, maximum drawdown, profit factor
- **Real-time Evaluation**: Streaming metrics for live trading
- **Custom Thresholds**: User-defined metric thresholds
- **Performance Benchmarking**: Comparison with baseline strategies

### Advanced Features

- **Multi-class Support**: Extension beyond binary classification
- **Time-series Metrics**: Sequence-aware evaluation metrics
- **Risk-adjusted Returns**: Risk metrics integration
- **Portfolio Metrics**: Multi-asset portfolio evaluation

### Integration Enhancements

- **Database Logging**: Persistent metric storage
- **API Endpoints**: RESTful metric calculation services
- **Visualization**: Chart and dashboard integration
- **Alerting**: Performance threshold notifications
