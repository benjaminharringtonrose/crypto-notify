# TradeModelTrainer Class Documentation

## Overview

The `TradeModelTrainer` class is the comprehensive training engine of the Bitcoin trading system that orchestrates the complete machine learning model training pipeline. It handles data preparation, model creation, training execution, performance monitoring, and weight persistence, providing a unified interface for training deep learning models for cryptocurrency price prediction.

## Architecture

The TradeModelTrainer implements a sophisticated training pipeline:

```
Data Preparation → Model Creation → Training Execution → Performance Monitoring → Weight Persistence
       ↓               ↓               ↓               ↓               ↓
   Historical Data   Neural Network   Epoch Training   Callbacks &     Firebase Storage
   Processing        Architecture     with Validation   Metrics         & Local Backup
```

### Key Design Principles

- **End-to-End Pipeline**: Complete training workflow from data to deployed model
- **Advanced Training Techniques**: Custom loss functions, callbacks, and optimization
- **Performance Monitoring**: Comprehensive metrics and validation tracking
- **Weight Management**: Best weight tracking and automatic restoration
- **Memory Efficiency**: Proper tensor disposal and memory management
- **Robust Validation**: Train/validation split with performance tracking

## Class Structure

```typescript
export class TradeModelTrainer {
  private readonly config: ModelConfig;
  private bucket: admin.storage.Bucket;
  private model: tf.LayersModel | null;
  private lossFn: Function;
  private dataProcessor: DataProcessor;
  private bestThreshold: number;
  private bestWeights: tf.Tensor[];
  private bestValF1Buy: number;

  constructor();

  private async trainModel(
    X: number[][][],
    y: number[]
  ): Promise<{ X_mean: tf.Tensor; X_std: tf.Tensor }>;
  private async saveModelWeights(
    X_mean: tf.Tensor,
    X_std: tf.Tensor
  ): Promise<void>;
  public async train(): Promise<void>;
  public getBestThreshold(): number;
}
```

### Core Dependencies

- **Firebase Admin**: Cloud storage for model weights
- **TensorFlow.js**: Core ML training framework
- **Custom Callbacks**: Advanced training monitoring and control
- **DataProcessor**: Historical data preparation and feature engineering
- **TradeModelFactory**: Neural network architecture creation
- **Metrics**: Custom loss functions and evaluation metrics

## Configuration

### Model Configuration

```typescript
private readonly config: ModelConfig = {
  timesteps: MODEL_CONFIG.TIMESTEPS,           // Number of time steps
  epochs: TRAINING_CONFIG.EPOCHS,              // Training epochs
  batchSize: TRAINING_CONFIG.BATCH_SIZE,       // Batch size for training
  initialLearningRate: TRAINING_CONFIG.INITIAL_LEARNING_RATE, // Starting learning rate
};
```

### Training Configuration Constants

```typescript
// From TRAINING_CONFIG
EPOCHS: number; // Total training epochs
BATCH_SIZE: number; // Training batch size
INITIAL_LEARNING_RATE: number; // Starting learning rate
OUTPUT_CLASSES: number; // Number of output classes (2: Buy/Sell)
SHUFFLE_CHUNK_SIZE: number; // Chunk size for data shuffling
TRAIN_SPLIT: number; // Training data proportion
PREFETCH_BUFFER: number; // Data prefetch buffer size
MS_TO_SECONDS: number; // Time conversion constant
BYTES_TO_MB: number; // Memory conversion constant
```

## Training Process

### 1. Data Preparation

#### Input Validation

```typescript
console.log(
  `Input data dimensions: [${X.length}, ${X[0].length}, ${X[0][0].length}]`
);

if (X[0].length !== this.config.timesteps) {
  throw new Error(
    `Data timestep mismatch: expected ${this.config.timesteps}, got ${X[0].length}`
  );
}
```

**Purpose**: Ensures data dimensions match expected model input shape
**Validation**: Checks timesteps, features, and sample count consistency

#### Tensor Creation

```typescript
// Note: Import FeatureDetector first: import { FeatureDetector } from "./FeatureDetector";
const X_tensor = tf.tensor3d(X, [
  X.length,
  this.config.timesteps,
  FeatureDetector.getFeatureCount(),
]);

const y_tensor = tf.tensor2d(
  y.map((label) => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]),
  [y.length, TRAINING_CONFIG.OUTPUT_CLASSES]
);
```

**Input Shape**: `[samples, timesteps, features]`
**Label Shape**: `[samples, 2]` (one-hot encoded: [Sell, Buy])
**Transformation**: Converts numeric labels to one-hot encoding

### 2. Feature Normalization

#### Statistics Computation

```typescript
const featureStats = this.dataProcessor.computeFeatureStats(X.flat(1));
const X_mean = tf.tensor1d(featureStats.mean);
const X_std = tf.tensor1d(featureStats.std);
```

**Purpose**: Computes mean and standard deviation for feature normalization
**Method**: Flattens all features across all samples for global statistics
**Output**: Feature-wise mean and standard deviation tensors

#### Normalization Application

```typescript
const X_normalized = X_tensor.sub(X_mean).div(X_std);
```

**Formula**: `(X - μ) / σ`
**Benefits**: Standardizes features to zero mean and unit variance
**Impact**: Improves training stability and convergence

### 3. Data Splitting and Shuffling

#### Chunked Shuffling

```typescript
const totalSamples = X.length;
const numChunks = Math.ceil(totalSamples / TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE);
const chunkedIndices: number[][] = [];

for (let i = 0; i < numChunks; i++) {
  const start = i * TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE;
  const end = Math.min(
    (i + 1) * TRAINING_CONFIG.SHUFFLE_CHUNK_SIZE,
    totalSamples
  );
  chunkedIndices.push(Array.from({ length: end - start }, (_, j) => start + j));
}

const shuffledChunks = [...chunkedIndices];
tf.util.shuffle(shuffledChunks);
const shuffledIndices = shuffledChunks.flat();
```

**Purpose**: Implements memory-efficient shuffling for large datasets
**Method**: Shuffles chunks of indices rather than entire dataset
**Benefits**: Maintains data order within chunks while ensuring randomness

#### Train-Validation Split

```typescript
const trainSize = Math.floor(totalSamples * TRAINING_CONFIG.TRAIN_SPLIT);
const trainIndices = shuffledIndices.slice(0, trainSize);
const valIndices = shuffledIndices.slice(trainSize);

const X_train = tf.gather(X_normalized, trainIndices);
const y_train = tf.gather(y_tensor, trainIndices);
const X_val = tf.gather(X_normalized, valIndices);
const y_val = tf.gather(y_tensor, valIndices);
```

**Split Ratio**: Configurable training/validation proportion
**Method**: Uses shuffled indices for unbiased split
**Tensor Operations**: Efficient data selection using `tf.gather`

### 4. Dataset Creation

#### Training Dataset

```typescript
const trainDataset = tf.data
  .zip({
    xs: tf.data.array(X_train.arraySync() as number[][][]),
    ys: tf.data.array(y_train.arraySync() as number[][]),
  })
  .batch(this.config.batchSize)
  .prefetch(TRAINING_CONFIG.PREFETCH_BUFFER);
```

**Structure**: Paired input-output data with batching
**Batching**: Configurable batch size for training
**Prefetching**: Optimized data loading for GPU training

#### Validation Dataset

```typescript
const valDataset = tf.data
  .zip({
    xs: tf.data.array(X_val.arraySync() as number[][][]),
    ys: tf.data.array(y_val.arraySync() as number[][]),
  })
  .batch(this.config.batchSize);
```

**Purpose**: Separate dataset for validation during training
**Batching**: Same batch size as training for consistency

### 5. Model Creation and Compilation

#### Model Initialization

```typescript
// Note: Import FeatureDetector first: import { FeatureDetector } from "./FeatureDetector";
const factory = new TradeModelFactory(
  this.config.timesteps,
  FeatureDetector.getFeatureCount()
);
this.model = factory.createModel();
```

**Architecture**: Hybrid CNN-LSTM model from TradeModelFactory
**Configuration**: Uses training configuration for model parameters

#### Model Compilation

```typescript
this.model.compile({
  optimizer: tf.train.adam(this.config.initialLearningRate),
  loss: this.lossFn, // Metrics.focalLoss
  metrics: [
    "binaryAccuracy",
    Metrics.precisionBuy,
    Metrics.precisionSell,
    Metrics.recallBuy,
    Metrics.recallSell,
    Metrics.customF1Buy,
    Metrics.customF1Sell,
  ],
});
```

**Optimizer**: Adam optimizer with configurable learning rate
**Loss Function**: Custom Focal Loss for handling class imbalance
**Metrics**: Comprehensive evaluation metrics for buy/sell predictions

## Advanced Training Features

### 1. Custom Callbacks

#### Prediction Logger Callback

```typescript
const predictionLoggerCallback = new PredictionLoggerCallback(
  X_val,
  y_val,
  this
);
```

**Purpose**: Logs predictions and performance during training
**Input**: Validation data for real-time monitoring
**Output**: Detailed prediction analysis and logging

#### Learning Rate Scheduling

```typescript
const lrCallback = new ExponentialDecayLRCallback();
```

**Purpose**: Automatically adjusts learning rate during training
**Method**: Exponential decay based on performance metrics
**Benefits**: Faster convergence and better final performance

#### Gradient Clipping

```typescript
const gradientClippingCallback = new GradientClippingCallback(
  this.lossFn,
  X_val,
  y_val
);
```

**Purpose**: Prevents gradient explosion during training
**Method**: Clips gradients to maximum threshold
**Benefits**: Training stability and convergence

### 2. Best Weight Tracking

#### Performance Monitoring

```typescript
onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
  if (logs) {
    const valF1Buy = logs["val_customF1Buy"] || 0;
    if (valF1Buy > this.bestValF1Buy) {
      this.bestValF1Buy = valF1Buy;
      this.bestWeights = this.model!.getWeights().map((w) => w.clone());
      console.log(
        `New best val_customF1Buy: ${this.bestValF1Buy.toFixed(4)} at epoch ${
          epoch + 1
        }`
      );
    }
  }
};
```

**Metric**: Tracks validation F1 score for buy predictions
**Storage**: Saves best performing weights during training
**Logging**: Reports new best performance achievements

#### Weight Restoration

```typescript
onTrainEnd: async () => {
  if (this.bestWeights.length > 0) {
    this.model!.setWeights(this.bestWeights);
    console.log(
      `Restored weights from best val_customF1Buy: ${this.bestValF1Buy.toFixed(
        4
      )}`
    );
    this.bestWeights.forEach((w) => w.dispose());
  }
};
```

**Purpose**: Restores best performing weights after training
**Timing**: Executes at end of training process
**Cleanup**: Disposes of stored weight copies

### 3. Training Monitoring

#### Epoch Progress Tracking

```typescript
const trainAcc = logs["binaryAccuracy"] || 0;
const valAcc = logs["val_binaryAccuracy"] || 0;
const accSpread = trainAcc - valAcc;

console.log(
  `Epoch ${epoch + 1} - Binary Accuracy: ${trainAcc.toFixed(
    4
  )}, Val Binary Accuracy: ${valAcc.toFixed(4)}, Spread: ${accSpread.toFixed(
    4
  )}`
);
```

**Metrics**: Training and validation accuracy
**Overfitting Detection**: Monitors accuracy spread
**Logging**: Real-time training progress updates

#### Class Distribution Monitoring

```typescript
const trainBuyCount = Array.from(y_train_array).filter((l) => l === 1).length;
console.log(
  `Training set - Buy: ${trainBuyCount}, Sell: ${
    trainSize - trainBuyCount
  }, Buy Ratio: ${(trainBuyCount / trainSize).toFixed(3)}`
);

const valBuyCount = Array.from(y_val_array).filter((l) => l === 1).length;
console.log(
  `Validation set - Buy: ${valBuyCount}, Sell: ${
    totalSamples - trainSize - valBuyCount
  }, Buy Ratio: ${(valBuyCount / (totalSamples - trainSize)).toFixed(3)}`
);
```

**Purpose**: Monitors class balance in training and validation sets
**Metrics**: Buy/sell counts and ratios
**Benefits**: Ensures balanced data distribution

## Weight Persistence

### 1. Weight Extraction

#### Layer-by-Layer Extraction

```typescript
const weights = {
  // Convolutional Layers
  conv1Weights: Array.from(
    await this.model.getLayer("conv1d").getWeights()[0].data()
  ),
  conv1Bias: Array.from(
    await this.model.getLayer("conv1d").getWeights()[1].data()
  ),
  conv2Weights: Array.from(
    await this.model.getLayer("conv1d_2").getWeights()[0].data()
  ),
  conv2Bias: Array.from(
    await this.model.getLayer("conv1d_2").getWeights()[1].data()
  ),

  // LSTM Layers
  lstm1Weights: Array.from(
    await this.model.getLayer("lstm1").getWeights()[0].data()
  ),
  lstm1RecurrentWeights: Array.from(
    await this.model.getLayer("lstm1").getWeights()[1].data()
  ),
  lstm1Bias: Array.from(
    await this.model.getLayer("lstm1").getWeights()[2].data()
  ),
  // ... additional LSTM layers

  // Dense Layers
  dense1Weights: Array.from(
    await this.model.getLayer("dense").getWeights()[0].data()
  ),
  dense1Bias: Array.from(
    await this.model.getLayer("dense").getWeights()[1].data()
  ),
  dense2Weights: Array.from(
    await this.model.getLayer("dense_1").getWeights()[0].data()
  ),
  dense2Bias: Array.from(
    await this.model.getLayer("dense_1").getWeights()[1].data()
  ),

  // Batch Normalization
  bnGamma: Array.from(
    await this.model.getLayer("batchNormalization").getWeights()[0].data()
  ),
  bnBeta: Array.from(
    await this.model.getLayer("batchNormalization").getWeights()[1].data()
  ),
  bnMovingMean: Array.from(
    await this.model.getLayer("batchNormalization").getWeights()[2].data()
  ),
  bnMovingVariance: Array.from(
    await this.model.getLayer("batchNormalization").getWeights()[3].data()
  ),

  // Feature Statistics
  featureMeans: Array.from(await X_mean.data()),
  featureStds: Array.from(await X_std.data()),
};
```

**Comprehensive Coverage**: All model layers and parameters
**Data Conversion**: Tensor data to JavaScript arrays
**Feature Statistics**: Normalization parameters for inference

### 2. Firebase Storage

#### File Upload

```typescript
const weightsJson = JSON.stringify({ weights });
const file = this.bucket.file(FILE_NAMES.WEIGHTS);
await file.save(weightsJson, { contentType: "application/json" });
console.log("Model weights saved to Firebase Storage");
```

**Format**: JSON serialization of all weights
**Storage**: Firebase Cloud Storage for persistence
**Metadata**: Proper content type for JSON data

## Usage Examples

### Basic Training

```typescript
import { TradeModelTrainer } from "./TradeModelTrainer";

// Initialize trainer
const trainer = new TradeModelTrainer();

// Execute complete training pipeline
try {
  await trainer.train();
  console.log("Training completed successfully");
} catch (error) {
  console.error("Training failed:", error);
}
```

### Custom Configuration

```typescript
// Modify training parameters
const trainer = new TradeModelTrainer();

// Access configuration
console.log(`Training epochs: ${trainer.config.epochs}`);
console.log(`Batch size: ${trainer.config.batchSize}`);
console.log(`Learning rate: ${trainer.config.initialLearningRate}`);

// Get best threshold for predictions
const bestThreshold = trainer.getBestThreshold();
console.log(`Best validation threshold: ${bestThreshold}`);
```

### Training Pipeline Integration

```typescript
// Complete ML pipeline
async function runMLPipeline() {
  const trainer = new TradeModelTrainer();

  try {
    // 1. Data preparation and training
    await trainer.train();

    // 2. Get training results
    const bestThreshold = trainer.getBestThreshold();

    // 3. Model evaluation
    console.log(`Training completed with best threshold: ${bestThreshold}`);
  } catch (error) {
    console.error("ML pipeline failed:", error);
    throw error;
  }
}
```

## Performance Considerations

### Memory Management

#### Tensor Disposal

```typescript
finally {
  X_tensor.dispose();
  y_tensor.dispose();
  X_normalized.dispose();
}
```

**Critical**: Proper cleanup prevents memory leaks
**Pattern**: Dispose of all intermediate tensors
**Timing**: Execute in finally block for guaranteed cleanup

#### Memory Monitoring

```typescript
console.log(
  `Memory after training: ${
    tf.memory().numBytes / TRAINING_CONFIG.BYTES_TO_MB
  } MB`
);
```

**Purpose**: Monitor memory usage during training
**Conversion**: Bytes to MB for readability
**Timing**: Checked after training completion

### Computational Efficiency

#### Batch Processing

- **Configurable Batch Size**: Optimized for available memory
- **Data Prefetching**: Efficient data loading for GPU training
- **Chunked Shuffling**: Memory-efficient data randomization

#### Training Optimization

- **Gradient Clipping**: Prevents training instability
- **Learning Rate Scheduling**: Optimized convergence
- **Best Weight Tracking**: Automatic performance optimization

## Error Handling

### Training Failures

#### Data Validation

```typescript
if (X[0].length !== this.config.timesteps) {
  throw new Error(
    `Data timestep mismatch: expected ${this.config.timesteps}, got ${X[0].length}`
  );
}
```

**Validation**: Input data shape consistency
**Error Messages**: Clear diagnostic information
**Fail-Fast**: Early error detection

#### Model Initialization

```typescript
if (!this.model) throw new Error("Model initialization failed");
```

**Check**: Model creation success
**Error**: Clear failure indication
**Recovery**: Proper error propagation

### Exception Handling

#### Training Pipeline

```typescript
try {
  const startTime = performance.now();
  const { X, y } = await this.dataProcessor.prepareData();
  const { X_mean, X_std } = await this.trainModel(X, y);
  await this.saveModelWeights(X_mean, X_std);
  const endTime = performance.now();
  const executionTime = (endTime - startTime) / TRAINING_CONFIG.MS_TO_SECONDS;
  console.log(`Execution time: ${executionTime} seconds`);
} catch (error) {
  console.error("Training failed:", error);
  throw error;
}
```

**Comprehensive**: Covers entire training pipeline
**Logging**: Detailed error information
**Propagation**: Re-throws for higher-level handling

## Dependencies

### External Dependencies

- **firebase-admin**: Cloud storage and persistence
- **@tensorflow/tfjs-node**: Core ML training framework
- **dotenv**: Environment variable management

### Internal Dependencies

- **PredictionLoggerCallback**: Training progress monitoring
- **ExponentialDecayLRCallback**: Learning rate scheduling
- **GradientClippingCallback**: Training stability
- **DataProcessor**: Data preparation and feature engineering
- **TradeModelFactory**: Neural network architecture
- **Metrics**: Custom loss functions and evaluation
- **FirebaseService**: Data persistence infrastructure

## Configuration

### Environment Variables

```typescript
dotenv.config();
```

**Purpose**: Load environment-specific configuration
**Usage**: API keys, storage paths, and environment settings

### Constants Integration

```typescript
import { FILE_NAMES, MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
```

**Centralized**: All configuration in constants file
**Consistent**: Shared across training components
**Maintainable**: Single source of truth for settings

## Testing

### Unit Testing Strategy

- **Data Validation**: Test input data requirements and validation
- **Model Creation**: Verify model initialization and compilation
- **Training Process**: Test training pipeline execution
- **Weight Persistence**: Verify weight saving and loading

### Integration Testing

- **End-to-End Pipeline**: Complete training workflow
- **Data Compatibility**: Real data integration testing
- **Performance Validation**: Training time and memory usage
- **Output Validation**: Model weights and statistics

### Test Data Requirements

- **Historical Data**: Real cryptocurrency price and volume data
- **Edge Cases**: Various data sizes and distributions
- **Performance Data**: Large datasets for scalability testing
- **Quality Scenarios**: Missing data, outliers, and noise

## Monitoring and Logging

### Training Progress

- **Epoch Updates**: Real-time training and validation metrics
- **Performance Tracking**: Best weight monitoring and logging
- **Memory Usage**: Tensor memory consumption tracking
- **Execution Time**: Complete training pipeline timing

### Debug Information

```typescript
console.log(
  `Input data dimensions: [${X.length}, ${X[0].length}, ${X[0][0].length}]`
);
console.log(
  `Training set - Buy: ${trainBuyCount}, Sell: ${
    trainSize - trainBuyCount
  }, Buy Ratio: ${(trainBuyCount / trainSize).toFixed(3)}`
);
console.log(
  `New best val_customF1Buy: ${this.bestValF1Buy.toFixed(4)} at epoch ${
    epoch + 1
  }`
);
```

**Comprehensive**: Detailed training information
**Formatted**: Readable output with proper precision
**Timing**: Real-time updates during training

## Future Enhancements

### Potential Improvements

- **Distributed Training**: Multi-GPU and multi-node training
- **Hyperparameter Optimization**: Automated parameter tuning
- **Model Versioning**: Training history and model comparison
- **Real-time Monitoring**: Live training dashboard integration

### Advanced Features

- **Transfer Learning**: Pre-trained model adaptation
- **Ensemble Methods**: Multiple model training and combination
- **AutoML**: Automated architecture search and optimization
- **Continuous Learning**: Incremental model updates

### Integration Enhancements

- **MLflow Integration**: Experiment tracking and model management
- **Kubernetes Deployment**: Scalable training infrastructure
- **Cloud Training**: AWS/GCP training optimization
- **Model Serving**: Real-time inference API endpoints
