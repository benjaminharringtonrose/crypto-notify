# Train Script Documentation

## Overview

The `train.ts` script is a comprehensive training automation tool that orchestrates the complete machine learning model training process for the Bitcoin trading system. It provides a simple, single-purpose interface for training neural network models that will be used for cryptocurrency price prediction and trading decision making.

## Architecture

The train script follows a streamlined training pipeline:

```
Data Preparation → Model Creation → Training Execution → Weight Persistence → Validation
      ↓                ↓                ↓                ↓                ↓
Historical BTC Data  Neural Network  TensorFlow.js    Firebase Storage  Performance
Feature Engineering  Architecture    Training Loop     Weight Management  Metrics
```

### Key Responsibilities

- **Data Orchestration**: Coordinates data fetching and preprocessing
- **Model Initialization**: Creates and configures neural network architecture
- **Training Execution**: Manages the complete training process
- **Weight Management**: Handles model weight persistence and retrieval
- **Performance Monitoring**: Tracks training progress and metrics
- **Error Handling**: Manages training failures and recovery

## Script Structure

```typescript
// train.ts
import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { DataProcessor } from "../bitcoin/DataProcessor";
import { TradeModelFactory } from "../bitcoin/TradeModelFactory";
import { FirebaseService } from "../api/FirebaseService";

async function main() {
  // Training orchestration logic
}

main().catch(console.error);
```

### Core Dependencies

The script depends on the TradeModelTrainer class from the bitcoin module:

```typescript
import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
```

## Training Process

### 1. Data Preparation

```typescript
// Initialize data processor
const dataProcessor = new DataProcessor();

// Prepare training data
const trainingData = await dataProcessor.prepareTrainingData();

console.log("Training Data Summary:");
console.log(`Sequences: ${trainingData.sequences.length}`);
console.log(`Features per sequence: ${trainingData.sequences[0][0].length}`);
console.log(`Timesteps per sequence: ${trainingData.sequences[0].length}`);
console.log(
  `Buy ratio: ${
    trainingData.metadata.buySamples / trainingData.metadata.totalSamples
  }`
);
```

### 2. Model Creation

```typescript
// Create model factory
const modelFactory = new TradeModelFactory();

// Build neural network architecture
const model = modelFactory.createModel({
  timesteps: MODEL_CONFIG.TIMESTEPS,
  featureCount: MODEL_CONFIG.FEATURE_COUNT,
  lstmUnits: [48, 24, 12],
  dropoutRate: 0.35,
  l2Regularization: 0.008,
});

console.log("Model Architecture Created");
model.summary();
```

### 3. Training Configuration

```typescript
// Configure training parameters
const trainingConfig = {
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  callbacks: [
    // Learning rate scheduling
    tf.callbacks.learningRateScheduler((epoch) => {
      const initialLR = 0.001;
      const decayRate = 0.95;
      return initialLR * Math.pow(decayRate, epoch);
    }),

    // Early stopping
    tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience: 10,
      restoreBestWeights: true,
    }),

    // Model checkpointing
    tf.callbacks.modelCheckpoint("file://./checkpoints", {
      monitor: "val_loss",
      saveBestOnly: true,
    }),
  ],
};
```

### 4. Training Execution

```typescript
// Initialize trainer
const trainer = new TradeModelTrainer(model, trainingConfig);

// Execute training
const trainingHistory = await trainer.train(
  trainingData.sequences,
  trainingData.labels
);

console.log("Training Completed");
console.log(
  `Final Loss: ${
    trainingHistory.history.loss[trainingHistory.history.loss.length - 1]
  }`
);
console.log(
  `Final Validation Loss: ${
    trainingHistory.history.val_loss[
      trainingHistory.history.val_loss.length - 1
    ]
  }`
);
```

### 5. Weight Persistence

```typescript
// Save model weights to Firebase
const firebaseService = FirebaseService.getInstance();
await firebaseService.saveWeights(model, "tradePredictorWeights.json");

console.log("Model weights saved to Firebase Storage");
```

## Configuration

### Model Architecture Parameters

```typescript
const MODEL_CONFIG = {
  TIMESTEPS: 30, // Sequence length
  FEATURE_COUNT: 36, // BTC features
  LSTM_UNITS_1: 48, // First LSTM layer units
  LSTM_UNITS_2: 24, // Second LSTM layer units
  LSTM_UNITS_3: 12, // Third LSTM layer units
  DROPOUT_RATE: 0.35, // Dropout rate for regularization
  L2_REGULARIZATION: 0.008, // L2 regularization strength
  CONV1D_FILTERS_1: 12, // First Conv1D filters
  CONV1D_FILTERS_2: 24, // Second Conv1D filters
  DENSE_UNITS_1: 24, // Dense layer units
  OUTPUT_UNITS: 2, // Output classes (buy/sell)
};
```

### Training Parameters

```typescript
const TRAINING_CONFIG = {
  EPOCHS: 100, // Maximum training epochs
  BATCH_SIZE: 32, // Training batch size
  VALIDATION_SPLIT: 0.2, // Validation data ratio
  INITIAL_LEARNING_RATE: 0.001, // Starting learning rate
  LEARNING_RATE_DECAY: 0.95, // Learning rate decay factor
  EARLY_STOPPING_PATIENCE: 10, // Early stopping patience
  MIN_DELTA: 0.001, // Minimum improvement threshold
};
```

### Data Configuration

```typescript
const DATA_CONFIG = {
  HISTORICAL_DAYS: 450, // Days of historical data
  LOOKAHEAD_DAYS: 7, // Labeling lookahead period
  PRICE_CHANGE_THRESHOLD: 0.02, // Buy/sell threshold
  MIN_DATA_LENGTH: 400, // Minimum required data length
  FEATURE_VALIDATION: true, // Enable feature validation
};
```

## Usage Examples

### Basic Training

```bash
# Run training script
npm run train:btc

# Output:
# Loading historical BTC data...
# Preparing training sequences...
# Creating model architecture...
# Starting training...
# Epoch 1/100 - loss: 0.6931 - val_loss: 0.6928
# Epoch 2/100 - loss: 0.6892 - val_loss: 0.6901
# ...
# Training completed successfully
# Model weights saved to Firebase
```

### Training with Custom Parameters

```typescript
// Custom training configuration
const customConfig = {
  epochs: 150,
  batchSize: 64,
  validationSplit: 0.3,
  callbacks: [
    // Custom learning rate schedule
    tf.callbacks.learningRateScheduler((epoch) => {
      if (epoch < 50) return 0.001;
      if (epoch < 100) return 0.0005;
      return 0.0001;
    }),

    // Custom early stopping
    tf.callbacks.earlyStopping({
      monitor: "val_accuracy",
      patience: 15,
      minDelta: 0.005,
    }),
  ],
};

const trainer = new TradeModelTrainer(model, customConfig);
await trainer.train(sequences, labels);
```

### Training with Validation

```typescript
// Split data for validation
const splitIndex = Math.floor(sequences.length * 0.8);
const trainSequences = sequences.slice(0, splitIndex);
const trainLabels = labels.slice(0, splitIndex);
const valSequences = sequences.slice(splitIndex);
const valLabels = labels.slice(splitIndex);

// Train with validation data
const history = await trainer.train(
  trainSequences,
  trainLabels,
  valSequences,
  valLabels
);

console.log("Validation Results:");
console.log(`Best Validation Loss: ${Math.min(...history.history.val_loss)}`);
console.log(
  `Best Validation Accuracy: ${Math.max(...history.history.val_accuracy)}`
);
```

## Error Handling

### Data Validation

```typescript
// Validate training data
if (trainingData.sequences.length < 100) {
  throw new Error("Insufficient training data");
}

if (trainingData.sequences[0][0].length !== MODEL_CONFIG.FEATURE_COUNT) {
  throw new Error("Feature count mismatch");
}

console.log("Training data validation passed");
```

### Model Validation

```typescript
// Validate model architecture
const inputShape = model.inputs[0].shape;
const outputShape = model.outputs[0].shape;

if (inputShape[1] !== MODEL_CONFIG.TIMESTEPS) {
  throw new Error("Model timesteps mismatch");
}

if (inputShape[2] !== MODEL_CONFIG.FEATURE_COUNT) {
  throw new Error("Model feature count mismatch");
}

console.log("Model architecture validation passed");
```

### Graceful Degradation

```typescript
try {
  // Attempt training
  const history = await trainer.train(sequences, labels);
  console.log("Training completed successfully");
} catch (error) {
  console.error("Training failed:", error.message);

  // Attempt recovery
  if (error.message.includes("out of memory")) {
    console.log("Reducing batch size and retrying...");
    const reducedConfig = { ...trainingConfig, batchSize: 16 };
    const trainer = new TradeModelTrainer(model, reducedConfig);
    await trainer.train(sequences, labels);
  } else {
    throw error;
  }
}
```

## Performance Monitoring

### Training Metrics

```typescript
// Monitor training progress
const monitorCallback = {
  onEpochEnd: (epoch: number, logs: any) => {
    console.log(`Epoch ${epoch + 1}:`);
    console.log(`  Loss: ${logs.loss.toFixed(4)}`);
    console.log(`  Accuracy: ${logs.accuracy.toFixed(4)}`);
    console.log(`  Val Loss: ${logs.val_loss.toFixed(4)}`);
    console.log(`  Val Accuracy: ${logs.val_accuracy.toFixed(4)}`);
  },
};

const trainer = new TradeModelTrainer(model, {
  ...trainingConfig,
  callbacks: [...trainingConfig.callbacks, monitorCallback],
});
```

### Memory Management

```typescript
// Monitor memory usage
const memoryCallback = {
  onEpochEnd: () => {
    const memoryInfo = tf.memory();
    console.log(
      `Memory Usage: ${(memoryInfo.numBytes / 1024 / 1024).toFixed(2)} MB`
    );
    console.log(`Tensors: ${memoryInfo.numTensors}`);
  },
};
```

## Dependencies

### External Dependencies

- **TensorFlow.js**: Core machine learning framework
- **Firebase Admin SDK**: Cloud storage for model weights
- **Node.js**: Runtime environment

### Internal Dependencies

- **TradeModelTrainer**: Training orchestration
- **DataProcessor**: Data preparation and preprocessing
- **TradeModelFactory**: Model architecture creation
- **FirebaseService**: Weight persistence and retrieval

## Testing

### Unit Testing

```typescript
// Test data preparation
describe("Data Preparation", () => {
  it("should prepare valid training data", async () => {
    const dataProcessor = new DataProcessor();
    const trainingData = await dataProcessor.prepareTrainingData();

    expect(trainingData.sequences.length).toBeGreaterThan(0);
    expect(trainingData.labels.length).toBe(trainingData.sequences.length);
    expect(trainingData.sequences[0][0].length).toBe(
      MODEL_CONFIG.FEATURE_COUNT
    );
  });
});

// Test model creation
describe("Model Creation", () => {
  it("should create valid model architecture", () => {
    const modelFactory = new TradeModelFactory();
    const model = modelFactory.createModel(MODEL_CONFIG);

    expect(model.inputs[0].shape[1]).toBe(MODEL_CONFIG.TIMESTEPS);
    expect(model.inputs[0].shape[2]).toBe(MODEL_CONFIG.FEATURE_COUNT);
    expect(model.outputs[0].shape[1]).toBe(MODEL_CONFIG.OUTPUT_UNITS);
  });
});
```

### Integration Testing

```typescript
// Test complete training pipeline
describe("Training Pipeline", () => {
  it("should complete training successfully", async () => {
    const dataProcessor = new DataProcessor();
    const modelFactory = new TradeModelFactory();

    const trainingData = await dataProcessor.prepareTrainingData();
    const model = modelFactory.createModel(MODEL_CONFIG);
    const trainer = new TradeModelTrainer(model, trainingConfig);

    const history = await trainer.train(
      trainingData.sequences,
      trainingData.labels
    );

    expect(history.history.loss.length).toBeGreaterThan(0);
    expect(history.history.val_loss.length).toBeGreaterThan(0);
  });
});
```

## Future Enhancements

### Potential Improvements

- **Hyperparameter Optimization**: Automated hyperparameter tuning
- **Cross-Validation**: K-fold cross-validation for robust evaluation
- **Model Ensembling**: Multiple model training and combination
- **Transfer Learning**: Pre-trained model adaptation

### Advanced Features

- **Distributed Training**: Multi-GPU training support
- **Real-time Training**: Continuous model updates
- **A/B Testing**: Model performance comparison
- **Automated Retraining**: Scheduled model updates

### Integration Enhancements

- **CI/CD Integration**: Automated training pipelines
- **Monitoring Dashboard**: Real-time training visualization
- **Alert System**: Training failure notifications
- **Version Control**: Model versioning and rollback
