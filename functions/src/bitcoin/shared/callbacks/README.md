# Training Callbacks

This directory contains the callback classes used during model training to handle various aspects of the training process.

## Overview

The training system has been updated to use a centralized logging approach that displays all epoch information in a clean, organized table format. This eliminates redundant logging and makes it easier to track training progress.

## Callbacks

### TrainingLoggerCallback (NEW - Centralized Logger)

**File**: `TrainingLoggerCallback.ts`

The main logging callback that consolidates all epoch information into a single table format.

**Features**:

- Displays all metrics in a clean, aligned table using `console.table()`
- Shows training and validation accuracy, loss, F1 scores
- Tracks learning rate, curriculum difficulty, effective weights
- Displays buy/sell counts, precision, recall metrics
- Shows gradient norm (when available)
- Provides training summary and complete history at the end
- Uses native Node.js `console.table()` for optimal formatting

**Table Format**:

Uses `console.table()` with columns: Epoch, Train Acc, Val Acc, Val Loss, Spread, F1 Buy, F1 Sell, Combined, LR, Diff%, EffWgt, Buy/Sell, Prec B/S, Recall B/S, Grad Norm

**Methods**:

- `onEpochEnd(epoch, logs)`: Processes and displays epoch data using `console.table()`
- `printTrainingSummary()`: Shows final training statistics in table format
- `printAllEpochsTable()`: Displays complete training history in a single table
- `getEpochData()`: Returns stored epoch data for external use
- `setGradientNorm(norm)`: Receives gradient norm from other callbacks

### PredictionLoggerCallback (DEPRECATED)

**File**: `PredictionLoggerCallback.ts`

This callback has been deprecated and simplified. All logging functionality has been moved to `TrainingLoggerCallback`.

**Current Status**: Minimal functionality for backward compatibility only.

### ExponentialDecayLRCallback

**File**: `ExponentialDecayLRCallback.ts`

Handles learning rate scheduling with exponential decay and warmup.

**Features**:

- Warmup period with gradual learning rate increase
- Exponential decay after warmup
- Automatic optimizer recompilation with new learning rate

**Changes**: Removed console logging - learning rate is now displayed in the centralized table.

### GradientClippingCallback

**File**: `GradientClippingCallback.ts`

Implements gradient clipping to prevent exploding gradients.

**Features**:

- Computes gradient norm every 5 epochs
- Applies clipping when norm exceeds threshold
- Communicates gradient norm to centralized logger

**Changes**:

- Removed direct console logging
- Added `setTrainingLogger()` method to communicate with centralized logger

### CurriculumLearningCallback

**File**: `CurriculumLearningCallback.ts`

Implements curriculum learning by gradually increasing data difficulty.

**Features**:

- 4-phase difficulty schedule (60% → 80% → 90% → 100%)
- Updates data processor with current difficulty level
- Provides difficulty information to centralized logger

**Changes**: Removed console logging - difficulty percentage is now displayed in the centralized table.

### EarlyStoppingCallback

**File**: `EarlyStoppingCallback.ts`

Monitors training progress and stops early if no improvement is detected.

**Features**:

- Configurable monitoring metric (val_loss, val_customF1, etc.)
- Patience-based early stopping
- Optional best weight restoration

## Usage

The callbacks are automatically configured in `TradeModelTrainer.ts`:

```typescript
// Create centralized training logger
const trainingLoggerCallback = new TrainingLoggerCallback(X_val, y_val, this);

// Set up other callbacks
const lrCallback = new ExponentialDecayLRCallback();
const gradientClippingCallback = new GradientClippingCallback(
  this.lossFn,
  X_val,
  y_val
);
const curriculumLearningCallback = new CurriculumLearningCallback(
  this.config.epochs,
  this.dataProcessor
);

// Connect callbacks
trainingLoggerCallback.setModel(this.model);
lrCallback.setModel(this.model);
gradientClippingCallback.setModel(this.model);
curriculumLearningCallback.setModel(this.model);

// Connect gradient clipping to training logger
gradientClippingCallback.setTrainingLogger(trainingLoggerCallback);
```

## Benefits of the New System

1. **Cleaner Output**: All epoch information is displayed in a single, organized table
2. **Reduced Redundancy**: Eliminates scattered console.log statements across multiple callbacks
3. **Better Readability**: Aligned columns make it easy to track progress
4. **Comprehensive View**: All metrics visible at once for better analysis
5. **Consistent Format**: Standardized display format across all training runs
6. **Training Summary**: Automatic summary at the end of training

## Example Output

The new system uses `console.table()` for clean, professional table formatting:

### Individual Epoch Display

Each epoch is displayed as a single row in a formatted table with proper column alignment.

### Training Summary

```
┌───────────────────────────┬────────────────────┐
│          (index)          │       Values       │
├───────────────────────────┼────────────────────┤
│       Total Epochs        │         3          │
│    Best Combined Score    │ '1.6030 (Epoch 3)' │
│   Final Validation Loss   │      '0.4100'      │
│ Final Validation Accuracy │      '0.8360'      │
│     Final F1 Buy/Sell     │  '0.8100/0.8340'   │
└───────────────────────────┴────────────────────┘
```

### Complete Training History

All epochs are displayed in a single, comprehensive table showing the full training progression with proper column headers and alignment.
