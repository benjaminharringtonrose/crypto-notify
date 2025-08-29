# Crypto Trading Model Architecture Documentation

## Overview

This document provides a comprehensive explanation of the crypto trading model architecture used in the crypto-notify system. The model is designed to predict buy/sell signals for Bitcoin trading based on historical price and volume data.

## Model Architecture Summary

The model follows a **Conv1D-LSTM-Dense** architecture specifically designed for time series prediction in cryptocurrency markets:

```
Input (35 timesteps × 26 features)
    ↓
Conv1D Layer (48 filters, kernel=3) + BatchNorm + Dropout(0.3)
    ↓
LSTM Layer (64 units, return_sequences=False) + RecurrentDropout(0.1)
    ↓
Dense Layer (32 units, ReLU) + Dropout(0.3)
    ↓
Output Layer (2 units, Softmax)
```

## Detailed Layer Specifications

### 1. Input Layer

- **Shape**: `[batch_size, timesteps, FeatureDetector.getFeatureCount()]`
- **Timesteps**: 35 days of historical data
- **Features**: 26 features (optimized from original 36 features)

### 2. Convolutional Layer (Feature Extraction)

#### Conv1D Layer

- **Filters**: 48 (proven optimal filter count from experiments)
- **Kernel Size**: 3 (proven optimal kernel size)
- **Activation**: ReLU
- **Input Shape**: `[timesteps, FeatureDetector.getFeatureCount()]`
- **Output Shape**: `[timesteps-2, 48]` (depends on kernel size)
- **Purpose**: Extract local temporal patterns in price movements
- **Batch Normalization**: Applied after convolution
- **Dropout**: 0.3 applied after batch normalization
- **Kernel Initializer**: HeNormal (optimal for ReLU activation)
- **L2 Regularization**: 0.001 (prevents overfitting)

### 3. LSTM Layer (Temporal Modeling)

#### LSTM Layer

- **Units**: 64 (proven optimal capacity for the dataset)
- **Return Sequences**: False (proven: returnSequences=true failed)
- **Input Shape**: `[timesteps, 48]` (from Conv1D output)
- **Output Shape**: `[64]` (flattened for dense layer)
- **Recurrent Dropout**: 0.1
- **Purpose**: Capture temporal dependencies and extract sequential patterns
- **Kernel Initializer**: HeNormal (proven optimal for ReLU activation)

### 4. Dense Layers (Feature Learning)

#### Dense Layer

- **Units**: 32 (proven optimal - 48 units caused severe overfitting)
- **Activation**: ReLU
- **Input Shape**: `[64]` (from LSTM output)
- **Output Shape**: `[32]`
- **Purpose**: Learn complex feature combinations and prepare for classification
- **Dropout**: 0.3 applied after activation

#### Output Layer

- **Units**: 2 (binary classification)
- **Activation**: Softmax
- **Input Shape**: `[32]` (from dense layer)
- **Output Shape**: `[2]`
- **Purpose**: Binary classification (Buy/Sell probabilities)

## Feature Engineering

### Input Features

The model uses 26 optimized technical indicators and features computed from Bitcoin price and volume data:

1. **Core Features (4 features)**:

   - High-low price range
   - Price volatility
   - Price position in recent range
   - Relative volume indicator

2. **Technical Indicators (5 features)**:

   - RSI momentum oscillator
   - MACD signal line
   - VWAP ratio
   - Average True Range (ATR)
   - On-Balance Volume (OBV)

3. **Enhanced Indicators (5 features)**:

   - Raw momentum
   - MACD histogram
   - Price to SMA7 ratio
   - Price to SMA21 ratio
   - Price to SMA50 ratio

4. **Market Regime Features (5 features)**:

   - Trend regime score
   - Volatility regime score
   - Ichimoku Tenkan-sen (9-period)
   - Ichimoku Kijun-sen (26-period)
   - Ichimoku cloud position

5. **Advanced Microstructure Features (7 features)**:
   - Williams %R momentum oscillator
   - Volume MA20
   - Volume oscillator
   - Money Flow Index (MFI)
   - Aroon Oscillator
   - Donchian Channels position
   - Parabolic SAR trend

## Training Configuration

### Hyperparameters

- **Epochs**: 35 (proven optimal - 50+ caused issues)
- **Batch Size**: 16 (proven optimal balance for gradient updates)
- **Initial Learning Rate**: 0.0005 (proven optimal baseline)
- **Minimum Learning Rate**: 0.00005
- **Patience (Early Stopping)**: 10 epochs
- **Train/Validation Split**: 80%/20%

### Regularization

- **L2 Regularization**: 0.001
- **Dropout Rate**: 0.3 (main), 0.1 (recurrent)
- **Gradient Clipping**: 1.0

### Loss Function

- **Focal Loss** with parameters:
  - Alpha: [0.45, 0.55] (class weights)
  - Gamma: 1.5 (focusing parameter)

### Optimizer

- **Adam** optimizer with exponential learning rate decay

## Data Processing Pipeline

### Data Preparation

1. **Historical Data Fetching**: 730 days of BTC data
2. **Feature Calculation**: 26 technical indicators computed
3. **Sequence Generation**: 35-day sliding windows
4. **Data Normalization**: Z-score normalization per feature
5. **Data Augmentation**: Noise injection and sequence balancing

## Model Performance Metrics

### Primary Metrics

- **Binary Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1 Score**: Balanced measure for buy/sell predictions
- **Validation Loss**: Training stability indicator

### Custom Metrics

- **F1 Buy**: F1 score for buy predictions
- **F1 Sell**: F1 score for sell predictions
- **Precision Buy/Sell**: Precision for each class
- **Recall Buy/Sell**: Recall for each class

## Model Storage and Deployment

### Weight Storage

- **Format**: JSON file stored in Firebase Storage
- **Components**: All layer weights, batch normalization parameters, feature statistics
- **File**: `tradePredictorWeights.json`

### Deployment

- **Environment**: Firebase Functions (Node.js)
- **Framework**: TensorFlow.js
- **Inference**: Real-time prediction for trading decisions

## Architecture Rationale

### Why Conv1D-LSTM-Dense?

1. **Conv1D Layer**: Extract local temporal patterns and reduce noise
2. **LSTM Layer**: Capture long-term dependencies and temporal relationships
3. **Dense Layer**: Learn complex feature combinations for final classification

### Design Decisions

1. **35 Timesteps**: Balance between sufficient history and computational efficiency
2. **26 Features**: Optimized feature set based on gradual feature optimization
3. **Single LSTM Layer**: Proven optimal for this dataset size
4. **Batch Normalization**: Stabilize training and improve convergence
5. **Dropout**: Prevent overfitting in deep architecture

## Performance Considerations

### Computational Complexity

- **Parameters**: ~35K trainable parameters
- **Memory Usage**: ~150MB during training
- **Training Time**: ~73 seconds for 35 epochs
- **Inference Time**: <100ms per prediction

### Optimization History

The current architecture is the result of systematic experimentation:

- **Experiment #1**: Threshold optimization (0.0015 → 0.001)
- **Experiment #3**: Conv1D filters optimization (32 → 48)
- **Experiment #61**: Advanced market microstructure features (36 → 26 optimized)

## Future Enhancements

### Potential Improvements

1. **Feature Selection**: Further optimize the 26-feature set
2. **Attention Mechanisms**: Add self-attention for better feature weighting
3. **Ensemble Methods**: Combine multiple model predictions
4. **Online Learning**: Continuous model updates with new data
