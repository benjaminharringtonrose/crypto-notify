# Crypto Trading Model Architecture Documentation

## Overview

This document provides a comprehensive explanation of the crypto trading model architecture used in the crypto-notify system. The model is designed to predict buy/sell signals for Bitcoin trading based on historical price and volume data.

## Model Architecture Summary

The model follows a **CNN-LSTM-Dense** hybrid architecture specifically designed for time series prediction in cryptocurrency markets:

```
Input (30 timesteps × 36 features)
    ↓
Conv1D Layer 1 (12 filters, kernel=5) + BatchNorm + Dropout(0.2)
    ↓
Conv1D Layer 2 (24 filters, kernel=3) + BatchNorm + Dropout(0.2)
    ↓
LSTM Layer 1 (48 units, return_sequences=True) + BatchNorm + Dropout(0.35)
    ↓
LSTM Layer 2 (24 units, return_sequences=True) + BatchNorm + Dropout(0.35)
    ↓
LSTM Layer 3 (12 units, return_sequences=False) + BatchNorm + Dropout(0.35)
    ↓
Dense Layer 1 (24 units, ReLU) + BatchNorm + Dropout(0.35)
    ↓
Dense Layer 2 (12 units, ReLU) + BatchNorm + Dropout(0.175)
    ↓
Output Layer (2 units, Softmax)
```

## Detailed Layer Specifications

### 1. Input Layer

- **Shape**: `[batch_size, 30, 36]`
- **Timesteps**: 30 days of historical data
- **Features**: 36 total features (BTC features only)

### 2. Convolutional Layers (Feature Extraction)

#### Conv1D Layer 1

- **Filters**: 12
- **Kernel Size**: 5
- **Activation**: ReLU
- **Input Shape**: `[30, 36]`
- **Output Shape**: `[26, 12]`
- **Purpose**: Extract local temporal patterns in price movements

#### Conv1D Layer 2

- **Filters**: 24
- **Kernel Size**: 3
- **Activation**: ReLU
- **Input Shape**: `[26, 12]`
- **Output Shape**: `[24, 24]`
- **Purpose**: Further refine feature extraction and increase feature dimensionality

### 3. LSTM Layers (Temporal Modeling)

#### LSTM Layer 1

- **Units**: 48
- **Return Sequences**: True
- **Input Shape**: `[24, 24]`
- **Output Shape**: `[24, 48]`
- **Dropout**: 0.1 (input), 0.1 (recurrent)
- **Purpose**: Capture medium-term temporal dependencies

#### LSTM Layer 2

- **Units**: 24
- **Return Sequences**: True
- **Input Shape**: `[24, 48]`
- **Output Shape**: `[24, 24]`
- **Dropout**: 0.1 (input), 0.1 (recurrent)
- **Purpose**: Refine temporal patterns and reduce dimensionality

#### LSTM Layer 3

- **Units**: 12
- **Return Sequences**: False
- **Input Shape**: `[24, 24]`
- **Output Shape**: `[12]`
- **Dropout**: 0.1 (input), 0.1 (recurrent)
- **Purpose**: Final temporal feature extraction and dimensionality reduction

### 4. Dense Layers (Feature Learning)

#### Dense Layer 1

- **Units**: 24
- **Activation**: ReLU
- **Input Shape**: `[12]`
- **Output Shape**: `[24]`
- **Purpose**: Learn complex feature combinations

#### Dense Layer 2

- **Units**: 12 (half of previous layer)
- **Activation**: ReLU
- **Input Shape**: `[24]`
- **Output Shape**: `[12]`
- **Purpose**: Further feature refinement

#### Output Layer

- **Units**: 2
- **Activation**: Softmax
- **Input Shape**: `[12]`
- **Output Shape**: `[2]`
- **Purpose**: Binary classification (Buy/Sell probabilities)

## Feature Engineering

### Input Features (36 total)

The model uses 36 technical indicators and features computed from Bitcoin price and volume data (as of v1.5.0 - Advanced Market Microstructure Features):

1. **Momentum Indicators**:

   - RSI (14-period) and previous RSI
   - Stochastic RSI and signal line
   - Price momentum and volatility-adjusted momentum

2. **Trend Indicators**:

   - SMA (7, 20, 21, 50, 200-day)
   - MACD line and signal line
   - Previous MACD line

3. **Price Information**:

   - Current price and previous price
   - VWAP (Volume Weighted Average Price)

4. **Volatility Indicators**:

   - ATR (Average True Range) and baseline
   - Bollinger Bands (upper, lower)
   - Z-score

5. **Volume Indicators**:

   - OBV (On-Balance Volume)
   - Volume oscillator and previous volume oscillator
   - Volume spike detection

6. **Pattern Recognition**:

   - Double top detection
   - Head and shoulders detection
   - Triple top detection
   - Triple bottom detection

7. **Market Regime Features**:

   - Volatility regime score
   - Trend regime score
   - Momentum regime score
   - Realized volatility
   - Overall regime score

8. **Advanced Technical Features**:
   - ADX proxy
   - Trend regime
   - Fibonacci levels
   - Price-to-moving average ratios
   - Volume ratios
   - Normalized indicators

## Training Configuration

### Hyperparameters

- **Epochs**: 120
- **Batch Size**: 32
- **Initial Learning Rate**: 0.0008
- **Minimum Learning Rate**: 0.000005
- **Patience (Early Stopping)**: 20 epochs
- **Train/Validation Split**: 85%/15%

### Regularization

- **L2 Regularization**: 0.008
- **Dropout Rate**: 0.35 (main), 0.175 (final dense layer)
- **Gradient Clipping**: 1.0

### Loss Function

- **Focal Loss** with parameters:
  - Alpha: [0.35, 0.65] (class weights)
  - Gamma: 1.5 (focusing parameter)

### Optimizer

- **Adam** optimizer with exponential learning rate decay

## Data Processing Pipeline

### Data Preparation

1. **Historical Data Fetching**: 1200 days of BTC data
2. **Feature Calculation**: 36 technical indicators computed
3. **Sequence Generation**: 30-day sliding windows
4. **Data Normalization**: Z-score normalization per feature
5. **Data Augmentation**: Noise injection and sequence balancing

### Curriculum Learning

- **Difficulty Progression**: Starts with easier samples, gradually increases complexity
- **Sample Filtering**: Based on volatility and trend complexity scores
- **Adaptive Training**: Adjusts training difficulty based on model performance

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

### Why CNN-LSTM-Dense?

1. **CNN Layers**: Extract local temporal patterns and reduce noise
2. **LSTM Layers**: Capture long-term dependencies and temporal relationships
3. **Dense Layers**: Learn complex feature combinations for final classification

### Design Decisions

1. **30 Timesteps**: Balance between sufficient history and computational efficiency
2. **36 Features**: Comprehensive technical analysis without overfitting
3. **3 LSTM Layers**: Gradual temporal abstraction (48→24→12 units)
4. **Batch Normalization**: Stabilize training and improve convergence
5. **Dropout**: Prevent overfitting in deep architecture

## Performance Considerations

### Computational Complexity

- **Parameters**: ~50K trainable parameters
- **Memory Usage**: ~200MB during training
- **Training Time**: ~2-4 hours for 120 epochs
- **Inference Time**: <100ms per prediction

### Optimization Opportunities

1. **Batch Size**: Can be increased to 64-128 for faster training
2. **Model Complexity**: LSTM units can be reduced for speed
3. **Feature Count**: Less important features can be removed
4. **Timesteps**: Can be reduced from 30 to 20-25

## Future Enhancements

### Potential Improvements

1. **Attention Mechanisms**: Add self-attention for better feature weighting
2. **Residual Connections**: Improve gradient flow in deep layers
3. **Multi-head Architecture**: Separate models for different market conditions
4. **Ensemble Methods**: Combine multiple model predictions
5. **Online Learning**: Continuous model updates with new data
