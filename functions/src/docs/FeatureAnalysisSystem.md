# Feature Analysis System Documentation

## Overview

The Feature Analysis System is a comprehensive suite of tools designed to systematically determine which features are beneficial for the Bitcoin trading model. This system addresses one of the most critical aspects of machine learning for trading: feature selection and optimization.

## Problem Statement

The current trading model uses 36 features (as defined in FeatureRegistry), but the actual FeatureCalculator produces 31 features. This mismatch, combined with the need to optimize model performance, requires a systematic approach to:

1. **Audit current features** - Identify what features are actually being used
2. **Analyze feature utility** - Determine which features contribute to trading success
3. **Remove redundancy** - Eliminate correlated features that add noise
4. **Optimize performance** - Find the minimal set of features for maximum trading performance

## System Architecture

```
Feature Analysis System
├── analyzeFeatures.ts        # Correlation & statistical analysis
├── featureAblationStudy.ts   # Model performance impact analysis
├── optimizeFeatureSet.ts     # Comprehensive optimization
├── fixFeatureRegistry.ts     # Registry alignment tool
└── FeatureAnalysisSystem.md  # This documentation
```

## Components

### 1. Feature Analyzer (`analyzeFeatures.ts`)

**Purpose**: Statistical analysis of feature utility and redundancy

**Methods**:

- **Correlation Analysis**: Pearson correlation between features and target variable
- **Statistical Significance**: T-tests to determine feature significance
- **Redundancy Analysis**: Inter-feature correlation matrix
- **Utility Scoring**: Combined utility and redundancy scores

**Output**:

- Feature rankings by correlation, significance, and utility
- Redundancy analysis showing highly correlated features
- Recommendations: "keep", "remove", or "investigate"

**Usage**:

```bash
npm run features:analyze
```

### 2. Feature Ablation Study (`featureAblationStudy.ts`)

**Purpose**: Measure impact of each feature on model performance

**Methods**:

- **Systematic Removal**: Train model with each feature removed individually
- **Performance Measurement**: Compare accuracy, returns, Sharpe ratio
- **Importance Scoring**: Calculate feature importance based on performance impact
- **Speed Analysis**: Measure training time improvements

**Output**:

- Feature importance rankings based on performance impact
- Critical features that significantly hurt performance when removed
- Potential noise features that improve performance when removed

**Usage**:

```bash
npm run features:ablation
```

### 3. Feature Set Optimizer (`optimizeFeatureSet.ts`)

**Purpose**: Comprehensive optimization using multiple selection strategies

**Strategies**:

1. **Top Correlation**: Best features by correlation with target
2. **Top Importance**: Best features by ablation importance
3. **Balanced Category**: Representative features from each category
4. **Low Redundancy**: Features with minimal inter-correlation
5. **Utility Optimized**: High utility, manageable redundancy
6. **Conservative**: Features important in multiple analyses
7. **Minimal**: Bare essentials for maximum efficiency

**Process**:

1. Generate candidate feature subsets using different strategies
2. Train models with each subset
3. Evaluate performance, speed, and efficiency
4. Select optimal subset based on balanced scoring

**Output**:

- Optimal feature subset with performance metrics
- Comparison of all strategies
- Efficiency gains (speed, memory reduction)
- Specific recommendations for implementation

**Usage**:

```bash
npm run features:optimize
```

### 4. Feature Registry Fixer (`fixFeatureRegistry.ts`)

**Purpose**: Align FeatureRegistry with actual FeatureCalculator implementation

**Functions**:

- **Audit**: Compare registry vs actual implementation
- **Analysis**: Determine actual features produced by FeatureCalculator
- **Generation**: Create corrected FeatureRegistry code
- **Validation**: Verify alignment between components

**Output**:

- Corrected FeatureRegistry.ts code
- Feature mapping and categorization
- Validation results

**Usage**:

```bash
npm run features:fix-registry
```

## Feature Categories

### Core Features (8 features)

- Price action: current price, previous price, price change %
- Volume data: current volume, volume momentum
- Basic ratios: price position, volume ratios

### Technical Indicators (7 features)

- Trend: SMA7, SMA21, SMA50
- Momentum: RSI (multiple periods), MACD line, signal line
- Volatility: ATR, Bollinger Bands

### Ratio Features (5 features)

- Price ratios: Price/SMA ratios
- Normalized indicators: RSI normalized, Bollinger position

### Enhanced Features (10 features)

- Secondary indicators: previous RSI, MACD histogram
- Normalized metrics: ATR ratio, volume ratios
- Pattern signals: momentum, squeeze, divergence

### Microstructure Features (6 features - Experiment #61)

- Ichimoku indicators: Tenkan-sen, Kijun-sen, cloud position
- Advanced oscillators: Williams %R, Stochastic %K, VPT

## Current Issues Identified

### 1. Registry Mismatch

- **Registry**: Defines 36 features
- **Calculator**: Produces 31 features
- **Impact**: Tensor shape mismatches, incorrect feature mapping

### 2. Potential Redundancy

- Multiple RSI variants (7-day, 14-day, 21-day)
- Similar ratio features (price/SMA ratios)
- Correlated volume indicators

### 3. Unknown Feature Importance

- Experimental features (Ichimoku, Williams %R) have unknown importance
- No systematic evaluation of feature contribution to trading performance
- Possible noise features reducing model performance

## Analysis Methodology

### Statistical Analysis

1. **Correlation Analysis**: Measure linear relationship with target
2. **Significance Testing**: T-tests between buy/sell classes
3. **Redundancy Analysis**: Inter-feature correlation matrix
4. **Utility Scoring**: Combined correlation and significance

### Performance Analysis

1. **Ablation Study**: Remove each feature and measure impact
2. **Subset Evaluation**: Train models with different feature combinations
3. **Trading Metrics**: Evaluate impact on actual trading performance
4. **Efficiency Metrics**: Measure training speed and memory usage

### Selection Criteria

1. **High Utility**: Strong correlation with trading success
2. **Low Redundancy**: Minimal overlap with other features
3. **Statistical Significance**: Proven predictive power
4. **Performance Impact**: Positive contribution to trading metrics

## Expected Benefits

### Performance Improvements

- **Higher Accuracy**: Remove noise features that confuse the model
- **Better Trading Returns**: Focus on features that predict profitable trades
- **Improved Sharpe Ratio**: Better risk-adjusted returns

### Efficiency Gains

- **Faster Training**: Fewer features = faster model training
- **Lower Memory Usage**: Reduced tensor sizes and memory footprint
- **Faster Inference**: Quicker predictions for live trading

### System Reliability

- **Consistent Architecture**: Aligned registry and implementation
- **Validated Features**: Only proven beneficial features
- **Maintainable Code**: Clear feature documentation and categorization

## Usage Workflow

### Complete Analysis

```bash
# Run complete feature analysis pipeline
npm run features:all
```

### Individual Components

```bash
# Fix registry mismatch first
npm run features:fix-registry

# Analyze feature utility
npm run features:analyze

# Study feature importance
npm run features:ablation

# Find optimal feature set
npm run features:optimize
```

### Integration Steps

1. **Run Analysis**: Execute feature analysis pipeline
2. **Review Results**: Examine recommendations and optimal feature set
3. **Update Code**: Modify FeatureCalculator to use optimized features
4. **Retrain Model**: Train model with optimized feature set
5. **Validate Performance**: Run backtests to confirm improvements

## Implementation Recommendations

### Phase 1: Immediate Fixes

1. **Fix Registry Mismatch**: Align FeatureRegistry with FeatureCalculator
2. **Run Analysis**: Execute basic feature analysis
3. **Identify Obvious Issues**: Remove clearly redundant features

### Phase 2: Optimization

1. **Comprehensive Analysis**: Run full ablation study
2. **Feature Selection**: Implement optimal feature subset
3. **Performance Validation**: Verify improvements through backtesting

### Phase 3: Continuous Improvement

1. **Regular Analysis**: Periodic feature utility reviews
2. **New Feature Evaluation**: Systematic testing of new features
3. **Performance Monitoring**: Track feature contribution over time

## Integration with Trading System

### Model Training

- Use optimized features in TradeModelTrainer
- Update model architecture for new feature count
- Adjust normalization statistics

### Prediction Pipeline

- Update FeatureCalculator.compute() method
- Ensure consistent feature ordering
- Validate tensor shapes

### Backtesting

- Test with historical data using new features
- Compare performance vs baseline
- Validate trading metrics improvement

## Monitoring and Maintenance

### Performance Tracking

- Monitor trading performance with optimized features
- Track feature contribution over time
- Identify performance degradation

### Regular Reviews

- Quarterly feature analysis
- Evaluation of new technical indicators
- Market regime analysis impact

### System Updates

- Update feature registry when adding new features
- Maintain documentation
- Version control for feature sets

## Conclusion

The Feature Analysis System provides a comprehensive, systematic approach to optimizing the feature set for the Bitcoin trading model. By combining statistical analysis, performance measurement, and systematic optimization, it enables data-driven decisions about which features truly contribute to trading success.

This system addresses the critical challenge of feature selection in trading models, where the quality of input features directly impacts profitability and risk management. The systematic approach ensures that only beneficial features are used, leading to improved performance, efficiency, and reliability of the trading system.
