# Training Metrics Targets Documentation

## Overview

This document defines the specific target metrics for improving the Bitcoin trading model's training performance. These targets are based on the current system analysis and designed to achieve optimal prediction accuracy for profitable trading decisions.

## Current Performance Baseline

**As of January 2025 (After Experiment #61 - Advanced Market Microstructure Features):**

### Model Performance Metrics:

- **Validation Accuracy**: 57.37% ⚠️ **NEEDS IMPROVEMENT**
- **Buy F1 Score**: 0.4442 ⚠️ **NEEDS IMPROVEMENT**
- **Sell F1 Score**: 0.6376 ✅ **ACCEPTABLE**
- **Buy Precision**: 66.34% ⚠️ **NEEDS IMPROVEMENT**
- **Sell Precision**: 54.76% ⚠️ **NEEDS IMPROVEMENT**
- **Buy Recall**: 29.91% ❌ **CRITICAL ISSUE**
- **Sell Recall**: 84.82% ✅ **EXCELLENT**
- **Matthews Correlation Coefficient**: 0.1763 ⚠️ **NEEDS IMPROVEMENT**

### Model Architecture:

- **Features**: 42 comprehensive technical indicators
- **Architecture**: Conv1D(48,3) → LSTM(64) → Dense(32) → Output(2)
- **Timesteps**: 35 days of historical data
- **Training Time**: ~73 seconds for 35 epochs

### Trading Performance Metrics:

- **Win Rate**: ~58-60% ⚠️ **NEEDS IMPROVEMENT**
- **Total Return**: ~15-20% ⚠️ **NEEDS IMPROVEMENT**
- **Annualized Return**: ~8-10% ⚠️ **NEEDS IMPROVEMENT**
- **Sharpe Ratio**: ~1.2-1.5 ⚠️ **NEEDS IMPROVEMENT**
- **Max Drawdown**: ~3-5% ✅ **ACCEPTABLE**

## Primary Target Metrics

### 1. Win Rate Target: 70%

**Current**: ~58-60%  
**Target**: 70%  
**Gap**: +10-12 percentage points

**Rationale**:

- Win rate is the most critical metric for profitable trading
- 70% win rate provides sufficient margin for transaction costs and slippage
- Achievable with improved model confidence and better strategy selection

**Measurement**:

- Track across all backtest periods (Recent, Middle, Older, Full)
- Monitor consistency across different market conditions
- Ensure win rate stability over time

### 2. Validation Accuracy Target: 65%

**Current**: 57.37%  
**Target**: 65%  
**Gap**: +7.63 percentage points

**Rationale**:

- Validation accuracy provides overall model performance measure
- Accounts for class imbalance in trading datasets
- More reliable than simple accuracy for imbalanced data

**Measurement**:

- Calculate as (Sensitivity + Specificity) / 2
- Monitor during training and validation
- Track across different market regimes

### 3. Buy/Sell F1 Score Balance Target: <0.1 Difference

**Current**: Buy F1 (0.4442) vs Sell F1 (0.6376) = 0.1934 difference  
**Target**: <0.1 difference  
**Gap**: Reduce imbalance by 0.0934

**Rationale**:

- Current model heavily favors sell signals
- Balanced buy/sell performance needed for complete trading strategy
- Prevents overfitting to one type of signal

**Measurement**:

- Track F1 scores for both buy and sell predictions
- Aim for |Buy F1 - Sell F1| < 0.1
- Monitor during training and validation

### 4. Buy Recall Target: 60%

**Current**: 29.91%  
**Target**: 60%  
**Gap**: +30.09 percentage points

**Rationale**:

- Buy recall is critically low and needs major improvement
- Higher buy recall enables more trading opportunities
- Essential for capturing upward price movements

**Measurement**:

- Track buy signal detection rate
- Monitor false negative rate for buy signals
- Ensure balanced buy/sell signal generation

## Secondary Target Metrics

### 5. Sharpe Ratio Target: >2.0

**Current**: 1.2-1.5  
**Target**: >2.0 consistently  
**Gap**: Improve risk-adjusted returns

**Rationale**:

- Sharpe ratio measures risk-adjusted returns
- > 2.0 indicates excellent risk-adjusted performance
- Ensures consistent performance across market conditions

### 6. Maximum Drawdown Target: <3%

**Current**: 3-5%  
**Target**: <3%  
**Gap**: Reduce worst-case drawdown

**Rationale**:

- Lower drawdown reduces capital risk
- Enables more aggressive position sizing
- Improves investor confidence

### 7. Matthews Correlation Target: >0.3

**Current**: 0.1763  
**Target**: >0.3  
**Gap**: +0.1237 improvement

**Rationale**:

- MCC is robust to class imbalance
- > 0.3 indicates good model performance
- Better measure than accuracy for imbalanced datasets

### 8. Combined F1 Score Target: >1.2

**Current**: 1.0818  
**Target**: >1.2  
**Gap**: +0.1182 improvement

**Rationale**:

- Combined F1 provides balanced performance measure
- > 1.2 indicates strong overall model performance
- Accounts for both buy and sell prediction quality

## Model-Specific Training Targets

### 9. Validation Loss Target: <0.08

**Current**: ~0.12  
**Target**: <0.08  
**Gap**: -0.04 improvement

**Rationale**:

- Lower validation loss indicates better generalization
- Prevents overfitting to training data
- Ensures model learns meaningful patterns

### 10. Class Balance Target: 40-60% Buy Ratio

**Current**: Varies from 0.000 to 0.674  
**Target**: 40-60% buy ratio consistently  
**Gap**: More balanced class distribution

**Rationale**:

- Balanced classes improve model learning
- Prevents bias toward majority class
- Enables better buy signal prediction

### 11. Training Stability Target: Consistent Learning

**Current**: Early stopping at epoch 27  
**Target**: Stable learning through full training  
**Gap**: Improve training stability

**Rationale**:

- Stable training indicates good optimization
- Prevents premature convergence
- Enables better model convergence

## Implementation Priority

### Phase 1 (Immediate - 1-2 weeks): Critical Fixes

1. **Buy Recall**: Target 50% (current: 29.91%)
2. **Validation Accuracy**: Target 60% (current: 57.37%)
3. **Buy F1 Score**: Target 0.55+ (current: 0.4442)
4. **Training Stability**: Eliminate early stopping issues

### Phase 2 (Short-term - 2-4 weeks): Balance Metrics

1. **Buy/Sell F1 Balance**: Target <0.15 difference
2. **Class Balance**: Target 45-55% buy ratio
3. **Validation Loss**: Target <0.10
4. **Matthews Correlation**: Target >0.25

### Phase 3 (Medium-term - 1-2 months): Performance Metrics

1. **Win Rate**: Target 65%+
2. **Validation Accuracy**: Target 65%+
3. **Sharpe Ratio**: Target >1.8
4. **Combined F1 Score**: Target >1.15

### Phase 4 (Long-term - 2-3 months): Optimal Performance

1. **Win Rate**: Target 70%
2. **Validation Accuracy**: Target 65%
3. **Buy/Sell F1 Balance**: Target <0.1 difference
4. **Buy Recall**: Target 60%
5. **Maximum Drawdown**: Target <3%

## Measurement and Tracking

### Daily Monitoring

- Training loss and validation loss
- Learning rate progression
- Class distribution in training data
- Model confidence distribution

### Weekly Evaluation

- Full backtest on recent data
- Strategy distribution analysis
- Win rate and Sharpe ratio calculation
- Buy/sell performance comparison

### Monthly Assessment

- Comprehensive backtest across all periods
- Model retraining with new data
- Performance trend analysis
- Strategy optimization

### Quarterly Review

- Complete system performance evaluation
- Model architecture optimization
- Feature engineering improvements
- Risk management adjustments

## Success Criteria

### Minimum Viable Performance

- Win Rate: 60%+
- Validation Accuracy: 60%+
- Buy F1 Score: 0.5+
- Buy Recall: 50%+

### Target Performance

- Win Rate: 70%
- Validation Accuracy: 65%
- Buy/Sell F1 Balance: <0.1 difference
- Buy Recall: 60%
- Sharpe Ratio: >2.0
- Maximum Drawdown: <3%

### Optimal Performance

- Win Rate: 75%+
- Validation Accuracy: 70%+
- Buy/Sell F1 Balance: <0.05 difference
- Buy Recall: 65%+
- Sharpe Ratio: >2.5
- Maximum Drawdown: <2%

## Risk Mitigation

### Overfitting Prevention

- Monitor validation loss vs training loss
- Implement early stopping with patience
- Use cross-validation for model selection
- Regular model retraining with fresh data

### Data Quality Assurance

- Validate feature distributions
- Monitor for data drift
- Ensure balanced class representation
- Check for data leakage

### Performance Degradation Prevention

- Track performance trends over time
- Implement model versioning
- Maintain performance baselines
- Regular model comparison and rollback capability

## Conclusion

These target metrics provide a clear roadmap for improving the Bitcoin trading model's performance. The phased approach ensures steady progress while maintaining system stability. Regular monitoring and adjustment of these targets will guide the development toward optimal trading performance.

The focus should be on achieving balanced performance across all metrics rather than optimizing any single metric at the expense of others. This balanced approach will result in a more robust and profitable trading system.

**Critical Focus Areas**:

1. **Buy Signal Performance**: Major improvement needed in buy recall and F1 score
2. **Class Balance**: Address severe sell bias in current model
3. **Training Stability**: Eliminate early stopping and convergence issues
4. **Overall Accuracy**: Improve validation accuracy from 57.37% to 65%+
