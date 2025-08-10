# Training Metrics Targets Documentation

## Overview

This document defines the specific target metrics for improving the Cardano trading model's training performance. These targets are based on the current system analysis and designed to achieve optimal prediction accuracy for profitable trading decisions.

## Current Performance Baseline

**As of December 2024 (After Iterative Training):**

### Model Performance Metrics:

- **Balanced Accuracy**: 83.78% ✅ **ACHIEVED TARGET!**
- **Buy F1 Score**: 0.8063 ✅ **EXCEEDED TARGET!**
- **Sell F1 Score**: 0.8989 ✅ **EXCEEDED TARGET!**
- **Buy Precision**: 0.9753 ✅ **EXCEEDED TARGET!**
- **Sell Precision**: 0.8243 ✅ **EXCEEDED TARGET!**
- **Buy Recall**: 0.6872 ✅ **EXCEEDED TARGET!**
- **Sell Recall**: 0.9883 ✅ **EXCEEDED TARGET!**
- **Matthews Correlation Coefficient**: 0.7350 ✅ **EXCEEDED TARGET!**

### Trading Performance Metrics:

- **Win Rate**: 47.50% ❌ **NEEDS IMPROVEMENT**
- **Total Return**: 23.96% ✅ **IMPROVED**
- **Annualized Return**: 10.61% ✅ **IMPROVED**
- **Sharpe Ratio**: 1.53 ❌ **NEEDS IMPROVEMENT**
- **Max Drawdown**: 3.78% ✅ **ACCEPTABLE**

## Primary Target Metrics

### 1. Win Rate Target: 75%

**Current**: ~58-60%  
**Target**: 75%  
**Gap**: +15-17 percentage points

**Rationale**:

- Win rate is the most critical metric for profitable trading
- 75% win rate provides sufficient margin for transaction costs and slippage
- Achievable with improved model confidence and better strategy selection

**Measurement**:

- Track across all backtest periods (Recent, Middle, Older, Full)
- Monitor consistency across different market conditions
- Ensure win rate stability over time

### 2. Balanced Accuracy Target: 75%

**Current**: 58.47%  
**Target**: 75%  
**Gap**: +16.53 percentage points

**Rationale**:

- Balanced accuracy provides overall model performance measure
- Accounts for class imbalance in trading datasets
- More reliable than simple accuracy for imbalanced data

**Measurement**:

- Calculate as (Sensitivity + Specificity) / 2
- Monitor during training and validation
- Track across different market regimes

### 3. Buy/Sell F1 Score Balance Target: <0.1 Difference

**Current**: Buy F1 (0.3893) vs Sell F1 (0.7714) = 0.3821 difference  
**Target**: <0.1 difference  
**Gap**: Reduce imbalance by 0.2821

**Rationale**:

- Current model heavily favors sell signals
- Balanced buy/sell performance needed for complete trading strategy
- Prevents overfitting to one type of signal

**Measurement**:

- Track F1 scores for both buy and sell predictions
- Aim for |Buy F1 - Sell F1| < 0.1
- Monitor during training and validation

### 4. Prediction Confidence Target: 0.7+ Average

**Current**: 0.4-0.6 range  
**Target**: 0.7+ average  
**Gap**: +0.1-0.3 improvement

**Rationale**:

- Higher confidence indicates stronger model conviction
- Reduces false positive trades
- Enables better position sizing decisions

**Measurement**:

- Track average confidence across all predictions
- Monitor confidence distribution (target: fewer low-confidence predictions)
- Ensure confidence correlates with accuracy

## Secondary Target Metrics

### 5. Sharpe Ratio Target: >2.0

**Current**: 1.24-6.67 (varies by period)  
**Target**: >2.0 consistently  
**Gap**: Improve worst-case performance

**Rationale**:

- Sharpe ratio measures risk-adjusted returns
- > 2.0 indicates excellent risk-adjusted performance
- Ensures consistent performance across market conditions

### 6. Maximum Drawdown Target: <2%

**Current**: 1-2.5%  
**Target**: <2%  
**Gap**: Reduce worst-case drawdown

**Rationale**:

- Lower drawdown reduces capital risk
- Enables more aggressive position sizing
- Improves investor confidence

### 7. Average Holding Period Target: <30 days

**Current**: 799 days  
**Target**: <30 days  
**Gap**: Dramatic reduction needed

**Rationale**:

- Shorter holding periods increase trading frequency
- Reduces exposure to adverse market movements
- Enables faster capital deployment

### 8. Strategy Diversification Target: <50% Single Strategy

**Current**: Heavy momentum strategy bias  
**Target**: <50% reliance on any single strategy  
**Gap**: Improve strategy distribution

**Rationale**:

- Diversified strategies reduce overfitting
- Adapts to different market conditions
- Improves overall system robustness

## Model-Specific Training Targets

### 9. Validation Loss Target: <0.08

**Current**: 0.0990 (best achieved)  
**Target**: <0.08  
**Gap**: -0.019 improvement

**Rationale**:

- Lower validation loss indicates better generalization
- Prevents overfitting to training data
- Ensures model learns meaningful patterns

### 10. Matthews Correlation Coefficient Target: >0.4

**Current**: 0.2072  
**Target**: >0.4  
**Gap**: +0.1928 improvement

**Rationale**:

- MCC is robust to class imbalance
- > 0.4 indicates good model performance
- Better measure than accuracy for imbalanced datasets

### 11. Precision-Recall Balance Target: <0.1 Difference

**Current**: Significant imbalance between precision and recall  
**Target**: <0.1 difference for both buy and sell  
**Gap**: Balance precision and recall

**Rationale**:

- Balanced precision-recall indicates good model calibration
- Prevents over-prediction or under-prediction bias
- Ensures reliable trading signals

## Training Process Targets

### 12. Training Convergence Target: <100 Epochs

**Current**: 120 epochs (early stopping at 91)  
**Target**: <100 epochs  
**Gap**: Faster convergence

**Rationale**:

- Faster convergence indicates better learning
- Reduces training time and computational cost
- Prevents overfitting from excessive training

### 13. Learning Rate Efficiency Target: Stable Learning

**Current**: Learning rate reduction from 0.0001 to 0.000001  
**Target**: More stable learning rate progression  
**Gap**: Better learning rate scheduling

**Rationale**:

- Stable learning indicates good optimization
- Prevents learning rate collapse
- Enables better model convergence

### 14. Class Balance Target: 40-60% Buy Ratio

**Current**: Varies from 0.000 to 0.674  
**Target**: 40-60% buy ratio consistently  
**Gap**: More balanced class distribution

**Rationale**:

- Balanced classes improve model learning
- Prevents bias toward majority class
- Enables better buy signal prediction

## Implementation Priority

### Phase 1 (Immediate - 1-2 weeks): Core Metrics

1. **Win Rate**: Target 65% (current: ~58-60%)
2. **Balanced Accuracy**: Target 65% (current: 58.47%)
3. **Buy F1 Score**: Target 0.5+ (current: 0.3893)
4. **Prediction Confidence**: Target 0.6+ average

### Phase 2 (Short-term - 2-4 weeks): Balance Metrics

1. **Buy/Sell F1 Balance**: Target <0.2 difference
2. **Precision-Recall Balance**: Target <0.15 difference
3. **Class Balance**: Target 45-55% buy ratio
4. **Validation Loss**: Target <0.09

### Phase 3 (Medium-term - 1-2 months): Advanced Metrics

1. **Win Rate**: Target 70%+
2. **Balanced Accuracy**: Target 70%+
3. **Sharpe Ratio**: Target >2.0 consistently
4. **Strategy Diversification**: Target <60% single strategy

### Phase 4 (Long-term - 2-3 months): Optimal Performance

1. **Win Rate**: Target 75%
2. **Balanced Accuracy**: Target 75%
3. **Buy/Sell F1 Balance**: Target <0.1 difference
4. **Prediction Confidence**: Target 0.7+ average
5. **Maximum Drawdown**: Target <2%

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

- Win Rate: 65%+
- Balanced Accuracy: 65%+
- Buy F1 Score: 0.5+
- Prediction Confidence: 0.6+ average

### Target Performance

- Win Rate: 75%
- Balanced Accuracy: 75%
- Buy/Sell F1 Balance: <0.1 difference
- Prediction Confidence: 0.7+ average
- Sharpe Ratio: >2.0
- Maximum Drawdown: <2%

### Optimal Performance

- Win Rate: 80%+
- Balanced Accuracy: 80%+
- Buy/Sell F1 Balance: <0.05 difference
- Prediction Confidence: 0.8+ average
- Sharpe Ratio: >3.0
- Maximum Drawdown: <1.5%

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

These target metrics provide a clear roadmap for improving the Cardano trading model's performance. The phased approach ensures steady progress while maintaining system stability. Regular monitoring and adjustment of these targets will guide the development toward optimal trading performance.

The focus should be on achieving balanced performance across all metrics rather than optimizing any single metric at the expense of others. This balanced approach will result in a more robust and profitable trading system.
