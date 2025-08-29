# Model Improvements Documentation

## Current Baseline (January 2025)

### Performance Metrics:

- **Balanced Accuracy**: 57.58% (Target: 65%)
- **Buy F1 Score**: 54.84%
- **Sell F1 Score**: 60.00%
- **Combined F1 Score**: 1.148
- **Matthews Correlation**: 0.1526
- **Optimal Threshold**: 0.650

### Current Configuration:

- **Features**: 35 (correlation-based selection from 57)
- **Architecture**: Conv1D(48,3) → BN → LSTM(64) → Dense(32) → Output(2)
- **Timesteps**: 35 days
- **Class Balance**: 1.0:1 ratio (undersampling)
- **Loss Function**: Focal Loss with class weights [1.2, 0.8]
- **Alpha Values**: [0.45, 0.55]

### Confusion Matrix (Optimized):

```
[ [ 189, 108 ],  # Sell: 189 correct, 108 incorrect
  [ 144, 153 ] ] # Buy: 144 incorrect, 153 correct
```

## Experiment Log

### Experiment 1: Advanced Market Microstructure Features

**Status**: COMPLETED - MIXED RESULTS  
**Change**: Added 7 sophisticated market microstructure features  
**Hypothesis**: Additional features will capture more market patterns and improve accuracy  
**Target**: Increase balanced accuracy from 57.58% to 60%+

**Results**:

- **Balanced Accuracy**: 58.59% ✅ **+1.01% improvement**
- **Buy F1**: 45.58% ❌ **-9.26% degradation**
- **Sell F1**: 66.58% ✅ **+6.58% improvement**
- **Combined F1**: 1.122 ✅ **-0.026 slight degradation**
- **Matthews Correlation**: 0.1955 ✅ **+0.0429 improvement**

**Analysis**:

- ✅ **Sell performance dramatically improved** (66.58% F1 vs 60.00%)
- ✅ **Sell precision improved** (55.81% vs 56.76%)
- ✅ **Sell recall improved** (82.49% vs 63.64%)
- ❌ **Buy performance degraded** (45.58% F1 vs 54.84%)
- ❌ **Buy recall dropped significantly** (34.68% vs 51.52%)

**Key Learning**: The new market microstructure features are helping with Sell signal detection but hurting Buy signal detection. This suggests the features may be biased toward detecting downward price movements.

**Decision**: **PARTIAL SUCCESS** - Features improved overall performance but created imbalance. Need to address Buy signal degradation.

**TODO**:

- [x] Implement order flow imbalance features
- [x] Add volume profile analysis
- [x] Create market microstructure indicators
- [x] Test with new features
- [x] Document results

### Experiment 2: Architecture Optimization

**Status**: COMPLETED - FAILURE  
**Change**: Increased model capacity (LSTM 128→64, Dense 64→32, added BN/dropout)  
**Hypothesis**: Current architecture may be underfitting, more capacity will improve both Buy and Sell performance  
**Target**: Increase balanced accuracy from 58.59% to 62%+ and improve Buy F1

**Results**:

- **Balanced Accuracy**: 48.99% ❌ **-9.60% degradation**
- **Buy F1**: 23.68% ❌ **-21.90% degradation**
- **Sell F1**: 61.69% ❌ **-4.89% degradation**
- **Combined F1**: 0.854 ❌ **-0.268 degradation**
- **Matthews Correlation**: -0.027 ❌ **-0.222 degradation**

**Analysis**:

- ❌ **Balanced accuracy dropped significantly** (48.99% vs 58.59%)
- ❌ **Buy F1 score collapsed** (23.68% vs 45.58%)
- ❌ **Buy recall dropped dramatically** (15.82% vs 34.68%)
- ❌ **Matthews correlation became negative** (-0.027 vs 0.1955)
- ❌ **Model severely overfitting** to Sell signals

**Key Learning**: The increased model capacity caused severe overfitting. The model became too complex for the dataset size and started memorizing training data instead of learning generalizable patterns.

**Decision**: **FAILURE** - Reverted to baseline architecture. Need to try different approach.

**TODO**:

- [x] Modify model architecture
- [x] Test with increased capacity
- [x] Document results
- [x] Revert to baseline

### Experiment 3: Loss Function Tuning

**Status**: PLANNED  
**Change**: Fine-tune focal loss parameters to improve balance  
**Hypothesis**: Current loss function may not be optimal for this dataset, tuning gamma and alpha will improve balance  
**Target**: Increase balanced accuracy from 58.59% to 62%+ and improve Buy F1

**Planned Changes**:

1. **Adjust gamma parameter** from 1.5 to 2.0 (more focus on hard examples)
2. **Fine-tune alpha values** from [0.45, 0.55] to [0.4, 0.6] (more weight to minority class)
3. **Test different class weight combinations** in focal loss
4. **Experiment with label smoothing** to improve generalization

**Success Criteria**:

- Balanced accuracy > 62%
- Buy F1 > 50%
- Sell F1 > 65%
- No degradation in Matthews correlation

**TODO**:

- [ ] Modify focal loss parameters
- [ ] Test different gamma values
- [ ] Test different alpha combinations
- [ ] Document results

## Performance Targets

### Primary Target: 65% Balanced Accuracy

- **Current**: 57.58%
- **Target**: 65.00%
- **Gap**: +7.42 percentage points

### Secondary Targets:

- Buy F1 Score: >60%
- Sell F1 Score: >65%
- Matthews Correlation: >0.25
- Combined F1 Score: >1.2

## Experiment Methodology

### Scientific Approach:

1. **Small, focused experiments** (one variable at a time)
2. **Clear hypotheses** for each experiment
3. **Systematic testing** and documentation
4. **Success/failure/mixed result** classifications

### Documentation Strategy:

- Update after each experiment
- Include: Status, Change, Hypothesis, Results, Analysis, Key Learning, Decision
- Track performance metrics: accuracy, F1 scores, combined scores
- Document both successes AND failures

### Iterative Principles:

- Small changes to isolate what works
- Always revert failed experiments to maintain stability
- Build on successful improvements incrementally
- Test architectural changes, hyperparameters, and feature engineering
