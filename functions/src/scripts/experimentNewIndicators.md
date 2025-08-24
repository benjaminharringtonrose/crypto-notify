# Technical Indicator Experimentation Prompt

## 🎯 Objective

**Systematically experiment with 5 new technical indicators** to improve our Bitcoin trading model performance using the gradual feature optimization script. **Keep any indicator that improves the model and move on to the next experiment.**

## 📊 Current Context

- **Model**: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)
- **Current Features**: 31 optimized features (up from 26 original)
- **Performance Target**: Improve combined F1 score, validation accuracy, and balanced accuracy
- **Tolerance**: 2% performance improvement threshold
- **Prediction Horizon**: 7 days out

## 🔧 Implementation Requirements

### 1. **New Indicator Integration**

- Add the new technical indicator to `FeatureCalculator.ts` in the `compute` method
- Add corresponding feature definition to `FeatureRegistry.ts`
- Ensure the indicator is added to the `optimizedFeatures` array
- Update the feature count in `FeatureDetector.ts` if needed

### 2. **Testing Process**

Use the gradual feature optimization script to test the new indicator:

```bash
# Test the new indicator specifically
npm run features:gradual -- --feature "newIndicatorName"

# Or test all features including the new one
npm run features:gradual
```

### 3. **Performance Evaluation Criteria**

Keep the new indicator if it meets ANY of these criteria:

- **Combined F1 Score**: Improves by >2%
- **Validation Accuracy**: Improves by >2%
- **Balanced Accuracy**: Improves by >2%
- **Matthews Correlation**: Improves by >2%
- **Sell F1 Score**: Improves significantly (addresses current imbalance)

## 📈 **ROUND 1 RESULTS (COMPLETED)**

### **Previous 5 Experiments Summary:**

1. **Stochastic RSI (StochRSI)** - ❌ **REMOVED** (mixed results, no clear improvement)
2. **Commodity Channel Index (CCI)** - ✅ **KEPT** (massive 39.05% decline in Buy F1 when removed)
3. **Money Flow Index (MFI)** - ✅ **KEPT** (massive 24.37% decline in Buy F1 when removed)
4. **Keltner Channels** - ❌ **REMOVED** (1.99% improvement in Combined F1 when removed)
5. **Aroon Oscillator** - ✅ **KEPT** (massive 32.38% decline in Sell F1 when removed)

### **Round 1 Feature Set:**

- **Total Features**: 28 (up from 26 original)
- **New Features Added**: CCI, MFI, Aroon Oscillator
- **Features Removed**: Keltner Channels

## 📈 **ROUND 2 RESULTS (COMPLETED)**

### **Round 2 Experiments Summary:**

1. **Williams %R** - ❌ **REMOVED** (6.22% improvement in Combined F1 when removed)
2. **Donchian Channels** - ✅ **KEPT** (16.40% decline in Combined F1 when removed)
3. **Parabolic SAR** - ✅ **KEPT** (6.80% decline in Combined F1 when removed)
4. **Chaikin Money Flow (CMF)** - ❌ **REMOVED** (32.02% improvement in Combined F1 when removed)
5. **Rate of Change (ROC)** - ❌ **REMOVED** (17.31% improvement in Combined F1 when removed)

### **Round 2 Feature Set:**

- **Total Features**: 31 (up from 26 original)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Removed**: Keltner Channels, Williams %R, CMF, ROC

## 📈 **ROUND 3: 5 NEW INDICATORS FOR 7-DAY PREDICTION**

Based on analysis for 7-day prediction capabilities, these indicators are selected for their trend-following and momentum characteristics:

### **1. Average Directional Index (ADX)** - Trend Strength

- **Status**: Not implemented yet
- **Expected Benefit**: Measures trend strength, crucial for 7-day predictions
- **Implementation**: Add `calculateADX()` method and integrate
- **7-Day Relevance**: Strong trends are more likely to continue over longer periods

### **2. Ichimoku Cloud** - Trend & Support/Resistance

- **Status**: Not implemented yet
- **Expected Benefit**: Comprehensive trend analysis with multiple timeframes
- **Implementation**: Add `calculateIchimoku()` method and integrate
- **7-Day Relevance**: Cloud levels provide strong support/resistance for longer predictions

### **3. Volume Weighted Average Price (VWAP)** - Price Action

- **Status**: Not implemented yet
- **Expected Benefit**: Volume-weighted price levels that act as dynamic support/resistance
- **Implementation**: Add `calculateVWAP()` method and integrate
- **7-Day Relevance**: VWAP levels are significant for longer-term price movements

### **4. Fibonacci Retracement** - Support/Resistance

- **Status**: Not implemented yet
- **Expected Benefit**: Natural support/resistance levels based on Fibonacci ratios
- **Implementation**: Add `calculateFibonacciLevels()` method and integrate
- **7-Day Relevance**: Fibonacci levels often act as reversal points over longer periods

### **5. On-Balance Volume (OBV)** - Volume Trend

- **Status**: Not implemented yet
- **Expected Benefit**: Volume-based trend confirmation, leading indicator
- **Implementation**: Add `calculateOBV()` method and integrate
- **7-Day Relevance**: Volume trends often precede price trends by several days

## 🎯 **EXPERIMENTATION PROCESS**

**For each of the 5 indicators:**

1. **Research the indicator**: Explain its calculation, interpretation, and typical use cases
2. **Implement the indicator**: Add it to FeatureCalculator.ts and FeatureRegistry.ts
3. **Test with gradual optimization**: Use the script to evaluate performance impact
4. **Analyze results**: Compare before/after metrics and make recommendation
5. **If beneficial**: Keep the indicator and update the optimized feature set
6. **If not beneficial**: Revert changes and explain why it didn't work
7. **Move to next indicator**: Continue with the next indicator in the list

## 📋 Expected Output Format

### **Indicator Analysis**

- **Name**: [Indicator Name]
- **Type**: [Momentum/Volatility/Trend/Volume/Advanced]
- **Calculation**: [Mathematical formula and implementation]
- **Interpretation**: [How to read the signals]
- **Expected Benefits**: [Why it might improve our 7-day model]

### **Implementation Details**

- **Feature Name**: [e.g., "adx", "ichimokuConversion", "vwap"]
- **Code Changes**: [Files modified and specific changes]
- **Integration**: [How it fits with existing features]

### **Performance Results**

```
📈 PERFORMANCE COMPARISON:
   Metric              | Baseline | Modified | Change
   --------------------|----------|----------|--------
   Validation Accuracy | 0.5222   | 0.5444   | 4.26%
   Buy F1 Score        | 0.6126   | 0.6612   | 7.92%
   Sell F1 Score       | 0.3768   | 0.3051   | -19.04%
   Combined F1 Score   | 0.9894   | 0.9662   | -2.34%
   Balanced Accuracy   | 0.5336   | 0.5699   | 6.80%
   Matthews Correlation| 0.0782   | 0.2131   | 172.40%
```

### **Recommendation**

- **Decision**: [REMOVE/KEEP/MINIMAL_IMPACT]
- **Reasoning**: [Detailed explanation of performance impact]
- **Next Steps**: [What to do with the indicator]

## 🔄 **Systematic Iteration Process**

**Complete all 5 experiments in sequence:**

1. **Experiment 1**: Average Directional Index (ADX)
2. **Experiment 2**: Ichimoku Cloud
3. **Experiment 3**: Volume Weighted Average Price (VWAP)
4. **Experiment 4**: Fibonacci Retracement
5. **Experiment 5**: On-Balance Volume (OBV)

**After each experiment:**

- If indicator improves performance → **KEEP** and add to optimized features
- If no improvement → **REMOVE** and revert changes
- **Continue to next indicator** regardless of outcome

## 📝 **Specific Request Format**

```
Please experiment with [INDICATOR_NAME] for our Bitcoin trading model.
This is experiment #[X] of 5 in our systematic indicator testing process.
[Brief description of why this indicator might help with 7-day predictions]
```

---

**Note**: This systematic approach ensures we test multiple indicators efficiently, keeping only those that provide measurable performance improvements while maintaining the model's current optimization level.
