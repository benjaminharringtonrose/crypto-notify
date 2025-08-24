# Technical Indicator Experimentation Prompt

## 🎯 Objective

**Systematically experiment with 5 new technical indicators** to improve our Bitcoin trading model performance using the gradual feature optimization script. **Keep any indicator that improves the model and move on to the next experiment.**

## 📊 Current Context

- **Model**: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)
- **Current Features**: 26 optimized features (see FeatureCalculator.ts)
- **Performance Target**: Improve combined F1 score, validation accuracy, and balanced accuracy
- **Tolerance**: 2% performance improvement threshold

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

## 📈 **5 SELECTED INDICATORS FOR EXPERIMENTATION**

Based on analysis of current feature set, these indicators are available but not in optimized features:

### **1. Stochastic RSI (StochRSI)** - Momentum

- **Status**: Already implemented in `calculateStochRSI()` but not in optimized features
- **Expected Benefit**: Combines RSI and Stochastic for better momentum signals
- **Implementation**: Add `indicators.stochRsi` to optimized features

### **2. Commodity Channel Index (CCI)** - Momentum

- **Status**: Not implemented yet
- **Expected Benefit**: Measures price deviations from average, good for overbought/oversold
- **Implementation**: Add `calculateCCI()` method and integrate

### **3. Money Flow Index (MFI)** - Volume-Based Momentum

- **Status**: Not implemented yet
- **Expected Benefit**: Volume-weighted RSI alternative, better for volume analysis
- **Implementation**: Add `calculateMFI()` method and integrate

### **4. Keltner Channels** - Volatility

- **Status**: Not implemented yet
- **Expected Benefit**: Volatility-based support/resistance levels
- **Implementation**: Add `calculateKeltnerChannels()` method and integrate

### **5. Aroon Indicator** - Trend

- **Status**: Not implemented yet
- **Expected Benefit**: Trend strength and reversal timing
- **Implementation**: Add `calculateAroon()` method and integrate

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
- **Expected Benefits**: [Why it might improve our model]

### **Implementation Details**

- **Feature Name**: [e.g., "stochRsi", "cci", "mfi"]
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

1. **Experiment 1**: Stochastic RSI (already implemented)
2. **Experiment 2**: Commodity Channel Index (CCI)
3. **Experiment 3**: Money Flow Index (MFI)
4. **Experiment 4**: Keltner Channels
5. **Experiment 5**: Aroon Indicator

**After each experiment:**

- If indicator improves performance → **KEEP** and add to optimized features
- If no improvement → **REMOVE** and revert changes
- **Continue to next indicator** regardless of outcome

## 📝 **Specific Request Format**

```
Please experiment with [INDICATOR_NAME] for our Bitcoin trading model.
This is experiment #[X] of 5 in our systematic indicator testing process.
[Brief description of why this indicator might help]
```

---

**Note**: This systematic approach ensures we test multiple indicators efficiently, keeping only those that provide measurable performance improvements while maintaining the model's current optimization level.
