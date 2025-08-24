# Technical Indicator Experimentation Prompt

## 🎯 Objective

**Systematically experiment with 5 new technical indicators** to improve our Bitcoin trading model performance using the gradual feature optimization script. **Keep any indicator that improves the model and move on to the next experiment.**

**IMPORTANT**: You must come up with 5 completely new technical indicators that we have NOT tested before. Do not retest any indicators that have already been experimented with in previous rounds.

## 📊 Current Context

- **Model**: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)
- **Current Features**: 38 optimized features (up from 26 original)
- **Performance Target**: Improve combined F1 score, validation accuracy, and balanced accuracy
- **Tolerance**: 2% performance improvement threshold
- **Prediction Horizon**: 7 days out

## 🔧 Implementation Requirements

### 1. **New Indicator Integration**

- **CRITICAL**: You must select 5 completely new technical indicators that have NOT been tested in any previous rounds
- Add the new technical indicator to `FeatureCalculator.ts` in the `compute` method
- Add corresponding feature definition to `FeatureRegistry.ts`
- Ensure the indicator is added to the `optimizedFeatures` array
- Update the feature count in `FeatureDetector.ts` if needed

### 2. **Indicator Selection Requirements**

**You must research and select 5 new indicators from the following categories that we have NOT tested:**

- **Advanced Momentum Indicators**: (e.g., Choppiness Index, Detrended Price Oscillator, Ease of Movement)
- **Advanced Volatility Indicators**: (e.g., Historical Volatility, Volatility Ratio, Volatility Index)
- **Advanced Trend Indicators**: (e.g., Trend Intensity Index, Trend Vigor, Trend Strength Index)
- **Advanced Volume Indicators**: (e.g., Volume Price Trend, Volume Rate of Change, Volume Weighted Average Price)
- **Advanced Oscillators**: (e.g., Ultimate Oscillator, Awesome Oscillator, Accelerator Oscillator)
- **Advanced Support/Resistance**: (e.g., Pivot Points, Camarilla Pivots, Woodie Pivots)
- **Advanced Price Action**: (e.g., Price Rate of Change, Price Momentum, Price Acceleration)
- **Advanced Market Microstructure**: (e.g., True Strength Index, Accumulation/Distribution Line, Price Momentum Oscillator)

**DO NOT retest any indicators from previous rounds:**

- Round 1: CCI, MFI, Aroon Oscillator, Keltner Channels, StochRSI
- Round 2: Williams %R, Donchian Channels, Parabolic SAR, CMF, ROC
- Round 3: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- Round 4: ROC, Bollinger Band Width, Pivot Points
- Round 5: VROC, PROC, VPT, StochRSI, VWMA
- Round 6: Center of Gravity Oscillator (COG)
- Round 7: TSI, ADL, PMO, Bollinger Band Width, Williams %R

### 3. **Testing Process**

Use the gradual feature optimization script to test the new indicator:

```bash
# Test the new indicator specifically
npm run features:gradual -- --feature "newIndicatorName"

# Or test all features including the new one
npm run features:gradual
```

### 4. **Performance Evaluation Criteria**

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

## 📈 **ROUND 3 RESULTS (COMPLETED)**

### **Round 3 Experiments Summary:**

1. **Average Directional Index (ADX)** - ✅ **KEPT** (8.84% decline in Combined F1 when removed)
2. **Ichimoku Cloud** - ✅ **KEPT** (6.32% decline in Combined F1 when removed)
3. **Fibonacci Retracement** - ✅ **KEPT** (4.45% decline in Combined F1 when removed)
4. **Stochastic K** - ✅ **KEPT** (3.68% decline in Combined F1 when removed)
5. **Price Acceleration** - ✅ **KEPT** (10.34% decline in Combined F1 when removed)

### **Round 3 Feature Set:**

- **Total Features**: 36 (up from 31 at start of Round 3)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Added from Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- **Features Removed**: Keltner Channels, Williams %R, CMF, ROC

### **Round 3 Analysis:**

All 5 indicators from Round 3 were kept, indicating they all provide significant value for the 7-day Bitcoin prediction model. The indicators cover:

- **Trend Strength**: ADX measures trend strength and direction
- **Support/Resistance**: Ichimoku Cloud provides comprehensive trend analysis
- **Price Levels**: Fibonacci Retracement identifies natural support/resistance levels
- **Momentum**: Stochastic K measures price momentum and overbought/oversold conditions
- **Acceleration**: Price Acceleration measures the rate of change in price momentum

The model now has a comprehensive set of 38 technical indicators covering momentum, trend, volatility, and microstructure features optimized for 7-day Bitcoin price prediction.

## 📈 **ROUND 5 RESULTS (COMPLETED)**

### **Round 5 Experiments Summary:**

1. **Volume Rate of Change (VROC)** - ❌ **REMOVED** (10.01% improvement in Combined F1 when removed)
2. **Price Rate of Change (PROC)** - ✅ **KEPT** (7.24% degradation in Combined F1 when removed)
3. **Volume Price Trend (VPT)** - ❌ **REMOVED** (21.26% improvement in Combined F1 when removed)
4. **Stochastic RSI (StochRSI)** - ✅ **KEPT** (15.30% degradation in Combined F1 when removed)
5. **Volume Weighted Moving Average (VWMA)** - ✅ **KEPT** (6.14% degradation in Combined F1 when removed)

### **Round 5 Feature Set:**

- **Total Features**: 38 (up from 37 at start of Round 5)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Added from Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- **Features Added from Round 5**: PROC, StochRSI, VWMA
- **Features Removed**: Keltner Channels, Williams %R, CMF, ROC, Bollinger Band Width, Pivot Points, VROC, VPT

## 📈 **ROUND 6 RESULTS (COMPLETED)**

### **Round 6 Experiments Summary:**

1. **Ultimate Oscillator (UO)** - ❌ **REMOVED** (16.22% improvement in Combined F1 when removed)
2. **Choppiness Index (CHOP)** - ❌ **REMOVED** (0.75% improvement in Combined F1 when removed, minimal impact)
3. **Detrended Price Oscillator (DPO)** - ❌ **REMOVED** (7.58% improvement in Combined F1 when removed)
4. **Center of Gravity Oscillator (COG)** - ✅ **KEPT** (10.74% degradation in Combined F1 when removed)
5. **Ease of Movement (EOM)** - ❌ **REMOVED** (0.29% improvement in Combined F1 when removed, minimal impact)

### **Round 6 Feature Set:**

- **Total Features**: 39 (down from 40 after removing EOM)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Added from Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- **Features Added from Round 5**: PROC, StochRSI, VWMA
- **Features Added from Round 6**: Center of Gravity Oscillator (COG)
- **Features Removed**: Keltner Channels, Williams %R, CMF, ROC, Bollinger Band Width, Pivot Points, VROC, VPT, Ultimate Oscillator, Choppiness Index, Detrended Price Oscillator, Ease of Movement (EOM)

## 📈 **ROUND 7 RESULTS (COMPLETED)**

### **Round 7 Experiments Summary:**

1. **True Strength Index (TSI)** - ✅ **KEPT** (6.43% degradation in Combined F1 when removed)
2. **Accumulation/Distribution Line (ADL)** - ❌ **REMOVED** (65.46% improvement in Combined F1 when removed)
3. **Price Momentum Oscillator (PMO)** - ✅ **KEPT** (10.56% degradation in Combined F1 when removed)
4. **Bollinger Band Width** - ✅ **KEPT** (3.92% degradation in Combined F1 when removed)
5. **Williams %R** - ✅ **KEPT** (13.34% degradation in Combined F1 when removed)

### **Round 7 Feature Set:**

- **Total Features**: 42 (up from 39 at start of Round 7)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Added from Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- **Features Added from Round 5**: PROC, StochRSI, VWMA
- **Features Added from Round 6**: Center of Gravity Oscillator (COG)
- **Features Added from Round 7**: TSI, PMO, Bollinger Band Width, Williams %R
- **Features Removed**: Keltner Channels, CMF, ROC, Bollinger Band Width, Pivot Points, VROC, VPT, Ultimate Oscillator, Choppiness Index, Detrended Price Oscillator, Ease of Movement (EOM), ADL

### **Round 7 Analysis:**

**Key Findings:**

- **True Strength Index (TSI)**: Kept due to 6.43% degradation in combined F1 score when removed, especially valuable for Buy F1 (13.21% degradation when removed)
- **Accumulation/Distribution Line (ADL)**: Removed due to massive 65.46% improvement in combined F1 score when removed, completely destroyed Buy F1 performance
- **Price Momentum Oscillator (PMO)**: Kept due to 10.56% degradation in combined F1 score when removed, especially valuable for Buy F1 (30.57% degradation when removed)
- **Bollinger Band Width**: Kept due to 3.92% degradation in combined F1 score when removed, valuable for Buy F1 (10.96% degradation when removed)
- **Williams %R**: Kept due to 13.34% degradation in combined F1 score when removed, especially valuable for Sell F1 (28.78% degradation when removed)

**Round 7 Insights:**

- **TSI** provides significant value for buy predictions and overall correlation
- **PMO** is highly valuable for both buy predictions and overall model performance
- **Bollinger Band Width** adds value, especially for buy predictions
- **Williams %R** is crucial for sell predictions and overall model balance
- **ADL** was adding substantial noise rather than value

The model now has a comprehensive set of 42 technical indicators optimized for 7-day Bitcoin price prediction, with each feature providing measurable value to the prediction accuracy.

### **Round 6 Analysis:**

**Key Findings:**

- **Ultimate Oscillator (UO)**: Removed due to 16.22% improvement in combined F1 score when removed
- **Choppiness Index (CHOP)**: Removed due to minimal impact (0.75% improvement in combined F1 when removed)
- **Detrended Price Oscillator (DPO)**: Removed due to 7.58% improvement in combined F1 score when removed
- **Center of Gravity Oscillator (COG)**: Kept due to 10.74% degradation in combined F1 score when removed
- **Ease of Movement (EOM)**: Removed due to minimal impact (0.29% improvement in combined F1 when removed)
- The COG showed significant value, especially for Buy F1 (25.52% degradation when removed)
- The EOM showed mixed results - improving Buy F1 by 14.09% but degrading Sell F1 by 14.25%
- One indicator from Round 6 has been kept (COG), indicating it provides unique value to the model
- The model now has 39 optimized features with the addition of the Center of Gravity Oscillator

### **Round 5 Analysis:**

**Key Findings:**

- **Volume Rate of Change (VROC)**: Removed due to 10.01% improvement in combined F1 score when removed
- **Price Rate of Change (PROC)**: Kept due to 7.24% degradation in combined F1 score when removed
- **Volume Price Trend (VPT)**: Removed due to 21.26% improvement in combined F1 score when removed
- **Stochastic RSI (StochRSI)**: Kept due to 15.30% degradation in combined F1 score when removed
- **Volume Weighted Moving Average (VWMA)**: Kept due to 6.14% degradation in combined F1 score when removed

**Round 5 Insights:**

- **PROC** provides significant value for sell predictions and overall correlation
- **StochRSI** is highly valuable for both buy and sell predictions
- **VWMA** adds value, especially for buy predictions
- **VROC and VPT** were adding noise rather than value

The model now has a comprehensive set of 38 technical indicators optimized for 7-day Bitcoin price prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 4 RESULTS (COMPLETED)**

### **Round 4 Experiments Summary:**

1. **Rate of Change (ROC)** - ❌ **REMOVED** (4.52% improvement in Combined F1 when removed)
2. **Bollinger Band Width** - ❌ **REMOVED** (8.31% improvement in Combined F1 when removed)
3. **Pivot Points** - ❌ **REMOVED** (7.39% improvement in Combined F1 when removed)

### **Round 4 Feature Set:**

- **Total Features**: 36 (same as end of Round 3)
- **Features Added from Round 1**: CCI, MFI, Aroon Oscillator
- **Features Added from Round 2**: Donchian Channels, Parabolic SAR
- **Features Added from Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration
- **Features Removed**: Keltner Channels, Williams %R, CMF, ROC, Bollinger Band Width, Pivot Points

### **Round 4 Analysis:**

All 3 indicators tested in Round 4 were removed, indicating they were adding noise to the model rather than providing value. This suggests the model has reached a high level of optimization with the current 36 features.

**Key Findings:**

- **Rate of Change (ROC)**: Removed due to 4.52% improvement in combined F1 score when removed
- **Bollinger Band Width**: Removed due to 8.31% improvement in combined F1 score when removed
- **Pivot Points**: Removed due to 7.39% improvement in combined F1 score when removed

The model maintains its optimal 36-feature configuration, demonstrating that the current feature set is well-optimized for 7-day Bitcoin price prediction.

## 🎯 **EXPERIMENTATION PROCESS**

**For each of the 5 NEW indicators (you must select 5 completely untested indicators):**

1. **Research and select a new indicator**: Choose from untested categories, explain its calculation, interpretation, and typical use cases
2. **Implement the indicator**: Add it to FeatureCalculator.ts and FeatureRegistry.ts
3. **Test with gradual optimization**: Use the script to evaluate performance impact
4. **Analyze results**: Compare before/after metrics and make recommendation
5. **If beneficial**: Keep the indicator and update the optimized feature set
6. **If not beneficial**: Revert changes and explain why it didn't work
7. **Move to next indicator**: Continue with the next untested indicator in the list

**IMPORTANT**: You must select 5 completely new indicators that have never been tested in any previous round. Do not retest any indicators from the list above.

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

## 📊 **COMPLETE EXPERIMENTATION SUMMARY**

### **All Rounds Summary:**

**Round 1**: 3 out of 5 indicators kept (CCI, MFI, Aroon Oscillator)
**Round 2**: 2 out of 5 indicators kept (Donchian Channels, Parabolic SAR)  
**Round 3**: 5 out of 5 indicators kept (ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration)
**Round 4**: 0 out of 3 indicators kept (All removed - ROC, Bollinger Band Width, Pivot Points)
**Round 5**: 3 out of 5 indicators kept (PROC, StochRSI, VWMA)

**Final Result**: The Bitcoin trading model now has **38 optimized features** (up from the original 26), with 13 new indicators from the five rounds providing measurable value for 7-day price prediction.

### **Current Optimal Feature Set:**

The model now includes a comprehensive set of technical indicators covering:

- **Momentum indicators** (RSI, CCI, MFI, Stochastic K, Aroon Oscillator, StochRSI)
- **Trend indicators** (MACD, ADX, Parabolic SAR, Ichimoku Cloud)
- **Volatility indicators** (ATR, Bollinger Bands, Donchian Channels)
- **Volume indicators** (OBV, VWAP, VWMA)
- **Support/Resistance indicators** (Fibonacci Retracement, Price Acceleration)
- **Rate of Change indicators** (PROC)
- **Core price action** (Price changes, volatility, position)

## 🔄 **Systematic Iteration Process**

**For future rounds, complete all 5 experiments in sequence with COMPLETELY NEW indicators:**

1. **Experiment 1**: [New Untested Indicator 1]
2. **Experiment 2**: [New Untested Indicator 2]
3. **Experiment 3**: [New Untested Indicator 3]
4. **Experiment 4**: [New Untested Indicator 4]
5. **Experiment 5**: [New Untested Indicator 5]

**After each experiment:**

- If indicator improves performance → **KEEP** and add to optimized features
- If no improvement → **REMOVE** and revert changes
- **Continue to next indicator** regardless of outcome

**CRITICAL REQUIREMENT**: You must research and select 5 completely new technical indicators that have never been tested in any previous round. Do not retest any indicators from the comprehensive list provided above.

## 📝 **Specific Request Format**

```
Please experiment with [NEW_UNTESTED_INDICATOR_NAME] for our Bitcoin trading model.
This is experiment #[X] of 5 in our systematic indicator testing process.
[Brief description of why this indicator might help with 7-day predictions]
```

---

**Note**: This systematic approach ensures we test multiple indicators efficiently, keeping only those that provide measurable performance improvements while maintaining the model's current optimization level.

**CRITICAL**: You must select 5 completely new technical indicators that have never been tested in any previous round. The comprehensive list of tested indicators is provided above to ensure no duplication.
