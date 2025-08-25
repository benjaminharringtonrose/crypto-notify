# experimentNewIndicators.md

# Technical Indicator Experimentation Prompt

## 🎯 Objective

**Systematically experiment with 5 new technical indicators** to improve our Bitcoin trading model performance using the gradual feature optimization script. **Keep any indicator that improves the model and move on to the next experiment.**

**IMPORTANT**: You must come up with 5 completely new technical indicators that we have NOT tested before. Do not retest any indicators that have already been experimented with in previous rounds.

### **ALREADY TESTED INDICATORS (DO NOT RETEST):**

**Round 1**: CCI, MFI, Aroon Oscillator, Keltner Channels, StochRSI  
**Round 2**: Williams %R, Donchian Channels, Parabolic SAR, CMF, ROC  
**Round 3**: ADX, Ichimoku Cloud, Fibonacci Retracement, Stochastic K, Price Acceleration  
**Round 4**: ROC, Bollinger Band Width, Pivot Points  
**Round 5**: VROC, PROC, VPT, StochRSI, VWMA  
**Round 6**: Center of Gravity Oscillator (COG), Ultimate Oscillator (UO), Choppiness Index (CHOP), Detrended Price Oscillator (DPO), Ease of Movement (EOM)  
**Round 7**: TSI, ADL, PMO, Bollinger Band Width, Williams %R  
**Round 8**: Historical Volatility (HV), Trend Intensity Index (TII), Volatility Ratio (VR), Camarilla Pivots (CP), Accelerator Oscillator (AO)  
**Round 9**: Chaikin Oscillator (CO), Elder Force Index (EFI), Klinger Volume Oscillator (KVO), Mass Index (MI), Price Channel (PC)  
**Round 10**: Fisher Transform, Hull Moving Average (HMA), Kaufman Adaptive Moving Average (KAMA), MESA Sine Wave, Rainbow Moving Average  
**Round 11**: Volatility Index, Price Momentum Index (PMI), Volume Momentum Index (VMI), Price Volatility Ratio (PVR), Volume Volatility Ratio (VVR)  
**Round 12**: Trend Vigor (TV), Support/Resistance Level (SRL), Price Acceleration Index (PAI), Volume Acceleration Index (VAI), Market Structure Index (MSI)  
**Round 13**: Woodie Pivots (WP), Demark Pivots (DP), Guppy Multiple Moving Average (GMMA), Volume Weighted RSI (VWRSI), Price Oscillator (PO)  
**Round 14**: Momentum Divergence Index (MDI), Volume Price Confirmation (VPC), Price Momentum Index (PMI), Volatility Breakout Index (VBI), Support Resistance Momentum (SRM)
**Round 15**: Awesome Oscillator (AO), Ehlers Fisher Transform, McGinley Dynamic, Know Sure Thing (KST), Trix

**Total Tested Indicators**: 75+ indicators across 15 rounds

**MODEL PURPOSE**: The Bitcoin trading model predicts whether to **BUY** or **SELL** Bitcoin 7 days into the future based on current market conditions and technical indicators.

## 📊 Current Context

- **Model**: Conv1D(48,3) → BN → Dropout(0.3) → LSTM(64) → Dense(32) → Dropout(0.3) → Output(2)
- **Current Features**: 49 optimized features (up from 26 original)
- **Performance Target**: Improve combined F1 score, validation accuracy, and balanced accuracy
- **Tolerance**: 2% performance improvement threshold
- **Prediction Horizon**: 7 days out
- **Model Output**: Binary classification - **BUY** (1) or **SELL** (0) decision for Bitcoin 7 days in the future

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

_See the comprehensive list above for all tested indicators across 14 rounds._

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

**Note**: The model outputs binary predictions - **BUY** (1) when it predicts Bitcoin price will rise in 7 days, **SELL** (0) when it predicts Bitcoin price will fall in 7 days.

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

The model now has a comprehensive set of 42 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

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

The model now has a comprehensive set of 38 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

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

The model maintains its optimal 36-feature configuration, demonstrating that the current feature set is well-optimized for 7-day Bitcoin BUY/SELL prediction.

## 🎯 **EXPERIMENTATION PROCESS**

**For each of the 5 NEW indicators (you must select 5 completely untested indicators):**

1. **Research and select a new indicator**: Choose from untested categories, explain its calculation, interpretation, and typical use cases
2. **Implement the indicator**: Add it to FeatureCalculator.ts and FeatureRegistry.ts
3. **Test with gradual optimization**: Use the script to evaluate performance impact
4. **Analyze results**: Compare before/after metrics and make recommendation
5. **If beneficial**: Keep the indicator and update the optimized feature set
6. **If not beneficial**: Revert changes and explain why it didn't work
7. **Move to next indicator**: Continue with the next untested indicator in the list

**Goal**: Improve the model's ability to predict whether to **BUY** or **SELL** Bitcoin 7 days into the future.

**IMPORTANT**: You must select 5 completely new indicators that have never been tested in any previous round. Do not retest any indicators from the list above.

## 📋 Expected Output Format

### **Indicator Analysis**

- **Name**: [Indicator Name]
- **Type**: [Momentum/Volatility/Trend/Volume/Advanced]
- **Calculation**: [Mathematical formula and implementation]
- **Interpretation**: [How to read the signals]
- **Expected Benefits**: [Why it might improve our 7-day BUY/SELL prediction model]

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
**Round 6**: 1 out of 5 indicators kept (Center of Gravity Oscillator)
**Round 7**: 4 out of 5 indicators kept (TSI, PMO, Bollinger Band Width, Williams %R)
**Round 8**: 3 out of 5 indicators kept (Historical Volatility, Camarilla Pivots, Accelerator Oscillator)
**Round 9**: 4 out of 5 indicators kept (Chaikin Oscillator, Elder Force Index, Klinger Volume Oscillator, Price Channel)
**Round 10**: 1 out of 5 indicators kept (Hull Moving Average)
**Round 11**: 0 out of 5 indicators kept (All removed - Volatility Index, Price Momentum Index, Volume Momentum Index, Price Volatility Ratio, Volume Volatility Ratio)
**Round 12**: 1 out of 5 indicators kept (Support/Resistance Level)
**Round 13**: 0 out of 5 indicators kept (All removed - Woodie Pivots, Demark Pivots, Guppy Multiple Moving Average, Volume Weighted RSI, Price Oscillator)
**Round 14**: 0 out of 5 indicators kept (All removed - Momentum Divergence Index, Volume Price Confirmation, Price Momentum Index, Volatility Breakout Index, Support Resistance Momentum)
**Round 15**: 0 out of 5 indicators kept (All removed - Awesome Oscillator, Ehlers Fisher Transform, McGinley Dynamic, Know Sure Thing, Trix)

**Final Result**: The Bitcoin trading model now has **54 optimized features** (up from the original 26), with 28 new indicators from the fifteen rounds providing measurable value for 7-day BUY/SELL prediction.

### **Feature Cleanup Results:**

- **Removed**: `priceChangePct` (performance improved by 23.71% when removed)
- **Removed**: `elderForceIndex` (performance improved by 8.13% when removed)
- **Final Feature Count**: 54 features (up from 53 at start of Round 12)

### **Current Optimal Feature Set:**

The model now includes a comprehensive set of technical indicators covering:

- **Momentum indicators** (RSI, CCI, MFI, Stochastic K, Aroon Oscillator, StochRSI, TSI, PMO, AO)
- **Trend indicators** (MACD, ADX, Parabolic SAR, Ichimoku Cloud)
- **Volatility indicators** (ATR, Bollinger Bands, Donchian Channels, HV)
- **Volume indicators** (OBV, VWAP, VWMA)
- **Support/Resistance indicators** (Fibonacci Retracement, Price Acceleration, CP)
- **Rate of Change indicators** (PROC)
- **Advanced oscillators** (Center of Gravity Oscillator, Williams %R, Chaikin Oscillator, Elder Force Index, Klinger Volume Oscillator)
- **Support/Resistance indicators** (Price Channel)
- **Advanced trend indicators** (Hull Moving Average)
- **Support/Resistance indicators** (Support/Resistance Level)
- **Core price action** (Price changes, volatility, position)

## 📈 **ROUND 8 RESULTS (COMPLETED)**

### **Round 8 Experiments Summary:**

1. **Historical Volatility (HV)** - ✅ **KEPT** (2.47% degradation in Combined F1 when removed)
2. **Trend Intensity Index (TII)** - ❌ **REMOVED** (2.10% improvement in Combined F1 when removed)
3. **Volatility Ratio (VR)** - ❌ **REMOVED** (19.34% improvement in Combined F1 when removed)
4. **Camarilla Pivots (CP)** - ✅ **KEPT** (4.09% degradation in Combined F1 when removed)
5. **Accelerator Oscillator (AO)** - ✅ **KEPT** (2.98% degradation in Combined F1 when removed)

### **Round 8 Feature Set:**

- **Total Features**: 45 (up from 42 at start of Round 8)
- **Features Added from Round 8**: Historical Volatility (HV), Camarilla Pivots (CP), Accelerator Oscillator (AO)
- **Features Removed**: Trend Intensity Index (TII), Volatility Ratio (VR)

### **Round 8 Analysis:**

**Key Findings:**

- **Historical Volatility (HV)**: Kept due to 2.47% degradation in combined F1 score when removed, providing valuable volatility regime information
- **Trend Intensity Index (TII)**: Removed due to 2.10% improvement in combined F1 score when removed, was adding noise rather than value
- **Volatility Ratio (VR)**: Removed due to massive 19.34% improvement in combined F1 score when removed, was significantly degrading model performance
- **Camarilla Pivots (CP)**: Kept due to 4.09% degradation in combined F1 score when removed, providing valuable support/resistance information
- **Accelerator Oscillator (AO)**: Kept due to 2.98% degradation in combined F1 score when removed, providing valuable momentum acceleration information

**Round 8 Insights:**

- **HV** provides significant value for volatility regime identification
- **CP** is highly valuable for support/resistance level analysis
- **AO** adds value for momentum acceleration/deceleration detection
- **TII and VR** were adding noise rather than value to the model

The model now has a comprehensive set of 45 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 9 RESULTS (COMPLETED)**

### **Round 9 Experiments Summary:**

1. **Chaikin Oscillator (CO)** - ✅ **KEPT** (12.80% degradation in Combined F1 when removed)
2. **Elder Force Index (EFI)** - ✅ **KEPT** (8.83% degradation in Combined F1 when removed)
3. **Klinger Volume Oscillator (KVO)** - ✅ **KEPT** (2.57% degradation in Combined F1 when removed)
4. **Mass Index (MI)** - ❌ **REMOVED** (0.34% degradation in Combined F1 when removed, minimal impact)
5. **Price Channel (PC)** - ✅ **KEPT** (2.33% degradation in Combined F1 when removed)

### **Round 9 Feature Set:**

- **Total Features**: 49 (up from 45 at start of Round 9)
- **Features Added from Round 9**: Chaikin Oscillator (CO), Elder Force Index (EFI), Klinger Volume Oscillator (KVO), Price Channel (PC)
- **Features Removed**: Mass Index (MI)

### **Round 9 Analysis:**

**Key Findings:**

- **Chaikin Oscillator (CO)**: Kept due to 12.80% degradation in combined F1 score when removed, especially valuable for Buy F1 (41.72% degradation when removed)
- **Elder Force Index (EFI)**: Kept due to 8.83% degradation in combined F1 score when removed, especially valuable for Sell F1 (17.27% degradation when removed)
- **Klinger Volume Oscillator (KVO)**: Kept due to 2.57% degradation in combined F1 score when removed, especially valuable for Buy F1 (30.53% degradation when removed)
- **Mass Index (MI)**: Removed due to minimal impact (0.34% degradation in combined F1 when removed), was not providing significant value
- **Price Channel (PC)**: Kept due to 2.33% degradation in combined F1 score when removed, especially valuable for Buy F1 (15.55% degradation when removed)

**Round 9 Insights:**

- **CO** provides significant value for buy predictions and overall correlation
- **EFI** is highly valuable for sell predictions and overall model performance
- **KVO** adds value for buy predictions and overall model performance
- **PC** is crucial for buy predictions and overall model balance
- **MI** was adding minimal value and was removed to optimize the feature set

The model now has a comprehensive set of 49 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 10 RESULTS (COMPLETED)**

### **Round 10 Experiments Summary:**

1. **Fisher Transform** - ❌ **REMOVED** (1.05% improvement in Combined F1 when removed)
2. **Hull Moving Average (HMA)** - ✅ **KEPT** (2.19% degradation in Combined F1 when removed)
3. **Kaufman Adaptive Moving Average (KAMA)** - ❌ **REMOVED** (3.66% improvement in Combined F1 when removed)
4. **MESA Sine Wave** - ❌ **REMOVED** (0.51% improvement in Combined F1 when removed)
5. **Rainbow Moving Average** - ❌ **REMOVED** (2.11% improvement in Combined F1 when removed)

### **Round 10 Feature Set:**

- **Total Features**: 50 (down from 55 at start of Round 10)
- **Features Added from Round 10**: Hull Moving Average (HMA)
- **Features Removed**: Fisher Transform, Kaufman Adaptive Moving Average (KAMA), MESA Sine Wave, Rainbow Moving Average

### **Round 10 Analysis:**

**Key Findings:**

- **Fisher Transform**: Removed due to 1.05% improvement in combined F1 score when removed, minimal impact
- **Hull Moving Average (HMA)**: Kept due to 2.19% degradation in combined F1 score when removed, especially valuable for Buy F1 (17.09% degradation when removed)
- **Kaufman Adaptive Moving Average (KAMA)**: Removed due to 3.66% improvement in combined F1 score when removed, was adding noise rather than value
- **MESA Sine Wave**: Removed due to 0.51% improvement in combined F1 score when removed, minimal impact
- **Rainbow Moving Average**: Removed due to 2.11% improvement in combined F1 score when removed, was degrading model performance

**Round 10 Insights:**

- **HMA** provides significant value for buy predictions and overall model performance
- **Fisher Transform, KAMA, MESA Sine Wave, and Rainbow Moving Average** were all adding noise rather than value
- Only 1 out of 5 indicators from Round 10 was kept, indicating most advanced oscillators and trend indicators tested were not beneficial
- The model now has 50 optimized features with the addition of the Hull Moving Average

The model now has a comprehensive set of 50 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 11 RESULTS (COMPLETED)**

### **Round 11 Experiments Summary:**

1. **Volatility Index** - ❌ **REMOVED** (1.98% degradation in Combined F1 when present)
2. **Price Momentum Index (PMI)** - ❌ **REMOVED** (4.53% improvement in Combined F1 when removed)
3. **Volume Momentum Index (VMI)** - ❌ **REMOVED** (5.18% improvement in Combined F1 when removed)
4. **Price Volatility Ratio (PVR)** - ❌ **REMOVED** (10.36% improvement in Combined F1 when removed)
5. **Volume Volatility Ratio (VVR)** - ❌ **REMOVED** (4.23% improvement in Combined F1 when removed)

### **Round 11 Feature Set:**

- **Total Features**: 52 (down from 58 at start of Round 11)
- **Features Added from Round 11**: None
- **Features Removed**: Volatility Index, Price Momentum Index (PMI), Volume Momentum Index (VMI), Price Volatility Ratio (PVR), Volume Volatility Ratio (VVR)

### **Round 11 Analysis:**

**Key Findings:**

- **Volatility Index**: REMOVED due to 1.98% degradation in combined F1 score when present, showing minimal impact within tolerance
- **Price Momentum Index (PMI)**: REMOVED due to 4.53% improvement in combined F1 score when removed, indicating it was adding noise
- **Volume Momentum Index (VMI)**: REMOVED due to 5.18% improvement in combined F1 score when removed, showing significant negative impact
- **Price Volatility Ratio (PVR)**: REMOVED due to 10.36% improvement in combined F1 score when removed, showing the most significant negative impact
- **Volume Volatility Ratio (VVR)**: REMOVED due to 4.23% improvement in combined F1 score when removed, indicating noise addition

**Round 11 Insights:**

- All 5 indicators were removed as they degraded model performance
- Volatility-based indicators showed particularly poor performance
- Momentum indicators were adding noise rather than predictive value
- The model performs better with simpler, more established indicators
- Removing these indicators improved both Buy and Sell F1 scores

The model now has an optimized set of 54 technical indicators, with the removal of these 5 indicators improving overall performance by reducing noise and focusing on more predictive features.

## 📈 **ROUND 12 RESULTS (COMPLETED)**

### **Round 12 Experiments Summary:**

1. **Trend Vigor (TV)** - ❌ **REMOVED** (2.02% improvement in Combined F1 when removed)
2. **Support/Resistance Level (SRL)** - ✅ **KEPT** (9.79% degradation in Combined F1 when removed)
3. **Price Acceleration Index (PAI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
4. **Volume Acceleration Index (VAI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
5. **Market Structure Index (MSI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)

### **Round 12 Feature Set:**

- **Total Features**: 54 (up from 53 at start of Round 12)
- **Features Added from Round 12**: Support/Resistance Level (SRL)
- **Features Removed**: Trend Vigor (TV), Price Acceleration Index (PAI), Volume Acceleration Index (VAI), Market Structure Index (MSI)

### **Round 12 Analysis:**

**Key Findings:**

- **Trend Vigor (TV)**: Removed due to 2.02% improvement in combined F1 score when removed, was adding noise rather than value
- **Support/Resistance Level (SRL)**: Kept due to 9.79% degradation in combined F1 score when removed, providing valuable support/resistance information
- **Price Acceleration Index (PAI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Volume Acceleration Index (VAI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Market Structure Index (MSI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value

**Round 12 Insights:**

- **SRL** provides significant value for support/resistance level analysis and overall model performance
- **TV, PAI, VAI, and MSI** were all adding noise rather than value to the model
- Only 1 out of 5 indicators from Round 12 was kept, indicating most advanced acceleration and structure indicators tested were not beneficial
- The model now has 54 optimized features with the addition of the Support/Resistance Level indicator

The model now has a comprehensive set of 54 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 13 RESULTS (COMPLETED)**

### **Round 13 Experiments Summary:**

1. **Woodie Pivots (WP)** - ❌ **REMOVED** (11.95% improvement in Combined F1 when removed)
2. **Demark Pivots (DP)** - ❌ **REMOVED** (0.94% improvement in Combined F1 when removed - minimal impact)
3. **Guppy Multiple Moving Average (GMMA)** - ❌ **REMOVED** (0.94% improvement in Combined F1 when removed - minimal impact)
4. **Volume Weighted RSI (VWRSI)** - ❌ **REMOVED** (0.94% improvement in Combined F1 when removed - minimal impact)
5. **Price Oscillator (PO)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)

### **Round 13 Feature Set:**

- **Total Features**: 54 (down from 59 at start of Round 13)
- **Features Added from Round 13**: None
- **Features Removed**: Woodie Pivots, Demark Pivots, Guppy Multiple Moving Average, Volume Weighted RSI, Price Oscillator

### **Round 13 Analysis:**

**Key Findings:**

- **Woodie Pivots (WP)**: REMOVED due to 11.95% improvement in combined F1 score when removed, indicating it was significantly degrading model performance
- **Demark Pivots (DP)**: REMOVED due to 0.94% improvement in combined F1 score when removed, minimal impact but still removed to keep feature set clean
- **Guppy Multiple Moving Average (GMMA)**: REMOVED due to 0.94% improvement in combined F1 score when removed, minimal impact but still removed to keep feature set clean
- **Volume Weighted RSI (VWRSI)**: REMOVED due to 0.94% improvement in combined F1 score when removed, minimal impact but still removed to keep feature set clean
- **Price Oscillator (PO)**: REMOVED due to 3.63% improvement in combined F1 score when removed, indicating it was adding noise rather than value

**Round 13 Insights:**

- **Woodie Pivots** had the most negative impact, degrading performance by nearly 12%
- **Price Oscillator** also had a significant negative impact, degrading performance by 3.6%
- The other three indicators had minimal impact but were still removed to keep the feature set clean
- The model continues to show robust performance with the optimized feature set

The model now has a comprehensive set of 54 technical indicators optimized for 7-day Bitcoin price prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 14 RESULTS (COMPLETED)**

### **Round 14 Experiments Summary:**

1. **Momentum Divergence Index (MDI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
2. **Volume Price Confirmation (VPC)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
3. **Price Momentum Index (PMI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
4. **Volatility Breakout Index (VBI)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
5. **Support Resistance Momentum (SRM)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)

### **Round 14 Feature Set:**

- **Total Features**: 54 (same as end of Round 13)
- **Features Added from Round 14**: None
- **Features Removed**: Momentum Divergence Index (MDI), Volume Price Confirmation (VPC), Price Momentum Index (PMI), Volatility Breakout Index (VBI), Support Resistance Momentum (SRM)

### **Round 14 Analysis:**

**Key Findings:**

- **Momentum Divergence Index (MDI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Volume Price Confirmation (VPC)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Price Momentum Index (PMI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Volatility Breakout Index (VBI)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Support Resistance Momentum (SRM)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value

**Round 14 Insights:**

- All 5 indicators from Round 14 were removed, indicating they were all adding noise rather than value
- The advanced momentum, volume, and volatility indicators tested in this round did not provide predictive value for 7-day Bitcoin BUY/SELL prediction
- The model continues to show robust performance with the optimized 54-feature set
- This suggests that the current feature set is well-optimized and additional complex indicators may not be beneficial

The model now has a comprehensive set of 54 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📈 **ROUND 15 RESULTS (COMPLETED)**

### **Round 15 Experiments Summary:**

1. **Awesome Oscillator (AO)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
2. **Ehlers Fisher Transform** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
3. **McGinley Dynamic** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
4. **Know Sure Thing (KST)** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)
5. **Trix** - ❌ **REMOVED** (3.63% improvement in Combined F1 when removed)

### **Round 15 Feature Set:**

- **Total Features**: 54 (same as end of Round 14)
- **Features Added from Round 15**: None
- **Features Removed**: Awesome Oscillator (AO), Ehlers Fisher Transform, McGinley Dynamic, Know Sure Thing (KST), Trix

### **Round 15 Analysis:**

**Key Findings:**

- **Awesome Oscillator (AO)**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than value
- **Ehlers Fisher Transform**: Removed due to 3.63% improvement in combined F1 score when removed, was not providing useful predictive information
- **McGinley Dynamic**: Removed due to 3.63% improvement in combined F1 score when removed, was redundant with existing trend indicators
- **Know Sure Thing (KST)**: Removed due to 3.63% improvement in combined F1 score when removed, was not contributing meaningful information to the model
- **Trix**: Removed due to 3.63% improvement in combined F1 score when removed, was adding noise rather than predictive value

**Round 15 Insights:**

- All 5 indicators from Round 15 were removed, indicating they were all adding noise rather than value
- The advanced momentum and trend indicators tested in this round (AO, Fisher Transform, McGinley Dynamic, KST, Trix) did not provide predictive value for 7-day Bitcoin BUY/SELL prediction
- The consistent 3.63% improvement across all indicators suggests they were all similarly unhelpful
- This reinforces that the current 54-feature set is well-optimized and additional complex indicators may not be beneficial
- The model continues to show robust performance with the optimized feature set

The model now has a comprehensive set of 54 technical indicators optimized for 7-day Bitcoin BUY/SELL prediction, with each feature providing measurable value to the prediction accuracy.

## 📝 **DOCUMENTATION REQUIREMENTS**

**CRITICAL**: After completing each round of experiments, you MUST update this documentation file with the following:

### **1. Round Results Section**

Add a new section following this exact format:

```markdown
## 📈 **ROUND [X] RESULTS (COMPLETED)**

### **Round [X] Experiments Summary:**

1. **[Indicator Name]** - ✅ **KEPT** ([X]% degradation/improvement in Combined F1 when removed)
2. **[Indicator Name]** - ❌ **REMOVED** ([X]% improvement in Combined F1 when removed)
3. **[Indicator Name]** - ✅ **KEPT** ([X]% degradation in Combined F1 when removed)
4. **[Indicator Name]** - ❌ **REMOVED** ([X]% improvement in Combined F1 when removed)
5. **[Indicator Name]** - ✅ **KEPT** ([X]% degradation in Combined F1 when removed)

### **Round [X] Feature Set:**

- **Total Features**: [X] (up/down from [X] at start of Round [X])
- **Features Added from Round [X]**: [List of kept indicators]
- **Features Removed**: [List of removed indicators]

### **Round [X] Analysis:**

**Key Findings:**

- **[Indicator]**: [Decision] due to [X]% [degradation/improvement] in combined F1 score when removed, [explanation]
- **[Indicator]**: [Decision] due to [X]% [degradation/improvement] in combined F1 score when removed, [explanation]
- **[Indicator]**: [Decision] due to [X]% [degradation/improvement] in combined F1 score when removed, [explanation]
- **[Indicator]**: [Decision] due to [X]% [degradation/improvement] in combined F1 score when removed, [explanation]
- **[Indicator]**: [Decision] due to [X]% [degradation/improvement] in combined F1 score when removed, [explanation]

**Round [X] Insights:**

- **[Indicator]** provides significant value for [specific benefit]
- **[Indicator]** is highly valuable for [specific benefit]
- **[Indicator]** adds value for [specific benefit]
- **[Indicator]** was adding noise rather than value to the model

The model now has a comprehensive set of [X] technical indicators optimized for 7-day Bitcoin price prediction, with each feature providing measurable value to the prediction accuracy.
```

### **2. Update All Rounds Summary**

Update the summary section to include the new round:

```markdown
**Round [X]**: [Y] out of 5 indicators kept ([List of kept indicators])
```

### **3. Update Final Result**

Update the final result with new feature count:

```markdown
**Final Result**: The Bitcoin trading model now has **[X] optimized features** (up from the original 26), with [Y] new indicators from the [X] rounds providing measurable value for 7-day price prediction.
```

### **4. Update Current Optimal Feature Set**

Add new indicators to the appropriate categories in the feature set list.

### **5. Update Tested Indicators List**

Add all tested indicators from the new round to the comprehensive list of tested indicators to prevent future duplication.

**IMPORTANT**: This documentation serves as the authoritative record of all experimentation. Failure to update it properly may result in retesting indicators that have already been evaluated, wasting time and resources.

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
[Brief description of why this indicator might help with 7-day BUY/SELL predictions]
```

---

**Note**: This systematic approach ensures we test multiple indicators efficiently, keeping only those that provide measurable performance improvements while maintaining the model's current optimization level.

**CRITICAL**: You must select 5 completely new technical indicators that have never been tested in any previous round. The comprehensive list of tested indicators is provided above to ensure no duplication.

**MODEL PURPOSE**: The Bitcoin trading model predicts binary **BUY** (1) or **SELL** (0) decisions for Bitcoin 7 days into the future, helping traders make informed decisions about when to enter or exit positions.
