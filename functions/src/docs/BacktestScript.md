# Backtest Script Documentation

## Overview

The `backtest.ts` script is a comprehensive backtesting utility that evaluates the performance of the Bitcoin trading system across multiple historical time periods. It implements sophisticated data fetching, chunked processing, and multi-period analysis to provide comprehensive insights into trading strategy performance under various market conditions.

## Purpose

The script performs extensive backtesting across four distinct time periods to validate trading strategy robustness and identify performance patterns across different market cycles:

1. **Recent Period** (April 2024 - April 2025): Current market conditions
2. **Middle Period** (November 2022 - April 2024): Post-bear market recovery
3. **Older Period** (January 2021 - October 2021): Bull market conditions
4. **Full Period** (January 2021 - April 2025): Complete historical analysis

## Architecture

```
Backtest Script → Data Fetching → Chunked Processing → Backtesting → Evaluation
       ↓               ↓                ↓                ↓            ↓
   Multi-Period    Coinbase API    Data Assembly    Strategy     Performance
   Analysis        Integration     & Validation     Execution    Metrics
```

### Key Components

- **Data Fetching**: Intelligent chunked retrieval from Coinbase API
- **Data Processing**: Merging and validation of historical data
- **Multi-Period Testing**: Comprehensive coverage of market cycles
- **Performance Evaluation**: Detailed backtest result analysis
- **Comprehensive Reporting**: Aggregated statistics and insights

## Script Structure

### Main Execution Flow

```typescript
async function runBacktest() {
  // 1. Initialize services
  // 2. Fetch data for each time period
  // 3. Execute backtests
  // 4. Evaluate results
  // 5. Print comprehensive summary
}

// Execute with error handling
runBacktest().catch((error) => {
  console.error("Backtest failed:", error);
  process.exit(1);
});
```

### Core Functions

#### `fetchHistoricalData()`

Intelligent data fetching with automatic chunking and retry logic.

```typescript
async function fetchHistoricalData(
  coinbaseService: CoinbaseService,
  productId: string,
  start: number,
  end: number
): Promise<HistoricalData>;
```

**Parameters**:

- `coinbaseService`: CoinbaseService instance for API access
- `productId`: Trading pair identifier (e.g., "BTC-USD")
- `start`: Start timestamp in milliseconds
- `end`: End timestamp in milliseconds

**Returns**:

- `HistoricalData`: Object containing arrays of prices and volumes

**Features**:

- **Automatic Chunking**: Breaks large requests into manageable chunks
- **Adaptive Sizing**: Reduces chunk size on failures
- **Error Recovery**: Implements retry logic with exponential backoff
- **Data Validation**: Ensures complete and accurate data retrieval

#### `runBacktest()`

Main execution function orchestrating the entire backtesting process.

```typescript
async function runBacktest() {
  // Initialize services and execute multi-period backtesting
}
```

#### `printFinalSummary()`

Comprehensive reporting function that aggregates results across all periods.

```typescript
function printFinalSummary(results: BacktestPeriodResult[]) {
  // Print performance summary table
  // Calculate aggregated statistics
  // Analyze strategy and confidence distributions
  // Provide performance insights
}
```

## Data Fetching Strategy

### Chunked Data Retrieval

The script implements intelligent data chunking to handle large historical datasets efficiently:

```typescript
let maxCandles = 300; // Start with 300, reduce if needed
const oneDaySec = TIME_CONVERSIONS.ONE_DAY_IN_SECONDS;
const chunks: { prices: number[]; volumes: number[] }[] = [];

let currentStart = startUnix;
while (currentStart < endUnix) {
  const currentEnd = Math.min(currentStart + maxCandles * oneDaySec, endUnix);
  // ... fetch chunk
  currentStart = currentEnd + oneDaySec; // Avoid overlap
}
```

**Chunking Benefits**:

- **API Rate Limits**: Respects Coinbase API constraints
- **Memory Efficiency**: Processes data in manageable chunks
- **Error Recovery**: Isolated failures don't affect entire dataset
- **Progress Tracking**: Visible progress through chunk processing

### Adaptive Chunk Sizing

Implements intelligent retry logic with dynamic chunk size adjustment:

```typescript
if (maxCandles > 100) {
  maxCandles = Math.max(100, Math.floor(maxCandles / 2));
  console.log(`Retrying with smaller chunk size: ${maxCandles} candles`);
  continue;
}
```

**Adaptive Features**:

- **Initial Size**: Starts with 300 candles per chunk
- **Failure Response**: Reduces chunk size on API failures
- **Minimum Limit**: Prevents chunks smaller than 100 candles
- **Exponential Backoff**: Gradually reduces chunk size for stability

### Data Merging and Validation

Ensures data integrity through proper merging and length validation:

```typescript
// Merge chunks, reversing to maintain chronological order
const mergedPrices: number[] = [];
const mergedVolumes: number[] = [];
for (const chunk of chunks.reverse()) {
  mergedPrices.push(...chunk.prices);
  mergedVolumes.push(...chunk.volumes);
}
```

**Validation Steps**:

- **Chronological Order**: Maintains proper time sequence
- **Length Consistency**: Ensures price and volume arrays match
- **Data Completeness**: Validates all chunks were processed
- **Overlap Prevention**: Avoids duplicate data points

## Time Period Analysis

### 1. Recent Period (April 2024 - April 2025)

**Duration**: 500 days  
**Market Context**: Current market conditions, recent trends  
**Purpose**: Validate strategy performance in current market environment

```typescript
const recentDays = 500;
const recentEnd = now;
const recentStart = recentEnd - recentDays * oneDayMs;

const recentBtcData = await fetchHistoricalData(
  coinbaseService,
  "BTC-USD",
  recentStart,
  recentEnd
);
```

### 2. Middle Period (November 2022 - April 2024)

**Duration**: 500 days  
**Market Context**: Post-bear market recovery, accumulation phase  
**Purpose**: Test strategy during market recovery and consolidation

```typescript
const middleDays = 500;
const middleEnd = recentStart - oneDayMs;
const middleStart = middleEnd - middleDays * oneDayMs;
```

### 3. Older Period (January 2021 - October 2021)

**Duration**: 300 days  
**Market Context**: Bull market conditions, strong upward trends  
**Purpose**: Validate strategy performance during favorable market conditions

```typescript
const olderDays = 300;
const olderEnd = new Date("2021-10-28").getTime();
const olderStart = new Date("2021-01-01").getTime();
```

### 4. Full Period (January 2021 - April 2025)

**Duration**: ~4.25 years  
**Market Context**: Complete market cycle including bull, bear, and recovery phases  
**Purpose**: Comprehensive strategy validation across full market cycle

```typescript
const fullStart = olderStart;
const fullResult = await backtester.backtest(
  fullBtcData,
  30,
  fullBtcData.prices.length - 1
);
```

## Backtesting Execution

### TradeModelBacktester Integration

Uses the TradeModelBacktester class for strategy execution:

```typescript
const backtester = new TradeModelBacktester(10000);

const recentResult = await backtester.backtest(
  recentBtcData,
  30,
  recentBtcData.prices.length - 1
);
```

**Backtesting Parameters**:

- **Initial Capital**: $10,000 starting balance
- **Lookback Period**: 30 days for feature calculation
- **End Index**: Full dataset length minus lookback period
- **Asset**: Bitcoin (BTC) data only

### Result Evaluation

Comprehensive evaluation of each backtest period:

```typescript
const recentEvaluation = await backtester.evaluateBacktest(recentResult);
results.push({
  period: "Recent (500d)",
  ...recentEvaluation,
});
```

**Evaluation Metrics**:

- **Total Returns**: Overall strategy performance
- **Risk Metrics**: Volatility, drawdown, Sharpe ratio
- **Trade Analysis**: Number of trades, win rate, average returns
- **Strategy Distribution**: Analysis of trading strategies used
- **Confidence Distribution**: Analysis of trade confidence levels

## Comprehensive Reporting

### Performance Summary Table

The script generates a detailed performance summary table:

```
📊 PERFORMANCE SUMMARY BY PERIOD:
Period          Return    Ann. Return Win Rate  Max DD  Sharpe  Trades  Avg Hold
Recent (500d)   15.23%    11.45%      68.5%     8.45%   1.23    45      12.3d
Middle (500d)   22.67%    16.78%      71.2%     6.78%   1.67    52      10.8d
Older (300d)    18.45%    22.34%      65.8%     9.12%   1.45    38      14.2d
Full Period     19.12%    4.23%       68.7%     12.34%  0.89    135     12.1d
```

### Aggregated Statistics

Provides comprehensive aggregated metrics across all periods:

```
📈 AGGREGATED STATISTICS:
   Total Combined Return: 75.47%
   Average Return per Period: 18.87%
   Overall Win Rate: 68.7%
   Total Trades: 270
   Winning Trades: 185
   Losing Trades: 85
   Average Holding Days: 12.1
```

### Strategy and Confidence Analysis

Analyzes trading strategy and confidence distributions:

```
🎯 STRATEGY DISTRIBUTION ANALYSIS:
   momentum        : 95 trades (35.2%)
   mean_reversion  : 78 trades (28.9%)
   breakout        : 52 trades (19.3%)
   trend_following : 45 trades (16.7%)

🎯 CONFIDENCE DISTRIBUTION ANALYSIS:
   0.8+        : 45 trades (16.7%)
   0.7-0.8     : 67 trades (24.8%)
   0.6-0.7     : 89 trades (33.0%)
   0.5-0.6     : 54 trades (20.0%)
   0.4-0.5     : 15 trades (5.6%)
```

### Performance Insights

Provides key insights about strategy performance:

```
💡 PERFORMANCE INSIGHTS:
   Best Performing Period: Middle (500d) (22.67%)
   Worst Performing Period: Recent (500d) (15.23%)
   Profitable Periods: 4/4 (100.0%)
```

## Error Handling

### Comprehensive Error Management

Implements robust error handling throughout the execution pipeline:

```typescript
} catch (error: any) {
  console.warn(`Fetch failed for ${productId}:`, error.message || error);
  if (maxCandles > 100) {
    maxCandles = Math.max(100, Math.floor(maxCandles / 2));
    console.log(`Retrying with smaller chunk size: ${maxCandles} candles`);
    continue;
  }
  throw error;
}
```

**Error Handling Features**:

- **Graceful Degradation**: Continues processing when possible
- **Automatic Retry**: Implements retry logic with backoff
- **Detailed Logging**: Comprehensive error information for debugging
- **Process Termination**: Clean exit on unrecoverable errors

### Process-Level Error Handling

Ensures clean script termination on critical failures:

```typescript
runBacktest().catch((error) => {
  console.error("Backtest failed:", error);
  process.exit(1);
});
```

**Termination Benefits**:

- **Error Visibility**: Clear error reporting to console
- **Clean Exit**: Proper process termination with exit code
- **Debugging Support**: Error details preserved for analysis
- **CI/CD Integration**: Proper exit codes for automation

## Logging and Monitoring

### Progress Tracking

Comprehensive logging throughout the execution process:

```typescript
console.log(
  `Fetching ${productId} data (Timestamp ${currentStart} to ${currentEnd})...`
);
console.log(`Fetched chunk for ${productId}: ${prices.length} candles`);
console.log(`Recent Data: ${recentBtcData.prices.length} candles`);
console.log(
  `Backtesting Recent ${recentDays} Days (Days 1 to ${
    recentBtcData.prices.length - 30
  })...`
);
```

**Logging Features**:

- **Data Fetching**: Progress updates for each chunk
- **Data Processing**: Validation and trimming information
- **Backtesting Progress**: Clear indication of current period
- **Performance Metrics**: Results and evaluation summaries

### Performance Monitoring

Tracks execution progress and performance metrics:

```typescript
console.log(`Fetched chunk for ${productId}: ${prices.length} candles`);
console.log(`Recent Data: ${recentBtcData.prices.length} candles`);
```

**Monitoring Benefits**:

- **Progress Visibility**: Clear indication of execution status
- **Performance Tracking**: Monitor data processing efficiency
- **Debugging Support**: Identify bottlenecks and issues
- **Resource Management**: Track memory and processing usage

## Usage

### Prerequisites

1. **Environment Variables**:

   ```bash
   COINBASE_API_KEY=your_coinbase_api_key
   COINBASE_API_SECRET=your_coinbase_api_secret
   ```

2. **Dependencies**: Ensure all required packages are installed
3. **Coinbase Access**: Valid API credentials with appropriate permissions

### Execution

```bash
# Run the backtest script
npm run backtest

# Or execute directly with ts-node
npx ts-node functions/src/scripts/backtest.ts
```

### Expected Output

```
Fetching BTC-USD data (Timestamp 1640995200 to 1641081600)...
Fetched chunk for BTC-USD: 300 candles
Recent Data: 500 candles
Backtesting Recent 500 Days (Days 1 to 470)...
[Backtest results and evaluation metrics]

Fetching BTC-USD data (Timestamp 1640995200 to 1641081600)...
Fetched chunk for BTC-USD: 300 candles
Middle Data: 500 candles
Backtesting Middle 500 Days (Days 1 to 470)...
[Backtest results and evaluation metrics]

... [continues for all periods]

====================================================================================================
🎯 COMPREHENSIVE BACKTEST SUMMARY
====================================================================================================

📊 PERFORMANCE SUMMARY BY PERIOD:
[Performance summary table]

📈 AGGREGATED STATISTICS:
[Aggregated statistics]

🎯 STRATEGY DISTRIBUTION ANALYSIS:
[Strategy distribution]

🎯 CONFIDENCE DISTRIBUTION ANALYSIS:
[Confidence distribution]

💡 PERFORMANCE INSIGHTS:
[Performance insights]
```

## Configuration

### Time Period Customization

Modify time periods by adjusting the duration variables:

```typescript
const recentDays = 500; // Recent period duration
const middleDays = 500; // Middle period duration
const olderDays = 300; // Older period duration
```

### Chunk Size Configuration

Adjust initial chunk size and retry behavior:

```typescript
let maxCandles = 300; // Initial chunk size
if (maxCandles > 100) {
  maxCandles = Math.max(100, Math.floor(maxCandles / 2));
}
```

### Backtesting Parameters

Customize backtesting configuration:

```typescript
const backtester = new TradeModelBacktester(10000); // Initial capital
const lookbackPeriod = 30; // Feature calculation period
```

## Performance Considerations

### Memory Management

- **Chunked Processing**: Prevents memory overflow with large datasets
- **Efficient Merging**: Optimized array operations for data assembly
- **Garbage Collection**: Proper cleanup of temporary variables

### API Optimization

- **Rate Limit Compliance**: Respects Coinbase API constraints
- **Efficient Chunking**: Optimal chunk sizes for API performance
- **Error Recovery**: Minimizes API call failures through retry logic

### Processing Efficiency

- **Sequential Data Fetching**: Efficient retrieval of BTC data
- **Optimized Validation**: Efficient length checking and trimming
- **Streaming Assembly**: Incremental data processing and merging

## Troubleshooting

### Common Issues

#### API Rate Limiting

**Symptoms**: Frequent fetch failures, reduced chunk sizes  
**Solution**: Increase delays between requests or reduce chunk sizes

#### Memory Issues

**Symptoms**: Script crashes on large datasets  
**Solution**: Reduce chunk sizes or implement streaming processing

#### Data Validation

**Symptoms**: Mismatched array lengths, backtesting errors  
**Solution**: Verify data trimming logic and validation steps

### Debug Mode

Enable detailed logging for troubleshooting:

```typescript
// Add debug logging
console.log(
  `Chunk processing: ${chunks.length} chunks, ${mergedPrices.length} total prices`
);
console.log(
  `Data validation: prices=${mergedPrices.length}, volumes=${mergedVolumes.length}`
);
```

### Performance Profiling

Monitor execution performance:

```typescript
const startTime = Date.now();
// ... execution code
const endTime = Date.now();
console.log(`Execution time: ${endTime - startTime}ms`);
```

## Future Enhancements

### Potential Improvements

- **Parallel Processing**: Concurrent execution of multiple time periods
- **Caching Layer**: Cache frequently accessed historical data
- **Configuration Files**: External configuration for time periods and parameters
- **Result Persistence**: Save backtest results to database or files

### Advanced Features

- **Monte Carlo Simulation**: Statistical analysis of strategy robustness
- **Walk-Forward Analysis**: Rolling window backtesting for time-varying performance
- **Multi-Asset Support**: Extended testing across additional cryptocurrencies
- **Risk Management**: Advanced position sizing and stop-loss strategies

### Integration Enhancements

- **Web Interface**: Interactive backtesting dashboard
- **API Endpoints**: RESTful API for backtesting execution
- **Scheduled Execution**: Automated backtesting at regular intervals
- **Result Comparison**: Benchmarking against multiple strategies
