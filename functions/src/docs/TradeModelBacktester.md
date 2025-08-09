# TradeModelBacktester Class Documentation

## Overview

The `TradeModelBacktester` class is a sophisticated backtesting engine for the Cardano trading system that simulates historical trading performance using real market data. It provides comprehensive analysis of trading strategies, performance metrics, and risk assessment by replaying historical market conditions with realistic trading constraints and commission structures.

## Architecture

The TradeModelBacktester implements a comprehensive backtesting framework:

```
Historical Data → Strategy Simulation → Trade Execution → Performance Analysis → Results
      ↓                ↓                    ↓                ↓              ↓
ADA/BTC Prices    Trading Decisions    Buy/Sell Orders   Metrics Calc    Reports
Volume Data       Risk Assessment      Position Mgmt      Risk Analysis   Performance
Market Context    Confidence Scoring   Capital Tracking   Drawdown Calc   Strategy Eval
```

### Key Responsibilities

- **Historical Simulation**: Replay trading decisions across historical market data
- **Strategy Evaluation**: Test trading strategies with realistic market conditions
- **Performance Analysis**: Calculate comprehensive performance metrics
- **Risk Assessment**: Measure drawdown, volatility, and risk-adjusted returns
- **Trade Tracking**: Monitor all trades, positions, and capital changes
- **Strategy Distribution**: Analyze strategy performance across different market conditions

## Class Structure

```typescript
export class TradeModelBacktester {
  private initialCapital: number;
  private tradingStrategy: TradingStrategy;

  constructor(initialCapital: number);

  async backtest(
    adaData: HistoricalData,
    btcData: HistoricalData,
    startIndex: number,
    endIndex: number
  ): Promise<BacktestTrade[]>;

  async evaluateBacktest(trades: BacktestTrade[]): Promise<void>;
}
```

### Constructor Parameters

- `initialCapital: number` - Starting capital in USD for the backtest simulation

### Core Interfaces

```typescript
interface BacktestTrade extends Trade {
  confidence: number; // Model confidence in the trade decision
  buyProb: number; // Probability of buy recommendation
  strategy: string; // Trading strategy used for the decision
  atrAdjustedHold: number; // ATR-adjusted hold duration
  holdToEndProfit?: number; // Profit if held until end of backtest
}
```

## Core Methods

### Public Methods

#### `backtest(adaData, btcData, startIndex, endIndex): Promise<BacktestTrade[]>`

Executes a comprehensive backtest simulation over a specified historical period.

**Parameters:**

- `adaData: HistoricalData` - Historical ADA price and volume data
- `btcData: HistoricalData` - Historical BTC price and volume data
- `startIndex: number` - Starting index in the historical data arrays
- `endIndex: number` - Ending index in the historical data arrays

**Returns:**

- `Promise<BacktestTrade[]>` - Array of all executed trades with performance data

**Process:**

1. **Data Validation**: Ensures all input arrays have equal lengths
2. **Strategy Simulation**: Iterates through each day applying trading strategy
3. **Trade Execution**: Simulates buy/sell decisions with realistic constraints
4. **Position Management**: Tracks holdings, capital, and trade performance
5. **Final Settlement**: Closes any remaining positions at end of backtest

**Key Features:**

- **Real-time Strategy Decisions**: Each day applies current market conditions
- **Position Tracking**: Monitors open positions, entry prices, and peak prices
- **Win Streak Analysis**: Tracks consecutive winning trades
- **ATR-Adjusted Hold**: Dynamic hold duration based on market volatility
- **Comprehensive Logging**: Detailed trade execution logs with strategy context

#### `evaluateBacktest(trades: BacktestTrade[]): Promise<void>`

Analyzes backtest results and generates comprehensive performance reports.

**Parameters:**

- `trades: BacktestTrade[]` - Array of completed trades from backtest

**Returns:**

- `Promise<void>` - Outputs detailed performance analysis to console

**Analysis Components:**

1. **Return Metrics**: Total return, annualized return, mean return
2. **Risk Metrics**: Maximum drawdown, standard deviation, Sharpe ratio
3. **Trade Statistics**: Win rate, total trades, average holding period
4. **Strategy Analysis**: Distribution of strategies used
5. **Confidence Analysis**: Distribution of trade confidence levels

## Backtesting Process

### Data Flow

```typescript
// 1. Initialize backtest parameters
const backtester = new TradeModelBacktester(10000); // $10,000 starting capital

// 2. Execute backtest over historical period
const trades = await backtester.backtest(
  adaHistoricalData,
  btcHistoricalData,
  0, // Start from beginning
  1000 // Test over 1000 days
);

// 3. Analyze results
await backtester.evaluateBacktest(trades);
```

### Trade Execution Logic

#### Buy Decision Processing

```typescript
if (trade.type === Recommendation.Buy && capital > 0) {
  // Update position and capital
  holdings += trade.adaAmount;
  capital -= trade.usdValue;
  lastBuyPrice = trade.price;
  peakPrice = trade.price;
  buyTimestamp = trade.timestamp;

  // Log trade details
  console.log(`Trade Opened: Price=${trade.price}, Strategy=${strategy}`);
}
```

#### Sell Decision Processing

```typescript
if (trade.type === Recommendation.Sell && holdings > 0) {
  // Calculate profit/loss
  const profit = ((trade.price - lastBuyPrice!) / lastBuyPrice!) * 100;
  const holdToEndProfit = ((finalPrice - lastBuyPrice!) / lastBuyPrice!) * 100;

  // Update position and capital
  capital += trade.usdValue;
  holdings -= trade.adaAmount;

  // Update win streak
  winStreak = profit > 0 ? winStreak + 1 : 0;
}
```

### Position Management

- **Capital Tracking**: Monitors available USD for new positions
- **Holdings Management**: Tracks ADA position size and entry price
- **Peak Price Monitoring**: Records highest price since entry for drawdown analysis
- **Win Streak Analysis**: Tracks consecutive profitable trades
- **Final Settlement**: Closes any remaining positions at backtest end

## Performance Metrics

### Return Metrics

#### Total Return

```typescript
totalReturn = ((finalCapital - initialCapital) / initialCapital) * 100;
```

#### Annualized Return

```typescript
annualizedReturn = ((1 + totalReturn) ** (365 / avgHoldingDays) - 1) * 100;
```

### Risk Metrics

#### Maximum Drawdown

```typescript
maxDrawdown = ((peakCapital - currentCapital) / peakCapital) * 100;
```

#### Sharpe Ratio

```typescript
sharpeRatio = (meanReturn / stdDev) * sqrt(365);
```

### Trade Statistics

#### Win Rate

```typescript
winRate = (winningTrades / totalTrades) * 100;
```

#### Average Holding Period

```typescript
avgHoldingDays = totalDaysHeld / totalTrades;
```

### Strategy Distribution Analysis

```typescript
const strategyDistribution = {
  Momentum: 0, // Momentum-based trading
  MeanReversion: 0, // Mean reversion strategies
  Breakout: 0, // Breakout detection
  TrendFollowing: 0, // Trend following approaches
};
```

### Confidence Distribution Analysis

```typescript
const confidenceDistribution = {
  "0.4-0.5": 0, // Low confidence trades
  "0.5-0.6": 0, // Below average confidence
  "0.6-0.7": 0, // Average confidence
  "0.7-0.8": 0, // Above average confidence
  "0.8+": 0, // High confidence trades
};
```

## Advanced Features

### ATR-Adjusted Hold Duration

The backtester implements dynamic hold duration based on market volatility:

```typescript
atrAdjustedHold = sellProb * 10;
```

This adjusts holding periods based on:

- **Market Volatility**: Higher ATR = shorter hold periods
- **Sell Probability**: Higher sell probability = shorter hold periods
- **Risk Management**: Dynamic position sizing based on market conditions

### Hold-to-End Profit Analysis

Tracks what profit would have been achieved by holding until the end of the backtest:

```typescript
holdToEndProfit = ((finalPrice - entryPrice) / entryPrice) * 100;
```

This helps evaluate:

- **Strategy Effectiveness**: Compare actual vs. buy-and-hold performance
- **Exit Timing**: Assess quality of sell decisions
- **Opportunity Cost**: Measure missed profits from early exits

### Win Streak Tracking

Monitors consecutive profitable trades to identify:

```typescript
winStreak = profit > 0 ? winStreak + 1 : 0;
```

- **Strategy Consistency**: Long win streaks indicate robust strategies
- **Market Regime Changes**: Streak breaks may signal market shifts
- **Risk Management**: Streak length affects position sizing decisions

## Usage Examples

### Basic Backtest Execution

```typescript
import { TradeModelBacktester } from "./TradeModelBacktester";
import { HistoricalData } from "../types";

// Initialize backtester with $10,000 starting capital
const backtester = new TradeModelBacktester(10000);

// Execute backtest over 2-year period
const trades = await backtester.backtest(
  adaData,
  btcData,
  0, // Start index
  730 // 2 years of daily data
);

// Analyze results
await backtester.evaluateBacktest(trades);
```

### Custom Backtest Periods

```typescript
// Test specific market periods
const bullMarketTrades = await backtester.backtest(
  adaData,
  btcData,
  500, // Start from day 500
  800 // End at day 800
);

const bearMarketTrades = await backtester.backtest(
  adaData,
  btcData,
  200, // Start from day 200
  500 // End at day 500
);

// Compare performance across different market conditions
await backtester.evaluateBacktest(bullMarketTrades);
await backtester.evaluateBacktest(bearMarketTrades);
```

### Performance Analysis

```typescript
// Analyze individual trade performance
for (const trade of trades) {
  if (trade.type === Recommendation.Sell) {
    console.log(`Trade: ${trade.strategy}`);
    console.log(`Profit: ${trade.holdToEndProfit?.toFixed(2)}%`);
    console.log(`Confidence: ${trade.confidence.toFixed(2)}`);
    console.log(`ATR Hold: ${trade.atrAdjustedHold.toFixed(2)}`);
  }
}

// Calculate custom metrics
const totalProfit = trades
  .filter((t) => t.type === Recommendation.Sell)
  .reduce((sum, t) => sum + (t.holdToEndProfit || 0), 0);

console.log(`Total Profit: ${totalProfit.toFixed(2)}%`);
```

## Error Handling

### Data Validation

```typescript
// Ensure data consistency
if (
  adaPrices.length !== btcPrices.length ||
  adaPrices.length !== adaVolumes.length ||
  adaPrices.length !== btcVolumes.length
) {
  throw new Error("Input data arrays must have equal lengths");
}

// Validate index bounds
if (endIndex >= adaPrices.length) {
  throw new Error("endIndex exceeds data length");
}
```

### Trade Execution Safety

- **Capital Validation**: Ensures sufficient capital before buy orders
- **Position Validation**: Verifies holdings before sell orders
- **Price Validation**: Handles missing or invalid price data
- **Timestamp Validation**: Manages date conversion and time calculations

## Performance Considerations

### Memory Management

- **Efficient Iteration**: Single pass through historical data
- **Array Slicing**: Minimal memory allocation for data subsets
- **Object Reuse**: Efficient trade object construction and updates

### Computational Efficiency

- **Batch Processing**: Processes multiple days in single iteration
- **Optimized Calculations**: Efficient profit/loss and metric calculations
- **Reduced Logging**: Conditional logging to minimize I/O overhead

## Dependencies

### External Dependencies

- **TradingStrategy**: Strategy decision engine for trade signals
- **HistoricalData**: Market data structure for prices and volumes
- **TIME_CONVERSIONS**: Time conversion constants for date calculations

### Internal Dependencies

- **Recommendation**: Trading decision types (Buy, Sell, Hold)
- **Trade**: Base trade interface with execution details
- **BacktestTrade**: Extended trade interface with backtest-specific data

## Configuration

### Backtest Parameters

```typescript
// Capital allocation
const initialCapital = 10000; // Starting USD amount

// Time period selection
const startIndex = 0; // Beginning of dataset
const endIndex = 1000; // End of dataset (1000 days)

// Data granularity
const dataFrequency = "daily"; // Daily price data
```

### Performance Thresholds

- **Minimum Capital**: $100 for position entry
- **Confidence Thresholds**: 0.4-0.8+ confidence ranges
- **Win Rate Targets**: 50%+ for profitable strategies
- **Drawdown Limits**: 20% maximum acceptable drawdown

## Testing

### Unit Testing Strategy

- **Mock Data**: Test with synthetic historical data
- **Edge Cases**: Test boundary conditions and data validation
- **Strategy Simulation**: Mock trading strategy responses
- **Performance Validation**: Verify metric calculations

### Integration Testing

- **Real Data**: Test with actual historical market data
- **Strategy Integration**: Test complete strategy decision flow
- **Performance Benchmarking**: Compare against known results
- **Stress Testing**: Test with large datasets and extreme conditions

### Test Data Requirements

- **Balanced Periods**: Include bull, bear, and sideways markets
- **Sufficient Length**: Minimum 1 year of daily data
- **Data Quality**: Clean, consistent price and volume data
- **Market Events**: Include significant market movements and volatility

## Monitoring and Logging

### Trade Execution Logs

```typescript
// Buy trade logging
console.log(
  `Trade Opened: Price=${price}, Strategy=${strategy}, PositionSize=${amount}`
);

// Sell trade logging
console.log(
  `Trade Closed: Entry=${entry}, Exit=${exit}, Profit=${profit}%, Days Held=${days}`
);
```

### Performance Monitoring

- **Real-time Metrics**: Track performance during backtest execution
- **Strategy Performance**: Monitor individual strategy effectiveness
- **Risk Metrics**: Track drawdown and volatility in real-time
- **Trade Quality**: Monitor confidence levels and success rates

## Future Enhancements

### Potential Improvements

- **Multi-Asset Support**: Extend beyond ADA/BTC to other cryptocurrencies
- **Advanced Risk Models**: Implement VaR, CVaR, and other risk metrics
- **Portfolio Optimization**: Multi-strategy portfolio backtesting
- **Real-time Backtesting**: Live backtesting with streaming data

### Advanced Features

- **Monte Carlo Simulation**: Statistical analysis of strategy performance
- **Regime Detection**: Automatic market condition identification
- **Dynamic Parameter Tuning**: Adaptive strategy parameters
- **Machine Learning Integration**: ML-based strategy optimization

### Integration Enhancements

- **Database Storage**: Persistent backtest results and analysis
- **API Endpoints**: RESTful backtesting services
- **Visualization**: Interactive charts and performance dashboards
- **Report Generation**: Automated PDF/HTML performance reports
