# TradingStrategy Class Documentation

## Overview

The `TradingStrategy` class is the intelligent decision-making engine of the Bitcoin trading system that dynamically selects and executes trading strategies based on market conditions, technical indicators, and machine learning predictions. It implements a sophisticated multi-strategy framework that adapts to changing market dynamics, combining quantitative analysis with risk management to optimize trading performance across different market regimes.

## Architecture

The TradingStrategy implements an adaptive, multi-strategy decision system:

```
Market Data → ML Predictions → Strategy Selection → Parameter Adaptation → Trade Decision
     ↓              ↓              ↓              ↓              ↓
  Price/Volume   Buy/Sell Probs   Dynamic       Risk/Reward    Execute/Hold
  & Indicators   & Confidence     Strategy      Optimization   with Position
```

### Key Design Principles

- **Adaptive Strategy Selection**: Dynamically switches between strategies based on market conditions
- **Multi-Factor Decision Making**: Combines ML predictions with technical analysis
- **Risk-Aware Position Sizing**: Adjusts position sizes based on volatility and confidence
- **Strategy Persistence**: Maintains strategy consistency unless overridden by strong signals
- **Dynamic Parameter Tuning**: Adapts strategy parameters based on current market regime
- **Comprehensive Risk Management**: Implements stop-loss, trailing stops, and profit-taking

## Class Structure

```typescript
export class TradingStrategy {
  private predictor: TradeModelPredictor;
  private currentStrategy: StrategyType;
  private basePositionSize: number;
  private slippage: number;
  private commission: number;
  private stopLossMultiplier: number;
  private trailingStop: number;
  private minHoldDays: number;
  private minConfidence: number;
  private profitTakeMultiplier: number;
  private buyProbThreshold: number;
  private sellProbThreshold: number;
  private smaPeriod: number;
  private breakoutThreshold: number;
  private strategyStartTimestamp: string | null;
  private strategyTradeCount: number;
  private recentIndicators: { [key: string]: number[] };

  constructor(params?: TradingStrategyParams);

  public async decideTrade(params: TradeDecisionParams): Promise<TradeDecisionResult>;
  public setStrategy(strategyType: StrategyType): void;
  public getCurrentStrategy(): StrategyType;

  private selectStrategy(...): StrategyType;
  private updateStrategyParameters(): void;
  private getBuyConditions(...): boolean;
  private getSellConditions(...): boolean;
  private calculateSMA(prices: number[], period: number): number;
  private calculateEMA(prices: number[], period: number): number;
  private updateRecentIndicators(...): void;
  private getThreeDayAverage(key: string): number;
  private shouldPersistStrategy(currentTimestamp: string, confidence: number): boolean;
}
```

### Core Dependencies

- **TradeModelPredictor**: Machine learning prediction engine
- **StrategyType**: Enum defining available trading strategies
- **PERIODS**: Technical analysis period constants
- **STRATEGY_CONFIG**: Strategy-specific configuration parameters
- **TIME_CONVERSIONS**: Time utility constants

## Strategy Types

### 1. Mean Reversion Strategy

**Market Conditions**: Range-bound, low volatility markets
**Core Logic**: Buy when price deviates significantly below moving average
**Key Parameters**:

- **Min Hold Days**: `STRATEGY_CONFIG.MEAN_REVERSION.MIN_HOLD_DAYS_DEFAULT`
- **Stop Loss**: `STRATEGY_CONFIG.MEAN_REVERSION.STOP_LOSS_MULTIPLIER_DEFAULT`
- **Trailing Stop**: `STRATEGY_CONFIG.MEAN_REVERSION.TRAILING_STOP_DEFAULT`
- **Profit Take**: `STRATEGY_CONFIG.MEAN_REVERSION.PROFIT_TAKE_MULTIPLIER_DEFAULT`
- **Min Confidence**: `STRATEGY_CONFIG.MEAN_REVERSION.MIN_CONFIDENCE_DEFAULT`
- **Buy Probability Threshold**: `STRATEGY_CONFIG.MEAN_REVERSION.BUY_PROB_THRESHOLD_DEFAULT`

**Buy Conditions**:

```typescript
case StrategyType.MeanReversion:
  const sma = this.calculateSMA(prices, this.smaPeriod);
  const deviation = (prices[prices.length - 1] - sma) / sma;
  return (
    buyProb > this.buyProbThreshold &&
    confidence >= this.minConfidence &&
    deviation < STRATEGY_CONFIG.NEGATIVE_DEVIATION_THRESHOLD &&
    shortMomentum > STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_MIN &&
    momentumDivergence > 0
  );
```

**Sell Conditions**:

```typescript
case StrategyType.MeanReversion:
  return (
    sellProb > this.sellProbThreshold ||
    momentum > STRATEGY_CONFIG.MOMENTUM_MAX ||
    priceChange >= STRATEGY_CONFIG.MEAN_REVERSION_THRESHOLD ||
    priceChange <= -stopLossLevel ||
    currentPrice <= trailingStopLevel
  );
```

### 2. Momentum Strategy

**Market Conditions**: Trending markets with strong momentum
**Core Logic**: Follow strong upward trends with momentum confirmation
**Key Parameters**:

- **Min Hold Days**: `STRATEGY_CONFIG.MOMENTUM.MIN_HOLD_DAYS_DEFAULT`
- **Stop Loss**: `STRATEGY_CONFIG.MOMENTUM.STOP_LOSS_MULTIPLIER_DEFAULT`
- **Trailing Stop**: `STRATEGY_CONFIG.MOMENTUM.TRAILING_STOP_DEFAULT`
- **Profit Take**: `STRATEGY_CONFIG.MOMENTUM.PROFIT_TAKE_MULTIPLIER_DEFAULT`
- **Min Confidence**: `STRATEGY_CONFIG.MOMENTUM.MIN_CONFIDENCE_DEFAULT`
- **Buy Probability Threshold**: `STRATEGY_CONFIG.MOMENTUM.BUY_PROB_THRESHOLD_DEFAULT`

**Buy Conditions**:

```typescript
case StrategyType.Momentum:
  const trendReversal = this.currentStrategy === StrategyType.Momentum &&
    prices.length > this.smaPeriod &&
    this.calculateSMA(prices.slice(-this.smaPeriod), this.smaPeriod) <
      prices[prices.length - 1] &&
    trendSlope > 0;

  return (
    buyProb > this.buyProbThreshold &&
    confidence >= this.minConfidence &&
    shortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD &&
    volatilityAdjustedMomentum > STRATEGY_CONFIG.MOMENTUM.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
    trendStrength > STRATEGY_CONFIG.MOMENTUM.TREND_STRENGTH_THRESHOLD &&
    (atrBreakout > dynamicBreakoutThreshold || trendReversal) &&
    currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_BOOST_THRESHOLD
  );
```

**Sell Conditions**:

```typescript
case StrategyType.Momentum:
  return (
    sellProb > this.sellProbThreshold + 0.05 ||
    momentum < STRATEGY_CONFIG.NEGATIVE_MOMENTUM_THRESHOLD ||
    shortMomentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
    trendStrength < STRATEGY_CONFIG.TREND_STRENGTH_REVERSAL_THRESHOLD ||
    priceChange <= -stopLossLevel ||
    currentPrice <= trailingStopLevel ||
    currentPrice >= profitTakeLevel
  );
```

### 3. Breakout Strategy

**Market Conditions**: High volatility, volume-driven breakouts
**Core Logic**: Enter positions on significant price breakouts with volume confirmation
**Key Parameters**:

- **Min Hold Days**: `STRATEGY_CONFIG.BREAKOUT.MIN_HOLD_DAYS_DEFAULT`
- **Stop Loss**: `STRATEGY_CONFIG.BREAKOUT.STOP_LOSS_MULTIPLIER_DEFAULT`
- **Trailing Stop**: `STRATEGY_CONFIG.BREAKOUT.TRAILING_STOP_DEFAULT`
- **Profit Take**: `STRATEGY_CONFIG.BREAKOUT.PROFIT_TAKE_MULTIPLIER_DEFAULT`
- **Min Confidence**: `STRATEGY_CONFIG.BREAKOUT.MIN_CONFIDENCE_DEFAULT`
- **Buy Probability Threshold**: `STRATEGY_CONFIG.BREAKOUT.BUY_PROB_THRESHOLD_DEFAULT`

**Buy Conditions**:

```typescript
case StrategyType.Breakout:
  return (
    buyProb > this.buyProbThreshold &&
    confidence >= this.minConfidence &&
    atrBreakout > dynamicBreakoutThreshold &&
    shortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD &&
    trendSlope > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD &&
    currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_MULTIPLIER
  );
```

**Sell Conditions**:

```typescript
case StrategyType.Breakout:
  return (
    sellProb > this.sellProbThreshold ||
    momentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
    priceChange <= -stopLossLevel ||
    currentPrice <= trailingStopLevel ||
    currentPrice >= profitTakeLevel
  );
```

### 4. Trend Following Strategy

**Market Conditions**: Strong trending markets with clear direction
**Core Logic**: Follow established trends with trend strength confirmation
**Key Parameters**:

- **Min Hold Days**: `STRATEGY_CONFIG.TREND_FOLLOWING.MIN_HOLD_DAYS_DEFAULT`
- **Stop Loss**: `STRATEGY_CONFIG.TREND_FOLLOWING.STOP_LOSS_MULTIPLIER_DEFAULT`
- **Trailing Stop**: `STRATEGY_CONFIG.TREND_FOLLOWING.TRAILING_STOP_DEFAULT`
- **Profit Take**: `STRATEGY_CONFIG.TREND_FOLLOWING.PROFIT_TAKE_MULTIPLIER_DEFAULT`
- **Min Confidence**: `STRATEGY_CONFIG.TREND_FOLLOWING.MIN_CONFIDENCE_DEFAULT`
- **Buy Probability Threshold**: `STRATEGY_CONFIG.TREND_FOLLOWING.BUY_PROB_THRESHOLD_DEFAULT`

**Buy Conditions**:

```typescript
case StrategyType.TrendFollowing:
  const emaLong = this.calculateEMA(prices.slice(-this.smaPeriod), this.smaPeriod);
  const emaShort = this.calculateEMA(prices.slice(-Math.floor(this.smaPeriod / 2)), Math.floor(this.smaPeriod / 2));

  return (
    buyProb > this.buyProbThreshold &&
    confidence >= this.minConfidence &&
    emaShort > emaLong &&
    trendSlope > STRATEGY_CONFIG.TREND_SLOPE_THRESHOLD &&
    trendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD
  );
```

**Sell Conditions**:

```typescript
case StrategyType.TrendFollowing:
  return (
    sellProb > this.sellProbThreshold + 0.05 ||
    trendStrength < STRATEGY_CONFIG.TREND_STRENGTH_REVERSAL_THRESHOLD ||
    momentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
    priceChange <= -stopLossLevel ||
    currentPrice <= trailingStopLevel ||
    currentPrice >= profitTakeLevel
  );
```

## Strategy Selection Logic

### Dynamic Strategy Selection

```typescript
private selectStrategy(
  shortMomentum: number,
  momentum: number,
  trendSlope: number,
  momentumDivergence: number,
  volatilityAdjustedMomentum: number,
  trendStrength: number,
  atrBreakout: number,
  prices: number[],
  volumes: number[],
  currentTimestamp: string,
  confidence: number
): StrategyType {
  // Update recent indicators for averaging
  this.updateRecentIndicators(
    shortMomentum, momentum, trendSlope, momentumDivergence,
    volatilityAdjustedMomentum, trendStrength, atrBreakout
  );

  // Calculate 3-day averages for stability
  const avgShortMomentum = this.getThreeDayAverage("shortMomentum");
  const avgMomentum = this.getThreeDayAverage("momentum");
  const avgTrendSlope = this.getThreeDayAverage("trendSlope");
  const avgMomentumDivergence = this.getThreeDayAverage("momentumDivergence");
  const avgVolatilityAdjustedMomentum = this.getThreeDayAverage("volatilityAdjustedMomentum");
  const avgTrendStrength = this.getThreeDayAverage("trendStrength");
  const avgAtrBreakout = this.getThreeDayAverage("atrBreakout");

  // Check strategy persistence
  if (this.shouldPersistStrategy(currentTimestamp, confidence)) {
    return this.currentStrategy;
  }

  let newStrategy: StrategyType;

  // Strategy selection priority:
  // 1. Trend Following for bullish markets
  if (Math.abs(avgTrendSlope) > STRATEGY_CONFIG.TREND_SLOPE_THRESHOLD &&
      emaShort > emaLong &&
      avgTrendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD) {
    newStrategy = StrategyType.TrendFollowing;
  }
  // 2. Momentum for volatile markets
  else if (avgShortMomentum > STRATEGY_CONFIG.MOMENTUM_THRESHOLD &&
           avgVolatilityAdjustedMomentum > STRATEGY_CONFIG.MOMENTUM.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
           avgTrendStrength > STRATEGY_CONFIG.MOMENTUM.TREND_STRENGTH_THRESHOLD &&
           emaShort > emaLong) {
    newStrategy = StrategyType.Momentum;
  }
  // 3. Breakout for high volume and volatility
  else if (avgAtrBreakout > this.breakoutThreshold &&
           currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_MULTIPLIER &&
           avgShortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD) {
    newStrategy = StrategyType.Breakout;
  }
  // 4. Mean Reversion as default
  else if (Math.abs(deviation) > STRATEGY_CONFIG.DEVIATION_THRESHOLD &&
           Math.abs(avgMomentum) < STRATEGY_CONFIG.MOMENTUM_THRESHOLD &&
           avgMomentumDivergence !== 0) {
    newStrategy = StrategyType.MeanReversion;
  } else {
    newStrategy = StrategyType.MeanReversion; // Default for stability
  }

  // Update strategy if changed
  if (newStrategy !== this.currentStrategy) {
    this.currentStrategy = newStrategy;
    this.strategyStartTimestamp = currentTimestamp;
    this.strategyTradeCount = 0;
    this.updateStrategyParameters();
  }

  return this.currentStrategy;
}
```

### Strategy Persistence Logic

```typescript
private shouldPersistStrategy(
  currentTimestamp: string,
  confidence: number
): boolean {
  if (!this.strategyStartTimestamp) return false;

  const daysSinceStart = (new Date(currentTimestamp).getTime() -
    new Date(this.strategyStartTimestamp).getTime()) /
    TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;

  return (
    (this.strategyTradeCount < STRATEGY_CONFIG.STRATEGY_PERSISTENCE_TRADES ||
     daysSinceStart < STRATEGY_CONFIG.STRATEGY_PERSISTENCE_DAYS) &&
    confidence < STRATEGY_CONFIG.STRATEGY_OVERRIDE_CONFIDENCE
  );
}
```

## Trade Decision Process

### Main Decision Method

```typescript
public async decideTrade({
  adaPrices,
  adaVolumes,
  btcPrices,
  btcVolumes,
  capital,
  holdings,
  lastBuyPrice,
  peakPrice,
  buyTimestamp,
  currentTimestamp,
  winStreak = 0,
}: TradeDecisionParams): Promise<TradeDecisionResult> {
  // 1. Get ML predictions
  const prediction = await this.predictor.predict(
    adaPrices, adaVolumes, btcPrices, btcVolumes
  );

  // 2. Validate confidence and ATR thresholds
  if (confidence < this.minConfidence || atr > STRATEGY_CONFIG.MAX_ATR_THRESHOLD) {
    return { trade: null, confidence, buyProb, sellProb };
  }

  // 3. Select appropriate strategy
  this.currentStrategy = this.selectStrategy(
    shortMomentum, momentum, trendSlope, momentumDivergence,
    volatilityAdjustedMomentum, trendStrength, atrBreakout,
    adaPrices, adaVolumes, currentTimestamp, confidence
  );

  // 4. Calculate dynamic parameters
  const dynamicBreakoutThreshold = this.calculateDynamicBreakoutThreshold(daysSinceLastTrade);
  const dynamicStopLossMultiplier = this.calculateDynamicStopLoss(confidence);
  const dynamicTrailingStop = this.calculateDynamicTrailingStop(momentum);
  const dynamicProfitTake = this.calculateDynamicProfitTake(momentum);
  const atrAdjustedHold = this.calculateATRAdjustedHold(atr);

  // 5. Execute buy or sell logic
  if (holdings > 0 && lastBuyPrice && buyTimestamp) {
    return this.evaluateSellDecision(/* parameters */);
  } else {
    return this.evaluateBuyDecision(/* parameters */);
  }
}
```

### Dynamic Parameter Calculation

#### Position Sizing

```typescript
// Volatility-adjusted position sizing
const volatilityAdjustedSize = Math.min(
  this.basePositionSize / (atr > 0 ? atr : 0.01),
  atr > STRATEGY_CONFIG.ATR_POSITION_THRESHOLD
    ? STRATEGY_CONFIG.POSITION_SIZE_MAX_HIGH_ATR
    : STRATEGY_CONFIG.POSITION_SIZE_MAX
);

// Trend-adjusted position sizing
const trendAdjustedSize =
  trendSlope > STRATEGY_CONFIG.TREND_SLOPE_BOOST_THRESHOLD
    ? volatilityAdjustedSize * STRATEGY_CONFIG.TREND_SLOPE_POSITION_BOOST
    : volatilityAdjustedSize;

// Confidence and win streak boosts
const confidenceBoost =
  confidence > STRATEGY_CONFIG.HIGH_CONFIDENCE_THRESHOLD
    ? STRATEGY_CONFIG.CONFIDENCE_BOOST_MULTIPLIER
    : 1.0;

const winStreakBoost =
  this.currentStrategy === StrategyType.Momentum && winStreak > 1
    ? 1 + winStreak * 0.2
    : 1.0;

// Final position size calculation
const positionSize = Math.min(
  trendAdjustedSize *
    confidenceBoost *
    winStreakBoost *
    Math.min(
      buyProb / this.buyProbThreshold,
      STRATEGY_CONFIG.BUY_PROB_MAX_MULTIPLIER
    ),
  STRATEGY_CONFIG.POSITION_SIZE_MAX
);
```

#### Dynamic Thresholds

```typescript
// Dynamic breakout threshold based on time since last trade
const dynamicBreakoutThreshold =
  daysSinceLastTrade > STRATEGY_CONFIG.DAYS_SINCE_TRADE_THRESHOLD
    ? STRATEGY_CONFIG.DYNAMIC_BREAKOUT_THRESHOLD
    : this.breakoutThreshold;

// Dynamic stop loss based on confidence
const dynamicStopLossMultiplier =
  confidence > STRATEGY_CONFIG.HIGH_CONFIDENCE_THRESHOLD
    ? this.stopLossMultiplier * 1.2
    : this.stopLossMultiplier;

// Dynamic trailing stop based on momentum
const dynamicTrailingStop =
  momentum > STRATEGY_CONFIG.MOMENTUM_MULTIPLIER
    ? this.trailingStop * 0.8
    : this.trailingStop;

// Dynamic profit take based on momentum
const dynamicProfitTake = Math.min(
  this.profitTakeMultiplier *
    (momentum > 0.1 ? STRATEGY_CONFIG.CONFIDENCE_BOOST_MULTIPLIER : 1.0),
  STRATEGY_CONFIG.MAX_PROFIT_TAKE
);
```

## Technical Indicators

### Moving Averages

#### Simple Moving Average (SMA)

```typescript
private calculateSMA(prices: number[], period: number): number {
  const window = prices.slice(-period);
  return window.length >= period
    ? window.reduce((sum, price) => sum + price, 0) / period
    : prices[prices.length - 1];
}
```

#### Exponential Moving Average (EMA)

```typescript
private calculateEMA(prices: number[], period: number): number {
  const k = 2 / (period + 1);
  let ema = prices[0];
  for (let i = 1; i < prices.length && i < period; i++) {
    ema = prices[i] * k + ema * (1 - k);
  }
  return ema;
}
```

### Indicator Tracking

#### Recent Indicators Management

```typescript
private recentIndicators: { [key: string]: number[] } = {
  shortMomentum: [],
  momentum: [],
  trendSlope: [],
  momentumDivergence: [],
  volatilityAdjustedMomentum: [],
  trendStrength: [],
  atrBreakout: [],
};

private updateRecentIndicators(
  shortMomentum: number,
  momentum: number,
  trendSlope: number,
  momentumDivergence: number,
  volatilityAdjustedMomentum: number,
  trendStrength: number,
  atrBreakout: number
) {
  const keys = ["shortMomentum", "momentum", "trendSlope", "momentumDivergence",
                "volatilityAdjustedMomentum", "trendStrength", "atrBreakout"];
  const values = [shortMomentum, momentum, trendSlope, momentumDivergence,
                  volatilityAdjustedMomentum, trendStrength, atrBreakout];

  keys.forEach((key, index) => {
    this.recentIndicators[key].push(values[index]);
    if (this.recentIndicators[key].length > 3) {
      this.recentIndicators[key].shift();
    }
  });
}

private getThreeDayAverage(key: string): number {
  const values = this.recentIndicators[key];
  return values.length >= 3
    ? values.reduce((sum, val) => sum + val, 0) / 3
    : values[values.length - 1] || 0;
}
```

## Configuration Constants

### Strategy Configuration Integration

```typescript
import { STRATEGY_CONFIG } from "../constants";

// Base configuration
BASE_POSITION_SIZE_DEFAULT: number;
SLIPPAGE: number;
COMMISSION: number;
STOP_LOSS_MULTIPLIER_DEFAULT: number;
TRAILING_STOP_DEFAULT: number;
MIN_HOLD_DAYS_DEFAULT: number;
MIN_CONFIDENCE_DEFAULT: number;
PROFIT_TAKE_MULTIPLIER_DEFAULT: number;
BUY_PROB_THRESHOLD_DEFAULT: number;
SELL_PROB_THRESHOLD_DEFAULT: number;
DYNAMIC_BREAKOUT_THRESHOLD: number;

// Strategy-specific configurations
MOMENTUM: {
  MIN_HOLD_DAYS_DEFAULT: number;
  STOP_LOSS_MULTIPLIER_DEFAULT: number;
  TRAILING_STOP_DEFAULT: number;
  PROFIT_TAKE_MULTIPLIER_DEFAULT: number;
  MIN_CONFIDENCE_DEFAULT: number;
  BUY_PROB_THRESHOLD_DEFAULT: number;
  VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD: number;
  TREND_STRENGTH_THRESHOLD: number;
}

MEAN_REVERSION: {
  /* similar structure */
}
BREAKOUT: {
  /* similar structure */
}
TREND_FOLLOWING: {
  /* similar structure */
}

// Thresholds and limits
TREND_SLOPE_THRESHOLD: number;
TREND_STRENGTH_THRESHOLD: number;
MOMENTUM_THRESHOLD: number;
SHORT_MOMENTUM_THRESHOLD: number;
VOLUME_MULTIPLIER: number;
DEVIATION_THRESHOLD: number;
MAX_ATR_THRESHOLD: number;
HIGH_CONFIDENCE_THRESHOLD: number;
STRATEGY_PERSISTENCE_TRADES: number;
STRATEGY_PERSISTENCE_DAYS: number;
STRATEGY_OVERRIDE_CONFIDENCE: number;
```

## Usage Examples

### Basic Strategy Initialization

```typescript
import { TradingStrategy } from "./TradingStrategy";

// Initialize with default parameters
const strategy = new TradingStrategy();

// Initialize with custom parameters
const customStrategy = new TradingStrategy({
  basePositionSize: 0.03,
  slippage: 0.001,
  commission: 0.005,
  stopLossMultiplier: 2.0,
  trailingStop: 0.05,
  minHoldDays: 3,
  minConfidence: 0.7,
  profitTakeMultiplier: 3.0,
  buyProbThreshold: 0.6,
  sellProbThreshold: 0.6,
  smaPeriod: 20,
  breakoutThreshold: 1.5,
});
```

### Trade Decision Execution

```typescript
// Execute trade decision
const decision = await strategy.decideTrade({
  adaPrices: [0.45, 0.46, 0.47, 0.48, 0.49],
  adaVolumes: [1000000, 1200000, 1100000, 1300000, 1400000],
  btcPrices: [45000, 45500, 46000, 46500, 47000],
  btcVolumes: [500, 600, 550, 650, 700],
  capital: 10000,
  holdings: 0,
  lastBuyPrice: undefined,
  peakPrice: undefined,
  buyTimestamp: undefined,
  currentTimestamp: "2024-01-15T10:00:00Z",
  winStreak: 0,
});

if (decision.trade) {
  console.log(
    `Trade executed: ${decision.trade.type} at ${decision.trade.price}`
  );
  console.log(`Confidence: ${decision.confidence.toFixed(4)}`);
  console.log(`Buy Probability: ${decision.buyProb.toFixed(4)}`);
  console.log(`Sell Probability: ${decision.sellProb.toFixed(4)}`);
} else {
  console.log("No trade executed");
}
```

### Strategy Management

```typescript
// Get current strategy
const currentStrategy = strategy.getCurrentStrategy();
console.log(`Current strategy: ${currentStrategy}`);

// Manually set strategy
strategy.setStrategy(StrategyType.Momentum);
console.log(`Strategy manually set to: ${strategy.getCurrentStrategy()}`);

// Monitor strategy changes
const decision = await strategy.decideTrade(/* params */);
console.log(`Strategy selected: ${strategy.getCurrentStrategy()}`);
```

### Complete Trading Workflow

```typescript
async function executeTradingCycle() {
  try {
    // 1. Initialize strategy
    const strategy = new TradingStrategy({
      basePositionSize: 0.04,
      minConfidence: 0.75,
      stopLossMultiplier: 2.5,
    });

    // 2. Get market data
    const marketData = await fetchMarketData();

    // 3. Execute trade decision
    const decision = await strategy.decideTrade({
      adaPrices: marketData.adaPrices,
      adaVolumes: marketData.adaVolumes,
      btcPrices: marketData.btcPrices,
      btcVolumes: marketData.btcVolumes,
      capital: accountBalance.capital,
      holdings: accountBalance.holdings,
      lastBuyPrice: lastTrade?.price,
      peakPrice: calculatePeakPrice(marketData.adaPrices),
      buyTimestamp: lastTrade?.timestamp,
      currentTimestamp: new Date().toISOString(),
      winStreak: calculateWinStreak(tradeHistory),
    });

    // 4. Execute trade if decision made
    if (decision.trade) {
      const tradeResult = await executeTrade(decision.trade);
      console.log(`Trade executed: ${tradeResult.status}`);

      // 5. Update strategy tracking
      console.log(`Strategy: ${strategy.getCurrentStrategy()}`);
      console.log(`Confidence: ${decision.confidence.toFixed(4)}`);
    }

    return decision;
  } catch (error) {
    console.error("Trading cycle failed:", error);
    throw error;
  }
}
```

## Error Handling

### Strategy Selection Errors

```typescript
private selectStrategy(...): StrategyType {
  try {
    // Strategy selection logic
    return newStrategy;
  } catch (error) {
    console.error('Strategy selection failed:', error);
    // Fallback to default strategy
    return StrategyType.MeanReversion;
  }
}
```

### Unknown Strategy Type

```typescript
private getBuyConditions(...): boolean {
  switch (this.currentStrategy) {
    case StrategyType.Momentum:
      return /* momentum conditions */;
    case StrategyType.MeanReversion:
      return /* mean reversion conditions */;
    case StrategyType.Breakout:
      return /* breakout conditions */;
    case StrategyType.TrendFollowing:
      return /* trend following conditions */;
    default:
      throw new Error(`Unknown strategy type: ${this.currentStrategy}`);
  }
}
```

### Parameter Validation

```typescript
constructor(params?: TradingStrategyParams) {
  const {
    basePositionSize = STRATEGY_CONFIG.BASE_POSITION_SIZE_DEFAULT,
    slippage = STRATEGY_CONFIG.SLIPPAGE,
    // ... other parameters
  } = params ?? {};

  // Validate parameter ranges
  if (basePositionSize <= 0 || basePositionSize > 1) {
    throw new Error('basePositionSize must be between 0 and 1');
  }

  if (slippage < 0 || slippage > 0.1) {
    throw new Error('slippage must be between 0 and 0.1');
  }

  // ... additional validation
}
```

## Performance Considerations

### Computational Efficiency

- **Indicator Caching**: Recent indicators stored in memory for quick access
- **Moving Average Optimization**: Efficient window-based calculations
- **Strategy Persistence**: Reduces unnecessary strategy recalculations
- **Parameter Caching**: Strategy parameters cached until strategy changes

### Memory Management

- **Limited History**: Only 3-day indicator history maintained
- **Efficient Data Structures**: Array-based storage for indicators
- **Garbage Collection**: Automatic cleanup of old indicator values

## Security Considerations

### Input Validation

- **Price Data Validation**: Ensures price arrays are valid and non-negative
- **Volume Data Validation**: Validates volume data integrity
- **Timestamp Validation**: Ensures proper timestamp format and ordering
- **Parameter Bounds**: Validates configuration parameter ranges

### Risk Management

- **Position Size Limits**: Hard caps on maximum position sizes
- **Confidence Thresholds**: Minimum confidence requirements for trades
- **ATR Limits**: Maximum ATR thresholds to prevent extreme volatility trades
- **Stop Loss Protection**: Automatic stop loss mechanisms

## Testing

### Unit Testing Strategy

- **Strategy Selection**: Test strategy selection logic with various market conditions
- **Buy/Sell Conditions**: Test decision logic for each strategy type
- **Parameter Updates**: Verify strategy parameter adaptation
- **Indicator Calculations**: Test technical indicator computations

### Integration Testing

- **End-to-End Decisions**: Complete trade decision workflow
- **Strategy Transitions**: Test strategy switching behavior
- **Parameter Persistence**: Verify parameter updates across strategy changes
- **Performance Testing**: Measure decision execution time

### Test Data Requirements

- **Market Scenarios**: Various market conditions (trending, ranging, volatile)
- **Strategy Transitions**: Data that triggers strategy changes
- **Edge Cases**: Extreme market conditions and parameter values
- **Historical Data**: Real market data for realistic testing

## Monitoring and Logging

### Strategy Performance Tracking

```typescript
// Strategy selection logging
console.log(`Strategy selected: ${this.currentStrategy}`);

// Trade execution logging
console.log(
  `Buy Trigger: ShortMomentum=${shortMomentum.toFixed(
    4
  )}, TrendSlope=${trendSlope.toFixed(4)}, Strategy=${this.currentStrategy}`
);

// Position sizing logging
console.log(
  `Position Size: ${positionSize.toFixed(4)}, ATR: ${atr.toFixed(
    4
  )}, MinConfidence: ${this.minConfidence.toFixed(2)}`
);
```

### Performance Metrics

- **Strategy Distribution**: Track frequency of each strategy type
- **Decision Timing**: Measure time to execute trade decisions
- **Parameter Changes**: Monitor strategy parameter adaptations
- **Trade Success Rates**: Track performance by strategy type

## Future Enhancements

### Potential Improvements

- **Machine Learning Integration**: Use ML for strategy selection optimization
- **Multi-Asset Support**: Extend to multiple cryptocurrency pairs
- **Advanced Risk Models**: Implement more sophisticated risk management
- **Real-Time Adaptation**: Dynamic parameter adjustment based on market feedback

### Advanced Features

- **Strategy Ensembles**: Combine multiple strategies for better performance
- **Market Regime Detection**: Automatic detection of market conditions
- **Portfolio Optimization**: Multi-strategy portfolio management
- **Backtesting Framework**: Comprehensive strategy backtesting capabilities

### Integration Enhancements

- **External Data Sources**: Integration with additional market data providers
- **Social Sentiment**: Incorporate social media and news sentiment
- **Economic Indicators**: Integration with macroeconomic data
- **Cross-Market Analysis**: Multi-market correlation analysis
