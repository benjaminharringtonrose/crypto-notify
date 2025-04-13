import { TradeModelPredictor } from "./TradeModelPredictor";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { TradingStrategy } from "./TradingStrategy";
import { MODEL_CONFIG, STRATEGY_CONFIG, TIME_CONVERSIONS } from "../constants";
import { HistoricalData, Recommendation } from "../types";

interface Trade {
  entryPrice: number;
  exitPrice: number | null;
  entryIndex: number;
  exitIndex: number | null;
  position: "buy" | "sell";
  profit: number | null;
  holdingDays: number | null;
  confidence: number;
  stopLoss: number;
  takeProfit: number;
  trailingStop: number;
  buyProb?: number;
  atr: number;
  strategy: string;
}

interface BacktestResult {
  totalReturn: number;
  annualizedReturn: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageHoldingDays: number;
  equityCurve: number[];
  trades: Trade[];
}

export class TradeModelBacktester {
  private predictor: TradeModelPredictor;
  private sequenceGenerator: FeatureSequenceGenerator;
  private tradingStrategy: TradingStrategy;
  private initialCapital: number;

  constructor(initialCapital: number = 10000) {
    this.predictor = new TradeModelPredictor();
    this.sequenceGenerator = new FeatureSequenceGenerator(
      MODEL_CONFIG.TIMESTEPS
    );
    this.tradingStrategy = new TradingStrategy();
    this.initialCapital = initialCapital;
  }

  private getStrategyConfig(strategy: string) {
    switch (strategy.toLowerCase()) {
      case "momentum":
        return STRATEGY_CONFIG.MOMENTUM;
      case "mean_reversion":
        return STRATEGY_CONFIG.MEAN_REVERSION;
      case "breakout":
        return STRATEGY_CONFIG.BREAKOUT;
      case "trend_following":
        return STRATEGY_CONFIG.TREND_FOLLOWING;
      default:
        return STRATEGY_CONFIG;
    }
  }

  public async backtest(
    adaData: HistoricalData,
    btcData: HistoricalData,
    startIndex: number,
    endIndex: number
  ): Promise<BacktestResult> {
    const { prices: adaPrices, volumes: adaVolumes } = adaData;
    const { prices: btcPrices, volumes: btcVolumes } = btcData;

    if (
      adaPrices.length !== btcPrices.length ||
      adaPrices.length !== adaVolumes.length ||
      adaPrices.length !== btcVolumes.length
    ) {
      throw new Error("Input data arrays must have equal lengths");
    }

    let capital = this.initialCapital;
    let equity = capital;
    let holdings = 0;
    let lastBuyPrice: number | undefined;
    let peakPrice: number | undefined;
    let buyTimestamp: string | undefined;
    let winStreak = 0;
    const equityCurve: number[] = [equity];
    const trades: Trade[] = [];
    const returns: number[] = [];

    if (
      startIndex < MODEL_CONFIG.TIMESTEPS ||
      endIndex <= startIndex ||
      endIndex >= adaPrices.length
    ) {
      throw new Error("Invalid index range or insufficient data");
    }

    for (let i = startIndex; i <= endIndex; i++) {
      const currentPrice = adaPrices[i];
      const currentTimestamp = new Date(
        Date.now() - (endIndex - i) * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
      ).toISOString();

      const sequence = this.sequenceGenerator.generateSequence(
        adaPrices.slice(0, i + 1),
        adaVolumes.slice(0, i + 1),
        btcPrices.slice(0, i + 1),
        btcVolumes.slice(0, i + 1),
        Math.max(0, i - MODEL_CONFIG.TIMESTEPS + 1),
        i
      );

      if (
        sequence.length !== MODEL_CONFIG.TIMESTEPS ||
        sequence[0].length !== MODEL_CONFIG.FEATURE_COUNT
      ) {
        console.warn(`Skipping index ${i} due to invalid sequence`);
        continue;
      }

      const { trade, confidence, buyProb } =
        await this.tradingStrategy.decideTrade({
          adaPrices: adaPrices.slice(0, i + 1),
          adaVolumes: adaVolumes.slice(0, i + 1),
          btcPrices: btcPrices.slice(0, i + 1),
          btcVolumes: btcVolumes.slice(0, i + 1),
          capital,
          holdings,
          lastBuyPrice,
          peakPrice,
          buyTimestamp,
          currentTimestamp,
          winStreak,
        });

      const prediction = await this.predictor.predict(
        adaPrices.slice(0, i + 1),
        adaVolumes.slice(0, i + 1),
        btcPrices.slice(0, i + 1),
        btcVolumes.slice(0, i + 1)
      );
      const atr = prediction.atr;
      const currentStrategy = this.tradingStrategy.getCurrentStrategy();
      const strategyConfig = this.getStrategyConfig(currentStrategy);
      const atrAdjustedHold = Math.max(
        3,
        Math.min(12, strategyConfig.MIN_HOLD_DAYS_DEFAULT * (1 + atr))
      );

      if (trade) {
        if (
          trade.type === Recommendation.Buy &&
          holdings === 0 &&
          capital > 0
        ) {
          holdings = trade.adaAmount;
          capital -= trade.usdValue;
          lastBuyPrice = trade.price;
          peakPrice = trade.price;
          buyTimestamp = trade.timestamp;
          const stopLoss =
            trade.price *
            (1 - strategyConfig.STOP_LOSS_MULTIPLIER_DEFAULT * atr);
          const takeProfit =
            trade.price *
            (1 + strategyConfig.PROFIT_TAKE_MULTIPLIER_DEFAULT * atr);
          const trailingStop =
            trade.price * (1 - strategyConfig.TRAILING_STOP_DEFAULT);

          trades.push({
            entryPrice: trade.price,
            exitPrice: null,
            entryIndex: i,
            exitIndex: null,
            position: "buy",
            profit: null,
            holdingDays: null,
            confidence,
            stopLoss,
            takeProfit,
            trailingStop,
            buyProb,
            atr,
            strategy: currentStrategy,
          });
          console.log(
            `Trade Opened: Price=${trade.price.toFixed(
              4
            )}, BuyProb=${buyProb.toFixed(4)}, Confidence=${confidence.toFixed(
              4
            )}, Strategy=${currentStrategy}, PositionSize=${(
              trade.usdValue / capital
            ).toFixed(4)}, ATRAdjustedHold=${atrAdjustedHold.toFixed(2)}`
          );
        } else if (trade.type === Recommendation.Sell && holdings > 0) {
          const currentTrade = trades[trades.length - 1];
          currentTrade.exitPrice = trade.price;
          currentTrade.exitIndex = i;
          currentTrade.holdingDays = i - currentTrade.entryIndex;
          currentTrade.profit =
            (trade.price - currentTrade.entryPrice) / currentTrade.entryPrice;

          capital += trade.usdValue;
          equity = capital;
          holdings = 0;
          const positionSize = trade.usdValue / equity;
          console.log(
            `Trade Closed: Entry=${currentTrade.entryPrice.toFixed(
              4
            )}, Exit=${trade.price.toFixed(4)}, Profit=${(
              currentTrade.profit * 100
            ).toFixed(2)}%, Days Held=${
              currentTrade.holdingDays
            }, Reason=Sell, Strategy=${
              currentTrade.strategy
            }, PositionSize=${positionSize.toFixed(
              4
            )}, ATRAdjustedHold=${atrAdjustedHold.toFixed(2)}`
          );

          if (currentTrade.profit > 0) {
            winStreak++;
          } else {
            winStreak = 0;
          }
          lastBuyPrice = undefined;
          peakPrice = undefined;
          buyTimestamp = undefined;
        }
      }

      if (holdings > 0 && peakPrice && currentPrice > peakPrice) {
        peakPrice = currentPrice;
      }
      equity = capital + holdings * currentPrice;
      equityCurve.push(equity);
      if (trades.length > 0 && trades[trades.length - 1].profit !== null) {
        returns.push(trades[trades.length - 1].profit!);
      }
    }

    if (holdings > 0 && endIndex < adaPrices.length) {
      const currentTrade = trades[trades.length - 1];
      currentTrade.exitPrice = adaPrices[endIndex];
      currentTrade.exitIndex = endIndex;
      currentTrade.holdingDays = endIndex - currentTrade.entryIndex;
      currentTrade.profit =
        (currentTrade.exitPrice - currentTrade.entryPrice) /
        currentTrade.entryPrice;

      const tradeValue = holdings * currentTrade.exitPrice;
      capital += tradeValue - STRATEGY_CONFIG.COMMISSION * tradeValue;
      equity = capital;
      holdings = 0;
      console.log(
        `Trade Closed (End of Backtest): Entry=${currentTrade.entryPrice.toFixed(
          4
        )}, Exit=${currentTrade.exitPrice.toFixed(4)}, Profit=${(
          currentTrade.profit * 100
        ).toFixed(2)}%, Days Held=${
          currentTrade.holdingDays
        }, Reason=End of Backtest, Strategy=${
          currentTrade.strategy
        }, PositionSize=${(tradeValue / equity).toFixed(4)}`
      );
      equityCurve.push(equity);
    }

    const totalReturn = (equity - this.initialCapital) / this.initialCapital;
    const days = endIndex - startIndex;
    const annualizedReturn =
      days > 0 ? Math.pow(1 + totalReturn, 365 / days) - 1 : 0;
    const winningTrades = trades.filter((t) => (t.profit || 0) > 0).length;
    const totalTrades = trades.length;
    const winRate = totalTrades > 0 ? winningTrades / totalTrades : 0;
    const averageHoldingDays =
      totalTrades > 0
        ? trades.reduce((sum, t) => sum + (t.holdingDays || 0), 0) / totalTrades
        : 0;

    let maxDrawdown = 0;
    let peak = equityCurve[0];
    for (const eq of equityCurve) {
      if (eq > peak) peak = eq;
      const drawdown = (peak - eq) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    const meanReturn =
      returns.length > 0
        ? returns.reduce((a, b) => a + b, 0) / returns.length
        : 0;
    const stdReturn =
      returns.length > 0
        ? Math.sqrt(
            returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) /
              returns.length
          )
        : 0;
    const sharpeRatio =
      stdReturn > 0 ? (meanReturn * Math.sqrt(365)) / stdReturn : 0;

    return {
      totalReturn,
      annualizedReturn,
      winRate,
      maxDrawdown,
      sharpeRatio,
      totalTrades,
      winningTrades,
      losingTrades: totalTrades - winningTrades,
      averageHoldingDays,
      equityCurve,
      trades,
    };
  }

  public async evaluateBacktest(result: BacktestResult): Promise<void> {
    console.log("Backtest Results:");
    console.log(`Total Return: ${(result.totalReturn * 100).toFixed(2)}%`);
    console.log(
      `Annualized Return: ${(result.annualizedReturn * 100).toFixed(2)}%`
    );
    console.log(`Win Rate: ${(result.winRate * 100).toFixed(2)}%`);
    console.log(`Max Drawdown: ${(result.maxDrawdown * 100).toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
    console.log(`Total Trades: ${result.totalTrades}`);
    console.log(`Winning Trades: ${result.winningTrades}`);
    console.log(`Losing Trades: ${result.losingTrades}`);
    console.log(
      `Average Holding Days: ${result.averageHoldingDays.toFixed(2)}`
    );

    const confidenceBuckets: { [key: string]: number } = {
      "0.4-0.5": 0,
      "0.5-0.6": 0,
      "0.6-0.7": 0,
      "0.7-0.8": 0,
      "0.8+": 0,
    };
    result.trades.forEach((trade) => {
      if (trade.confidence >= 0.8) confidenceBuckets["0.8+"]++;
      else if (trade.confidence >= 0.7) confidenceBuckets["0.7-0.8"]++;
      else if (trade.confidence >= 0.6) confidenceBuckets["0.6-0.7"]++;
      else if (trade.confidence >= 0.5) confidenceBuckets["0.5-0.6"]++;
      else confidenceBuckets["0.4-0.5"]++;
    });
    console.log("\nTrade Confidence Distribution:");
    Object.entries(confidenceBuckets).forEach(([bucket, count]) => {
      console.log(`${bucket}: ${count} trades`);
    });

    const strategyBuckets: { [key: string]: number } = {
      Momentum: 0,
      MeanReversion: 0,
      Breakout: 0,
      TrendFollowing: 0,
    };
    result.trades.forEach((trade) => {
      const strategyKey = trade.strategy.toLowerCase();
      if (strategyKey === "momentum") strategyBuckets.Momentum++;
      else if (strategyKey === "mean_reversion")
        strategyBuckets.MeanReversion++;
      else if (strategyKey === "breakout") strategyBuckets.Breakout++;
      else if (strategyKey === "trend_following")
        strategyBuckets.TrendFollowing++;
    });
    console.log("\nTrade Strategy Distribution:");
    Object.entries(strategyBuckets).forEach(([strategy, count]) => {
      console.log(`${strategy}: ${count} trades`);
    });
  }
}
