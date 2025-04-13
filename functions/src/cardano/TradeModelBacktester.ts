import { TradeModelPredictor } from "./TradeModelPredictor";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { MODEL_CONFIG, STRATEGY_CONFIG } from "../constants";
import { HistoricalData } from "../types";

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
  private initialCapital: number;
  private positionSize: number;

  constructor(initialCapital: number = 10000) {
    this.predictor = new TradeModelPredictor();
    this.sequenceGenerator = new FeatureSequenceGenerator(
      MODEL_CONFIG.TIMESTEPS
    );
    this.initialCapital = initialCapital;
    this.positionSize = STRATEGY_CONFIG.BASE_POSITION_SIZE_DEFAULT;
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
    const equityCurve: number[] = [equity];
    const trades: Trade[] = [];
    let openTrade: Trade | null = null;
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

      // Generate sequence for prediction
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

      // Get prediction
      const prediction = await this.predictor.predict(
        adaPrices.slice(0, i + 1),
        adaVolumes.slice(0, i + 1),
        btcPrices.slice(0, i + 1),
        btcVolumes.slice(0, i + 1)
      );

      // Manage open trade
      if (openTrade) {
        const daysHeld =
          openTrade.exitIndex !== null
            ? openTrade.exitIndex - openTrade.entryIndex
            : i - openTrade.entryIndex;
        const currentProfit =
          ((currentPrice - openTrade.entryPrice) / openTrade.entryPrice) *
          (openTrade.position === "buy" ? 1 : -1);

        // Check stop-loss, take-profit, trailing stop, or minimum hold period
        if (
          currentPrice <= openTrade.stopLoss ||
          currentPrice >= openTrade.takeProfit ||
          (currentPrice <= openTrade.trailingStop &&
            openTrade.position === "buy") ||
          daysHeld >= STRATEGY_CONFIG.MIN_HOLD_DAYS_DEFAULT
        ) {
          openTrade.exitPrice = currentPrice;
          openTrade.exitIndex = i;
          openTrade.profit = currentProfit;
          openTrade.holdingDays = daysHeld;

          capital +=
            (currentProfit -
              STRATEGY_CONFIG.COMMISSION -
              STRATEGY_CONFIG.SLIPPAGE) *
            capital *
            this.positionSize;
          equity = capital;
          trades.push(openTrade);
          openTrade = null;
        } else if (openTrade.position === "buy") {
          // Update trailing stop
          const newTrailingStop =
            currentPrice * (1 - STRATEGY_CONFIG.TRAILING_STOP_DEFAULT);
          openTrade.trailingStop = Math.max(
            openTrade.trailingStop,
            newTrailingStop
          );
        }
      }

      // Open new trade if no open position
      if (!openTrade) {
        if (
          prediction.buyProb >= STRATEGY_CONFIG.BUY_PROB_THRESHOLD_DEFAULT &&
          prediction.confidence >= STRATEGY_CONFIG.MIN_CONFIDENCE_DEFAULT &&
          prediction.volatilityAdjustedMomentum >
            STRATEGY_CONFIG.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
          prediction.trendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD
        ) {
          const atr = prediction.atr;
          const stopLoss =
            currentPrice *
            (1 - STRATEGY_CONFIG.STOP_LOSS_MULTIPLIER_DEFAULT * atr);
          const takeProfit =
            currentPrice *
            (1 + STRATEGY_CONFIG.PROFIT_TAKE_MULTIPLIER_DEFAULT * atr);
          const trailingStop =
            currentPrice * (1 - STRATEGY_CONFIG.TRAILING_STOP_DEFAULT);

          openTrade = {
            entryPrice: currentPrice,
            exitPrice: null,
            entryIndex: i,
            exitIndex: null,
            position: "buy",
            profit: null,
            holdingDays: null,
            confidence: prediction.confidence,
            stopLoss,
            takeProfit,
            trailingStop,
          };
        }
      }

      equityCurve.push(equity);
      if (trades.length > 0) {
        const lastProfit = trades[trades.length - 1].profit || 0;
        returns.push(lastProfit);
      }
    }

    // Close any open trade at the end
    if (openTrade && endIndex < adaPrices.length) {
      openTrade.exitPrice = adaPrices[endIndex];
      openTrade.exitIndex = endIndex;
      const profit =
        ((openTrade.exitPrice - openTrade.entryPrice) / openTrade.entryPrice) *
        (openTrade.position === "buy" ? 1 : -1);
      openTrade.profit = profit;
      openTrade.holdingDays = openTrade.exitIndex - openTrade.entryIndex;
      capital +=
        (profit - STRATEGY_CONFIG.COMMISSION - STRATEGY_CONFIG.SLIPPAGE) *
        capital *
        this.positionSize;
      equity = capital;
      trades.push(openTrade);
      equityCurve.push(equity);
    }

    // Calculate metrics
    const totalReturn = (equity - this.initialCapital) / this.initialCapital;
    const days = endIndex - startIndex; // Assuming daily data
    const annualizedReturn =
      days > 0 ? Math.pow(1 + totalReturn, 365 / days) - 1 : 0;
    const winningTrades = trades.filter((t) => (t.profit || 0) > 0).length;
    const totalTrades = trades.length;
    const winRate = totalTrades > 0 ? winningTrades / totalTrades : 0;
    const averageHoldingDays =
      totalTrades > 0
        ? trades.reduce((sum, t) => sum + (t.holdingDays || 0), 0) / totalTrades
        : 0;

    // Calculate max drawdown
    let maxDrawdown = 0;
    let peak = equityCurve[0];
    for (const eq of equityCurve) {
      if (eq > peak) peak = eq;
      const drawdown = (peak - eq) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    // Calculate Sharpe ratio
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

    // Analyze trade distribution
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
  }
}
