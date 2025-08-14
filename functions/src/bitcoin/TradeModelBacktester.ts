import { TradingStrategy } from "./TradingStrategy";
import { HistoricalData, Recommendation, Trade } from "../types";
import { TIME_CONVERSIONS } from "../constants";

interface BacktestTrade extends Trade {
  confidence: number;
  buyProb: number;
  strategy: string;
  atrAdjustedHold: number;
  holdToEndProfit?: number; // Added for hold-to-end logging
}

export class TradeModelBacktester {
  private initialCapital: number;
  private tradingStrategy: TradingStrategy;

  constructor(initialCapital: number) {
    this.initialCapital = initialCapital;
    this.tradingStrategy = new TradingStrategy();
  }

  async backtest(
    btcData: HistoricalData,
    startIndex: number,
    endIndex: number
  ): Promise<BacktestTrade[]> {
    const btcPrices = btcData.prices;
    const btcVolumes = btcData.volumes;

    if (btcPrices.length !== btcVolumes.length) {
      throw new Error("Input data arrays must have equal lengths");
    }

    if (endIndex >= btcPrices.length) {
      throw new Error("endIndex exceeds data length");
    }

    let capital = this.initialCapital;
    let holdings = 0;
    let lastBuyPrice: number | undefined;
    let peakPrice: number | undefined;
    let buyTimestamp: string | undefined;
    let winStreak = 0;
    const trades: BacktestTrade[] = [];

    for (let i = startIndex; i <= endIndex; i++) {
      const currentTimestamp = new Date(
        Date.now() -
          (btcPrices.length - i - 1) * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
      ).toISOString();

      const { trade, confidence, buyProb, sellProb } =
        await this.tradingStrategy.decideTrade({
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

      if (!trade) continue;

      if (trade.type === Recommendation.Buy && capital > 0) {
        holdings += trade.btcAmount;
        capital -= trade.usdValue;
        lastBuyPrice = trade.price;
        peakPrice = trade.price;
        buyTimestamp = trade.timestamp;
        console.log(
          `Trade Opened: Price=${trade.price.toFixed(
            4
          )}, BuyProb=${buyProb.toFixed(4)}, Confidence=${confidence.toFixed(
            4
          )}, Strategy=${this.tradingStrategy.getCurrentStrategy()}, PositionSize=${trade.btcAmount.toFixed(
            4
          )}, ATRAdjustedHold=${(sellProb * 10).toFixed(2)}`
        );
        trades.push({
          ...trade,
          confidence,
          buyProb,
          strategy: this.tradingStrategy.getCurrentStrategy(),
          atrAdjustedHold: sellProb * 10,
        });
      } else if (trade.type === Recommendation.Sell && holdings > 0) {
        const profit = ((trade.price - lastBuyPrice!) / lastBuyPrice!) * 100;
        const holdToEndProfit =
          ((btcPrices[btcPrices.length - 1] - lastBuyPrice!) / lastBuyPrice!) *
          100; // Added
        capital += trade.usdValue;
        holdings -= trade.btcAmount;
        peakPrice = undefined;
        buyTimestamp = undefined;
        winStreak = profit > 0 ? winStreak + 1 : 0;
        console.log(
          `Trade Closed: Entry=${lastBuyPrice!.toFixed(
            4
          )}, Exit=${trade.price.toFixed(4)}, Profit=${profit.toFixed(
            2
          )}%, HoldToEndProfit=${holdToEndProfit.toFixed(2)}%, Days Held=${
            buyTimestamp
              ? Math.round(
                  (new Date(trade.timestamp).getTime() -
                    new Date(buyTimestamp).getTime()) /
                    TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
                )
              : 0
          }, Reason=Sell, Strategy=${this.tradingStrategy.getCurrentStrategy()}, PositionSize=${trade.btcAmount.toFixed(
            4
          )}, ATRAdjustedHold=${(sellProb * 10).toFixed(2)}`
        );
        trades.push({
          ...trade,
          confidence,
          buyProb,
          strategy: this.tradingStrategy.getCurrentStrategy(),
          atrAdjustedHold: sellProb * 10,
          holdToEndProfit, // Added
        });
        lastBuyPrice = undefined;
      }
    }

    if (holdings > 0 && lastBuyPrice && buyTimestamp) {
      const finalPrice = btcPrices[btcPrices.length - 1];
      const profit = ((finalPrice - lastBuyPrice) / lastBuyPrice) * 100;
      const holdToEndProfit = profit; // Same as profit at end
      const usdReceived =
        holdings * finalPrice - this.tradingStrategy["commission"];
      console.log(
        `Trade Closed (End of Backtest): Entry=${lastBuyPrice.toFixed(
          4
        )}, Exit=${finalPrice.toFixed(4)}, Profit=${profit.toFixed(
          2
        )}%, HoldToEndProfit=${holdToEndProfit.toFixed(
          2
        )}%, Days Held=${Math.round(
          (new Date().getTime() - new Date(buyTimestamp).getTime()) /
            TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
        )}, Reason=End of Backtest, Strategy=${this.tradingStrategy.getCurrentStrategy()}, PositionSize=${holdings.toFixed(
          4
        )}`
      );
      trades.push({
        type: Recommendation.Sell,
        price: finalPrice,
        timestamp: new Date().toISOString(),
        btcAmount: holdings,
        usdValue: usdReceived,
        buyPrice: lastBuyPrice,
        confidence: 0,
        buyProb: 0,
        strategy: this.tradingStrategy.getCurrentStrategy(),
        atrAdjustedHold: 0,
        holdToEndProfit,
      });
      capital += usdReceived;
      holdings = 0;
    }

    return trades;
  }

  async evaluateBacktest(trades: BacktestTrade[]): Promise<{
    totalReturn: number;
    annualizedReturn: number;
    winRate: number;
    maxDrawdown: number;
    sharpeRatio: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    averageHoldingDays: number;
    strategyDistribution: { [key: string]: number };
    confidenceDistribution: { [key: string]: number };
  }> {
    let capital = this.initialCapital;
    let peakCapital = capital;
    let maxDrawdown = 0;
    let totalReturn = 0;
    let winningTrades = 0;
    let losingTrades = 0;
    let totalDaysHeld = 0;
    const confidenceDistribution: { [key: string]: number } = {
      "0.4-0.5": 0,
      "0.5-0.6": 0,
      "0.6-0.7": 0,
      "0.7-0.8": 0,
      "0.8+": 0,
    };
    const strategyDistribution: { [key: string]: number } = {
      momentum: 0,
      mean_reversion: 0,
      breakout: 0,
      trend_following: 0,
    };

    const returns: number[] = [];

    for (const trade of trades) {
      if (trade.type === Recommendation.Buy) {
        capital -= trade.usdValue;
      } else {
        capital += trade.usdValue;
        const profit = trade.buyPrice
          ? ((trade.price - trade.buyPrice) / trade.buyPrice) * 100
          : 0;
        returns.push(profit / 100);
        peakCapital = Math.max(peakCapital, capital);
        const drawdown =
          peakCapital > 0 ? (peakCapital - capital) / peakCapital : 0;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
        totalReturn = (capital - this.initialCapital) / this.initialCapital;
        if (profit > 0) {
          winningTrades++;
        } else {
          losingTrades++;
        }
        totalDaysHeld +=
          trade.timestamp &&
          trades.find(
            (t) =>
              t.type === Recommendation.Buy && t.timestamp < trade.timestamp
          )
            ? Math.round(
                (new Date(trade.timestamp).getTime() -
                  new Date(
                    trades.find(
                      (t) =>
                        t.type === Recommendation.Buy &&
                        t.timestamp < trade.timestamp
                    )!.timestamp
                  ).getTime()) /
                  TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
              )
            : 0;

        // Only count completed trades (sell trades) for distribution
        if (trade.confidence >= 0.8) confidenceDistribution["0.8+"]++;
        else if (trade.confidence >= 0.7) confidenceDistribution["0.7-0.8"]++;
        else if (trade.confidence >= 0.6) confidenceDistribution["0.6-0.7"]++;
        else if (trade.confidence >= 0.5) confidenceDistribution["0.5-0.6"]++;
        else confidenceDistribution["0.4-0.5"]++;

        strategyDistribution[trade.strategy]++;
      }
    }

    const totalTrades = winningTrades + losingTrades;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
    const annualizedReturn =
      totalTrades > 0
        ? ((1 + totalReturn) ** (365 / (totalDaysHeld / totalTrades)) - 1) * 100
        : 0;
    const meanReturn =
      returns.length > 0
        ? returns.reduce((sum, r) => sum + r, 0) / returns.length
        : 0;
    const variance =
      returns.length > 0
        ? returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) /
          returns.length
        : 0;
    const stdDev = variance > 0 ? Math.sqrt(variance) : 0;
    const sharpeRatio = stdDev > 0 ? (meanReturn / stdDev) * Math.sqrt(365) : 0;
    const avgHoldingDays = totalTrades > 0 ? totalDaysHeld / totalTrades : 0;

    console.log(`Backtest Results:`);
    console.log(`Total Return: ${totalReturn * 100}%`);
    console.log(`Annualized Return: ${annualizedReturn.toFixed(2)}%`);
    console.log(`Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`Max Drawdown: ${(maxDrawdown * 100).toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${sharpeRatio.toFixed(2)}`);
    console.log(`Total Trades: ${totalTrades}`);
    console.log(`Winning Trades: ${winningTrades}`);
    console.log(`Losing Trades: ${losingTrades}`);
    console.log(`Average Holding Days: ${avgHoldingDays.toFixed(2)}`);
    console.log(`\nTrade Confidence Distribution:`);
    for (const [range, count] of Object.entries(confidenceDistribution)) {
      console.log(`${range}: ${count} trades`);
    }
    console.log(`\nTrade Strategy Distribution:`);
    for (const [strategy, count] of Object.entries(strategyDistribution)) {
      console.log(`${strategy}: ${count} trades`);
    }

    return {
      totalReturn,
      annualizedReturn,
      winRate,
      maxDrawdown,
      sharpeRatio,
      totalTrades,
      winningTrades,
      losingTrades,
      averageHoldingDays: avgHoldingDays,
      strategyDistribution,
      confidenceDistribution,
    };
  }
}
