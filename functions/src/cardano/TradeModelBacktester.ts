import * as tf from "@tensorflow/tfjs-node";
import { BacktestResult, Trade, Recommendation } from "../types";
import { FirebaseService } from "../api/FirebaseService";
import { CryptoCompareService } from "../api/CryptoCompareService";
import { TradingBot } from "./TradingBot";
import { MODEL_CONSTANTS } from "../constants";

FirebaseService.getInstance();
const cryptoCompare = new CryptoCompareService();

export class TradeModelBacktester {
  private bot: TradingBot;
  private initialCapital: number;

  constructor(
    initialCapital: number = 10000,
    basePositionSize: number = MODEL_CONSTANTS.BASE_POSITION_SIZE_DEFAULT,
    slippage: number = MODEL_CONSTANTS.SLIPPAGE,
    commission: number = MODEL_CONSTANTS.COMMISSION,
    stopLossMultiplier: number = MODEL_CONSTANTS.STOP_LOSS_MULTIPLIER_DEFAULT,
    trailingStop: number = MODEL_CONSTANTS.TRAILING_STOP_DEFAULT,
    minHoldDays: number = MODEL_CONSTANTS.MIN_HOLD_DAYS_DEFAULT,
    minConfidence: number = MODEL_CONSTANTS.MIN_CONFIDENCE_DEFAULT,
    profitTakeMultiplier: number = MODEL_CONSTANTS.PROFIT_TAKE_MULTIPLIER_DEFAULT,
    logitThreshold: number = MODEL_CONSTANTS.LOGIT_THRESHOLD_DEFAULT,
    buyProbThreshold: number = MODEL_CONSTANTS.BUY_PROB_THRESHOLD_DEFAULT,
    sellProbThreshold: number = MODEL_CONSTANTS.SELL_PROB_THRESHOLD_DEFAULT
  ) {
    this.bot = new TradingBot({
      basePositionSize,
      slippage,
      commission,
      stopLossMultiplier,
      trailingStop,
      minHoldDays,
      minConfidence,
      profitTakeMultiplier,
      logitThreshold,
      buyProbThreshold,
      sellProbThreshold,
    });
    this.initialCapital = initialCapital;
  }

  public async backtest(
    startDate: Date,
    endDate: Date,
    adaSymbol: string = "ADA",
    btcSymbol: string = "BTC"
  ): Promise<BacktestResult> {
    const days = Math.ceil(
      (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)
    );
    const adaData = await cryptoCompare.getHistoricalData(adaSymbol, days);
    const btcData = await cryptoCompare.getHistoricalData(btcSymbol, days);

    if (adaData.prices.length !== btcData.prices.length) {
      throw new Error("Mismatch in ADA and BTC historical data lengths");
    }

    let capital = this.initialCapital;
    let holdings = 0;
    const trades: Trade[] = [];
    const portfolioHistory: { timestamp: string; value: number }[] = [];
    const returns: number[] = [];
    let consecutiveBuys = 0;
    let lastBuyPrice: number | undefined;
    let peakPrice: number | undefined;
    let buyTimestamp: string | undefined;
    let maxWinStreak = 0;
    let maxLossStreak = 0;
    let winStreak = 0;
    let lossStreak = 0;

    console.log(`Starting Capital: $${capital.toFixed(2)}`);
    console.log(
      `Backtest period: ${startDate.toISOString()} to ${endDate.toISOString()}`
    );

    for (let i = MODEL_CONSTANTS.TIMESTEPS; i < adaData.prices.length; i++) {
      const currentTimestamp = new Date(
        startDate.getTime() + i * 24 * 60 * 60 * 1000
      ).toISOString();

      const adaPrices = adaData.prices.slice(0, i + 1);
      const adaVolumes = adaData.volumes.slice(0, i + 1);
      const btcPrices = btcData.prices.slice(0, i + 1);
      const btcVolumes = btcData.volumes.slice(0, i + 1);
      const currentPrice = adaPrices[adaPrices.length - 1];

      const portfolioValue = capital + holdings * currentPrice;
      portfolioHistory.push({
        timestamp: currentTimestamp,
        value: portfolioValue,
      });

      const { trade, confidence } = await this.bot.decideTrade({
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
      });

      if (trade) {
        if (trade.type === Recommendation.Buy && consecutiveBuys < 4) {
          const minConfidence =
            consecutiveBuys === 0
              ? 0.55 // Relaxed from 0.6
              : consecutiveBuys === 1
              ? 0.6 // Relaxed from 0.65
              : consecutiveBuys === 2
              ? 0.65 // Relaxed from 0.7
              : 0.7; // Relaxed from 0.75
          if (confidence >= minConfidence) {
            capital -= trade.usdValue;
            holdings += trade.adaAmount;
            lastBuyPrice = trade.price;
            peakPrice = trade.price;
            consecutiveBuys++;
            buyTimestamp = currentTimestamp;
            trades.push(trade);
            console.log(
              `Buy Executed: Price: $${trade.price.toFixed(
                4
              )}, Amount: ${trade.adaAmount.toFixed(
                2
              )} ADA, Confidence: ${confidence.toFixed(
                2
              )}, Consecutive: ${consecutiveBuys}`
            );
          }
        } else if (trade.type === Recommendation.Sell) {
          const profitLoss = trade.usdValue - holdings * (lastBuyPrice || 0);
          capital += trade.usdValue;
          trades.push(trade);
          console.log(
            `Sell Executed: Price: $${trade.price.toFixed(
              4
            )}, P/L: $${profitLoss.toFixed(
              2
            )}, Confidence: ${confidence.toFixed(2)}`
          );
          console.log(`Trade Success: ${profitLoss > 0 ? "Win" : "Loss"}`);
          if (profitLoss > 0) {
            winStreak++;
            lossStreak = 0;
            maxWinStreak = Math.max(maxWinStreak, winStreak);
          } else {
            lossStreak++;
            winStreak = 0;
            maxLossStreak = Math.max(maxLossStreak, lossStreak);
          }
          console.log(`Win Streak: ${winStreak}, Loss Streak: ${lossStreak}`);
          holdings = 0;
          consecutiveBuys = 0;
          lastBuyPrice = undefined;
          peakPrice = undefined;
          buyTimestamp = undefined;
        }
      }

      if (holdings > 0) {
        peakPrice = Math.max(peakPrice || lastBuyPrice!, currentPrice);
      }

      if (i > MODEL_CONSTANTS.TIMESTEPS) {
        const prevValue = portfolioHistory[portfolioHistory.length - 2].value;
        returns.push((portfolioValue - prevValue) / prevValue);
        if (i % 30 === 0) {
          console.log(
            `Portfolio Trend at ${currentTimestamp}: Value: $${portfolioValue.toFixed(
              2
            )}, Change: ${(
              ((portfolioValue - prevValue) / prevValue) *
              100
            ).toFixed(2)}%, Rolling Sharpe (30d): ${this.calculateRollingSharpe(
              returns.slice(-30),
              30
            ).toFixed(2)}`
          );
        }
      }
    }

    const finalValue =
      capital + holdings * adaData.prices[adaData.prices.length - 1];
    console.log(`Ending Capital: $${finalValue.toFixed(2)}`);
    const totalReturn =
      (finalValue - this.initialCapital) / this.initialCapital;
    const totalTrades = trades.length;
    const wins = trades.filter(
      (t) => t.type === Recommendation.Sell && t.price > (t.buyPrice || 0)
    ).length;
    const winRate = totalTrades > 0 ? wins / (totalTrades / 2) : 0;
    const avgWin =
      trades
        .filter(
          (t) => t.type === Recommendation.Sell && t.price > (t.buyPrice || 0)
        )
        .reduce(
          (sum, t) => sum + (t.usdValue - (t.buyPrice || 0) * t.adaAmount),
          0
        ) / (wins || 1);
    const avgLoss =
      trades
        .filter(
          (t) => t.type === Recommendation.Sell && t.price <= (t.buyPrice || 0)
        )
        .reduce(
          (sum, t) => sum + (t.usdValue - (t.buyPrice || 0) * t.adaAmount),
          0
        ) / (totalTrades / 2 - wins || 1);

    const sharpeRatio = this.calculateSharpeRatio(returns);
    const sortinoRatio = this.calculateSortinoRatio(returns);
    const maxDrawdown = this.calculateMaxDrawdown(portfolioHistory);
    const rollingSharpe = this.calculateRollingSharpe(returns, 30);

    console.log(
      `Average Trade Duration: ${this.calculateAvgTradeDuration(trades).toFixed(
        2
      )} days`
    );
    console.log(`Total Return: ${(totalReturn * 100).toFixed(2)}%`);
    console.log(`Win Rate: ${(winRate * 100).toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${sharpeRatio.toFixed(2)}`);
    console.log(`Rolling Sharpe (30-day): ${rollingSharpe.toFixed(2)}`);
    console.log(`Sortino Ratio: ${sortinoRatio.toFixed(2)}`);
    console.log(`Max Drawdown: ${(maxDrawdown * 100).toFixed(2)}%`);
    console.log(
      `Max Win Streak: ${maxWinStreak}, Max Loss Streak: ${maxLossStreak}`
    );
    console.log(
      `Avg Win: $${avgWin.toFixed(2)}, Avg Loss: $${avgLoss.toFixed(2)}`
    );

    return {
      totalReturn,
      totalTrades,
      winRate,
      sharpeRatio,
      maxDrawdown,
      portfolioHistory,
      trades,
    };
  }

  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    const returnsTensor = tf.tensor1d(returns);
    const mean = tf.mean(returnsTensor).dataSync()[0];
    const std = tf
      .sqrt(tf.mean(tf.square(returnsTensor.sub(mean))))
      .dataSync()[0];
    returnsTensor.dispose();
    return std === 0 ? 0 : (mean / std) * Math.sqrt(252);
  }

  private calculateRollingSharpe(returns: number[], window: number): number {
    if (returns.length < window) return 0;
    const recentReturns = returns.slice(-window);
    const returnsTensor = tf.tensor1d(recentReturns);
    const mean = tf.mean(returnsTensor).dataSync()[0];
    const std = tf
      .sqrt(tf.mean(tf.square(returnsTensor.sub(mean))))
      .dataSync()[0];
    returnsTensor.dispose();
    return std === 0 ? 0 : (mean / std) * Math.sqrt(252);
  }

  private calculateSortinoRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    const returnsTensor = tf.tensor1d(returns);
    const mean = tf.mean(returnsTensor).dataSync()[0];
    const downsideReturns = returns.map((r) => (r < 0 ? r : 0));
    const downsideStd = tf
      .sqrt(tf.mean(tf.square(tf.tensor1d(downsideReturns))))
      .dataSync()[0];
    returnsTensor.dispose();
    return downsideStd === 0 ? 0 : (mean / downsideStd) * Math.sqrt(252);
  }

  private calculateMaxDrawdown(
    history: { timestamp: string; value: number }[]
  ): number {
    let maxPeak = -Infinity;
    let maxDrawdown = 0;
    for (const { value } of history) {
      maxPeak = Math.max(maxPeak, value);
      const drawdown = (maxPeak - value) / maxPeak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    return maxDrawdown;
  }

  private calculateAvgTradeDuration(trades: Trade[]): number {
    return trades.length > 1
      ? trades
          .filter((t) => t.type === Recommendation.Sell)
          .reduce((sum, sellTrade, idx) => {
            const buyTrade = trades[idx * 2];
            return (
              sum +
              (new Date(sellTrade.timestamp).getTime() -
                new Date(buyTrade.timestamp).getTime()) /
                (1000 * 60 * 60 * 24)
            );
          }, 0) /
          (trades.length / 2)
      : 0;
  }
}
