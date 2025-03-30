import * as tf from "@tensorflow/tfjs-node";
import { BacktestResult, Trade, Recommendation } from "../types";
import { FirebaseService } from "../api/FirebaseService";
import { CryptoCompareService } from "../api/CryptoCompareService";
import { TradingBot } from "./TradingBot";

FirebaseService.getInstance();

const cryptoCompare = new CryptoCompareService();

export class TradeModelBacktester {
  private bot: TradingBot;
  private initialCapital: number;

  constructor(
    initialCapital: number = 10000,
    basePositionSize: number = 0.05,
    slippage: number = 0.001,
    commission: number = 0.1,
    stopLossMultiplier: number = 4.0,
    trailingStop: number = 0.08,
    minHoldDays: number = 4,
    minConfidence: number = 0.55,
    profitTakeMultiplier: number = 3.5,
    logitThreshold: number = 0.05,
    buyProbThreshold: number = 0.5,
    sellProbThreshold: number = 0.33
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
    let winStreak = 0;
    let lossStreak = 0;

    console.log(`Starting Capital: $${capital.toFixed(2)}`);
    console.log(
      `Backtest period: ${startDate.toISOString()} to ${endDate.toISOString()}`
    );

    for (
      let i = this.bot["predictor"]["timesteps"];
      i < adaData.prices.length;
      i++
    ) {
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
        if (trade.type === Recommendation.Buy && consecutiveBuys < 1) {
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
            )} ADA, Confidence: ${confidence.toFixed(2)}`
          );
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
          if (profitLoss > 0) winStreak++, (lossStreak = 0);
          else lossStreak++, (winStreak = 0);
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

      if (i > this.bot["predictor"]["timesteps"]) {
        const prevValue = portfolioHistory[portfolioHistory.length - 2].value;
        returns.push((portfolioValue - prevValue) / prevValue);
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

    const sharpeRatio = this.calculateSharpeRatio(returns);
    const maxDrawdown = this.calculateMaxDrawdown(portfolioHistory);

    const avgTradeDuration =
      trades.length > 1
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
          (totalTrades / 2)
        : 0;
    console.log(`Average Trade Duration: ${avgTradeDuration.toFixed(2)} days`);
    console.log(`Total Return: ${(totalReturn * 100).toFixed(2)}%`);
    console.log(`Win Rate: ${(winRate * 100).toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${sharpeRatio.toFixed(2)}`);
    console.log(`Max Drawdown: ${(maxDrawdown * 100).toFixed(2)}%`);

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
    const meanTensor = tf.scalar(mean);
    const squaredDiffs = tf.square(returnsTensor.sub(meanTensor));
    const variance = tf.mean(squaredDiffs).dataSync()[0];
    const std = Math.sqrt(variance);
    returnsTensor.dispose();
    meanTensor.dispose();
    squaredDiffs.dispose();
    return std === 0 ? 0 : (mean / std) * Math.sqrt(252);
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
}
