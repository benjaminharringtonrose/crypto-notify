import * as tf from "@tensorflow/tfjs-node";
import { TradeModelPredictor } from "./TradeModelPredictor";
import { BacktestResult, Trade, Recommendation } from "../types";
import { FirebaseService } from "../api/FirebaseService";
import { CryptoCompareService } from "../api/CryptoCompareService";

FirebaseService.getInstance();

const cryptoCompare = new CryptoCompareService();

export class TradeModelBacktester {
  private predictor: TradeModelPredictor;
  private initialCapital: number;
  private basePositionSize: number;
  private slippage: number;
  private commission: number;
  private stopLossMultiplier: number;
  private trailingStop: number;
  private minHoldDays: number;
  private minConfidence: number;
  private profitTakeMultiplier: number;
  private logitThreshold: number;
  private buyProbThreshold: number;
  private sellProbThreshold: number;

  constructor(
    initialCapital: number = 10000,
    basePositionSize: number = 0.05,
    slippage: number = 0.001,
    commission: number = 0.1,
    stopLossMultiplier: number = 4.0,
    trailingStop: number = 0.08,
    minHoldDays: number = 4,
    minConfidence: number = 0.55, // Reduced from 0.58
    profitTakeMultiplier: number = 3.5,
    logitThreshold: number = 0.05, // Reduced from 0.1
    buyProbThreshold: number = 0.5, // Reduced from 0.52
    sellProbThreshold: number = 0.33
  ) {
    this.predictor = new TradeModelPredictor();
    this.initialCapital = initialCapital;
    this.basePositionSize = basePositionSize;
    this.slippage = slippage;
    this.commission = commission;
    this.stopLossMultiplier = stopLossMultiplier;
    this.trailingStop = trailingStop;
    this.minHoldDays = minHoldDays;
    this.minConfidence = minConfidence;
    this.profitTakeMultiplier = profitTakeMultiplier;
    this.logitThreshold = logitThreshold;
    this.buyProbThreshold = buyProbThreshold;
    this.sellProbThreshold = sellProbThreshold;
  }

  private calculateATR(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 0.01;
    const trueRanges = [];
    for (let i = 1; i < prices.length; i++) {
      trueRanges.push(Math.abs(prices[i] - prices[i - 1]));
    }
    const recentRanges = trueRanges.slice(-period);
    return recentRanges.reduce((sum, range) => sum + range, 0) / period;
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
    let adaHoldings = 0;
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

    for (let i = this.predictor["timesteps"]; i < adaData.prices.length; i++) {
      const timestamp = new Date(
        startDate.getTime() + i * 24 * 60 * 60 * 1000
      ).toISOString();
      const adaSlice = adaData.prices.slice(0, i + 1);
      const adaVolSlice = adaData.volumes.slice(0, i + 1);
      const btcSlice = btcData.prices.slice(0, i + 1);
      const btcVolSlice = btcData.volumes.slice(0, i + 1);
      const currentPrice = adaSlice[adaSlice.length - 1];

      const portfolioValue = capital + adaHoldings * currentPrice;
      portfolioHistory.push({ timestamp, value: portfolioValue });

      const {
        buyLogit,
        sellLogit,
        buyProb,
        sellProb,
        confidence,
        momentum,
        trendSlope,
      } = await this.predictor.predict(
        adaSlice,
        adaVolSlice,
        btcSlice,
        btcVolSlice
      );
      const atr = this.calculateATR(
        adaSlice.slice(-this.predictor["timesteps"])
      );
      const dynamicStopLossMultiplier =
        atr > 0.02 ? this.stopLossMultiplier * 0.8 : this.stopLossMultiplier;
      const dynamicTrailingStop =
        atr > 0.02 ? this.trailingStop * 1.2 : this.trailingStop;
      const dynamicProfitTake = Math.min(
        this.profitTakeMultiplier * (momentum > 0.1 ? 1.2 : 1.0),
        4.0
      );

      console.log(
        `Logit check: BuyLogit=${buyLogit.toFixed(
          4
        )}, SellLogit=${sellLogit.toFixed(4)}, Diff=${(
          buyLogit - sellLogit
        ).toFixed(4)} vs ${this.logitThreshold}, ` +
          `BuyProb=${buyProb.toFixed(4)} vs ${
            this.buyProbThreshold
          }, SellProb=${sellProb.toFixed(4)} vs ${this.sellProbThreshold}`
      );

      if (adaHoldings > 0) {
        peakPrice = Math.max(peakPrice || lastBuyPrice!, currentPrice);
      }

      if (adaHoldings > 0 && lastBuyPrice && peakPrice && buyTimestamp) {
        const daysHeld =
          (new Date(timestamp).getTime() - new Date(buyTimestamp).getTime()) /
          (1000 * 60 * 60 * 24);
        const priceChange = (currentPrice - lastBuyPrice) / lastBuyPrice;
        const stopLossLevel = dynamicStopLossMultiplier * atr;
        const profitTakeLevel = lastBuyPrice + dynamicProfitTake * atr;
        const trailingStopLevel = peakPrice * (1 - dynamicTrailingStop);

        if (daysHeld >= this.minHoldDays) {
          if (
            (sellLogit > buyLogit + this.logitThreshold &&
              sellProb > this.sellProbThreshold) ||
            (momentum < -0.1 && sellProb > 0.3 && trendSlope < -0.01) // Adjusted momentum and added trend check
          ) {
            const effectivePrice = currentPrice * (1 - this.slippage);
            const usdReceived = adaHoldings * effectivePrice - this.commission;
            capital += usdReceived;
            const profitLoss = usdReceived - adaHoldings * lastBuyPrice;
            trades.push({
              type: Recommendation.Sell,
              price: effectivePrice,
              timestamp,
              adaAmount: adaHoldings,
              usdValue: usdReceived,
              buyPrice: lastBuyPrice,
            });
            console.log(
              `Sell Triggered (${
                momentum < -0.1 ? "Momentum" : "Signal"
              }): Price: $${effectivePrice.toFixed(
                4
              )}, P/L: $${profitLoss.toFixed(2)}, ` +
                `Confidence: ${confidence.toFixed(2)}, ATR: ${atr.toFixed(
                  4
                )}, Trend Slope: ${trendSlope.toFixed(4)}`
            );
            if (profitLoss > 0) winStreak++, (lossStreak = 0);
            else lossStreak++, (winStreak = 0);
            console.log(`Win Streak: ${winStreak}, Loss Streak: ${lossStreak}`);
            adaHoldings = 0;
            consecutiveBuys = 0;
            lastBuyPrice = undefined;
            peakPrice = undefined;
            buyTimestamp = undefined;
          } else if (
            priceChange <= -stopLossLevel ||
            currentPrice <= trailingStopLevel ||
            currentPrice >= profitTakeLevel
          ) {
            const effectivePrice = currentPrice * (1 - this.slippage);
            const usdReceived = adaHoldings * effectivePrice - this.commission;
            capital += usdReceived;
            const profitLoss = usdReceived - adaHoldings * lastBuyPrice;
            trades.push({
              type: Recommendation.Sell,
              price: effectivePrice,
              timestamp,
              adaAmount: adaHoldings,
              usdValue: usdReceived,
              buyPrice: lastBuyPrice,
            });
            console.log(
              `Sell Triggered (${
                priceChange <= -stopLossLevel
                  ? "Stop"
                  : currentPrice >= profitTakeLevel
                  ? "Profit"
                  : "Trailing"
              }): ` +
                `Price: $${effectivePrice.toFixed(
                  4
                )}, P/L: $${profitLoss.toFixed(
                  2
                )}, Confidence: ${confidence.toFixed(2)}, ATR: ${atr.toFixed(
                  4
                )}, ` +
                `Profit Take Multiplier: ${dynamicProfitTake.toFixed(
                  2
                )}, Trend Slope: ${trendSlope.toFixed(4)}`
            );
            if (profitLoss > 0) winStreak++, (lossStreak = 0);
            else lossStreak++, (winStreak = 0);
            console.log(`Win Streak: ${winStreak}, Loss Streak: ${lossStreak}`);
            adaHoldings = 0;
            consecutiveBuys = 0;
            lastBuyPrice = undefined;
            peakPrice = undefined;
            buyTimestamp = undefined;
          }
        }
      } else if (
        buyLogit > sellLogit + this.logitThreshold &&
        buyProb > this.buyProbThreshold &&
        confidence >= this.minConfidence &&
        capital > 0 &&
        consecutiveBuys < 1
      ) {
        const volatilityAdjustedSize = Math.min(
          this.basePositionSize / (atr > 0 ? atr : 0.01),
          0.15
        );
        const trendAdjustedSize =
          trendSlope > 0.02
            ? volatilityAdjustedSize * 1.2
            : volatilityAdjustedSize;
        const confidenceBoost = confidence > 0.65 ? 1.1 : 1.0; // Boost size for high confidence
        const positionSize = Math.min(
          trendAdjustedSize *
            confidenceBoost *
            Math.min(buyProb / this.buyProbThreshold, 2.0),
          0.15
        );
        const tradeAmount = Math.min(capital * positionSize, capital);
        const effectivePrice = currentPrice * (1 + this.slippage);
        const adaBought = (tradeAmount - this.commission) / effectivePrice;
        capital -= tradeAmount;
        adaHoldings += adaBought;
        lastBuyPrice = effectivePrice;
        peakPrice = effectivePrice;
        consecutiveBuys++;
        buyTimestamp = timestamp;

        trades.push({
          type: Recommendation.Buy,
          price: effectivePrice,
          timestamp,
          adaAmount: adaBought,
          usdValue: tradeAmount,
        });
        console.log(
          `Buy Executed: Price: $${effectivePrice.toFixed(
            4
          )}, Amount: ${adaBought.toFixed(
            2
          )} ADA, Confidence: ${confidence.toFixed(2)}, ` +
            `Position Size: ${(positionSize * 100).toFixed(
              1
            )}%, ATR: ${atr.toFixed(4)}, Trend Slope: ${trendSlope.toFixed(4)}`
        );
      } else if (buyProb > 0.5 && confidence < this.minConfidence) {
        console.log(
          `Missed Buy Opportunity: Price: $${currentPrice.toFixed(
            4
          )}, BuyProb: ${buyProb.toFixed(4)}, Confidence: ${confidence.toFixed(
            2
          )} < ${this.minConfidence}`
        );
      }

      if (i > this.predictor["timesteps"]) {
        const prevValue = portfolioHistory[portfolioHistory.length - 2].value;
        returns.push((portfolioValue - prevValue) / prevValue);
      }
    }

    const finalValue =
      capital + adaHoldings * adaData.prices[adaData.prices.length - 1];
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
