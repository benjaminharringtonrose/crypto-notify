import * as tf from "@tensorflow/tfjs-node";
import { TradeModelPredictor } from "./TradeModelPredictor";
import { getHistoricalData } from "../api/getHistoricalData";
import { BacktestResult, Trade, Recommendation } from "../types";
import { FirebaseService } from "../api/FirebaseService";

FirebaseService.getInstance();

export class TradeModelBacktester {
  private predictor: TradeModelPredictor;
  private initialCapital: number;
  private basePositionSize: number;
  private slippage: number;
  private commission: number;
  private buyThreshold: number;
  private sellThreshold: number;
  private stopLossMultiplier: number;
  private trailingStop: number;
  private minHoldDays: number;
  private minConfidence: number;

  constructor(
    initialCapital: number = 10000,
    basePositionSize: number = 0.1, // V6 value
    slippage: number = 0.001,
    commission: number = 0.1,
    buyThreshold: number = 0.75, // V6 value
    sellThreshold: number = 0.65, // V6 value
    stopLossMultiplier: number = 1.5, // V6 value
    trailingStop: number = 0.05, // V6 value
    minHoldDays: number = 2, // V6 value
    minConfidence: number = 0.7 // V6 value
  ) {
    this.predictor = new TradeModelPredictor();
    this.initialCapital = initialCapital;
    this.basePositionSize = basePositionSize;
    this.slippage = slippage;
    this.commission = commission;
    this.buyThreshold = buyThreshold;
    this.sellThreshold = sellThreshold;
    this.stopLossMultiplier = stopLossMultiplier;
    this.trailingStop = trailingStop;
    this.minHoldDays = minHoldDays;
    this.minConfidence = minConfidence;
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

    const adaData = await getHistoricalData(adaSymbol, days);
    const btcData = await getHistoricalData(btcSymbol, days);

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

    console.log(`Starting Capital: $${capital.toFixed(2)}`);

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

      const { buyProb, sellProb, confidence } = await this.predictor.predict(
        adaSlice,
        adaVolSlice,
        btcSlice,
        btcVolSlice
      );
      const atr = this.predictor["calculateATR"](
        adaSlice.slice(-this.predictor["timesteps"]).map((p) => [p])
      );

      if (adaHoldings > 0) {
        peakPrice = Math.max(peakPrice || lastBuyPrice!, currentPrice);
      }

      if (adaHoldings > 0 && lastBuyPrice && peakPrice && buyTimestamp) {
        const priceChange = (currentPrice - lastBuyPrice) / lastBuyPrice;
        const stopLossLevel =
          this.stopLossMultiplier * atr * (confidence / 0.75);
        const trailingStopLevel = peakPrice * (1 - this.trailingStop);

        if (
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel
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
            `Sell Triggered: Price: $${effectivePrice.toFixed(
              4
            )}, P/L: $${profitLoss.toFixed(2)}, Buy Signal: ${buyProb.toFixed(
              2
            )}, Sell Signal: ${sellProb.toFixed(
              2
            )}, Confidence: ${confidence.toFixed(2)}`
          );
          adaHoldings = 0;
          consecutiveBuys = 0;
          lastBuyPrice = undefined;
          peakPrice = undefined;
          buyTimestamp = undefined;
        }
      }

      if (
        buyProb >= this.buyThreshold &&
        confidence >= this.minConfidence &&
        capital > 0 &&
        consecutiveBuys < 2
      ) {
        const positionSize = this.basePositionSize * (buyProb / 0.75);
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
          )} ADA, Buy Signal: ${buyProb.toFixed(
            2
          )}, Confidence: ${confidence.toFixed(2)}`
        );
      } else if (
        sellProb >= this.sellThreshold &&
        confidence >= this.minConfidence &&
        adaHoldings > 0 &&
        buyTimestamp &&
        (new Date(timestamp).getTime() - new Date(buyTimestamp).getTime()) /
          (1000 * 60 * 60 * 24) >=
          this.minHoldDays
      ) {
        const effectivePrice = currentPrice * (1 - this.slippage);
        const usdReceived = adaHoldings * effectivePrice - this.commission;
        capital += usdReceived;
        const profitLoss = usdReceived - adaHoldings * lastBuyPrice!;
        trades.push({
          type: Recommendation.Sell,
          price: effectivePrice,
          timestamp,
          adaAmount: adaHoldings,
          usdValue: usdReceived,
          buyPrice: lastBuyPrice,
        });
        console.log(
          `Sell Triggered: Price: $${effectivePrice.toFixed(
            4
          )}, P/L: $${profitLoss.toFixed(2)}, Buy Signal: ${buyProb.toFixed(
            2
          )}, Sell Signal: ${sellProb.toFixed(
            2
          )}, Confidence: ${confidence.toFixed(2)}`
        );
        adaHoldings = 0;
        consecutiveBuys = 0;
        lastBuyPrice = undefined;
        peakPrice = undefined;
        buyTimestamp = undefined;
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
