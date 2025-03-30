import { TradeModelPredictor } from "./TradeModelPredictor";
import { Trade, Recommendation } from "../types";

interface TradingBotParams {
  basePositionSize: number;
  slippage: number;
  commission: number;
  stopLossMultiplier: number;
  trailingStop: number;
  minHoldDays: number;
  minConfidence: number;
  profitTakeMultiplier: number;
  logitThreshold: number;
  buyProbThreshold: number;
  sellProbThreshold: number;
}

export class TradingBot {
  private predictor: TradeModelPredictor;
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

  constructor({
    basePositionSize = 0.05,
    slippage = 0.001,
    commission = 0.1,
    stopLossMultiplier = 4.0,
    trailingStop = 0.08,
    minHoldDays = 4,
    minConfidence = 0.55,
    profitTakeMultiplier = 3.5,
    logitThreshold = 0.05,
    buyProbThreshold = 0.5,
    sellProbThreshold = 0.33,
  }: TradingBotParams) {
    this.predictor = new TradeModelPredictor();
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
  }: {
    adaPrices: number[];
    adaVolumes: number[];
    btcPrices: number[];
    btcVolumes: number[];
    capital: number;
    holdings: number;
    lastBuyPrice: number | undefined;
    peakPrice: number | undefined;
    buyTimestamp: string | undefined;
    currentTimestamp: string;
  }): Promise<{ trade: Trade | null; confidence: number }> {
    const currentPrice = adaPrices[adaPrices.length - 1];
    const prediction = await this.predictor.predict(
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes
    );
    const {
      buyLogit,
      sellLogit,
      buyProb,
      sellProb,
      confidence,
      momentum,
      trendSlope,
    } = prediction;
    const atr = this.calculateATR(
      adaPrices.slice(-this.predictor["timesteps"])
    );
    const dynamicStopLossMultiplier =
      atr > 0.02 ? this.stopLossMultiplier * 0.8 : this.stopLossMultiplier;
    const dynamicTrailingStop =
      atr > 0.02 ? this.trailingStop * 1.2 : this.trailingStop;
    const dynamicProfitTake = Math.min(
      this.profitTakeMultiplier * (momentum > 0.1 ? 1.2 : 1.0),
      4.0
    );

    if (holdings > 0 && lastBuyPrice && buyTimestamp) {
      const daysHeld =
        (new Date(currentTimestamp).getTime() -
          new Date(buyTimestamp).getTime()) /
        (1000 * 60 * 60 * 24);
      const priceChange = (currentPrice - lastBuyPrice) / lastBuyPrice;
      const stopLossLevel = dynamicStopLossMultiplier * atr;
      const profitTakeLevel = lastBuyPrice + dynamicProfitTake * atr;
      const trailingStopLevel =
        (peakPrice || lastBuyPrice) * (1 - dynamicTrailingStop);

      if (daysHeld >= this.minHoldDays) {
        if (
          (sellLogit > buyLogit + this.logitThreshold &&
            sellProb > this.sellProbThreshold) ||
          (momentum < -0.1 && sellProb > 0.3 && trendSlope < -0.01)
        ) {
          const effectivePrice = currentPrice * (1 - this.slippage);
          const usdReceived = holdings * effectivePrice - this.commission;
          return {
            trade: {
              type: Recommendation.Sell,
              price: effectivePrice,
              timestamp: currentTimestamp,
              adaAmount: holdings,
              usdValue: usdReceived,
              buyPrice: lastBuyPrice,
            },
            confidence,
          };
        } else if (
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        ) {
          const effectivePrice = currentPrice * (1 - this.slippage);
          const usdReceived = holdings * effectivePrice - this.commission;
          return {
            trade: {
              type: Recommendation.Sell,
              price: effectivePrice,
              timestamp: currentTimestamp,
              adaAmount: holdings,
              usdValue: usdReceived,
              buyPrice: lastBuyPrice,
            },
            confidence,
          };
        }
      }
    } else if (
      buyLogit > sellLogit + this.logitThreshold &&
      buyProb > this.buyProbThreshold &&
      confidence >= this.minConfidence &&
      capital > 0
    ) {
      const volatilityAdjustedSize = Math.min(
        this.basePositionSize / (atr > 0 ? atr : 0.01),
        0.15
      );
      const trendAdjustedSize =
        trendSlope > 0.02
          ? volatilityAdjustedSize * 1.2
          : volatilityAdjustedSize;
      const confidenceBoost = confidence > 0.65 ? 1.1 : 1.0;
      const positionSize = Math.min(
        trendAdjustedSize *
          confidenceBoost *
          Math.min(buyProb / this.buyProbThreshold, 2.0),
        0.15
      );
      const tradeAmount = Math.min(capital * positionSize, capital);
      const effectivePrice = currentPrice * (1 + this.slippage);
      const adaBought = (tradeAmount - this.commission) / effectivePrice;

      return {
        trade: {
          type: Recommendation.Buy,
          price: effectivePrice,
          timestamp: currentTimestamp,
          adaAmount: adaBought,
          usdValue: tradeAmount,
        },
        confidence,
      };
    }

    return { trade: null, confidence };
  }
}
