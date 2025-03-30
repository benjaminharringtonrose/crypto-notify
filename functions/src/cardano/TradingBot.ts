import { TradeModelPredictor } from "./TradeModelPredictor";
import { Trade, Recommendation } from "../types";
import { MODEL_CONSTANTS } from "../constants";

interface TradingBotParams {
  basePositionSize?: number;
  slippage?: number;
  commission?: number;
  stopLossMultiplier?: number;
  trailingStop?: number;
  minHoldDays?: number;
  minConfidence?: number;
  profitTakeMultiplier?: number;
  logitThreshold?: number;
  buyProbThreshold?: number;
  sellProbThreshold?: number;
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
    basePositionSize = MODEL_CONSTANTS.BASE_POSITION_SIZE_DEFAULT,
    slippage = MODEL_CONSTANTS.SLIPPAGE,
    commission = MODEL_CONSTANTS.COMMISSION,
    stopLossMultiplier = MODEL_CONSTANTS.STOP_LOSS_MULTIPLIER_DEFAULT,
    trailingStop = MODEL_CONSTANTS.TRAILING_STOP_DEFAULT,
    minHoldDays = MODEL_CONSTANTS.MIN_HOLD_DAYS_DEFAULT,
    minConfidence = MODEL_CONSTANTS.MIN_CONFIDENCE_DEFAULT,
    profitTakeMultiplier = MODEL_CONSTANTS.PROFIT_TAKE_MULTIPLIER_DEFAULT,
    logitThreshold = MODEL_CONSTANTS.LOGIT_THRESHOLD_DEFAULT,
    buyProbThreshold = MODEL_CONSTANTS.BUY_PROB_THRESHOLD_DEFAULT,
    sellProbThreshold = MODEL_CONSTANTS.SELL_PROB_THRESHOLD_DEFAULT,
  }: TradingBotParams = {}) {
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
      atr,
    } = prediction;

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
          (momentum < -0.03 && trendSlope < 0) || // Version 3 momentum exit
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        ) {
          const effectivePrice = currentPrice * (1 - this.slippage);
          const usdReceived = holdings * effectivePrice - this.commission;
          console.log(
            `Sell Trigger: StopLoss=${stopLossLevel.toFixed(
              4
            )}, Trailing=${trailingStopLevel.toFixed(
              4
            )}, ProfitTake=${profitTakeLevel.toFixed(4)}`
          );
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
        trendSlope > 0.05 && atr < 0.02 ? 0.2 : 0.15
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
        0.2
      );
      const tradeAmount = Math.min(capital * positionSize, capital);
      const effectivePrice = currentPrice * (1 + this.slippage);
      const adaBought = (tradeAmount - this.commission) / effectivePrice;

      console.log(
        `Position Size: ${positionSize.toFixed(4)}, ATR: ${atr.toFixed(
          4
        )}, MinConfidence: ${this.minConfidence.toFixed(2)}`
      );
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
