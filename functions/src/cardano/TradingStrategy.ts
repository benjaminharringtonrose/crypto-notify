import { TradeModelPredictor } from "./TradeModelPredictor";
import { Trade, Recommendation, StrategyType } from "../types";
import { MODEL_CONSTANTS } from "../constants";

interface TradingStrategyParams {
  basePositionSize?: number;
  slippage?: number;
  commission?: number;
  stopLossMultiplier?: number;
  trailingStop?: number;
  minHoldDays?: number;
  minConfidence?: number;
  profitTakeMultiplier?: number;
  buyProbThreshold?: number;
  sellProbThreshold?: number;
  smaPeriod?: number;
  breakoutThreshold?: number;
}

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

  constructor({
    basePositionSize = MODEL_CONSTANTS.BASE_POSITION_SIZE_DEFAULT,
    slippage = MODEL_CONSTANTS.SLIPPAGE,
    commission = MODEL_CONSTANTS.COMMISSION,
    stopLossMultiplier = MODEL_CONSTANTS.STOP_LOSS_MULTIPLIER_DEFAULT,
    trailingStop = MODEL_CONSTANTS.TRAILING_STOP_DEFAULT,
    minHoldDays = MODEL_CONSTANTS.MIN_HOLD_DAYS_DEFAULT,
    minConfidence = MODEL_CONSTANTS.MIN_CONFIDENCE_DEFAULT,
    profitTakeMultiplier = MODEL_CONSTANTS.PROFIT_TAKE_MULTIPLIER_DEFAULT,
    buyProbThreshold = MODEL_CONSTANTS.BUY_PROB_THRESHOLD_DEFAULT,
    sellProbThreshold = MODEL_CONSTANTS.SELL_PROB_THRESHOLD_DEFAULT,
    smaPeriod = 20,
    breakoutThreshold = 0.8, // Reduced from 1.0
  }: TradingStrategyParams = {}) {
    this.predictor = new TradeModelPredictor();
    this.currentStrategy = StrategyType.Momentum;
    this.basePositionSize = basePositionSize;
    this.slippage = slippage;
    this.commission = commission;
    this.stopLossMultiplier = stopLossMultiplier;
    this.trailingStop = trailingStop;
    this.minHoldDays = minHoldDays;
    this.minConfidence = minConfidence;
    this.profitTakeMultiplier = profitTakeMultiplier;
    this.buyProbThreshold = buyProbThreshold;
    this.sellProbThreshold = sellProbThreshold;
    this.smaPeriod = smaPeriod;
    this.breakoutThreshold = breakoutThreshold;
  }

  private calculateSMA(prices: number[], period: number): number {
    const window = prices.slice(-period);
    return window.length >= period
      ? window.reduce((sum, price) => sum + price, 0) / period
      : prices[prices.length - 1];
  }

  private calculateEMA(prices: number[], period: number): number {
    const k = 2 / (period + 1);
    let ema = prices[0];
    for (let i = 1; i < prices.length && i < period; i++) {
      ema = prices[i] * k + ema * (1 - k);
    }
    return ema;
  }

  private selectStrategy(
    shortMomentum: number,
    momentum: number,
    trendSlope: number,
    momentumDivergence: number,
    volatilityAdjustedMomentum: number,
    trendStrength: number,
    atrBreakout: number,
    prices: number[],
    volumes: number[]
  ): StrategyType {
    const currentPrice = prices[prices.length - 1];
    const currentVolume = volumes[volumes.length - 1];
    const avgVolume =
      volumes.slice(-this.smaPeriod).reduce((sum, v) => sum + v, 0) /
      Math.min(volumes.length, this.smaPeriod);
    const sma = this.calculateSMA(prices, this.smaPeriod);
    const deviation = (currentPrice - sma) / sma;
    const emaLong = this.calculateEMA(
      prices.slice(-this.smaPeriod),
      this.smaPeriod
    );
    const emaShort = this.calculateEMA(
      prices.slice(-Math.floor(this.smaPeriod / 2)),
      Math.floor(this.smaPeriod / 2)
    );

    if (
      shortMomentum > 0.02 &&
      volatilityAdjustedMomentum > 0.2 &&
      trendStrength > 0.005 &&
      emaShort > emaLong
    ) {
      console.log("Selected Strategy: MomentumTrendHybrid");
      return StrategyType.Momentum; // Hybrid approach
    }

    if (
      shortMomentum > 0.02 &&
      volatilityAdjustedMomentum > 0.2 &&
      trendStrength > 0.005
    ) {
      console.log("Selected Strategy: Momentum");
      return StrategyType.Momentum;
    }

    if (
      Math.abs(deviation) > 0.02 &&
      Math.abs(momentum) < 0.01 &&
      momentumDivergence !== 0
    ) {
      console.log("Selected Strategy: Mean Reversion");
      return StrategyType.MeanReversion;
    }

    if (
      atrBreakout > this.breakoutThreshold &&
      currentVolume > avgVolume * 1.3 &&
      shortMomentum > 0.01
    ) {
      console.log("Selected Strategy: Breakout");
      return StrategyType.Breakout;
    }

    if (
      Math.abs(trendSlope) > 0.01 &&
      emaShort > emaLong &&
      trendStrength > 0.01
    ) {
      console.log("Selected Strategy: Trend Following");
      return StrategyType.TrendFollowing;
    }

    console.log("Selected Strategy: Momentum (default)");
    return StrategyType.Momentum;
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
    winStreak = 0,
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
    winStreak?: number;
  }): Promise<{
    trade: Trade | null;
    confidence: number;
    buyProb: number;
    sellProb: number;
  }> {
    const currentPrice = adaPrices[adaPrices.length - 1];
    const currentVolume = adaVolumes[adaVolumes.length - 1];
    const prediction = await this.predictor.predict(
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes
    );
    const {
      buyProb,
      sellProb,
      confidence,
      momentum,
      shortMomentum,
      trendSlope,
      atr,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
    } = prediction;

    this.currentStrategy = this.selectStrategy(
      shortMomentum,
      momentum,
      trendSlope,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
      adaPrices,
      adaVolumes
    );

    const daysSinceLastTrade = buyTimestamp
      ? (new Date(currentTimestamp).getTime() -
          new Date(buyTimestamp).getTime()) /
        (1000 * 60 * 60 * 24)
      : Infinity;

    const dynamicBreakoutThreshold =
      daysSinceLastTrade > 15 ? 0.7 : this.breakoutThreshold;

    const dynamicStopLossMultiplier =
      confidence > 0.7
        ? this.stopLossMultiplier * 0.8
        : this.stopLossMultiplier;

    const dynamicTrailingStop =
      momentum > 0.05 ? this.trailingStop * 0.8 : this.trailingStop * 1.2;

    const dynamicProfitTake = Math.min(
      this.profitTakeMultiplier * (momentum > 0.1 ? 1.2 : 1.0),
      4.0
    );

    if (atr > MODEL_CONSTANTS.MAX_ATR_THRESHOLD) {
      console.log(`Trade skipped: ATR ${atr.toFixed(4)} exceeds threshold`);
      return {
        trade: null,
        confidence,
        buyProb,
        sellProb,
      };
    }

    if (holdings > 0 && lastBuyPrice && buyTimestamp) {
      const daysHeld =
        (new Date(currentTimestamp).getTime() -
          new Date(buyTimestamp).getTime()) /
        (1000 * 60 * 60 * 24);

      const priceChange = (currentPrice - lastBuyPrice) / lastBuyPrice;

      const stopLossLevel = dynamicStopLossMultiplier * atr;

      const profitTakeLevel = lastBuyPrice + dynamicProfitTake * atr;

      const trailingStopLevel =
        priceChange >= MODEL_CONSTANTS.MIN_PROFIT_THRESHOLD
          ? (peakPrice || lastBuyPrice) * (1 - dynamicTrailingStop)
          : Infinity;

      if (daysHeld >= this.minHoldDays) {
        const sellConditions = this.getSellConditions(
          sellProb,
          momentum,
          shortMomentum,
          trendStrength,
          priceChange,
          stopLossLevel,
          trailingStopLevel,
          profitTakeLevel,
          currentPrice
        );
        if (sellConditions) {
          const effectivePrice = currentPrice * (1 - this.slippage);
          const usdReceived = holdings * effectivePrice - this.commission;
          console.log(
            `Sell Trigger: StopLoss=${stopLossLevel.toFixed(
              4
            )}, Trailing=${trailingStopLevel.toFixed(
              4
            )}, ProfitTake=${profitTakeLevel.toFixed(
              4
            )}, Trend Strength=${trendStrength.toFixed(4)}, Strategy=${
              this.currentStrategy
            }`
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
            buyProb,
            sellProb,
          };
        }
      }
    } else {
      const buyConditions = this.getBuyConditions(
        buyProb,
        confidence,
        shortMomentum,
        trendSlope,
        momentumDivergence,
        volatilityAdjustedMomentum,
        trendStrength,
        atrBreakout,
        adaPrices,
        adaVolumes,
        dynamicBreakoutThreshold
      );
      if (buyConditions && capital > 0) {
        const volatilityAdjustedSize = Math.min(
          this.basePositionSize / (atr > 0 ? atr : 0.01),
          trendSlope > 0.05 && atr < 0.02 ? 0.25 : 0.2
        );
        const trendAdjustedSize =
          trendSlope > 0.02
            ? volatilityAdjustedSize * 1.2
            : volatilityAdjustedSize;
        const confidenceBoost = confidence > 0.7 ? 1.2 : 1.0;
        const winStreakBoost =
          this.currentStrategy === StrategyType.Momentum && winStreak > 1
            ? 1 + winStreak * 0.2
            : 1.0;
        const positionSize = Math.min(
          trendAdjustedSize *
            confidenceBoost *
            winStreakBoost *
            Math.min(buyProb / this.buyProbThreshold, 2.0),
          confidence > 0.7 && trendStrength > 0.1 ? 0.35 : 0.25 // Dynamic max size
        );
        const tradeAmount = Math.min(capital * positionSize, capital);
        const effectivePrice = currentPrice * (1 + this.slippage);
        const adaBought = (tradeAmount - this.commission) / effectivePrice;

        console.log(
          `Buy Trigger: ShortMomentum=${shortMomentum.toFixed(
            4
          )}, TrendSlope=${trendSlope.toFixed(
            4
          )}, Divergence=${momentumDivergence.toFixed(
            4
          )}, Vol-Adj Momentum=${volatilityAdjustedMomentum.toFixed(
            4
          )}, Trend Strength=${trendStrength.toFixed(
            4
          )}, ATR Breakout=${atrBreakout.toFixed(
            4
          )}, Volume=${currentVolume.toFixed(2)}, Strategy=${
            this.currentStrategy
          }`
        );
        console.log(
          `Position Size: ${positionSize.toFixed(4)}, ATR: ${atr.toFixed(
            4
          )}, MinConfidence: ${this.minConfidence.toFixed(
            2
          )}, WinStreakBoost: ${winStreakBoost.toFixed(2)}`
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
          buyProb,
          sellProb,
        };
      }
    }

    return {
      trade: null,
      confidence,
      buyProb,
      sellProb,
    };
  }

  private getBuyConditions(
    buyProb: number,
    confidence: number,
    shortMomentum: number,
    trendSlope: number,
    momentumDivergence: number,
    volatilityAdjustedMomentum: number,
    trendStrength: number,
    atrBreakout: number,
    prices: number[],
    volumes: number[],
    dynamicBreakoutThreshold: number
  ): boolean {
    const currentVolume = volumes[volumes.length - 1];
    const avgVolume =
      volumes.slice(-this.smaPeriod).reduce((sum, v) => sum + v, 0) /
      Math.min(volumes.length, this.smaPeriod);
    const trendReversal =
      this.currentStrategy === StrategyType.Momentum &&
      prices.length > this.smaPeriod &&
      this.calculateSMA(prices.slice(-this.smaPeriod), this.smaPeriod) <
        prices[prices.length - 1] &&
      trendSlope > 0;

    switch (this.currentStrategy) {
      case StrategyType.Momentum:
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          shortMomentum > 0.01 &&
          volatilityAdjustedMomentum > 0.2 &&
          trendStrength > 0.005 &&
          (atrBreakout > dynamicBreakoutThreshold || trendReversal) &&
          currentVolume > avgVolume * 1.2
        );
      case StrategyType.MeanReversion:
        const sma = this.calculateSMA(prices, this.smaPeriod);
        const deviation = (prices[prices.length - 1] - sma) / sma;
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          deviation < -0.015 &&
          shortMomentum > -0.01 &&
          momentumDivergence > 0
        );
      case StrategyType.Breakout:
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          atrBreakout > dynamicBreakoutThreshold &&
          shortMomentum > 0.01 &&
          trendSlope > 0.005 &&
          currentVolume > avgVolume * 1.3
        );
      case StrategyType.TrendFollowing:
        const emaLong = this.calculateEMA(
          prices.slice(-this.smaPeriod),
          this.smaPeriod
        );
        const emaShort = this.calculateEMA(
          prices.slice(-Math.floor(this.smaPeriod / 2)),
          Math.floor(this.smaPeriod / 2)
        );
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          emaShort > emaLong &&
          trendSlope > 0.005 &&
          trendStrength > 0.01
        );
      default:
        throw new Error(`Unknown strategy type: ${this.currentStrategy}`);
    }
  }

  private getSellConditions(
    sellProb: number,
    momentum: number,
    shortMomentum: number,
    trendStrength: number,
    priceChange: number,
    stopLossLevel: number,
    trailingStopLevel: number,
    profitTakeLevel: number,
    currentPrice: number
  ): boolean {
    switch (this.currentStrategy) {
      case StrategyType.Momentum:
        return (
          sellProb > this.sellProbThreshold ||
          momentum < -0.03 ||
          shortMomentum < -0.01 ||
          trendStrength < 0 ||
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        );
      case StrategyType.MeanReversion:
        return (
          sellProb > this.sellProbThreshold ||
          momentum > 0.03 ||
          priceChange >= 0.015 ||
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel
        );
      case StrategyType.Breakout:
        return (
          sellProb > this.sellProbThreshold ||
          momentum < -0.01 ||
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        );
      case StrategyType.TrendFollowing:
        return (
          sellProb > this.sellProbThreshold ||
          trendStrength < 0 ||
          momentum < -0.01 ||
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        );
      default:
        throw new Error(`Unknown strategy type: ${this.currentStrategy}`);
    }
  }

  public setStrategy(strategyType: StrategyType): void {
    this.currentStrategy = strategyType;
    console.log(`Strategy manually set to: ${strategyType}`);
  }

  public getCurrentStrategy(): StrategyType {
    return this.currentStrategy;
  }
}
