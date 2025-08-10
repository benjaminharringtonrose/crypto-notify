import { TradeModelPredictor } from "./TradeModelPredictor";
import { Trade, Recommendation, StrategyType } from "../types";
import { PERIODS, STRATEGY_CONFIG, TIME_CONVERSIONS } from "../constants";

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
  private strategyStartTimestamp: string | null = null;
  private strategyTradeCount: number = 0;
  private recentIndicators: { [key: string]: number[] } = {
    shortMomentum: [],
    momentum: [],
    trendSlope: [],
    momentumDivergence: [],
    volatilityAdjustedMomentum: [],
    trendStrength: [],
    atrBreakout: [],
  };

  constructor(params?: TradingStrategyParams) {
    const {
      basePositionSize = STRATEGY_CONFIG.BASE_POSITION_SIZE_DEFAULT,
      slippage = STRATEGY_CONFIG.SLIPPAGE,
      commission = STRATEGY_CONFIG.COMMISSION,
      stopLossMultiplier = STRATEGY_CONFIG.STOP_LOSS_MULTIPLIER_DEFAULT,
      trailingStop = STRATEGY_CONFIG.TRAILING_STOP_DEFAULT,
      minHoldDays = STRATEGY_CONFIG.MIN_HOLD_DAYS_DEFAULT,
      minConfidence = STRATEGY_CONFIG.MIN_CONFIDENCE_DEFAULT,
      profitTakeMultiplier = STRATEGY_CONFIG.PROFIT_TAKE_MULTIPLIER_DEFAULT,
      buyProbThreshold = STRATEGY_CONFIG.BUY_PROB_THRESHOLD_DEFAULT,
      sellProbThreshold = STRATEGY_CONFIG.SELL_PROB_THRESHOLD_DEFAULT,
      smaPeriod = PERIODS.SMA_MEDIUM,
      breakoutThreshold = STRATEGY_CONFIG.DYNAMIC_BREAKOUT_THRESHOLD,
    } = params ?? {};
    this.predictor = new TradeModelPredictor();
    this.currentStrategy = StrategyType.MeanReversion; // Default
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

  private updateRecentIndicators(
    shortMomentum: number,
    momentum: number,
    trendSlope: number,
    momentumDivergence: number,
    volatilityAdjustedMomentum: number,
    trendStrength: number,
    atrBreakout: number
  ) {
    const keys = [
      "shortMomentum",
      "momentum",
      "trendSlope",
      "momentumDivergence",
      "volatilityAdjustedMomentum",
      "trendStrength",
      "atrBreakout",
    ];
    const values = [
      shortMomentum,
      momentum,
      trendSlope,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
    ];
    keys.forEach((key, index) => {
      this.recentIndicators[key].push(values[index]);
      if (this.recentIndicators[key].length > 3) {
        this.recentIndicators[key].shift();
      }
    });
  }

  private getThreeDayAverage(key: string): number {
    const values = this.recentIndicators[key];
    return values.length >= 3
      ? values.reduce((sum, val) => sum + val, 0) / 3
      : values[values.length - 1] || 0;
  }

  private shouldPersistStrategy(
    currentTimestamp: string,
    confidence: number
  ): boolean {
    if (!this.strategyStartTimestamp) return false;
    const daysSinceStart =
      (new Date(currentTimestamp).getTime() -
        new Date(this.strategyStartTimestamp).getTime()) /
      TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
    return (
      (this.strategyTradeCount < STRATEGY_CONFIG.STRATEGY_PERSISTENCE_TRADES ||
        daysSinceStart < STRATEGY_CONFIG.STRATEGY_PERSISTENCE_DAYS) &&
      confidence < STRATEGY_CONFIG.STRATEGY_OVERRIDE_CONFIDENCE
    );
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
    volumes: number[],
    currentTimestamp: string,
    confidence: number
  ): StrategyType {
    this.updateRecentIndicators(
      shortMomentum,
      momentum,
      trendSlope,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout
    );

    const avgShortMomentum = this.getThreeDayAverage("shortMomentum");
    const avgMomentum = this.getThreeDayAverage("momentum");
    const avgTrendSlope = this.getThreeDayAverage("trendSlope");
    const avgMomentumDivergence = this.getThreeDayAverage("momentumDivergence");
    const avgVolatilityAdjustedMomentum = this.getThreeDayAverage(
      "volatilityAdjustedMomentum"
    );
    const avgTrendStrength = this.getThreeDayAverage("trendStrength");
    const avgAtrBreakout = this.getThreeDayAverage("atrBreakout");

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

    if (this.shouldPersistStrategy(currentTimestamp, confidence)) {
      return this.currentStrategy;
    }

    let newStrategy: StrategyType;
    // Prioritize Trend-Following for bullish markets
    if (
      Math.abs(avgTrendSlope) > STRATEGY_CONFIG.TREND_SLOPE_THRESHOLD &&
      emaShort > emaLong &&
      avgTrendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD
    ) {
      newStrategy = StrategyType.TrendFollowing;
    }
    // Momentum for volatile markets
    else if (
      avgShortMomentum > STRATEGY_CONFIG.MOMENTUM_THRESHOLD &&
      avgVolatilityAdjustedMomentum >
        STRATEGY_CONFIG.MOMENTUM.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
      avgTrendStrength > STRATEGY_CONFIG.MOMENTUM.TREND_STRENGTH_THRESHOLD &&
      emaShort > emaLong
    ) {
      newStrategy = StrategyType.Momentum;
    }
    // Breakout for high volume and volatility
    else if (
      avgAtrBreakout > this.breakoutThreshold &&
      currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_MULTIPLIER &&
      avgShortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD
    ) {
      newStrategy = StrategyType.Breakout;
    }
    // Mean-Reversion for range-bound markets
    else if (
      Math.abs(deviation) > STRATEGY_CONFIG.DEVIATION_THRESHOLD &&
      Math.abs(avgMomentum) < STRATEGY_CONFIG.MOMENTUM_THRESHOLD &&
      avgMomentumDivergence !== 0
    ) {
      newStrategy = StrategyType.MeanReversion;
    } else {
      newStrategy = StrategyType.MeanReversion; // Default for stability
    }

    if (newStrategy !== this.currentStrategy) {
      this.currentStrategy = newStrategy;
      this.strategyStartTimestamp = currentTimestamp;
      this.strategyTradeCount = 0;
      this.updateStrategyParameters();
    }

    return this.currentStrategy;
  }

  private updateStrategyParameters() {
    switch (this.currentStrategy) {
      case StrategyType.Momentum:
        this.minHoldDays = STRATEGY_CONFIG.MOMENTUM.MIN_HOLD_DAYS_DEFAULT;
        this.stopLossMultiplier =
          STRATEGY_CONFIG.MOMENTUM.STOP_LOSS_MULTIPLIER_DEFAULT;
        this.trailingStop = STRATEGY_CONFIG.MOMENTUM.TRAILING_STOP_DEFAULT;
        this.profitTakeMultiplier =
          STRATEGY_CONFIG.MOMENTUM.PROFIT_TAKE_MULTIPLIER_DEFAULT;
        this.minConfidence = STRATEGY_CONFIG.MOMENTUM.MIN_CONFIDENCE_DEFAULT;
        this.buyProbThreshold =
          STRATEGY_CONFIG.MOMENTUM.BUY_PROB_THRESHOLD_DEFAULT;
        break;
      case StrategyType.MeanReversion:
        this.minHoldDays = STRATEGY_CONFIG.MEAN_REVERSION.MIN_HOLD_DAYS_DEFAULT;
        this.stopLossMultiplier =
          STRATEGY_CONFIG.MEAN_REVERSION.STOP_LOSS_MULTIPLIER_DEFAULT;
        this.trailingStop =
          STRATEGY_CONFIG.MEAN_REVERSION.TRAILING_STOP_DEFAULT;
        this.profitTakeMultiplier =
          STRATEGY_CONFIG.MEAN_REVERSION.PROFIT_TAKE_MULTIPLIER_DEFAULT;
        this.minConfidence =
          STRATEGY_CONFIG.MEAN_REVERSION.MIN_CONFIDENCE_DEFAULT;
        this.buyProbThreshold =
          STRATEGY_CONFIG.MEAN_REVERSION.BUY_PROB_THRESHOLD_DEFAULT;
        break;
      case StrategyType.Breakout:
        this.minHoldDays = STRATEGY_CONFIG.BREAKOUT.MIN_HOLD_DAYS_DEFAULT;
        this.stopLossMultiplier =
          STRATEGY_CONFIG.BREAKOUT.STOP_LOSS_MULTIPLIER_DEFAULT;
        this.trailingStop = STRATEGY_CONFIG.BREAKOUT.TRAILING_STOP_DEFAULT;
        this.profitTakeMultiplier =
          STRATEGY_CONFIG.BREAKOUT.PROFIT_TAKE_MULTIPLIER_DEFAULT;
        this.minConfidence = STRATEGY_CONFIG.BREAKOUT.MIN_CONFIDENCE_DEFAULT;
        this.buyProbThreshold =
          STRATEGY_CONFIG.BREAKOUT.BUY_PROB_THRESHOLD_DEFAULT;
        break;
      case StrategyType.TrendFollowing:
        this.minHoldDays =
          STRATEGY_CONFIG.TREND_FOLLOWING.MIN_HOLD_DAYS_DEFAULT;
        this.stopLossMultiplier =
          STRATEGY_CONFIG.TREND_FOLLOWING.STOP_LOSS_MULTIPLIER_DEFAULT;
        this.trailingStop =
          STRATEGY_CONFIG.TREND_FOLLOWING.TRAILING_STOP_DEFAULT;
        this.profitTakeMultiplier =
          STRATEGY_CONFIG.TREND_FOLLOWING.PROFIT_TAKE_MULTIPLIER_DEFAULT;
        this.minConfidence =
          STRATEGY_CONFIG.TREND_FOLLOWING.MIN_CONFIDENCE_DEFAULT;
        this.buyProbThreshold =
          STRATEGY_CONFIG.TREND_FOLLOWING.BUY_PROB_THRESHOLD_DEFAULT;
        break;
      default:
        this.minHoldDays = STRATEGY_CONFIG.MIN_HOLD_DAYS_DEFAULT;
        this.stopLossMultiplier = STRATEGY_CONFIG.STOP_LOSS_MULTIPLIER_DEFAULT;
        this.trailingStop = STRATEGY_CONFIG.TRAILING_STOP_DEFAULT;
        this.profitTakeMultiplier =
          STRATEGY_CONFIG.PROFIT_TAKE_MULTIPLIER_DEFAULT;
        this.minConfidence = STRATEGY_CONFIG.MIN_CONFIDENCE_DEFAULT;
        this.buyProbThreshold = STRATEGY_CONFIG.BUY_PROB_THRESHOLD_DEFAULT;
    }
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

    if (
      confidence < this.minConfidence ||
      atr > STRATEGY_CONFIG.MAX_ATR_THRESHOLD
    ) {
      return { trade: null, confidence, buyProb, sellProb };
    }

    this.currentStrategy = this.selectStrategy(
      shortMomentum,
      momentum,
      trendSlope,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
      adaPrices,
      adaVolumes,
      currentTimestamp,
      confidence
    );

    const daysSinceLastTrade = buyTimestamp
      ? (new Date(currentTimestamp).getTime() -
          new Date(buyTimestamp).getTime()) /
        TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS
      : Infinity;

    const dynamicBreakoutThreshold =
      daysSinceLastTrade > STRATEGY_CONFIG.DAYS_SINCE_TRADE_THRESHOLD
        ? STRATEGY_CONFIG.DYNAMIC_BREAKOUT_THRESHOLD
        : this.breakoutThreshold;

    const dynamicStopLossMultiplier =
      confidence > STRATEGY_CONFIG.HIGH_CONFIDENCE_THRESHOLD
        ? this.stopLossMultiplier * 1.2
        : this.stopLossMultiplier;

    const dynamicTrailingStop =
      momentum > STRATEGY_CONFIG.MOMENTUM_MULTIPLIER
        ? this.trailingStop * 0.8
        : this.trailingStop;

    const dynamicProfitTake = Math.min(
      this.profitTakeMultiplier *
        (momentum > 0.1 ? STRATEGY_CONFIG.CONFIDENCE_BOOST_MULTIPLIER : 1.0),
      STRATEGY_CONFIG.MAX_PROFIT_TAKE
    );

    const atrAdjustedHold = Math.max(
      this.currentStrategy === StrategyType.MeanReversion ? 3 : 5,
      Math.min(12, this.minHoldDays * (1 + atr))
    );

    if (holdings > 0 && lastBuyPrice && buyTimestamp) {
      const daysHeld =
        (new Date(currentTimestamp).getTime() -
          new Date(buyTimestamp).getTime()) /
        TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;

      const priceChange = (currentPrice - lastBuyPrice) / lastBuyPrice;

      const stopLossLevel = dynamicStopLossMultiplier * atr;
      const profitTakeLevel = lastBuyPrice + dynamicProfitTake * atr;
      const trailingStopLevel =
        priceChange >= STRATEGY_CONFIG.MIN_PROFIT_THRESHOLD
          ? (peakPrice || lastBuyPrice) * (1 - dynamicTrailingStop)
          : Infinity;

      if (daysHeld >= atrAdjustedHold) {
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
          this.strategyTradeCount++;
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
      // Trade quality filtering - only take high-quality trades
      const tradeQuality = buyProb * confidence > 0.25; // Combined quality metric
      const minProfitPotential = 0.05; // 5% minimum profit potential
      const profitPotential = (dynamicProfitTake - currentPrice) / currentPrice;
      const hasProfitPotential = profitPotential >= minProfitPotential;

      if (buyConditions && tradeQuality && hasProfitPotential && capital > 0) {
        const volatilityAdjustedSize = Math.min(
          this.basePositionSize / (atr > 0 ? atr : 0.01),
          atr > STRATEGY_CONFIG.ATR_POSITION_THRESHOLD
            ? STRATEGY_CONFIG.POSITION_SIZE_MAX_HIGH_ATR
            : STRATEGY_CONFIG.POSITION_SIZE_MAX
        );
        const trendAdjustedSize =
          trendSlope > STRATEGY_CONFIG.TREND_SLOPE_BOOST_THRESHOLD
            ? volatilityAdjustedSize *
              STRATEGY_CONFIG.TREND_SLOPE_POSITION_BOOST
            : volatilityAdjustedSize;
        const confidenceBoost =
          confidence > STRATEGY_CONFIG.HIGH_CONFIDENCE_THRESHOLD
            ? STRATEGY_CONFIG.CONFIDENCE_BOOST_MULTIPLIER
            : 1.0;
        const winStreakBoost =
          this.currentStrategy === StrategyType.Momentum && winStreak > 1
            ? 1 + winStreak * 0.2
            : 1.0;
        const positionSize = Math.min(
          trendAdjustedSize *
            confidenceBoost *
            winStreakBoost *
            Math.min(
              buyProb / this.buyProbThreshold,
              STRATEGY_CONFIG.BUY_PROB_MAX_MULTIPLIER
            ),
          STRATEGY_CONFIG.POSITION_SIZE_MAX // Cap at 0.05
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
        this.strategyTradeCount++;
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
          shortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD &&
          volatilityAdjustedMomentum >
            STRATEGY_CONFIG.MOMENTUM.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
          trendStrength > STRATEGY_CONFIG.MOMENTUM.TREND_STRENGTH_THRESHOLD &&
          (atrBreakout > dynamicBreakoutThreshold || trendReversal) &&
          currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_BOOST_THRESHOLD
        );
      case StrategyType.MeanReversion:
        const sma = this.calculateSMA(prices, this.smaPeriod);
        const deviation = (prices[prices.length - 1] - sma) / sma;
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          deviation < STRATEGY_CONFIG.NEGATIVE_DEVIATION_THRESHOLD &&
          shortMomentum > STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_MIN &&
          momentumDivergence > 0
        );
      case StrategyType.Breakout:
        return (
          buyProb > this.buyProbThreshold &&
          confidence >= this.minConfidence &&
          atrBreakout > dynamicBreakoutThreshold &&
          shortMomentum > STRATEGY_CONFIG.SHORT_MOMENTUM_THRESHOLD &&
          trendSlope > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD &&
          currentVolume > avgVolume * STRATEGY_CONFIG.VOLUME_MULTIPLIER
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
          trendSlope > STRATEGY_CONFIG.TREND_SLOPE_THRESHOLD &&
          trendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD
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
        // Enhanced sell conditions with momentum-based exits
        const momentumExitMomentum =
          shortMomentum < -0.01 && trendStrength < 0.05;
        const profitPotentialExitMomentum = priceChange >= 0.05; // Exit if 5% profit reached
        return (
          sellProb > this.sellProbThreshold + 0.05 ||
          momentum < STRATEGY_CONFIG.NEGATIVE_MOMENTUM_THRESHOLD ||
          shortMomentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
          trendStrength < STRATEGY_CONFIG.TREND_STRENGTH_REVERSAL_THRESHOLD ||
          momentumExitMomentum || // New: momentum-based exit
          profitPotentialExitMomentum || // New: profit potential exit
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        );
      case StrategyType.MeanReversion:
        // Enhanced sell conditions with momentum-based exits
        const momentumExitMeanRev =
          shortMomentum < -0.01 && trendStrength < 0.05;
        const profitPotentialExitMeanRev = priceChange >= 0.03; // Exit if 3% profit reached
        return (
          sellProb > this.sellProbThreshold ||
          momentum > STRATEGY_CONFIG.MOMENTUM_MAX ||
          priceChange >= STRATEGY_CONFIG.MEAN_REVERSION_THRESHOLD ||
          momentumExitMeanRev || // New: momentum-based exit
          profitPotentialExitMeanRev || // New: profit potential exit
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel
        );
      case StrategyType.Breakout:
        return (
          sellProb > this.sellProbThreshold ||
          momentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
          priceChange <= -stopLossLevel ||
          currentPrice <= trailingStopLevel ||
          currentPrice >= profitTakeLevel
        );
      case StrategyType.TrendFollowing:
        return (
          sellProb > this.sellProbThreshold + 0.05 ||
          trendStrength < STRATEGY_CONFIG.TREND_STRENGTH_REVERSAL_THRESHOLD ||
          momentum < STRATEGY_CONFIG.NEGATIVE_SHORT_MOMENTUM_THRESHOLD ||
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
    this.strategyStartTimestamp = null;
    this.strategyTradeCount = 0;
    this.updateStrategyParameters();
    console.log(`Strategy manually set to: ${strategyType}`);
  }

  public getCurrentStrategy(): StrategyType {
    return this.currentStrategy;
  }
}
