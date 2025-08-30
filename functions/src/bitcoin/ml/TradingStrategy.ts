import { TradeModelPredictor } from "./TradeModelPredictor";
import { Trade, Recommendation, StrategyType } from "../../types";
import { PERIODS, STRATEGY_CONFIG, TIME_CONVERSIONS } from "../../constants";
import FeatureCalculator from "../shared/FeatureCalculator";

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

interface MarketRegime {
  volatilityRegime: string;
  trendRegime: string;
  momentumRegime: string;
  realizedVolatility: number;
  regimeScore: number;
}

interface StrategyWeights {
  momentum: number;
  mean_reversion: number;
  breakout: number;
  trend_following: number;
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
  private strategyTradeCount: number = 0;
  private featureCalculator: FeatureCalculator;

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
    this.featureCalculator = new FeatureCalculator();
  }

  // Market Regime Detection and Dynamic Strategy Adaptation
  private detectMarketRegime(
    prices: number[],
    volumes: number[],
    dayIndex: number,
    currentPrice: number
  ): MarketRegime {
    return this.featureCalculator.calculateMarketRegimeFeatures(
      prices,
      volumes,
      dayIndex,
      currentPrice
    );
  }

  private calculateStrategyWeights(
    marketRegime: MarketRegime
  ): StrategyWeights {
    const { volatilityRegime, trendRegime, momentumRegime } = marketRegime;

    // Enhanced base weights with more balanced distribution
    let weights: StrategyWeights = {
      momentum: 0.25,
      mean_reversion: 0.25,
      breakout: 0.25,
      trend_following: 0.25,
    };

    // Volatility-based adjustments with more nuanced weighting
    switch (volatilityRegime) {
      case "EXTREME_HIGH":
        weights.breakout = 0.45;
        weights.momentum = 0.25;
        weights.mean_reversion = 0.15;
        weights.trend_following = 0.15;
        break;
      case "HIGH":
        weights.breakout = 0.35;
        weights.momentum = 0.3;
        weights.mean_reversion = 0.2;
        weights.trend_following = 0.15;
        break;
      case "MEDIUM":
        weights.breakout = 0.25;
        weights.momentum = 0.3;
        weights.mean_reversion = 0.25;
        weights.trend_following = 0.2;
        break;
      case "LOW":
        weights.breakout = 0.15;
        weights.momentum = 0.25;
        weights.mean_reversion = 0.35;
        weights.trend_following = 0.25;
        break;
      case "VERY_LOW":
        weights.breakout = 0.1;
        weights.momentum = 0.2;
        weights.mean_reversion = 0.4;
        weights.trend_following = 0.3;
        break;
    }

    // Trend-based adjustments with enhanced logic
    switch (trendRegime) {
      case "STRONG_UPTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.4);
        weights.momentum = Math.max(weights.momentum, 0.3);
        weights.mean_reversion = Math.min(weights.mean_reversion, 0.2);
        weights.breakout = Math.min(weights.breakout, 0.1);
        break;
      case "UPTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.35);
        weights.momentum = Math.max(weights.momentum, 0.3);
        weights.mean_reversion = Math.min(weights.mean_reversion, 0.25);
        weights.breakout = Math.min(weights.breakout, 0.1);
        break;
      case "WEAK_UPTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.3);
        weights.momentum = Math.max(weights.momentum, 0.25);
        weights.mean_reversion = Math.min(weights.mean_reversion, 0.3);
        weights.breakout = Math.min(weights.breakout, 0.15);
        break;
      case "STRONG_DOWNTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.4);
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.3);
        weights.momentum = Math.min(weights.momentum, 0.2);
        weights.breakout = Math.min(weights.breakout, 0.1);
        break;
      case "DOWNTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.35);
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.3);
        weights.momentum = Math.min(weights.momentum, 0.25);
        weights.breakout = Math.min(weights.breakout, 0.1);
        break;
      case "WEAK_DOWNTREND":
        weights.trend_following = Math.max(weights.trend_following, 0.3);
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.3);
        weights.momentum = Math.min(weights.momentum, 0.25);
        weights.breakout = Math.min(weights.breakout, 0.15);
        break;
      case "SIDEWAYS":
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.4);
        weights.breakout = Math.max(weights.breakout, 0.3);
        weights.momentum = Math.min(weights.momentum, 0.2);
        weights.trend_following = Math.min(weights.trend_following, 0.1);
        break;
    }

    // Momentum-based adjustments with enhanced logic
    switch (momentumRegime) {
      case "STRONG_MOMENTUM":
        weights.momentum = Math.max(weights.momentum, 0.4);
        weights.trend_following = Math.max(weights.trend_following, 0.3);
        weights.breakout = Math.min(weights.breakout, 0.2);
        weights.mean_reversion = Math.min(weights.mean_reversion, 0.1);
        break;
      case "MOMENTUM":
        weights.momentum = Math.max(weights.momentum, 0.35);
        weights.trend_following = Math.max(weights.trend_following, 0.25);
        weights.breakout = Math.min(weights.breakout, 0.25);
        weights.mean_reversion = Math.min(weights.mean_reversion, 0.15);
        break;
      case "STRONG_REVERSAL":
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.4);
        weights.breakout = Math.max(weights.breakout, 0.3);
        weights.momentum = Math.min(weights.momentum, 0.2);
        weights.trend_following = Math.min(weights.trend_following, 0.1);
        break;
      case "REVERSAL":
        weights.mean_reversion = Math.max(weights.mean_reversion, 0.35);
        weights.breakout = Math.max(weights.breakout, 0.25);
        weights.momentum = Math.min(weights.momentum, 0.25);
        weights.trend_following = Math.min(weights.trend_following, 0.15);
        break;
      case "NEUTRAL":
        // Keep current weights, no adjustment needed
        break;
    }

    // Normalize weights to sum to 1
    const totalWeight = Object.values(weights).reduce(
      (sum, weight) => sum + weight,
      0
    );
    Object.keys(weights).forEach((key) => {
      weights[key as keyof StrategyWeights] /= totalWeight;
    });

    return weights;
  }

  private selectStrategyByRegime(
    marketRegime: MarketRegime,
    strategyWeights: StrategyWeights
  ): StrategyType {
    // Enhanced strategy selection with more sophisticated logic

    // High confidence regime (clear market conditions)
    if (marketRegime.regimeScore > 75) {
      // Use weighted random selection with bias toward highest weight
      const maxWeight = Math.max(...Object.values(strategyWeights));
      const maxWeightStrategy = Object.entries(strategyWeights).find(
        ([_, weight]) => weight === maxWeight
      )?.[0] as keyof StrategyWeights;

      // 70% chance to select highest weight strategy, 30% chance for weighted random
      if (Math.random() < 0.7) {
        return this.getStrategyTypeFromKey(maxWeightStrategy);
      }

      // Weighted random selection
      const random = Math.random();
      let cumulativeWeight = 0;

      if (random < (cumulativeWeight += strategyWeights.momentum)) {
        return StrategyType.Momentum;
      }
      if (random < (cumulativeWeight += strategyWeights.mean_reversion)) {
        return StrategyType.MeanReversion;
      }
      if (random < (cumulativeWeight += strategyWeights.breakout)) {
        return StrategyType.Breakout;
      }
      return StrategyType.TrendFollowing;
    }

    // Medium confidence regime (moderate market conditions)
    else if (marketRegime.regimeScore > 50) {
      // Use weighted random selection
      const random = Math.random();
      let cumulativeWeight = 0;

      if (random < (cumulativeWeight += strategyWeights.momentum)) {
        return StrategyType.Momentum;
      }
      if (random < (cumulativeWeight += strategyWeights.mean_reversion)) {
        return StrategyType.MeanReversion;
      }
      if (random < (cumulativeWeight += strategyWeights.breakout)) {
        return StrategyType.Breakout;
      }
      return StrategyType.TrendFollowing;
    }

    // Low confidence regime (unclear market conditions)
    else if (marketRegime.regimeScore > 25) {
      // Prefer mean reversion and trend following for stability
      const stableWeights = {
        momentum: strategyWeights.momentum * 0.5,
        mean_reversion: strategyWeights.mean_reversion * 1.5,
        breakout: strategyWeights.breakout * 0.3,
        trend_following: strategyWeights.trend_following * 1.2,
      };

      // Normalize stable weights
      const totalWeight = Object.values(stableWeights).reduce(
        (sum, w) => sum + w,
        0
      );
      Object.keys(stableWeights).forEach((key) => {
        stableWeights[key as keyof StrategyWeights] /= totalWeight;
      });

      const random = Math.random();
      let cumulativeWeight = 0;

      if (random < (cumulativeWeight += stableWeights.momentum)) {
        return StrategyType.Momentum;
      }
      if (random < (cumulativeWeight += stableWeights.mean_reversion)) {
        return StrategyType.MeanReversion;
      }
      if (random < (cumulativeWeight += stableWeights.breakout)) {
        return StrategyType.Breakout;
      }
      return StrategyType.TrendFollowing;
    }

    // Very low confidence regime (highly uncertain market)
    else {
      // Default to mean reversion for stability
      return StrategyType.MeanReversion;
    }
  }

  private getStrategyTypeFromKey(key: keyof StrategyWeights): StrategyType {
    switch (key) {
      case "momentum":
        return StrategyType.Momentum;
      case "mean_reversion":
        return StrategyType.MeanReversion;
      case "breakout":
        return StrategyType.Breakout;
      case "trend_following":
        return StrategyType.TrendFollowing;
      default:
        return StrategyType.MeanReversion;
    }
  }

  private adaptStrategyParameters(
    marketRegime: MarketRegime,
    strategyWeights: StrategyWeights
  ): void {
    const { volatilityRegime, trendRegime, momentumRegime } = marketRegime;

    // Adapt position sizing based on volatility
    let positionMultiplier = 1.0;
    switch (volatilityRegime) {
      case "EXTREME_HIGH":
        positionMultiplier = 0.7; // Reduce position size in extreme volatility
        break;
      case "HIGH":
        positionMultiplier = 0.85;
        break;
      case "VERY_LOW":
        positionMultiplier = 1.3; // Increase position size in low volatility
        break;
    }

    // Adapt confidence thresholds based on regime
    let confidenceMultiplier = 1.0;
    if (
      momentumRegime === "STRONG_MOMENTUM" ||
      trendRegime === "STRONG_UPTREND"
    ) {
      confidenceMultiplier = 0.9; // Lower confidence threshold in strong trends
    } else if (
      momentumRegime === "STRONG_REVERSAL" ||
      volatilityRegime === "EXTREME_HIGH"
    ) {
      confidenceMultiplier = 1.2; // Higher confidence threshold in uncertain conditions
    }

    // Apply adaptations
    this.minConfidence *= confidenceMultiplier;
    this.basePositionSize *= positionMultiplier;
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

  // Removed unused methods for market regime detection

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
    // Use market regime detection for strategy selection
    const dayIndex = prices.length - 1;
    const currentPrice = prices[dayIndex];

    const marketRegime = this.detectMarketRegime(
      prices,
      volumes,
      dayIndex,
      currentPrice
    );
    const strategyWeights = this.calculateStrategyWeights(marketRegime);

    // Adapt strategy parameters based on market regime
    this.adaptStrategyParameters(marketRegime, strategyWeights);

    // Select strategy using regime-based approach
    const selectedStrategy = this.selectStrategyByRegime(
      marketRegime,
      strategyWeights
    );

    // Log regime information for debugging
    console.log(
      `Market Regime: Volatility=${marketRegime.volatilityRegime}, Trend=${marketRegime.trendRegime}, Momentum=${marketRegime.momentumRegime}, Score=${marketRegime.regimeScore}`
    );
    console.log(
      `Strategy Weights: Momentum=${strategyWeights.momentum.toFixed(
        2
      )}, MeanReversion=${strategyWeights.mean_reversion.toFixed(
        2
      )}, Breakout=${strategyWeights.breakout.toFixed(
        2
      )}, TrendFollowing=${strategyWeights.trend_following.toFixed(2)}`
    );
    console.log(`Selected Strategy: ${selectedStrategy}`);

    // Update current strategy if changed
    if (selectedStrategy !== this.currentStrategy) {
      this.currentStrategy = selectedStrategy;
      this.strategyTradeCount = 0;
      this.updateStrategyParameters();
    }

    return this.currentStrategy;
  }

  private updateStrategyParameters() {
    // Use default configuration values for all strategies
    this.minHoldDays = STRATEGY_CONFIG.MIN_HOLD_DAYS_DEFAULT;
    this.stopLossMultiplier = STRATEGY_CONFIG.STOP_LOSS_MULTIPLIER_DEFAULT;
    this.trailingStop = STRATEGY_CONFIG.TRAILING_STOP_DEFAULT;
    this.profitTakeMultiplier = STRATEGY_CONFIG.PROFIT_TAKE_MULTIPLIER_DEFAULT;
    this.minConfidence = STRATEGY_CONFIG.MIN_CONFIDENCE_DEFAULT;
    this.buyProbThreshold = STRATEGY_CONFIG.BUY_PROB_THRESHOLD_DEFAULT;
  }

  public async decideTrade({
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
    const currentPrice = btcPrices[btcPrices.length - 1];
    const currentVolume = btcVolumes[btcVolumes.length - 1];
    const prediction = await this.predictor.predict(btcPrices, btcVolumes);
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
      btcPrices,
      btcVolumes,
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

    const dynamicProfitTake =
      currentPrice *
      (1 +
        Math.min(
          this.profitTakeMultiplier *
            (momentum > 0.1
              ? STRATEGY_CONFIG.CONFIDENCE_BOOST_MULTIPLIER
              : 1.0),
          STRATEGY_CONFIG.MAX_PROFIT_TAKE
        ));

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
              btcAmount: holdings,
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
        btcPrices,
        btcVolumes,
        dynamicBreakoutThreshold
      );

      // Trade quality filtering - much more permissive
      const tradeQuality = buyProb * confidence > 0.01;
      const minProfitPotential = 0.001;
      const profitPotential = (dynamicProfitTake - currentPrice) / currentPrice;
      const hasProfitPotential = profitPotential >= minProfitPotential;

      if (buyConditions && tradeQuality && hasProfitPotential && capital > 0) {
        // Balanced position sizing for optimal returns
        const basePositionSize = this.basePositionSize;

        // Balanced confidence boost
        const confidenceBoost = confidence > 0.52 ? 1.3 : 1.0;

        // Balanced volatility adjustment
        const volatilityAdjustment = atr > 0.05 ? 0.9 : 1.1; // Moderate adjustment

        // Balanced trend-based adjustment
        const trendAdjustment = trendStrength > 0.1 ? 1.2 : 1.0;

        // Balanced momentum-based adjustment
        const momentumAdjustment = momentum > 0.02 ? 1.15 : 1.0;

        // Balanced buy probability boost
        const buyProbBoost = buyProb > 0.52 ? 1.1 : 1.0;

        // Calculate position size with balanced safety checks
        const positionSize = Math.max(
          0.008, // Back to successful setting
          Math.min(
            basePositionSize *
              confidenceBoost *
              volatilityAdjustment *
              trendAdjustment *
              momentumAdjustment *
              buyProbBoost,
            STRATEGY_CONFIG.POSITION_SIZE_MAX
          )
        );

        // Calculate trade amount
        const tradeAmount = Math.min(capital * positionSize, capital);
        const effectivePrice = currentPrice * (1 + this.slippage);

        // Fix: Calculate BTC amount correctly (commission is percentage of trade amount)
        const commissionAmount = tradeAmount * this.commission;
        const netTradeAmount = tradeAmount - commissionAmount;
        const btcBought = netTradeAmount / effectivePrice;

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
          )}, Trade Amount: $${tradeAmount.toFixed(
            2
          )}, BTC Bought: ${btcBought.toFixed(4)}`
        );
        this.strategyTradeCount++;
        return {
          trade: {
            type: Recommendation.Buy,
            price: effectivePrice,
            timestamp: currentTimestamp,
            btcAmount: btcBought,
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
            STRATEGY_CONFIG.VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD &&
          trendStrength > STRATEGY_CONFIG.TREND_STRENGTH_THRESHOLD &&
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
        // Ultra-aggressive dynamic profit-taking with trailing stops
        const momentumProfitTake = momentum > 0.02 ? 0.2 : 0.12; // Increased from 0.15/0.08 for breakthrough returns
        const trailingProfitStop = priceChange > 0.15 ? 0.08 : 0.05; // Trailing profit stop that moves with price
        const profitPotentialExitMomentum =
          priceChange >= momentumProfitTake ||
          (priceChange > 0.1 &&
            priceChange < momentumProfitTake - trailingProfitStop); // Trailing stop logic
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
        // Ultra-aggressive dynamic profit-taking with trailing stops
        const trendProfitTake = trendStrength > 0.1 ? 0.18 : 0.1; // Increased from 0.12/0.06 for breakthrough returns
        const trailingProfitStopMeanRev = priceChange > 0.12 ? 0.06 : 0.04; // Trailing profit stop for mean reversion
        const profitPotentialExitMeanRev =
          priceChange >= trendProfitTake ||
          (priceChange > 0.08 &&
            priceChange < trendProfitTake - trailingProfitStopMeanRev); // Trailing stop logic
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
    this.strategyTradeCount = 0;
    this.updateStrategyParameters();
    console.log(`Strategy manually set to: ${strategyType}`);
  }

  public getCurrentStrategy(): StrategyType {
    return this.currentStrategy;
  }
}
