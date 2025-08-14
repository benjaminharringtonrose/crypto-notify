import { Indicators } from "../types";
import { MODEL_CONFIG, PERIODS } from "../constants";

interface ComputeParams {
  prices: number[];
  volumes: number[];
  dayIndex: number;
  currentPrice: number;
  isBTC: boolean;
}

interface CalculateIndicatorsParams {
  prices: number[];
  volumes: number[];
  dayIndex: number;
  currentPrice: number;
}

export default class FeatureCalculator {
  public calculateVWAP(
    prices: number[],
    volumes: number[],
    period: number = PERIODS.VWAP
  ): number {
    if (prices.length < period || volumes.length < period)
      return prices[prices.length - 1];
    const recentPrices = prices.slice(-period);
    const recentVolumes = volumes.slice(-period);
    const priceVolumeSum = recentPrices.reduce(
      (sum, price, i) => sum + price * recentVolumes[i],
      0
    );
    const volumeSum = recentVolumes.reduce((sum, vol) => sum + vol, 0);
    return volumeSum > 0
      ? priceVolumeSum / volumeSum
      : prices[prices.length - 1];
  }

  public calculateATR(prices: number[], period: number = PERIODS.ATR): number {
    if (prices.length < period + 1) return 0;
    const trueRanges = [];
    for (let i = 1; i < prices.length; i++) {
      trueRanges.push(Math.abs(prices[i] - prices[i - 1]));
    }
    const recentRanges = trueRanges.slice(-period);
    return recentRanges.reduce((sum, range) => sum + range, 0) / period;
  }

  public calculateADX(
    high: number[],
    low: number[],
    close: number[],
    period: number = PERIODS.ADX
  ): number {
    if (
      high.length < period + 1 ||
      low.length < period + 1 ||
      close.length < period + 1
    )
      return 0;
    let plusDM = 0,
      minusDM = 0,
      trSum = 0;
    for (let i = 1; i < high.length; i++) {
      const upMove = high[i] - high[i - 1];
      const downMove = low[i - 1] - low[i];
      plusDM += upMove > downMove && upMove > 0 ? upMove : 0;
      minusDM += downMove > upMove && downMove > 0 ? downMove : 0;
      trSum += Math.max(
        high[i] - low[i],
        Math.abs(high[i] - close[i - 1]),
        Math.abs(low[i] - close[i - 1])
      );
    }
    const plusDI = (plusDM / trSum) * 100;
    const minusDI = (minusDM / trSum) * 100;
    const dx = (Math.abs(plusDI - minusDI) / (plusDI + minusDI)) * 100;
    return dx || 0;
  }

  public calculateSlice(data: number[], periods: number, offset = 0): number[] {
    return data.slice(-periods - offset, offset === 0 ? undefined : -offset);
  }

  public calculateRSI(prices: number[], period = PERIODS.RSI): number {
    if (prices.length < period + 1) return 50;
    let gains = 0,
      losses = 0;
    for (let i = 1; i <= period; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses += Math.abs(change);
    }
    let avgGain = gains / period;
    let avgLoss = losses / period;
    for (let i = period + 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      const gain = change > 0 ? change : 0;
      const loss = change < 0 ? Math.abs(change) : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    const rs = avgLoss > 0 ? avgGain / avgLoss : Infinity;
    return rs === Infinity ? 100 : 100 - 100 / (1 + rs);
  }

  public calculateEMA(prices: number[], period: number): number {
    if (prices.length === 0) return 0;
    const alpha = 2 / (period + 1);
    let ema = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema = alpha * prices[i] + (1 - alpha) * ema;
    }
    return ema;
  }

  // Market Regime Detection Methods (using existing features)
  public calculateRealizedVolatility(
    prices: number[],
    period: number = 20
  ): number {
    if (prices.length < period + 1) return 0;

    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }

    const recentReturns = returns.slice(-period);
    const meanReturn =
      recentReturns.reduce((sum, ret) => sum + ret, 0) / period;
    const variance =
      recentReturns.reduce(
        (sum, ret) => sum + Math.pow(ret - meanReturn, 2),
        0
      ) / period;

    return Math.sqrt(variance * 252); // Annualized volatility
  }

  // Regime Score Conversion Methods
  private volatilityRegimeToScore(regime: string): number {
    switch (regime) {
      case "EXTREME_HIGH":
        return 5;
      case "HIGH":
        return 4;
      case "MEDIUM":
        return 3;
      case "LOW":
        return 2;
      case "VERY_LOW":
        return 1;
      default:
        return 3;
    }
  }

  private trendRegimeToScore(regime: string): number {
    switch (regime) {
      case "STRONG_UPTREND":
        return 7;
      case "UPTREND":
        return 6;
      case "WEAK_UPTREND":
        return 5;
      case "SIDEWAYS":
        return 4;
      case "WEAK_DOWNTREND":
        return 3;
      case "DOWNTREND":
        return 2;
      case "STRONG_DOWNTREND":
        return 1;
      default:
        return 4;
    }
  }

  private momentumRegimeToScore(regime: string): number {
    switch (regime) {
      case "STRONG_MOMENTUM":
        return 5;
      case "MOMENTUM":
        return 4;
      case "NEUTRAL":
        return 3;
      case "REVERSAL":
        return 2;
      case "STRONG_REVERSAL":
        return 1;
      default:
        return 3;
    }
  }

  // Market Regime Detection Methods
  public classifyVolatilityRegime(realizedVolatility: number): string {
    // More nuanced volatility classification with better thresholds
    if (realizedVolatility > 1.0) return "EXTREME_HIGH";
    if (realizedVolatility > 0.6) return "HIGH";
    if (realizedVolatility > 0.35) return "MEDIUM";
    if (realizedVolatility > 0.2) return "LOW";
    return "VERY_LOW";
  }

  public classifyTrendRegime(
    price: number,
    sma20: number,
    sma50: number,
    sma200: number
  ): string {
    // Enhanced trend classification with percentage-based thresholds
    const shortTrend = (price - sma20) / sma20;
    const mediumTrend = (sma20 - sma50) / sma50;
    const longTrend = (sma50 - sma200) / sma200;

    // Strong trends require significant deviation
    if (shortTrend > 0.05 && mediumTrend > 0.03 && longTrend > 0.02) {
      return "STRONG_UPTREND";
    }
    if (shortTrend < -0.05 && mediumTrend < -0.03 && longTrend < -0.02) {
      return "STRONG_DOWNTREND";
    }

    // Moderate trends
    if (shortTrend > 0.02 && mediumTrend > 0.01) return "UPTREND";
    if (shortTrend < -0.02 && mediumTrend < -0.01) return "DOWNTREND";

    // Weak trends
    if (shortTrend > 0.005) return "WEAK_UPTREND";
    if (shortTrend < -0.005) return "WEAK_DOWNTREND";

    return "SIDEWAYS";
  }

  public classifyMomentumRegime(
    rsi: number,
    momentum: number,
    macdLine: number,
    signalLine: number
  ): string {
    // Enhanced momentum classification with more nuanced logic
    const rsiMomentum =
      rsi > 75 ? "OVERBOUGHT" : rsi < 25 ? "OVERSOLD" : "NEUTRAL";
    const priceMomentum =
      momentum > 0.01
        ? "STRONG_POSITIVE"
        : momentum > 0
        ? "POSITIVE"
        : momentum < -0.01
        ? "STRONG_NEGATIVE"
        : "NEGATIVE";
    const macdMomentum =
      macdLine > signalLine * 1.1
        ? "STRONG_POSITIVE"
        : macdLine > signalLine
        ? "POSITIVE"
        : macdLine < signalLine * 0.9
        ? "STRONG_NEGATIVE"
        : "NEGATIVE";

    // Strong momentum combinations
    if (
      rsiMomentum === "NEUTRAL" &&
      priceMomentum === "STRONG_POSITIVE" &&
      macdMomentum === "STRONG_POSITIVE"
    ) {
      return "STRONG_MOMENTUM";
    }
    if (
      rsiMomentum === "NEUTRAL" &&
      priceMomentum === "STRONG_NEGATIVE" &&
      macdMomentum === "STRONG_NEGATIVE"
    ) {
      return "STRONG_REVERSAL";
    }

    // Moderate momentum
    if (priceMomentum === "POSITIVE" && macdMomentum === "POSITIVE") {
      return "MOMENTUM";
    }
    if (priceMomentum === "NEGATIVE" && macdMomentum === "NEGATIVE") {
      return "REVERSAL";
    }

    // RSI-based momentum
    if (rsiMomentum === "OVERBOUGHT" && priceMomentum === "NEGATIVE") {
      return "REVERSAL";
    }
    if (rsiMomentum === "OVERSOLD" && priceMomentum === "POSITIVE") {
      return "MOMENTUM";
    }

    return "NEUTRAL";
  }

  public calculateMarketRegimeFeatures(
    prices: number[],
    volumes: number[],
    dayIndex: number,
    currentPrice: number
  ): {
    volatilityRegime: string;
    trendRegime: string;
    momentumRegime: string;
    realizedVolatility: number;
    regimeScore: number;
  } {
    const realizedVolatility = this.calculateRealizedVolatility(
      prices.slice(0, dayIndex + 1),
      20
    );

    const volatilityRegime = this.classifyVolatilityRegime(realizedVolatility);

    const sma20 = this.calculateSMA(
      prices.slice(Math.max(0, dayIndex - 19), dayIndex + 1)
    );
    const sma50 = this.calculateSMA(
      prices.slice(Math.max(0, dayIndex - 49), dayIndex + 1)
    );
    const sma200 = this.calculateSMA(
      prices.slice(Math.max(0, dayIndex - 199), dayIndex + 1)
    );

    const trendRegime = this.classifyTrendRegime(
      currentPrice,
      sma20,
      sma50,
      sma200
    );

    const rsi = this.calculateRSI(prices.slice(0, dayIndex + 1));
    const momentum = dayIndex >= 10 ? currentPrice - prices[dayIndex - 10] : 0;
    const ema12 = this.calculateEMA(
      prices.slice(Math.max(0, dayIndex - 11), dayIndex + 1),
      12
    );
    const ema26 = this.calculateEMA(
      prices.slice(Math.max(0, dayIndex - 25), dayIndex + 1),
      26
    );
    const macdLine = ema12 - ema26;
    const signalLine = this.calculateEMA(
      prices.slice(Math.max(0, dayIndex - 8), dayIndex + 1).map((_, i) => {
        const slice = prices.slice(
          Math.max(0, dayIndex - 25 - i),
          dayIndex + 1 - i
        );
        return (
          this.calculateEMA(slice.slice(-12), 12) - this.calculateEMA(slice, 26)
        );
      }),
      9
    );

    const momentumRegime = this.classifyMomentumRegime(
      rsi,
      momentum,
      macdLine,
      signalLine
    );

    // Calculate regime score (0-100) for strategy weighting
    let regimeScore = 50; // Neutral base

    // Volatility adjustment
    if (volatilityRegime === "EXTREME_HIGH") regimeScore += 20;
    else if (volatilityRegime === "HIGH") regimeScore += 10;
    else if (volatilityRegime === "VERY_LOW") regimeScore -= 10;

    // Trend adjustment
    if (trendRegime === "STRONG_UPTREND") regimeScore += 15;
    else if (trendRegime === "UPTREND") regimeScore += 8;
    else if (trendRegime === "STRONG_DOWNTREND") regimeScore -= 15;
    else if (trendRegime === "DOWNTREND") regimeScore -= 8;

    // Momentum adjustment
    if (momentumRegime === "STRONG_MOMENTUM") regimeScore += 15;
    else if (momentumRegime === "MOMENTUM") regimeScore += 8;
    else if (momentumRegime === "STRONG_REVERSAL") regimeScore -= 15;
    else if (momentumRegime === "REVERSAL") regimeScore -= 8;

    return {
      volatilityRegime,
      trendRegime,
      momentumRegime,
      realizedVolatility,
      regimeScore: Math.max(0, Math.min(100, regimeScore)),
    };
  }

  public calculateSMA(prices: number[]): number {
    return prices.length > 0
      ? prices.reduce((sum, price) => sum + price, 0) / prices.length
      : 0;
  }

  public calculateStdDev(values: number[], mean: number): number {
    if (values.length < 1) return 0;
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      values.length;
    return Math.sqrt(variance);
  }

  public detectTripleBottom(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean {
    if (prices.length < 20 || volumes.length < 20) return false;
    const recentPrices = prices.slice(-20);
    const recentVolumes = volumes.slice(-20);
    const valleys = [];
    for (let i = 1; i < recentPrices.length - 1; i++) {
      if (
        recentPrices[i] < recentPrices[i - 1] &&
        recentPrices[i] < recentPrices[i + 1]
      ) {
        valleys.push({
          price: recentPrices[i],
          volume: recentVolumes[i],
          index: prices.length - 20 + i,
        });
      }
    }
    if (valleys.length < 3) return false;
    const lastThreeValleys = valleys.slice(-3);
    const priceRange = Math.max(...recentPrices) - Math.min(...recentPrices);
    const areSimilar = lastThreeValleys.every(
      (v) => Math.abs(v.price - lastThreeValleys[0].price) < priceRange * 0.05
    );
    const volumeDecreasing =
      lastThreeValleys[1].volume < lastThreeValleys[0].volume &&
      lastThreeValleys[2].volume < lastThreeValleys[1].volume;
    const avgValleyVolume =
      lastThreeValleys.reduce((sum, v) => sum + v.volume, 0) / 3;
    const breakoutVolume = recentVolumes[recentVolumes.length - 1];
    const isVolumeSpike = breakoutVolume > avgValleyVolume * 1.5;
    const recentHigh = Math.max(...recentPrices.slice(-5));
    const isBreakingUp = currentPrice > recentHigh;
    return areSimilar && volumeDecreasing && isVolumeSpike && isBreakingUp;
  }

  public calculateFibonacciLevels(
    prices: number[],
    period: number = PERIODS.FIBONACCI
  ): { levels: number[]; high: number; low: number } {
    if (prices.length < period) {
      return {
        levels: [0, 0, 0, 0, 0],
        high: prices[prices.length - 1] || 0,
        low: prices[0] || 0,
      };
    }
    const recentPrices = prices.slice(-period);
    const high = Math.max(...recentPrices);
    const low = Math.min(...recentPrices);
    const range = high - low;
    const levels = [
      low + range * 0.236,
      low + range * 0.382,
      low + range * 0.5,
      low + range * 0.618,
      high,
    ];
    return { levels, high, low };
  }

  public detectHeadAndShoulders(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean {
    let leftShoulder = 0,
      head = 0,
      rightShoulder = 0;
    let leftTrough = Infinity,
      rightTrough = Infinity;
    let leftShoulderIndex = -1,
      headIndex = -1,
      rightShoulderIndex = -1;
    for (let i = 1; i < prices.length - 1; i++) {
      if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
        if (leftShoulder === 0) {
          leftShoulder = prices[i];
          leftShoulderIndex = i;
        } else if (
          prices[i] > leftShoulder &&
          head === 0 &&
          i > leftShoulderIndex + 2
        ) {
          head = prices[i];
          headIndex = i;
        } else if (
          head !== 0 &&
          prices[i] < head &&
          Math.abs(prices[i] - leftShoulder) / leftShoulder < 0.1 &&
          i > headIndex + 2
        ) {
          rightShoulder = prices[i];
          rightShoulderIndex = i;
          break;
        }
      }
    }
    if (rightShoulder !== 0) {
      for (let i = leftShoulderIndex + 1; i < headIndex; i++) {
        if (prices[i] < leftTrough) leftTrough = prices[i];
      }
      for (let i = headIndex + 1; i < rightShoulderIndex; i++) {
        if (prices[i] < rightTrough) rightTrough = prices[i];
      }
    }
    const neckline = Math.min(leftTrough, rightTrough);
    const avgVolume =
      volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
    const headVolume = headIndex >= 0 ? volumes[headIndex] : 0;
    const rightShoulderVolume =
      rightShoulderIndex >= 0 ? volumes[rightShoulderIndex] : 0;
    return (
      rightShoulder !== 0 &&
      leftTrough !== Infinity &&
      rightTrough !== Infinity &&
      currentPrice < neckline &&
      (headVolume > avgVolume * 1.5 || rightShoulderVolume < headVolume * 0.8)
    );
  }

  public calculateStochRSI(
    prices: number[],
    rsiPeriod: number = PERIODS.STOCH_RSI,
    stochPeriod: number = PERIODS.STOCH_RSI,
    smoothPeriod: number = PERIODS.STOCH_SMOOTH
  ): { stochRsi: number; signal: number } {
    if (prices.length < rsiPeriod + stochPeriod)
      return { stochRsi: 50, signal: 50 };
    const rsiValues: number[] = [];
    for (let i = 0; i <= prices.length - rsiPeriod; i++) {
      const slice = prices.slice(i, i + rsiPeriod + 1);
      const rsi = this.calculateRSI(slice);
      rsiValues.push(rsi);
    }
    const recentRsis = rsiValues.slice(-stochPeriod);
    const currentRsi = recentRsis[recentRsis.length - 1];
    const lowestRsi = Math.min(...recentRsis);
    const highestRsi = Math.max(...recentRsis);
    const stochRsi =
      highestRsi === lowestRsi
        ? 50
        : ((currentRsi - lowestRsi) / (highestRsi - lowestRsi)) * 100;
    const recentStochRsis = rsiValues
      .slice(-stochPeriod - smoothPeriod + 1)
      .map((_, i) => {
        const subRsis = rsiValues.slice(i, i + stochPeriod);
        const subLow = Math.min(...subRsis);
        const subHigh = Math.max(...subRsis);
        return subHigh === subLow
          ? 50
          : ((subRsis[subRsis.length - 1] - subLow) / (subHigh - subLow)) * 100;
      });
    const signal =
      recentStochRsis.slice(-smoothPeriod).reduce((sum, val) => sum + val, 0) /
      smoothPeriod;
    return { stochRsi, signal };
  }

  public detectDoubleTop(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean {
    let firstTop = 0,
      secondTop = 0,
      trough = Infinity;
    let firstTopIndex = -1,
      secondTopIndex = -1;
    for (let i = 1; i < prices.length - 1; i++) {
      if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
        if (firstTop === 0) {
          firstTop = prices[i];
          firstTopIndex = i;
        } else if (
          Math.abs(prices[i] - firstTop) / firstTop < 0.05 &&
          i > firstTopIndex + 2
        ) {
          secondTop = prices[i];
          secondTopIndex = i;
          break;
        }
      }
    }
    if (secondTop !== 0) {
      for (let i = firstTopIndex + 1; i < secondTopIndex; i++) {
        if (prices[i] < trough) trough = prices[i];
      }
    }
    const avgVolume =
      volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
    const secondTopVolume = secondTopIndex >= 0 ? volumes[secondTopIndex] : 0;
    const postTopVolume =
      secondTopIndex + 1 < volumes.length
        ? volumes[secondTopIndex + 1]
        : volumes[volumes.length - 1];
    return (
      secondTop !== 0 &&
      trough !== Infinity &&
      currentPrice < trough &&
      (secondTopVolume > avgVolume * 2 || postTopVolume < secondTopVolume * 0.8)
    );
  }

  public detectTripleTop(
    prices: number[],
    volumes: number[],
    currentPrice: number
  ): boolean {
    let firstTop = 0,
      secondTop = 0,
      thirdTop = 0;
    let firstTrough = Infinity,
      secondTrough = Infinity;
    let firstTopIndex = -1,
      secondTopIndex = -1,
      thirdTopIndex = -1;
    for (let i = 1; i < prices.length - 1; i++) {
      if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
        if (firstTop === 0) {
          firstTop = prices[i];
          firstTopIndex = i;
        } else if (
          Math.abs(prices[i] - firstTop) / firstTop < 0.05 &&
          i > firstTopIndex + 2 &&
          secondTop === 0
        ) {
          secondTop = prices[i];
          secondTopIndex = i;
        } else if (
          secondTop !== 0 &&
          Math.abs(prices[i] - secondTop) / secondTop < 0.05 &&
          i > secondTopIndex + 2
        ) {
          thirdTop = prices[i];
          thirdTopIndex = i;
          break;
        }
      }
    }
    if (thirdTop !== 0) {
      for (let i = firstTopIndex + 1; i < secondTopIndex; i++) {
        if (prices[i] < firstTrough) firstTrough = prices[i];
      }
      for (let i = secondTopIndex + 1; i < thirdTopIndex; i++) {
        if (prices[i] < secondTrough) secondTrough = prices[i];
      }
    }
    const supportLevel = Math.min(firstTrough, secondTrough);
    const avgVolume =
      volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
    const thirdTopVolume = thirdTopIndex >= 0 ? volumes[thirdTopIndex] : 0;
    const postTopVolume =
      thirdTopIndex + 1 < volumes.length
        ? volumes[thirdTopIndex + 1]
        : volumes[volumes.length - 1];
    return (
      thirdTop !== 0 &&
      firstTrough !== Infinity &&
      secondTrough !== Infinity &&
      currentPrice < supportLevel &&
      (thirdTopVolume > avgVolume * 2 || postTopVolume < thirdTopVolume * 0.8)
    );
  }

  public calculateIndicators({
    prices,
    volumes,
    dayIndex,
    currentPrice,
  }: CalculateIndicatorsParams): Indicators {
    const rsi = this.calculateRSI(this.calculateSlice(prices, PERIODS.RSI + 1));
    const prevRsi = this.calculateRSI(
      this.calculateSlice(prices, PERIODS.RSI + 1, 1)
    );
    const sma7 = this.calculateSMA(
      this.calculateSlice(prices, PERIODS.SMA_SHORT)
    );
    const sma21 = this.calculateSMA(
      this.calculateSlice(prices, PERIODS.SMA_LONG)
    );
    const prevSma7 = this.calculateSMA(
      this.calculateSlice(prices, PERIODS.SMA_SHORT, 1)
    );
    const prevSma21 = this.calculateSMA(
      this.calculateSlice(prices, PERIODS.SMA_LONG, 1)
    );
    const ema12 = this.calculateEMA(
      this.calculateSlice(prices, PERIODS.EMA_SHORT),
      PERIODS.EMA_SHORT
    );
    const ema26 = this.calculateEMA(
      this.calculateSlice(prices, PERIODS.EMA_LONG),
      PERIODS.EMA_LONG
    );
    const macdLine = ema12 - ema26;
    const prevEma12 = this.calculateEMA(
      this.calculateSlice(prices, PERIODS.EMA_SHORT, 1),
      PERIODS.EMA_SHORT
    );
    const prevEma26 = this.calculateEMA(
      this.calculateSlice(prices, PERIODS.EMA_LONG, 1),
      PERIODS.EMA_LONG
    );
    const prevMacdLine = prevEma12 - prevEma26;
    const signalLine = this.calculateEMA(
      prices.slice(-PERIODS.MACD_SIGNAL).map((_, i) => {
        const slice = this.calculateSlice(prices, PERIODS.EMA_LONG + i, i);
        return (
          this.calculateEMA(
            slice.slice(-PERIODS.EMA_SHORT),
            PERIODS.EMA_SHORT
          ) - this.calculateEMA(slice, PERIODS.EMA_LONG)
        );
      }),
      PERIODS.MACD_SIGNAL
    );
    const sma20 = this.calculateSMA(
      this.calculateSlice(prices, PERIODS.SMA_MEDIUM)
    );
    const stdDev20 = this.calculateStdDev(
      this.calculateSlice(prices, PERIODS.SMA_MEDIUM),
      sma20
    );
    const upperBand = sma20 + 2 * stdDev20;
    const lowerBand = sma20 - 2 * stdDev20;
    const obvValues = [0];
    let obv = 0;
    for (let i = 1; i <= dayIndex && i < prices.length; i++) {
      const priceChange = prices[i] - prices[i - 1];
      obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
      obvValues.push(obv);
    }
    const atr = this.calculateATR(prices.slice(0, dayIndex + 1));
    const atrBaseline = this.calculateATR(
      prices.slice(0, Math.max(0, dayIndex))
    );
    const zScore =
      dayIndex >= PERIODS.SMA_MEDIUM ? (currentPrice - sma20) / stdDev20 : 0;
    const vwap = this.calculateVWAP(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1)
    );
    const { stochRsi, signal: stochRsiSignal } = this.calculateStochRSI(
      prices.slice(0, dayIndex + 1)
    );
    const prevStochRsi =
      dayIndex > 0
        ? this.calculateStochRSI(prices.slice(0, dayIndex)).stochRsi
        : 0;
    const { levels } = this.calculateFibonacciLevels(
      prices.slice(0, dayIndex + 1)
    );
    const fib61_8 = levels[3] || 0;
    const volSmaShort = this.calculateSMA(
      this.calculateSlice(volumes, PERIODS.VOL_SMA_SHORT)
    );
    const volSmaLong = this.calculateSMA(
      this.calculateSlice(volumes, PERIODS.VOL_SMA_LONG)
    );
    const volumeOscillator =
      volSmaLong !== 0 ? ((volSmaShort - volSmaLong) / volSmaLong) * 100 : 0;
    const prevVolSmaShort = this.calculateSMA(
      this.calculateSlice(volumes, PERIODS.VOL_SMA_SHORT, 1)
    );
    const prevVolSmaLong = this.calculateSMA(
      this.calculateSlice(volumes, PERIODS.VOL_SMA_LONG, 1)
    );
    const prevVolumeOscillator =
      prevVolSmaLong !== 0
        ? ((prevVolSmaShort - prevVolSmaLong) / prevVolSmaLong) * 100
        : 0;
    const currentVolume = dayIndex >= 0 ? volumes[dayIndex] : 0;
    const volSma5 = this.calculateSMA(
      this.calculateSlice(volumes, PERIODS.VOL_SMA_SHORT)
    );
    const isVolumeSpike = currentVolume > volSma5 * 2;
    const isDoubleTop = this.detectDoubleTop(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1),
      currentPrice
    );
    const isHeadAndShoulders = this.detectHeadAndShoulders(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1),
      currentPrice
    );
    const isTripleTop = this.detectTripleTop(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1),
      currentPrice
    );
    const isTripleBottom = this.detectTripleBottom(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1),
      currentPrice
    );
    const momentum =
      dayIndex >= PERIODS.MOMENTUM
        ? currentPrice - prices[dayIndex - PERIODS.MOMENTUM]
        : 0;
    const priceChangePct =
      dayIndex >= 1
        ? ((currentPrice - prices[dayIndex - 1]) / prices[dayIndex - 1]) * 100
        : 0;
    const volAdjustedMomentum =
      dayIndex >= PERIODS.MOMENTUM && atr !== 0
        ? (currentPrice - prices[dayIndex - PERIODS.MOMENTUM]) / atr
        : 0;
    const sma50 =
      dayIndex >= PERIODS.SMA_50
        ? this.calculateSMA(
            prices.slice(dayIndex - PERIODS.SMA_50 + 1, dayIndex + 1)
          )
        : sma20;
    const sma200 =
      dayIndex >= PERIODS.SMA_200
        ? this.calculateSMA(
            prices.slice(dayIndex - PERIODS.SMA_200 + 1, dayIndex + 1)
          )
        : sma20;
    const trendRegime = sma50 !== 0 ? (sma50 - sma200) / sma200 : 0;
    const adxProxy = this.calculateADX(
      prices.slice(0, dayIndex + 1),
      prices.slice(0, dayIndex + 1),
      prices.slice(0, dayIndex + 1)
    );

    // Market Regime Features (using existing features)
    const marketRegimeFeatures = this.calculateMarketRegimeFeatures(
      prices,
      volumes,
      dayIndex,
      currentPrice
    );

    return {
      rsi,
      prevRsi,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
      currentPrice,
      upperBand,
      lowerBand,
      obvValues,
      obv,
      atr,
      atrBaseline,
      zScore,
      vwap,
      stochRsi,
      stochRsiSignal,
      prevStochRsi,
      fib61_8,
      volumeOscillator,
      prevVolumeOscillator,
      isDoubleTop,
      isHeadAndShoulders,
      prevMacdLine,
      isTripleTop,
      isTripleBottom,
      isVolumeSpike,
      momentum,
      priceChangePct,
      sma20,
      sma50,
      sma200,
      prices,
      volAdjustedMomentum,
      trendRegime,
      adxProxy,
      // Market Regime Features (converted to numeric)
      volatilityRegimeScore: this.volatilityRegimeToScore(
        marketRegimeFeatures.volatilityRegime
      ),
      trendRegimeScore: this.trendRegimeToScore(
        marketRegimeFeatures.trendRegime
      ),
      momentumRegimeScore: this.momentumRegimeToScore(
        marketRegimeFeatures.momentumRegime
      ),
      realizedVolatility: marketRegimeFeatures.realizedVolatility,
      regimeScore: marketRegimeFeatures.regimeScore,
    };
  }

  public compute({
    prices,
    volumes,
    dayIndex,
    currentPrice,
    isBTC,
  }: ComputeParams): number[] {
    if (!prices || !volumes || dayIndex < 0 || dayIndex >= prices.length) {
      return Array(MODEL_CONFIG.BTC_FEATURE_COUNT).fill(0);
    }

    const indicators = this.calculateIndicators({
      prices,
      volumes,
      dayIndex,
      currentPrice,
    });
    const baseFeatures = [
      indicators.rsi,
      indicators.prevRsi,
      indicators.sma7,
      indicators.sma21,
      indicators.prevSma7,
      indicators.prevSma21,
      indicators.macdLine,
      indicators.signalLine,
      indicators.currentPrice,
      indicators.upperBand,
      indicators.lowerBand,
      indicators.obvValues[indicators.obvValues.length - 1] / 1e6,
      indicators.obv,
      indicators.atr,
      indicators.atrBaseline,
      indicators.zScore,
      indicators.vwap,
      indicators.stochRsi,
      indicators.stochRsiSignal,
      indicators.prevStochRsi,
      indicators.fib61_8,
      dayIndex > 0 ? prices[dayIndex - 1] : prices[0],
      indicators.volumeOscillator,
      indicators.prevVolumeOscillator,
      indicators.isDoubleTop ? 1 : 0,
      indicators.isHeadAndShoulders ? 1 : 0,
      indicators.prevMacdLine,
      indicators.isTripleTop ? 1 : 0,
      indicators.isTripleBottom ? 1 : 0,
      indicators.isVolumeSpike ? 1 : 0,
      indicators.momentum,
      indicators.priceChangePct,
      indicators.sma20,
      indicators.volAdjustedMomentum,
      indicators.trendRegime,
      indicators.adxProxy,
      // Market Regime Features
      indicators.volatilityRegimeScore,
      indicators.trendRegimeScore,
      indicators.momentumRegimeScore,
      indicators.realizedVolatility,
      indicators.regimeScore,
      // Additional technical indicators
      indicators.sma50 || indicators.sma20,
      indicators.sma200 || indicators.sma20,
      // Price-based features
      currentPrice / (indicators.sma20 || currentPrice),
      currentPrice / (indicators.sma50 || currentPrice),
      currentPrice / (indicators.sma200 || currentPrice),
      // Volume-based features
      volumes[dayIndex] /
        (volumes
          .slice(Math.max(0, dayIndex - 20), dayIndex + 1)
          .reduce((a, b) => a + b, 0) /
          Math.min(21, dayIndex + 1)),
      volumes[dayIndex] /
        (volumes
          .slice(Math.max(0, dayIndex - 50), dayIndex + 1)
          .reduce((a, b) => a + b, 0) /
          Math.min(51, dayIndex + 1)),
      // Momentum features
      indicators.momentum / (indicators.atr || 1),
      indicators.priceChangePct / (indicators.atr || 1),
      // RSI-based features
      indicators.rsi / 100,
      indicators.prevRsi / 100,
      // MACD-based features
      indicators.macdLine / (indicators.currentPrice || 1),
      indicators.signalLine / (indicators.currentPrice || 1),
      // Bollinger Bands features
      (indicators.currentPrice - indicators.upperBand) /
        (indicators.upperBand - indicators.lowerBand || 1),
      (indicators.currentPrice - indicators.lowerBand) /
        (indicators.upperBand - indicators.lowerBand || 1),
      // Stochastic features
      indicators.stochRsi / 100,
      indicators.stochRsiSignal / 100,
      // Fibonacci features
      indicators.fib61_8 / (indicators.currentPrice || 1),
      // ATR-based features
      indicators.atr / (indicators.currentPrice || 1),
      indicators.atrBaseline / (indicators.currentPrice || 1),
      // VWAP features
      indicators.vwap / (indicators.currentPrice || 1),
    ];

    return baseFeatures;
  }
}
