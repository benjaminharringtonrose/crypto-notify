import { Indicators, MarketData } from "../types";
import { PERIODS } from "../constants";

export default class FeatureCalculator {
  private prices: number[];
  private volumes: number[];
  private dayIndex: number;
  private currentPrice: number;
  private isBTC: boolean;
  private btcPrice?: number;

  constructor({
    prices,
    volumes,
    dayIndex,
    currentPrice,
    isBTC = false,
    btcPrice,
  }: MarketData & { isBTC?: boolean; btcPrice?: number }) {
    this.prices = prices;
    this.volumes = volumes;
    this.dayIndex = dayIndex;
    this.currentPrice = currentPrice;
    this.isBTC = isBTC;
    this.btcPrice = btcPrice;
  }

  private calculateVWAP(
    prices: number[],
    volumes: number[],
    period: number = 7
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
    return priceVolumeSum / volumeSum;
  }

  private calculateATR(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 0;
    const trueRanges = [];
    for (let i = 1; i < prices.length; i++) {
      trueRanges.push(Math.abs(prices[i] - prices[i - 1]));
    }
    const recentRanges = trueRanges.slice(-period);
    return recentRanges.reduce((sum, range) => sum + range, 0) / period;
  }

  private calculateSlice(
    data: number[],
    periods: number,
    offset = 0
  ): number[] {
    return data.slice(-periods - offset, offset === 0 ? undefined : -offset);
  }

  private calculateRSI(prices: number[], period = 14): number | undefined {
    if (prices.length < period + 1) return undefined;
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

    const rs = avgGain / avgLoss;
    return 100 - 100 / (1 + rs);
  }

  private calculateEMA(prices: number[], period: number): number {
    const k = 2 / (period + 1);
    let ema = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema = prices[i] * k + ema * (1 - k);
    }
    return ema;
  }

  private calculateSMA(prices: number[]): number {
    return prices.reduce((sum, price) => sum + price, 0) / prices.length;
  }

  private calculateStdDev(values: number[], mean: number): number {
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      values.length;
    return Math.sqrt(variance);
  }

  private detectTripleBottom(
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

  private calculateFibonacciLevels(
    prices: number[],
    period: number = 30
  ): { levels: number[]; high: number; low: number } {
    if (prices.length < period)
      return {
        levels: [0, 0, 0, 0, 0],
        high: prices[prices.length - 1],
        low: prices[0],
      };

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

  private detectHeadAndShoulders(
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

  private calculateStochRSI(
    prices: number[],
    rsiPeriod: number = 14,
    stochPeriod: number = 14,
    smoothPeriod: number = 3
  ): { stochRsi: number; signal: number } {
    if (prices.length < rsiPeriod + stochPeriod)
      return { stochRsi: 50, signal: 50 };

    const rsiValues: number[] = [];
    for (let i = 0; i <= prices.length - rsiPeriod; i++) {
      const slice = prices.slice(i, i + rsiPeriod + 1);
      const rsi = this.calculateRSI(slice);
      if (rsi) rsiValues.push(rsi);
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

  private detectDoubleTop(
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

  private detectTripleTop(
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

  public calculateIndicators(): Indicators {
    const rsi = this.calculateRSI(
      this.calculateSlice(this.prices, PERIODS.RSI + 1)
    );
    const prevRsi = this.calculateRSI(
      this.calculateSlice(this.prices, PERIODS.RSI + 1, 1)
    );
    const sma7 = this.calculateSMA(
      this.calculateSlice(this.prices, PERIODS.SMA_SHORT)
    );
    const sma21 = this.calculateSMA(
      this.calculateSlice(this.prices, PERIODS.SMA_LONG)
    );
    const prevSma7 = this.calculateSMA(
      this.calculateSlice(this.prices, PERIODS.SMA_SHORT, 1)
    );
    const prevSma21 = this.calculateSMA(
      this.calculateSlice(this.prices, PERIODS.SMA_LONG, 1)
    );
    const ema12 = this.calculateEMA(
      this.calculateSlice(this.prices, PERIODS.EMA_SHORT),
      PERIODS.EMA_SHORT
    );
    const ema26 = this.calculateEMA(
      this.calculateSlice(this.prices, PERIODS.EMA_LONG),
      PERIODS.EMA_LONG
    );
    const macdLine = ema12 - ema26;
    const prevEma12 = this.calculateEMA(
      this.calculateSlice(this.prices, PERIODS.EMA_SHORT, 1),
      PERIODS.EMA_SHORT
    );
    const prevEma26 = this.calculateEMA(
      this.calculateSlice(this.prices, PERIODS.EMA_LONG, 1),
      PERIODS.EMA_LONG
    );
    const prevMacdLine = prevEma12 - prevEma26;
    const signalLine = this.calculateEMA(
      this.prices.slice(-PERIODS.MACD_SIGNAL).map((_, i) => {
        const slice = this.calculateSlice(this.prices, PERIODS.EMA_LONG + i, i);
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
      this.calculateSlice(this.prices, PERIODS.SMA_MEDIUM)
    );
    const stdDev20 = this.calculateStdDev(
      this.calculateSlice(this.prices, PERIODS.SMA_MEDIUM),
      sma20
    );
    const upperBand = sma20 + 2 * stdDev20;
    const lowerBand = sma20 - 2 * stdDev20;
    const obvValues = [0];
    let obv = 0;
    for (let i = 1; i <= this.dayIndex && i < this.prices.length; i++) {
      const priceChange = this.prices[i] - this.prices[i - 1];
      obv +=
        priceChange > 0
          ? this.volumes[i]
          : priceChange < 0
          ? -this.volumes[i]
          : 0;
      obvValues.push(obv);
    }
    const atr = this.calculateATR(
      this.prices.slice(0, this.dayIndex + 1),
      PERIODS.ATR
    );
    const atrBaseline = this.calculateATR(
      this.prices.slice(0, Math.max(0, this.dayIndex)),
      PERIODS.ATR
    );
    const zScore =
      this.dayIndex >= PERIODS.SMA_MEDIUM
        ? (this.currentPrice - sma20) / stdDev20
        : 0;
    const vwap = this.calculateVWAP(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      PERIODS.VWAP
    );
    const { stochRsi, signal: stochRsiSignal } = this.calculateStochRSI(
      this.prices.slice(0, this.dayIndex + 1),
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_SMOOTH
    );
    const prevStochRsi =
      this.dayIndex > 0
        ? this.calculateStochRSI(
            this.prices.slice(0, this.dayIndex),
            PERIODS.STOCH_RSI,
            PERIODS.STOCH_RSI,
            PERIODS.STOCH_SMOOTH
          ).stochRsi
        : 0;
    const { levels } = this.calculateFibonacciLevels(
      this.prices.slice(0, this.dayIndex + 1),
      PERIODS.FIBONACCI
    );
    const fib61_8 = levels[3] || 0;
    const volSmaShort = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_SHORT)
    );
    const volSmaLong = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_LONG)
    );
    const volumeOscillator =
      volSmaLong !== 0 ? ((volSmaShort - volSmaLong) / volSmaLong) * 100 : 0;
    const prevVolSmaShort = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_SHORT, 1)
    );
    const prevVolSmaLong = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_LONG, 1)
    );
    const prevVolumeOscillator =
      prevVolSmaLong !== 0
        ? ((prevVolSmaShort - prevVolSmaLong) / prevVolSmaLong) * 100
        : 0;
    const currentVolume = this.dayIndex >= 0 ? this.volumes[this.dayIndex] : 0;
    const volSma5 = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_SHORT)
    );
    const isVolumeSpike = currentVolume > volSma5 * 2;
    const isDoubleTop = this.detectDoubleTop(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      this.currentPrice
    );
    const isHeadAndShoulders = this.detectHeadAndShoulders(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      this.currentPrice
    );
    const isTripleTop = this.detectTripleTop(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      this.currentPrice
    );
    const isTripleBottom = this.detectTripleBottom(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      this.currentPrice
    );
    const momentum =
      this.dayIndex >= PERIODS.MOMENTUM
        ? this.currentPrice - this.prices[this.dayIndex - PERIODS.MOMENTUM]
        : 0;
    const priceChangePct =
      this.dayIndex >= 1
        ? ((this.currentPrice - this.prices[this.dayIndex - 1]) /
            this.prices[this.dayIndex - 1]) *
          100
        : 0;
    const volAdjustedMomentum =
      this.dayIndex >= PERIODS.MOMENTUM && atr !== 0
        ? (this.currentPrice - this.prices[this.dayIndex - PERIODS.MOMENTUM]) /
          atr
        : 0;
    const sma50 =
      this.dayIndex >= 49
        ? this.calculateSMA(
            this.prices.slice(this.dayIndex - 49, this.dayIndex + 1)
          )
        : sma20;
    const adxProxy = sma50 !== 0 ? Math.abs((sma7 - sma50) / sma50) : 0;

    return {
      rsi,
      prevRsi,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
      currentPrice: this.currentPrice,
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
      prices: this.prices,
      volAdjustedMomentum,
      adxProxy,
    };
  }

  public compute(): number[] {
    if (
      !this.prices ||
      !this.volumes ||
      this.dayIndex < 0 ||
      this.dayIndex >= this.prices.length
    ) {
      return Array(this.isBTC ? 29 : 32).fill(0);
    }

    const indicators = this.calculateIndicators();
    const baseFeatures = [
      indicators.rsi ?? 0,
      indicators.prevRsi ?? 0,
      indicators.sma7,
      indicators.sma21,
      indicators.prevSma7,
      indicators.prevSma21,
      indicators.macdLine,
      indicators.signalLine,
      indicators.currentPrice,
      indicators.upperBand,
      indicators.obvValues[indicators.obvValues.length - 1] / 1e6,
      indicators.atr,
      indicators.atrBaseline,
      indicators.zScore,
      indicators.vwap,
      indicators.stochRsi,
      indicators.prevStochRsi,
      indicators.fib61_8,
      this.dayIndex > 0 ? this.prices[this.dayIndex - 1] : this.prices[0],
      indicators.volumeOscillator,
      indicators.prevVolumeOscillator,
      indicators.isDoubleTop ? 1 : 0,
      indicators.isHeadAndShoulders ? 1 : 0,
      indicators.prevMacdLine,
      indicators.isTripleTop ? 1 : 0,
      indicators.isVolumeSpike ? 1 : 0,
      indicators.momentum ?? 0,
      indicators.priceChangePct ?? 0,
      indicators.volAdjustedMomentum,
    ];

    return this.isBTC
      ? baseFeatures // 29 features
      : [
          ...baseFeatures,
          indicators.isTripleBottom ? 1 : 0,
          indicators.adxProxy,
          this.btcPrice ? this.currentPrice / this.btcPrice : 0,
        ]; // 32 features
  }
}
