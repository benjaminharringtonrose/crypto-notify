import { MarketData } from "../types";
import { PERIODS } from "../constants";

export class FeatureCalculator {
  private prices: number[];
  private volumes: number[];
  private dayIndex: number;
  private currentPrice: number;
  private isBTC: boolean;

  constructor({
    prices,
    volumes,
    dayIndex,
    currentPrice,
    isBTC = false,
  }: MarketData & { isBTC?: boolean }) {
    this.prices = prices;
    this.volumes = volumes;
    this.dayIndex = dayIndex;
    this.currentPrice = currentPrice;
    this.isBTC = isBTC;
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

  private calculateIfEnoughData<T>(
    minDays: number,
    dayIndex: number,
    calculateFn: () => T
  ): T | number {
    return dayIndex >= minDays - 1 ? calculateFn() || 0 : 0;
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
      if (rsi) {
        rsiValues.push(rsi);
      }
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
    let firstTop = 0;
    let secondTop = 0;
    let trough = Infinity;
    let firstTopIndex = -1;
    let secondTopIndex = -1;

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
        if (prices[i] < trough) {
          trough = prices[i];
        }
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
    let firstTop = 0;
    let secondTop = 0;
    let thirdTop = 0;
    let firstTrough = Infinity;
    let secondTrough = Infinity;
    let firstTopIndex = -1;
    let secondTopIndex = -1;
    let thirdTopIndex = -1;

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

  public calculateIndicators(): {
    rsi: number | undefined;
    prevRsi: number | undefined;
    sma7: number;
    sma21: number;
    prevSma7: number;
    prevSma21: number;
    macdLine: number;
    signalLine: number;
    upperBand: number;
    lowerBand: number;
    obvValues: number[];
    obv: number;
    atr: number;
    atrBaseline: number;
    zScore: number;
    vwap: number;
    stochRsi: number;
    stochRsiSignal: number;
    prevStochRsi: number;
    fib61_8: number;
    volumeOscillator: number;
    prevVolumeOscillator: number;
    isDoubleTop: boolean;
    isHeadAndShoulders: boolean;
    prevMacdLine: number;
    isTripleTop: boolean;
    isTripleBottom: boolean;
    isVolumeSpike: boolean;
    momentum: number;
    priceChangePct: number;
    sma20: number;
  } {
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
    for (let i = 1; i < this.prices.length; i++) {
      const priceChange = this.prices[i] - this.prices[i - 1];
      obv +=
        priceChange > 0
          ? this.volumes[i]
          : priceChange < 0
          ? -this.volumes[i]
          : 0;
      obvValues.push(obv);
    }

    const atr = this.calculateATR(this.prices, PERIODS.ATR);
    const atrBaseline = this.calculateATR(
      this.prices.slice(0, -1),
      PERIODS.ATR
    );
    const zScore = (this.currentPrice - sma20) / stdDev20;
    const vwap = this.calculateVWAP(this.prices, this.volumes, PERIODS.VWAP);
    const { stochRsi, signal: stochRsiSignal } = this.calculateStochRSI(
      this.prices,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_SMOOTH
    );
    const prevStochRsi = this.calculateStochRSI(
      this.prices.slice(0, -1),
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_SMOOTH
    ).stochRsi;
    const { levels } = this.calculateFibonacciLevels(
      this.prices,
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
    const currentVolume = this.volumes[this.volumes.length - 1];
    const volSma5 = this.calculateSMA(
      this.calculateSlice(this.volumes, PERIODS.VOL_SMA_SHORT)
    );
    const isVolumeSpike = currentVolume > volSma5 * 2;

    const isDoubleTop = this.detectDoubleTop(
      this.prices,
      this.volumes,
      this.currentPrice
    );
    const isHeadAndShoulders = this.detectHeadAndShoulders(
      this.prices,
      this.volumes,
      this.currentPrice
    );
    const isTripleTop = this.detectTripleTop(
      this.prices,
      this.volumes,
      this.currentPrice
    );
    const isTripleBottom = this.detectTripleBottom(
      this.prices,
      this.volumes,
      this.currentPrice
    );

    const momentum =
      this.prices.length >= PERIODS.MOMENTUM
        ? this.currentPrice - this.prices[this.prices.length - PERIODS.MOMENTUM]
        : 0;
    const priceChangePct =
      this.prices.length >= 2
        ? ((this.currentPrice - this.prices[this.prices.length - 2]) /
            this.prices[this.prices.length - 2]) *
          100
        : 0;

    return {
      rsi,
      prevRsi,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
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
    };
  }

  private calculateRSIValues(): { rsi: number; prevRsi: number } {
    const rsi = this.calculateIfEnoughData(
      PERIODS.RSI,
      this.dayIndex,
      () =>
        this.calculateRSI(
          this.prices.slice(this.dayIndex - PERIODS.RSI + 1, this.dayIndex + 1)
        ) || 0
    ) as number;
    const prevRsi = this.calculateIfEnoughData(
      PERIODS.RSI + 1,
      this.dayIndex,
      () =>
        this.calculateRSI(
          this.prices.slice(this.dayIndex - PERIODS.RSI, this.dayIndex)
        ) || 0
    ) as number;
    return { rsi, prevRsi };
  }

  private calculateSMAValues(): {
    smaShort: number;
    smaLong: number;
    prevSmaShort: number;
    prevSmaLong: number;
  } {
    const smaShort =
      (this.calculateIfEnoughData(PERIODS.SMA_SHORT, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(
            this.dayIndex - PERIODS.SMA_SHORT + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.prices[this.dayIndex];
    const smaLong =
      (this.calculateIfEnoughData(PERIODS.SMA_LONG, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(
            this.dayIndex - PERIODS.SMA_LONG + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.prices[this.dayIndex];
    const prevSmaShort =
      (this.calculateIfEnoughData(PERIODS.SMA_SHORT + 1, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(this.dayIndex - PERIODS.SMA_SHORT, this.dayIndex)
        )
      ) as number) ??
      this.prices[this.dayIndex - 1] ??
      this.prices[this.dayIndex];
    const prevSmaLong =
      (this.calculateIfEnoughData(PERIODS.SMA_LONG + 1, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(this.dayIndex - PERIODS.SMA_LONG, this.dayIndex)
        )
      ) as number) ??
      this.prices[this.dayIndex - 1] ??
      this.prices[this.dayIndex];
    return { smaShort, smaLong, prevSmaShort, prevSmaLong };
  }

  private calculateMACDValues(): {
    macdLine: number;
    signalLine: number;
    prevMacdLine: number;
  } {
    const emaShort =
      (this.calculateIfEnoughData(PERIODS.EMA_SHORT, this.dayIndex, () =>
        this.calculateEMA(
          this.prices.slice(
            this.dayIndex - PERIODS.EMA_SHORT + 1,
            this.dayIndex + 1
          ),
          PERIODS.EMA_SHORT
        )
      ) as number) ?? this.prices[this.dayIndex];
    const emaLong =
      (this.calculateIfEnoughData(PERIODS.EMA_LONG, this.dayIndex, () =>
        this.calculateEMA(
          this.prices.slice(
            this.dayIndex - PERIODS.EMA_LONG + 1,
            this.dayIndex + 1
          ),
          PERIODS.EMA_LONG
        )
      ) as number) ?? this.prices[this.dayIndex];
    const macdLine = emaShort - emaLong;

    const prevEmaShort =
      (this.calculateIfEnoughData(PERIODS.EMA_SHORT + 1, this.dayIndex, () =>
        this.calculateEMA(
          this.prices.slice(this.dayIndex - PERIODS.EMA_SHORT, this.dayIndex),
          PERIODS.EMA_SHORT
        )
      ) as number) ??
      this.prices[this.dayIndex - 1] ??
      this.prices[this.dayIndex];
    const prevEmaLong =
      (this.calculateIfEnoughData(PERIODS.EMA_LONG + 1, this.dayIndex, () =>
        this.calculateEMA(
          this.prices.slice(this.dayIndex - PERIODS.EMA_LONG, this.dayIndex),
          PERIODS.EMA_LONG
        )
      ) as number) ??
      this.prices[this.dayIndex - 1] ??
      this.prices[this.dayIndex];
    const prevMacdLine = prevEmaShort - prevEmaLong;

    const signalLine =
      (this.calculateIfEnoughData(
        PERIODS.EMA_LONG + PERIODS.MACD_SIGNAL,
        this.dayIndex,
        () => {
          const macdValues = this.prices
            .slice(this.dayIndex - PERIODS.MACD_SIGNAL + 1, this.dayIndex + 1)
            .map((_, i) => {
              const slice = this.prices.slice(
                this.dayIndex - PERIODS.EMA_LONG - i,
                this.dayIndex - PERIODS.EMA_SHORT - i + 1
              );
              return (
                this.calculateEMA(slice, PERIODS.EMA_SHORT) -
                this.calculateEMA(slice, PERIODS.EMA_LONG)
              );
            });
          return this.calculateEMA(macdValues, PERIODS.MACD_SIGNAL);
        }
      ) as number) ?? 0;

    return { macdLine, signalLine, prevMacdLine };
  }

  private calculateBollingerValues(): { smaMedium: number; upperBand: number } {
    const smaMedium =
      (this.calculateIfEnoughData(PERIODS.BOLLINGER, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(
            this.dayIndex - PERIODS.BOLLINGER + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.prices[this.dayIndex];
    const stdDev =
      (this.calculateIfEnoughData(PERIODS.BOLLINGER, this.dayIndex, () =>
        this.calculateStdDev(
          this.prices.slice(
            this.dayIndex - PERIODS.BOLLINGER + 1,
            this.dayIndex + 1
          ),
          smaMedium
        )
      ) as number) ?? 0;
    const upperBand = smaMedium + 2 * stdDev;
    return { smaMedium, upperBand };
  }

  private calculateOBV(): number {
    if (this.dayIndex < 1) return 0;
    let obv = 0;
    for (let i = 1; i <= this.dayIndex; i++) {
      const priceChange = this.prices[i] - this.prices[i - 1];
      obv +=
        priceChange > 0
          ? this.volumes[i]
          : priceChange < 0
          ? -this.volumes[i]
          : 0;
    }
    return obv / 1e6;
  }

  private calculateVolumeOscillatorValues(): {
    volumeOscillator: number;
    prevVolumeOscillator: number;
  } {
    const volumeSmaShort =
      (this.calculateIfEnoughData(PERIODS.VOL_SMA_SHORT, this.dayIndex, () =>
        this.calculateSMA(
          this.volumes.slice(
            this.dayIndex - PERIODS.VOL_SMA_SHORT + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.volumes[this.dayIndex];
    const volumeSmaLong =
      (this.calculateIfEnoughData(PERIODS.VOL_SMA_LONG, this.dayIndex, () =>
        this.calculateSMA(
          this.volumes.slice(
            this.dayIndex - PERIODS.VOL_SMA_LONG + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.volumes[this.dayIndex];
    const volumeOscillator =
      volumeSmaLong !== 0
        ? ((volumeSmaShort - volumeSmaLong) / volumeSmaLong) * 100
        : 0;

    const prevVolumeSmaShort =
      (this.calculateIfEnoughData(
        PERIODS.VOL_SMA_SHORT + 1,
        this.dayIndex,
        () =>
          this.calculateSMA(
            this.volumes.slice(
              this.dayIndex - PERIODS.VOL_SMA_SHORT,
              this.dayIndex
            )
          )
      ) as number) ??
      this.volumes[this.dayIndex - 1] ??
      this.volumes[this.dayIndex];
    const prevVolumeSmaLong =
      (this.calculateIfEnoughData(PERIODS.VOL_SMA_LONG + 1, this.dayIndex, () =>
        this.calculateSMA(
          this.volumes.slice(
            this.dayIndex - PERIODS.VOL_SMA_LONG,
            this.dayIndex
          )
        )
      ) as number) ??
      this.volumes[this.dayIndex - 1] ??
      this.volumes[this.dayIndex];
    const prevVolumeOscillator =
      prevVolumeSmaLong !== 0
        ? ((prevVolumeSmaShort - prevVolumeSmaLong) / prevVolumeSmaLong) * 100
        : 0;

    return { volumeOscillator, prevVolumeOscillator };
  }

  private calculateTechnicalIndicators(): {
    atr: number;
    atrBaseline: number;
    zScore: number;
    vwap: number;
  } {
    const atr =
      (this.calculateIfEnoughData(PERIODS.ATR, this.dayIndex, () =>
        this.calculateATR(this.prices.slice(0, this.dayIndex + 1), PERIODS.ATR)
      ) as number) ?? 0;
    const atrBaseline =
      (this.calculateIfEnoughData(PERIODS.ATR + 1, this.dayIndex, () =>
        this.calculateATR(this.prices.slice(0, this.dayIndex), PERIODS.ATR)
      ) as number) ?? 0;
    const smaMedium =
      (this.calculateIfEnoughData(PERIODS.BOLLINGER, this.dayIndex, () =>
        this.calculateSMA(
          this.prices.slice(
            this.dayIndex - PERIODS.BOLLINGER + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.prices[this.dayIndex];
    const stdDev =
      (this.calculateIfEnoughData(PERIODS.BOLLINGER, this.dayIndex, () =>
        this.calculateStdDev(
          this.prices.slice(
            this.dayIndex - PERIODS.BOLLINGER + 1,
            this.dayIndex + 1
          ),
          smaMedium
        )
      ) as number) ?? 0;
    const zScore =
      (this.calculateIfEnoughData(
        PERIODS.BOLLINGER,
        this.dayIndex,
        () => (this.currentPrice - smaMedium) / stdDev
      ) as number) ?? 0;
    const vwap =
      (this.calculateIfEnoughData(PERIODS.VWAP, this.dayIndex, () =>
        this.calculateVWAP(
          this.prices.slice(0, this.dayIndex + 1),
          this.volumes.slice(0, this.dayIndex + 1),
          PERIODS.VWAP
        )
      ) as number) ?? this.prices[this.dayIndex];
    return { atr, atrBaseline, zScore, vwap };
  }

  private calculateStochRSIValues(): {
    stochRsi: number;
    prevStochRsi: number;
  } {
    const { stochRsi } = (this.calculateIfEnoughData(
      PERIODS.STOCH_RSI,
      this.dayIndex,
      () =>
        this.calculateStochRSI(
          this.prices.slice(0, this.dayIndex + 1),
          PERIODS.STOCH_RSI,
          PERIODS.STOCH_RSI,
          PERIODS.STOCH_SMOOTH
        )
    ) as { stochRsi: number }) ?? { stochRsi: 0 };
    const { stochRsi: prevStochRsi } = (this.calculateIfEnoughData(
      PERIODS.STOCH_RSI + 1,
      this.dayIndex,
      () =>
        this.calculateStochRSI(
          this.prices.slice(0, this.dayIndex),
          PERIODS.STOCH_RSI,
          PERIODS.STOCH_RSI,
          PERIODS.STOCH_SMOOTH
        )
    ) as { stochRsi: number }) ?? { stochRsi: 0 };
    return { stochRsi, prevStochRsi };
  }

  private calculateFibonacci(): number {
    const { levels } = (this.calculateIfEnoughData(
      PERIODS.FIBONACCI,
      this.dayIndex,
      () =>
        this.calculateFibonacciLevels(
          this.prices.slice(
            this.dayIndex - PERIODS.FIBONACCI + 1,
            this.dayIndex + 1
          ),
          PERIODS.FIBONACCI
        )
    ) as { levels: number[] }) ?? { levels: [0, 0, 0, 0] };
    return levels[3] || 0;
  }

  private calculatePatterns(): {
    isDoubleTop: boolean;
    isHeadAndShoulders: boolean;
    isTripleTop: boolean;
    isVolumeSpike: boolean;
    isTripleBottom?: boolean;
  } {
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
    const volumeSmaShort =
      (this.calculateIfEnoughData(PERIODS.VOL_SMA_SHORT, this.dayIndex, () =>
        this.calculateSMA(
          this.volumes.slice(
            this.dayIndex - PERIODS.VOL_SMA_SHORT + 1,
            this.dayIndex + 1
          )
        )
      ) as number) ?? this.volumes[this.dayIndex];
    const currentVolume = this.volumes[this.dayIndex] ?? 0;
    const isVolumeSpike = currentVolume > volumeSmaShort * 2;

    if (this.isBTC) {
      return { isDoubleTop, isHeadAndShoulders, isTripleTop, isVolumeSpike };
    }
    const isTripleBottom = this.detectTripleBottom(
      this.prices.slice(0, this.dayIndex + 1),
      this.volumes.slice(0, this.dayIndex + 1),
      this.currentPrice
    );
    return {
      isDoubleTop,
      isHeadAndShoulders,
      isTripleTop,
      isVolumeSpike,
      isTripleBottom,
    };
  }

  private calculateMomentumValues(): {
    momentum: number;
    priceChangePct: number;
  } {
    const momentum =
      (this.calculateIfEnoughData(
        PERIODS.MOMENTUM,
        this.dayIndex,
        () =>
          this.currentPrice - this.prices[this.dayIndex - PERIODS.MOMENTUM + 1]
      ) as number) ?? 0;
    const priceChangePct =
      this.dayIndex >= 1
        ? ((this.currentPrice - this.prices[this.dayIndex - 1]) /
            this.prices[this.dayIndex - 1]) *
          100
        : 0;
    return { momentum, priceChangePct };
  }

  public compute(): number[] {
    if (
      !this.prices ||
      !this.volumes ||
      this.dayIndex < 0 ||
      this.dayIndex >= this.prices.length
    ) {
      return Array(this.isBTC ? 28 : 29).fill(0);
    }

    const { rsi, prevRsi } = this.calculateRSIValues();
    const { smaShort, smaLong, prevSmaShort, prevSmaLong } =
      this.calculateSMAValues();
    const { macdLine, signalLine, prevMacdLine } = this.calculateMACDValues();
    const { upperBand } = this.calculateBollingerValues();
    const normalizedOBV = this.calculateOBV();
    const { volumeOscillator, prevVolumeOscillator } =
      this.calculateVolumeOscillatorValues();
    const { atr, atrBaseline, zScore, vwap } =
      this.calculateTechnicalIndicators();
    const { stochRsi, prevStochRsi } = this.calculateStochRSIValues();
    const fib618 = this.calculateFibonacci();
    const {
      isDoubleTop,
      isHeadAndShoulders,
      isTripleTop,
      isVolumeSpike,
      isTripleBottom,
    } = this.calculatePatterns();
    const { momentum, priceChangePct } = this.calculateMomentumValues();

    const baseFeatures = [
      rsi,
      prevRsi,
      smaShort,
      smaLong,
      prevSmaShort,
      prevSmaLong,
      macdLine,
      signalLine,
      this.currentPrice,
      upperBand,
      normalizedOBV,
      atr,
      atrBaseline,
      zScore,
      vwap,
      stochRsi,
      prevStochRsi,
      fib618,
      this.prices[this.dayIndex - 1] || this.prices[0],
      volumeOscillator,
      prevVolumeOscillator,
      isDoubleTop ? 1 : 0,
      isHeadAndShoulders ? 1 : 0,
      prevMacdLine,
      isTripleTop ? 1 : 0,
      isVolumeSpike ? 1 : 0,
      momentum,
      priceChangePct,
    ];

    return this.isBTC
      ? baseFeatures
      : [...baseFeatures, isTripleBottom ? 1 : 0];
  }
}
