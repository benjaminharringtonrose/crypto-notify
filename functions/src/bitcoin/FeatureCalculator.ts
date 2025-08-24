import { Indicators } from "../types";
import { PERIODS } from "../constants";
import { FeatureDetector } from "./FeatureDetector";

interface ComputeParams {
  prices: number[];
  volumes: number[];
  dayIndex: number;
  currentPrice: number;
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

  // EXPERIMENT #61: Advanced Market Microstructure Features
  public calculateIchimokuTenkanSen(
    prices: number[],
    period: number = 9
  ): number {
    if (prices.length < period) return prices[prices.length - 1];
    const recentPrices = prices.slice(-period);
    const high = Math.max(...recentPrices);
    const low = Math.min(...recentPrices);
    return (high + low) / 2;
  }

  public calculateIchimokuKijunSen(
    prices: number[],
    period: number = 26
  ): number {
    if (prices.length < period) return prices[prices.length - 1];
    const recentPrices = prices.slice(-period);
    const high = Math.max(...recentPrices);
    const low = Math.min(...recentPrices);
    return (high + low) / 2;
  }

  public calculateIchimokuCloudPosition(
    currentPrice: number,
    prices: number[]
  ): number {
    // Senkou Span A (short-term cloud boundary)
    const tenkanSen = this.calculateIchimokuTenkanSen(prices, 9);
    const kijunSen = this.calculateIchimokuKijunSen(prices, 26);
    const senkouSpanA = (tenkanSen + kijunSen) / 2;

    // Senkou Span B (long-term cloud boundary)
    const senkouSpanB = this.calculateIchimokuKijunSen(prices, 52);

    // Cloud boundaries
    const cloudTop = Math.max(senkouSpanA, senkouSpanB);
    const cloudBottom = Math.min(senkouSpanA, senkouSpanB);

    // Position relative to cloud: +1 = above, 0 = in cloud, -1 = below
    if (currentPrice > cloudTop) return 1;
    if (currentPrice < cloudBottom) return -1;
    return 0;
  }

  public calculateIchimoku(
    high: number[],
    low: number[],
    close: number[]
  ): {
    conversion: number;
    base: number;
    spanA: number;
    spanB: number;
    position: number;
  } {
    if (high.length < 52 || low.length < 52 || close.length < 52) {
      return { conversion: 0, base: 0, spanA: 0, spanB: 0, position: 0 };
    }

    const currentPrice = close[close.length - 1];

    // Tenkan-sen (Conversion Line) - 9-period
    const tenkanSen = this.calculateIchimokuTenkanSen(close, 9);

    // Kijun-sen (Base Line) - 26-period
    const kijunSen = this.calculateIchimokuKijunSen(close, 26);

    // Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2
    const senkouSpanA = (tenkanSen + kijunSen) / 2;

    // Senkou Span B (Leading Span B) - 52-period
    const senkouSpanB = this.calculateIchimokuKijunSen(close, 52);

    // Calculate position relative to cloud
    const cloudTop = Math.max(senkouSpanA, senkouSpanB);
    const cloudBottom = Math.min(senkouSpanA, senkouSpanB);
    const position =
      currentPrice > cloudTop ? 1 : currentPrice < cloudBottom ? -1 : 0;

    return {
      conversion: tenkanSen,
      base: kijunSen,
      spanA: senkouSpanA,
      spanB: senkouSpanB,
      position: position,
    };
  }

  public calculateWilliamsR(prices: number[], period: number = 14): number {
    if (prices.length < period) return -50;
    const recentPrices = prices.slice(-period);
    const highest = Math.max(...recentPrices);
    const lowest = Math.min(...recentPrices);
    const currentPrice = prices[prices.length - 1];

    if (highest === lowest) return -50;
    return ((highest - currentPrice) / (highest - lowest)) * -100;
  }

  public calculateFibonacciRetracement(
    high: number[],
    low: number[],
    close: number[],
    period: number = 20
  ): {
    level236: number;
    level382: number;
    level500: number;
    level618: number;
    position: number;
  } {
    if (high.length < period || low.length < period || close.length < period) {
      return {
        level236: 0,
        level382: 0,
        level500: 0,
        level618: 0,
        position: 0,
      };
    }

    const recentHighs = high.slice(-period);
    const recentLows = low.slice(-period);
    const currentPrice = close[close.length - 1];

    // Find swing high and swing low
    const swingHigh = Math.max(...recentHighs);
    const swingLow = Math.min(...recentLows);
    const range = swingHigh - swingLow;

    if (range === 0) {
      return {
        level236: swingHigh,
        level382: swingHigh,
        level500: swingHigh,
        level618: swingHigh,
        position: 0,
      };
    }

    // Calculate Fibonacci retracement levels
    const level236 = swingHigh - range * 0.236;
    const level382 = swingHigh - range * 0.382;
    const level500 = swingHigh - range * 0.5;
    const level618 = swingHigh - range * 0.618;

    // Calculate position relative to Fibonacci levels
    let position = 0;
    if (currentPrice > swingHigh) {
      position = 1; // Above swing high
    } else if (currentPrice > level236) {
      position = 0.8; // Between swing high and 23.6%
    } else if (currentPrice > level382) {
      position = 0.6; // Between 23.6% and 38.2%
    } else if (currentPrice > level500) {
      position = 0.4; // Between 38.2% and 50%
    } else if (currentPrice > level618) {
      position = 0.2; // Between 50% and 61.8%
    } else if (currentPrice > swingLow) {
      position = -0.2; // Between 61.8% and swing low
    } else {
      position = -1; // Below swing low
    }

    return { level236, level382, level500, level618, position };
  }

  public calculateCCI(
    high: number[],
    low: number[],
    close: number[],
    period: number = 20
  ): number {
    if (high.length < period || low.length < period || close.length < period) {
      return 0;
    }

    // Calculate typical prices
    const typicalPrices: number[] = [];
    for (let i = 0; i < close.length; i++) {
      typicalPrices.push((high[i] + low[i] + close[i]) / 3);
    }

    // Get recent typical prices for the period
    const recentTypicalPrices = typicalPrices.slice(-period);
    const currentTypicalPrice =
      recentTypicalPrices[recentTypicalPrices.length - 1];

    // Calculate SMA of typical prices
    const smaTypicalPrice =
      recentTypicalPrices.reduce((sum, price) => sum + price, 0) / period;

    // Calculate mean deviation
    const deviations = recentTypicalPrices.map((price) =>
      Math.abs(price - smaTypicalPrice)
    );
    const meanDeviation =
      deviations.reduce((sum, dev) => sum + dev, 0) / period;

    // Calculate CCI
    if (meanDeviation === 0) return 0;
    return (currentTypicalPrice - smaTypicalPrice) / (0.015 * meanDeviation);
  }

  public calculateMFI(
    high: number[],
    low: number[],
    close: number[],
    volume: number[],
    period: number = 14
  ): number {
    if (
      high.length < period + 1 ||
      low.length < period + 1 ||
      close.length < period + 1 ||
      volume.length < period + 1
    ) {
      return 50;
    }

    // Calculate typical prices
    const typicalPrices: number[] = [];
    for (let i = 0; i < close.length; i++) {
      typicalPrices.push((high[i] + low[i] + close[i]) / 3);
    }

    // Calculate raw money flow (typical price * volume)
    const moneyFlow: number[] = [];
    for (let i = 0; i < typicalPrices.length; i++) {
      moneyFlow.push(typicalPrices[i] * volume[i]);
    }

    // Calculate positive and negative money flow
    let positiveMoneyFlow = 0;
    let negativeMoneyFlow = 0;

    for (let i = 1; i <= period; i++) {
      const currentTypicalPrice = typicalPrices[typicalPrices.length - i];
      const previousTypicalPrice = typicalPrices[typicalPrices.length - i - 1];
      const currentVolume = volume[volume.length - i];

      if (currentTypicalPrice > previousTypicalPrice) {
        positiveMoneyFlow += currentTypicalPrice * currentVolume;
      } else if (currentTypicalPrice < previousTypicalPrice) {
        negativeMoneyFlow += currentTypicalPrice * currentVolume;
      }
    }

    // Calculate Money Flow Index
    if (negativeMoneyFlow === 0) return 100;
    const moneyRatio = positiveMoneyFlow / negativeMoneyFlow;
    return 100 - 100 / (1 + moneyRatio);
  }

  public calculateKeltnerChannels(
    high: number[],
    low: number[],
    close: number[],
    period: number = 20,
    multiplier: number = 2
  ): { upper: number; middle: number; lower: number; position: number } {
    if (high.length < period || low.length < period || close.length < period) {
      return { upper: 0, middle: 0, lower: 0, position: 0.5 };
    }

    // Calculate typical prices
    const typicalPrices: number[] = [];
    for (let i = 0; i < close.length; i++) {
      typicalPrices.push((high[i] + low[i] + close[i]) / 3);
    }

    // Get recent data for the period
    const recentTypicalPrices = typicalPrices.slice(-period);
    const recentHighs = high.slice(-period);
    const recentLows = low.slice(-period);
    const currentPrice = close[close.length - 1];

    // Calculate middle line (EMA of typical prices)
    const middle = this.calculateEMA(recentTypicalPrices, period);

    // Calculate True Range
    const trueRanges: number[] = [];
    for (let i = 1; i < recentHighs.length; i++) {
      const tr = Math.max(
        recentHighs[i] - recentLows[i],
        Math.abs(recentHighs[i] - recentTypicalPrices[i - 1]),
        Math.abs(recentLows[i] - recentTypicalPrices[i - 1])
      );
      trueRanges.push(tr);
    }

    // Calculate ATR (Average True Range)
    const atr = trueRanges.reduce((sum, tr) => sum + tr, 0) / trueRanges.length;

    // Calculate upper and lower bands
    const upper = middle + multiplier * atr;
    const lower = middle - multiplier * atr;

    // Calculate position within the channel (0 = at lower band, 1 = at upper band)
    const position =
      upper === lower ? 0.5 : (currentPrice - lower) / (upper - lower);

    return { upper, middle, lower, position };
  }

  public calculateAroon(
    high: number[],
    low: number[],
    period: number = 25
  ): { aroonUp: number; aroonDown: number; aroonOscillator: number } {
    if (high.length < period || low.length < period) {
      return { aroonUp: 50, aroonDown: 50, aroonOscillator: 0 };
    }

    // Get recent data for the period
    const recentHighs = high.slice(-period);
    const recentLows = low.slice(-period);

    // Find the highest high and lowest low in the period
    const highestHigh = Math.max(...recentHighs);
    const lowestLow = Math.min(...recentLows);

    // Find the days since the highest high and lowest low
    let daysSinceHighest = period;
    let daysSinceLowest = period;

    for (let i = recentHighs.length - 1; i >= 0; i--) {
      if (recentHighs[i] === highestHigh) {
        daysSinceHighest = recentHighs.length - 1 - i;
        break;
      }
    }

    for (let i = recentLows.length - 1; i >= 0; i--) {
      if (recentLows[i] === lowestLow) {
        daysSinceLowest = recentLows.length - 1 - i;
        break;
      }
    }

    // Calculate Aroon Up and Down
    const aroonUp = ((period - daysSinceHighest) / period) * 100;
    const aroonDown = ((period - daysSinceLowest) / period) * 100;

    // Calculate Aroon Oscillator
    const aroonOscillator = aroonUp - aroonDown;

    return { aroonUp, aroonDown, aroonOscillator };
  }

  public calculateDonchianChannels(
    high: number[],
    low: number[],
    close: number[],
    period: number = 20
  ): { upper: number; middle: number; lower: number; position: number } {
    if (high.length < period || low.length < period || close.length < period) {
      return { upper: 0, middle: 0, lower: 0, position: 0.5 };
    }

    // Get recent data for the period
    const recentHighs = high.slice(-period);
    const recentLows = low.slice(-period);
    const currentPrice = close[close.length - 1];

    // Calculate Donchian Channels
    const upper = Math.max(...recentHighs);
    const lower = Math.min(...recentLows);
    const middle = (upper + lower) / 2;

    // Calculate position within the channel (0 = at lower band, 1 = at upper band)
    const position =
      upper === lower ? 0.5 : (currentPrice - lower) / (upper - lower);

    return { upper, middle, lower, position };
  }

  public calculateParabolicSAR(
    high: number[],
    low: number[],
    close: number[],
    acceleration: number = 0.02,
    maximum: number = 0.2
  ): { sar: number; trend: number } {
    if (high.length < 2 || low.length < 2 || close.length < 2) {
      return { sar: 0, trend: 0 };
    }

    // Initialize SAR
    let sar = low[0];
    let af = acceleration; // Acceleration Factor
    let ep = high[0]; // Extreme Point
    let trend = 1; // 1 for uptrend, -1 for downtrend

    // Calculate SAR for the last few periods
    for (let i = 1; i < close.length; i++) {
      const currentHigh = high[i];
      const currentLow = low[i];

      if (trend === 1) {
        // Uptrend
        if (currentLow > sar) {
          // Continue uptrend
          sar = sar + af * (ep - sar);
          if (currentHigh > ep) {
            ep = currentHigh;
            af = Math.min(af + acceleration, maximum);
          }
        } else {
          // Reverse to downtrend
          trend = -1;
          sar = ep;
          ep = currentLow;
          af = acceleration;
        }
      } else {
        // Downtrend
        if (currentHigh < sar) {
          // Continue downtrend
          sar = sar + af * (ep - sar);
          if (currentLow < ep) {
            ep = currentLow;
            af = Math.min(af + acceleration, maximum);
          }
        } else {
          // Reverse to uptrend
          trend = 1;
          sar = ep;
          ep = currentHigh;
          af = acceleration;
        }
      }
    }

    // Normalize SAR to a position relative to current price
    const currentPrice = close[close.length - 1];
    const sarPosition = (currentPrice - sar) / currentPrice;

    return { sar, trend: trend * Math.abs(sarPosition) };
  }

  public calculateCMF(
    high: number[],
    low: number[],
    close: number[],
    volume: number[],
    period: number = 20
  ): number {
    if (
      high.length < period ||
      low.length < period ||
      close.length < period ||
      volume.length < period
    ) {
      return 0;
    }

    let moneyFlowVolume = 0;
    let totalVolume = 0;

    // Calculate CMF for the last 'period' days
    for (let i = high.length - period; i < high.length; i++) {
      const highPrice = high[i];
      const lowPrice = low[i];
      const closePrice = close[i];
      const vol = volume[i];

      // Calculate Money Flow Multiplier
      const moneyFlowMultiplier =
        (closePrice - lowPrice - (highPrice - closePrice)) /
        (highPrice - lowPrice);

      // Calculate Money Flow Volume
      const mfv = moneyFlowMultiplier * vol;

      moneyFlowVolume += mfv;
      totalVolume += vol;
    }

    // Calculate CMF
    return totalVolume > 0 ? moneyFlowVolume / totalVolume : 0;
  }

  public calculateStochasticK(prices: number[], period: number = 14): number {
    if (prices.length < period) return 50;
    const recentPrices = prices.slice(-period);
    const highest = Math.max(...recentPrices);
    const lowest = Math.min(...recentPrices);
    const currentPrice = prices[prices.length - 1];

    if (highest === lowest) return 50;
    return ((currentPrice - lowest) / (highest - lowest)) * 100;
  }

  public calculateVPT(prices: number[], volumes: number[]): number {
    if (prices.length < 2 || volumes.length < 2) return 0;

    let vpt = 0;
    for (let i = 1; i < prices.length; i++) {
      const priceChange = prices[i] - prices[i - 1];
      const priceChangePercent = priceChange / prices[i - 1];
      vpt += volumes[i] * priceChangePercent;
    }

    // Normalize by average volume to make it scale-independent
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    return avgVolume > 0 ? vpt / avgVolume : 0;
  }

  public calculatePriceAcceleration(
    prices: number[],
    period: number = 7
  ): number {
    if (prices.length < period + 2) return 0;

    // Calculate first derivative (velocity) - rate of change
    const velocity =
      (prices[prices.length - 1] - prices[prices.length - 1 - period]) / period;

    // Calculate second derivative (acceleration) - change in velocity
    const prevVelocity =
      (prices[prices.length - 1 - period] -
        prices[prices.length - 1 - 2 * period]) /
      period;

    // Acceleration is the change in velocity
    const acceleration = velocity - prevVelocity;

    // Normalize by current price to make it scale-independent
    const currentPrice = prices[prices.length - 1];
    return currentPrice > 0 ? acceleration / currentPrice : 0;
  }

  public calculatePROC(prices: number[], period: number = 10): number {
    if (prices.length < period + 1) return 0;

    const currentPrice = prices[prices.length - 1];
    const pastPrice = prices[prices.length - 1 - period];

    // Calculate percentage change in price
    return pastPrice > 0 ? ((currentPrice - pastPrice) / pastPrice) * 100 : 0;
  }

  public calculateVWMA(
    prices: number[],
    volumes: number[],
    period: number = 20
  ): number {
    if (prices.length < period || volumes.length < period) return 0;

    const recentPrices = prices.slice(-period);
    const recentVolumes = volumes.slice(-period);

    // Calculate volume-weighted moving average
    const volumeSum = recentVolumes.reduce((sum, vol) => sum + vol, 0);
    const vwma =
      recentPrices.reduce((sum, price, index) => {
        return sum + price * recentVolumes[index];
      }, 0) / volumeSum;

    return vwma;
  }

  // EXPERIMENT #4: Center of Gravity Oscillator (COG)
  public calculateCenterOfGravityOscillator(
    prices: number[],
    period: number = 10
  ): number {
    if (prices.length < period) {
      return 0; // Neutral value if insufficient data
    }

    const recentPrices = prices.slice(-period);

    // Calculate weighted sum where more recent prices have higher weights
    let weightedSum = 0;
    let weightSum = 0;

    for (let i = 0; i < recentPrices.length; i++) {
      const weight = i + 1; // Weight increases with recency (1, 2, 3, ..., period)
      weightedSum += recentPrices[i] * weight;
      weightSum += weight;
    }

    // Calculate the center of gravity
    const centerOfGravity = weightedSum / weightSum;

    // Calculate the simple moving average for comparison
    const simpleAverage =
      recentPrices.reduce((sum, price) => sum + price, 0) / period;

    // Calculate COG oscillator: difference between COG and SMA, normalized
    const cogOscillator =
      simpleAverage > 0
        ? ((centerOfGravity - simpleAverage) / simpleAverage) * 100
        : 0;

    return cogOscillator;
  }

  // EXPERIMENT #7-1: True Strength Index (TSI)
  public calculateTSI(
    prices: number[],
    firstPeriod: number = 25,
    secondPeriod: number = 13
  ): number {
    if (prices.length < firstPeriod + secondPeriod + 1) return 0;

    // Calculate price changes
    const priceChanges = [];
    for (let i = 1; i < prices.length; i++) {
      priceChanges.push(prices[i] - prices[i - 1]);
    }

    // First smoothing: EMA of price changes
    const firstEMA = this.calculateEMA(priceChanges, firstPeriod);

    // Second smoothing: EMA of the first EMA
    const secondEMA = this.calculateEMA([firstEMA], secondPeriod);

    // Calculate absolute price changes
    const absPriceChanges = priceChanges.map((change) => Math.abs(change));

    // First smoothing: EMA of absolute price changes
    const firstAbsEMA = this.calculateEMA(absPriceChanges, firstPeriod);

    // Second smoothing: EMA of the first absolute EMA
    const secondAbsEMA = this.calculateEMA([firstAbsEMA], secondPeriod);

    // Calculate TSI
    return secondAbsEMA !== 0 ? (secondEMA / secondAbsEMA) * 100 : 0;
  }

  // EXPERIMENT #7-3: Price Momentum Oscillator (PMO)
  public calculatePMO(
    prices: number[],
    firstPeriod: number = 35,
    secondPeriod: number = 20
  ): number {
    if (prices.length < firstPeriod + secondPeriod + 1) return 0;

    // Calculate rate of change
    const roc = [];
    for (let i = 1; i < prices.length; i++) {
      roc.push(((prices[i] - prices[i - 1]) / prices[i - 1]) * 100);
    }

    // First smoothing: EMA of ROC
    const firstEMA = this.calculateEMA(roc, firstPeriod);

    // Second smoothing: EMA of the first EMA
    const secondEMA = this.calculateEMA([firstEMA], secondPeriod);

    return secondEMA;
  }

  // EXPERIMENT #7-4: Bollinger Band Width
  public calculateBollingerBandWidth(
    prices: number[],
    period: number = 20,
    stdDevMultiplier: number = 2
  ): number {
    if (prices.length < period) return 0;

    const recentPrices = prices.slice(-period);
    const sma = recentPrices.reduce((sum, price) => sum + price, 0) / period;

    // Calculate standard deviation
    const variance =
      recentPrices.reduce((sum, price) => {
        return sum + Math.pow(price - sma, 2);
      }, 0) / period;
    const stdDev = Math.sqrt(variance);

    // Calculate Bollinger Bands
    const upperBand = sma + stdDevMultiplier * stdDev;
    const lowerBand = sma - stdDevMultiplier * stdDev;

    // Calculate Band Width as percentage of middle band
    return sma > 0 ? ((upperBand - lowerBand) / sma) * 100 : 0;
  }

  // EXPERIMENT #8-1: Historical Volatility (HV)
  public calculateHistoricalVolatility(
    prices: number[],
    period: number = 20
  ): number {
    if (prices.length < period + 1) return 0;

    // Calculate logarithmic returns
    const logReturns = [];
    for (let i = 1; i < prices.length; i++) {
      logReturns.push(Math.log(prices[i] / prices[i - 1]));
    }

    // Get recent log returns
    const recentLogReturns = logReturns.slice(-period);

    // Calculate mean of log returns
    const meanReturn =
      recentLogReturns.reduce((sum, ret) => sum + ret, 0) / period;

    // Calculate variance
    const variance =
      recentLogReturns.reduce((sum, ret) => {
        return sum + Math.pow(ret - meanReturn, 2);
      }, 0) / period;

    // Calculate standard deviation
    const stdDev = Math.sqrt(variance);

    // Annualize volatility (assuming daily data, multiply by sqrt(365))
    const annualizedVolatility = stdDev * Math.sqrt(365) * 100; // Convert to percentage

    return annualizedVolatility;
  }

  // EXPERIMENT #8-4: Camarilla Pivots (CP)
  public calculateCamarillaPivots(
    prices: number[],
    period: number = 20
  ): number {
    if (prices.length < period + 1) return 0;

    // Get the previous day's high, low, and close
    const prevHigh = Math.max(...prices.slice(-period, -1));
    const prevLow = Math.min(...prices.slice(-period, -1));
    const prevClose = prices[prices.length - 2];
    const currentPrice = prices[prices.length - 1];

    // Calculate Camarilla pivot levels
    const range = prevHigh - prevLow;

    // H3 and L3 are the key reversal levels
    const h3 = prevClose + (range * 1.1) / 4;
    const l3 = prevClose - (range * 1.1) / 4;

    // H4 and L4 are extreme levels
    const h4 = prevClose + (range * 1.1) / 2;
    const l4 = prevClose - (range * 1.1) / 2;

    // Calculate position relative to Camarilla levels
    let position = 0;

    if (currentPrice > h4) {
      position = 1.0; // Above extreme resistance
    } else if (currentPrice > h3) {
      position = 0.75; // Above key resistance
    } else if (currentPrice > prevClose) {
      position = 0.5; // Above previous close
    } else if (currentPrice > l3) {
      position = 0.25; // Above key support
    } else if (currentPrice > l4) {
      position = 0.0; // Above extreme support
    } else {
      position = -0.25; // Below extreme support
    }

    return position;
  }

  // EXPERIMENT #8-5: Accelerator Oscillator (AO)
  public calculateAcceleratorOscillator(
    prices: number[],
    shortPeriod: number = 5,
    longPeriod: number = 34
  ): number {
    if (prices.length < longPeriod) return 0;

    // Calculate midpoints (high + low) / 2
    // Since we only have close prices, we'll use close prices as approximation
    const midpoints = prices.map((price) => price);

    // Calculate short-term SMA of midpoints
    const shortSMA =
      midpoints.slice(-shortPeriod).reduce((sum, price) => sum + price, 0) /
      shortPeriod;

    // Calculate long-term SMA of midpoints
    const longSMA =
      midpoints.slice(-longPeriod).reduce((sum, price) => sum + price, 0) /
      longPeriod;

    // Calculate Accelerator Oscillator
    const ao = shortSMA - longSMA;

    // Normalize by the long-term SMA to get a percentage
    return longSMA > 0 ? (ao / longSMA) * 100 : 0;
  }

  public calculateSlice(data: number[], periods: number, offset = 0): number[] {
    return data.slice(-periods - offset, offset === 0 ? undefined : -offset);
  }

  public calculateRSI(prices: number[], period: number = PERIODS.RSI): number {
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
      // EXPERIMENT #61: Advanced Market Microstructure Features
      ichimokuTenkanSen: this.calculateIchimokuTenkanSen(
        prices.slice(0, dayIndex + 1)
      ),
      ichimokuKijunSen: this.calculateIchimokuKijunSen(
        prices.slice(0, dayIndex + 1)
      ),
      ichimokuCloudPosition: this.calculateIchimokuCloudPosition(
        currentPrice,
        prices.slice(0, dayIndex + 1)
      ),
      williamsR: this.calculateWilliamsR(prices.slice(0, dayIndex + 1)),
      stochasticK: this.calculateStochasticK(prices.slice(0, dayIndex + 1)),
      vpt: this.calculateVPT(
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      cci: this.calculateCCI(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ),
      mfi: this.calculateMFI(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      aroonOscillator: this.calculateAroon(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ).aroonOscillator,
      donchianPosition: this.calculateDonchianChannels(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ).position,
      parabolicSAR: this.calculateParabolicSAR(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ).trend,
      adx: this.calculateADX(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ),
      ichimokuPosition: this.calculateIchimoku(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ).position,
      fibonacciPosition: this.calculateFibonacciRetracement(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ).position,
      priceAcceleration: this.calculatePriceAcceleration(
        prices.slice(0, dayIndex + 1)
      ),
      proc: this.calculatePROC(prices.slice(0, dayIndex + 1)),
      vwma: this.calculateVWMA(
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      centerOfGravityOscillator: this.calculateCenterOfGravityOscillator(
        prices.slice(0, dayIndex + 1)
      ),
      tsi: this.calculateTSI(prices.slice(0, dayIndex + 1)),
      pmo: this.calculatePMO(prices.slice(0, dayIndex + 1)),
      bollingerBandWidth: this.calculateBollingerBandWidth(
        prices.slice(0, dayIndex + 1)
      ),
      historicalVolatility: this.calculateHistoricalVolatility(
        prices.slice(0, dayIndex + 1)
      ),
      camarillaPivots: this.calculateCamarillaPivots(
        prices.slice(0, dayIndex + 1)
      ),
      acceleratorOscillator: this.calculateAcceleratorOscillator(
        prices.slice(0, dayIndex + 1)
      ),
      chaikinOscillator: this.calculateChaikinOscillator(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      elderForceIndex: this.calculateElderForceIndex(
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      klingerVolumeOscillator: this.calculateKlingerVolumeOscillator(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1)
      ),
      massIndex: this.calculateMassIndex(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ),
      priceChannel: this.calculatePriceChannel(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ),
      fisherTransform: this.calculateFisherTransform(
        prices.slice(0, dayIndex + 1),
        prices.slice(0, dayIndex + 1)
      ),
      hullMovingAverage: this.calculateHullMovingAverage(
        prices.slice(0, dayIndex + 1)
      ),
      kaufmanAdaptiveMovingAverage: this.calculateKaufmanAdaptiveMovingAverage(
        prices.slice(0, dayIndex + 1)
      ),
      mesaSineWave: this.calculateMESASineWave(prices.slice(0, dayIndex + 1)),
      rainbowMovingAverage: this.calculateRainbowMovingAverage(
        prices.slice(0, dayIndex + 1)
      ),
    };
  }

  public compute({
    prices,
    volumes,
    dayIndex,
    currentPrice,
  }: ComputeParams): number[] {
    if (!prices || !volumes || dayIndex < 0 || dayIndex >= prices.length) {
      return Array(FeatureDetector.getFeatureCount()).fill(0);
    }

    const indicators = this.calculateIndicators({
      prices,
      volumes,
      dayIndex,
      currentPrice,
    });

    // OPTIMIZED FEATURE SET: 26 Most Important Features (Based on Gradual Optimization)
    // Selected features that provide maximum trading performance with minimal redundancy
    const optimizedFeatures = [
      // 1-5: Core Price Action & Volatility (5 features)
      indicators.priceChangePct, // Price change percentage
      dayIndex >= 20
        ? Math.max(...prices.slice(dayIndex - 19, dayIndex + 1)) -
          Math.min(...prices.slice(dayIndex - 19, dayIndex + 1))
        : 0, // highLowRange
      dayIndex >= 5
        ? this.calculateStdDev(
            prices.slice(dayIndex - 4, dayIndex + 1),
            prices
              .slice(dayIndex - 4, dayIndex + 1)
              .reduce((a, b) => a + b, 0) / 5
          )
        : 0, // priceVolatility
      dayIndex >= 20
        ? (currentPrice -
            Math.min(...prices.slice(dayIndex - 19, dayIndex + 1))) /
          (Math.max(...prices.slice(dayIndex - 19, dayIndex + 1)) -
            Math.min(...prices.slice(dayIndex - 19, dayIndex + 1)) || 1)
        : 0.5, // pricePosition
      volumes[dayIndex] /
        (volumes
          .slice(Math.max(0, dayIndex - 19), dayIndex + 1)
          .reduce((a, b) => a + b, 0) /
          Math.min(20, dayIndex + 1)), // relativeVolume

      // 6-10: Technical Indicators (5 features)
      indicators.rsi, // RSI momentum oscillator
      indicators.signalLine, // MACD signal line
      currentPrice / (indicators.vwap || currentPrice), // vwapRatio
      indicators.atr, // Average True Range
      indicators.obv, // On-Balance Volume

      // 11-15: Enhanced Indicators (5 features)
      indicators.momentum, // Raw momentum
      indicators.macdLine - indicators.signalLine, // MACD histogram
      currentPrice / (indicators.sma7 || currentPrice), // priceSMA7Ratio
      currentPrice / (indicators.sma21 || currentPrice), // priceSMA21Ratio
      currentPrice / (indicators.sma50 || indicators.sma20 || currentPrice), // priceSMA50Ratio

      // 16-20: Market Regime Features (5 features)
      indicators.trendRegime, // Trend regime score
      indicators.volatilityRegimeScore / 5.0, // volatilityRegime (normalized)
      indicators.ichimokuTenkanSen, // Ichimoku Tenkan-sen (9-period)
      indicators.ichimokuKijunSen, // Ichimoku Kijun-sen (26-period)
      indicators.ichimokuCloudPosition, // Position relative to Ichimoku cloud

      // 21-27: Advanced Microstructure Features (7 features)
      indicators.williamsR, // Williams %R momentum oscillator
      volumes
        .slice(Math.max(0, dayIndex - 19), dayIndex + 1)
        .reduce((a, b) => a + b, 0) / Math.min(20, dayIndex + 1), // volumeMA20
      indicators.volumeOscillator, // Volume oscillator
      dayIndex >= 20
        ? (indicators.upperBand - indicators.lowerBand) /
          indicators.currentPrice
        : 0.1, // bollingerSqueeze
      dayIndex >= 1
        ? Math.sign(indicators.currentPrice - prices[dayIndex - 1]) *
          Math.sign(indicators.rsi - indicators.prevRsi)
        : 0, // rsiDivergence
      indicators.cci, // Commodity Channel Index (CCI)
      indicators.mfi, // Money Flow Index (MFI)
      indicators.aroonOscillator, // Aroon Oscillator
      indicators.donchianPosition, // Donchian Channels position
      indicators.parabolicSAR, // Parabolic SAR trend
      indicators.adx, // Average Directional Index (ADX)
      indicators.ichimokuPosition, // Ichimoku Cloud position
      indicators.fibonacciPosition, // Fibonacci Retracement position
      indicators.stochasticK, // Stochastic K oscillator
      indicators.priceAcceleration, // Price acceleration indicator
      indicators.proc, // Price Rate of Change (PROC)
      indicators.stochRsi, // Stochastic RSI (StochRSI)
      indicators.vwma, // Volume Weighted Moving Average (VWMA)
      indicators.centerOfGravityOscillator, // Center of Gravity Oscillator (COG) - EXPERIMENT #4
      indicators.tsi, // True Strength Index (TSI) - EXPERIMENT #7-1
      indicators.pmo, // Price Momentum Oscillator (PMO) - EXPERIMENT #7-3
      indicators.bollingerBandWidth, // Bollinger Band Width - EXPERIMENT #7-4
      indicators.historicalVolatility, // Historical Volatility (HV) - EXPERIMENT #8-1
      indicators.camarillaPivots, // Camarilla Pivots (CP) - EXPERIMENT #8-4
      indicators.acceleratorOscillator, // Accelerator Oscillator (AO) - EXPERIMENT #8-5
      indicators.chaikinOscillator, // Chaikin Oscillator (CO) - EXPERIMENT #9-1
      indicators.elderForceIndex, // Elder Force Index (EFI) - EXPERIMENT #9-2
      indicators.klingerVolumeOscillator, // Klinger Volume Oscillator (KVO) - EXPERIMENT #9-3
      indicators.massIndex, // Mass Index (MI) - EXPERIMENT #9-4
      indicators.priceChannel, // Price Channel (PC) - EXPERIMENT #9-5
      indicators.fisherTransform, // Fisher Transform - EXPERIMENT #10-1
      indicators.hullMovingAverage, // Hull Moving Average (HMA) - EXPERIMENT #10-2
      indicators.kaufmanAdaptiveMovingAverage, // Kaufman Adaptive Moving Average (KAMA) - EXPERIMENT #10-3
      indicators.mesaSineWave, // MESA Sine Wave - EXPERIMENT #10-4
      indicators.rainbowMovingAverage, // Rainbow Moving Average - EXPERIMENT #10-5
    ];

    return optimizedFeatures;
  }

  // EXPERIMENT #9-1: Chaikin Oscillator (CO) - Advanced Volume Indicator
  public calculateChaikinOscillator(
    high: number[],
    low: number[],
    close: number[],
    volume: number[],
    fastPeriod: number = 3,
    slowPeriod: number = 10
  ): number {
    if (
      high.length < slowPeriod ||
      low.length < slowPeriod ||
      close.length < slowPeriod ||
      volume.length < slowPeriod
    )
      return 0;

    // Calculate Money Flow Multiplier
    const moneyFlowMultipliers: number[] = [];
    for (let i = 0; i < close.length; i++) {
      const highLow = high[i] - low[i];
      const closeLow = close[i] - low[i];
      const highClose = high[i] - close[i];

      if (highLow === 0) {
        moneyFlowMultipliers.push(0);
      } else {
        const mfm = (closeLow - highClose) / highLow;
        moneyFlowMultipliers.push(mfm);
      }
    }

    // Calculate Money Flow Volume
    const moneyFlowVolumes: number[] = [];
    for (let i = 0; i < volume.length; i++) {
      moneyFlowVolumes.push(moneyFlowMultipliers[i] * volume[i]);
    }

    // Calculate Accumulation/Distribution Line (ADL)
    const adl: number[] = [];
    let cumulativeADL = 0;
    for (let i = 0; i < moneyFlowVolumes.length; i++) {
      cumulativeADL += moneyFlowVolumes[i];
      adl.push(cumulativeADL);
    }

    // Calculate EMAs of ADL
    const fastEMA = this.calculateEMA(adl, fastPeriod);
    const slowEMA = this.calculateEMA(adl, slowPeriod);

    // Chaikin Oscillator = Fast EMA - Slow EMA
    return fastEMA - slowEMA;
  }

  // EXPERIMENT #9-2: Elder Force Index (EFI) - Advanced Volume/Momentum Indicator
  public calculateElderForceIndex(
    prices: number[],
    volumes: number[],
    period: number = 13
  ): number {
    if (prices.length < period + 1 || volumes.length < period + 1) return 0;

    // Calculate Force Index for each day
    const forceIndexes: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      const priceChange = prices[i] - prices[i - 1];
      const forceIndex = priceChange * volumes[i];
      forceIndexes.push(forceIndex);
    }

    // Calculate EMA of Force Index
    const emaForceIndex = this.calculateEMA(forceIndexes, period);

    return emaForceIndex;
  }

  // EXPERIMENT #9-3: Klinger Volume Oscillator (KVO) - Advanced Volume Indicator
  public calculateKlingerVolumeOscillator(
    high: number[],
    low: number[],
    close: number[],
    volume: number[],
    shortPeriod: number = 34,
    longPeriod: number = 55
  ): number {
    if (
      high.length < longPeriod ||
      low.length < longPeriod ||
      close.length < longPeriod ||
      volume.length < longPeriod
    )
      return 0;

    // Calculate Trend Direction
    const trends: number[] = [];
    for (let i = 1; i < close.length; i++) {
      const sum1 = high[i] + low[i] + close[i];
      const sum2 = high[i - 1] + low[i - 1] + close[i - 1];
      trends.push(sum1 > sum2 ? 1 : sum1 < sum2 ? -1 : 0);
    }

    // Calculate Daily Force
    const dailyForces: number[] = [];
    for (let i = 1; i < close.length; i++) {
      const highLow = high[i] - low[i];
      const closeLow = close[i] - low[i];
      const highClose = high[i] - close[i];

      let force = 0;
      if (highLow !== 0) {
        const sv = (closeLow - highClose) / highLow;
        force = volume[i] * Math.abs(sv) * 2 * trends[i - 1];
      }
      dailyForces.push(force);
    }

    // Calculate EMAs
    const shortEMA = this.calculateEMA(dailyForces, shortPeriod);
    const longEMA = this.calculateEMA(dailyForces, longPeriod);

    // KVO = Short EMA - Long EMA
    return shortEMA - longEMA;
  }

  // EXPERIMENT #9-4: Mass Index (MI) - Advanced Volatility Indicator
  public calculateMassIndex(
    high: number[],
    low: number[],
    emaPeriod: number = 9,
    sumPeriod: number = 25
  ): number {
    if (high.length < sumPeriod || low.length < sumPeriod) return 0;

    // Calculate EMA of high-low range
    const ranges: number[] = [];
    for (let i = 0; i < high.length; i++) {
      ranges.push(high[i] - low[i]);
    }

    const emaRange = this.calculateEMA(ranges, emaPeriod);

    // Calculate EMA ratio
    const emaRatios: number[] = [];
    for (let i = 0; i < ranges.length; i++) {
      if (emaRange !== 0) {
        emaRatios.push(ranges[i] / emaRange);
      } else {
        emaRatios.push(1);
      }
    }

    // Calculate sum of EMA ratios over the sum period
    const recentRatios = emaRatios.slice(-sumPeriod);
    const massIndex = recentRatios.reduce((sum, ratio) => sum + ratio, 0);

    return massIndex;
  }

  // EXPERIMENT #9-5: Price Channel (PC) - Advanced Support/Resistance Indicator
  public calculatePriceChannel(
    high: number[],
    low: number[],
    period: number = 20
  ): number {
    if (high.length < period || low.length < period) return 0;

    // Calculate highest high and lowest low over the period
    const recentHighs = high.slice(-period);
    const recentLows = low.slice(-period);

    const highestHigh = Math.max(...recentHighs);
    const lowestLow = Math.min(...recentLows);

    // Calculate current price position within the channel
    const currentPrice = high[high.length - 1]; // Use current high as proxy for current price
    const channelRange = highestHigh - lowestLow;

    if (channelRange === 0) return 0.5; // Middle of channel if no range

    // Return position within channel (0 = bottom, 1 = top)
    return (currentPrice - lowestLow) / channelRange;
  }

  // EXPERIMENT #10-1: Fisher Transform - Advanced Oscillator
  public calculateFisherTransform(
    high: number[],
    low: number[],
    period: number = 10
  ): number {
    if (high.length < period || low.length < period) return 0;

    // Calculate median price (HLC/3)
    const medianPrices: number[] = [];
    for (let i = 0; i < high.length; i++) {
      medianPrices.push((high[i] + low[i] + (high[i] + low[i]) / 2) / 3);
    }

    // Calculate highest and lowest median prices over the period
    const recentMedianPrices = medianPrices.slice(-period);
    const highestMedian = Math.max(...recentMedianPrices);
    const lowestMedian = Math.min(...recentMedianPrices);

    // Calculate value1 (normalized median price)
    const currentMedian = medianPrices[medianPrices.length - 1];
    const range = highestMedian - lowestMedian;

    if (range === 0) return 0;

    const value1 =
      0.33 * 2 * ((currentMedian - lowestMedian) / range - 0.5) +
      0.67 *
        (medianPrices.length > 1 ? medianPrices[medianPrices.length - 2] : 0);

    // Apply Fisher Transform
    let fisher = 0;
    if (value1 > 0.99) {
      fisher = 0.999;
    } else if (value1 < -0.99) {
      fisher = -0.999;
    } else {
      fisher = 0.5 * Math.log((1 + value1) / (1 - value1));
    }

    return fisher;
  }

  // EXPERIMENT #10-2: Hull Moving Average (HMA) - Advanced Trend Indicator
  public calculateHullMovingAverage(
    prices: number[],
    period: number = 20
  ): number {
    if (prices.length < period) return prices[prices.length - 1];

    // Calculate WMA for half period
    const halfPeriod = Math.floor(period / 2);
    const sqrtPeriod = Math.floor(Math.sqrt(period));

    // Calculate WMA for full period
    let fullWMA = 0;
    let fullWeightSum = 0;
    for (let i = 0; i < period; i++) {
      const weight = period - i;
      fullWMA += prices[prices.length - period + i] * weight;
      fullWeightSum += weight;
    }
    fullWMA = fullWMA / fullWeightSum;

    // Calculate WMA for half period
    let halfWMA = 0;
    let halfWeightSum = 0;
    for (let i = 0; i < halfPeriod; i++) {
      const weight = halfPeriod - i;
      halfWMA += prices[prices.length - halfPeriod + i] * weight;
      halfWeightSum += weight;
    }
    halfWMA = halfWMA / halfWeightSum;

    // Calculate raw HMA: 2 * halfWMA - fullWMA
    const rawHMA = 2 * halfWMA - fullWMA;

    // Apply final WMA smoothing
    let finalWMA = 0;
    let finalWeightSum = 0;
    for (let i = 0; i < sqrtPeriod; i++) {
      const weight = sqrtPeriod - i;
      // Use rawHMA as the value (simplified for single value)
      finalWMA += rawHMA * weight;
      finalWeightSum += weight;
    }
    finalWMA = finalWMA / finalWeightSum;

    return finalWMA;
  }

  // EXPERIMENT #10-3: Kaufman Adaptive Moving Average (KAMA) - Advanced Trend Indicator
  public calculateKaufmanAdaptiveMovingAverage(
    prices: number[],
    period: number = 10,
    fastEMA: number = 2,
    slowEMA: number = 30
  ): number {
    if (prices.length < period + 1) return prices[prices.length - 1];

    // Calculate Efficiency Ratio (ER)
    let volatility = 0;

    for (let i = 1; i <= period; i++) {
      const priceChange = Math.abs(
        prices[prices.length - i] - prices[prices.length - i - 1]
      );
      volatility += priceChange;
    }

    const netChange = Math.abs(
      prices[prices.length - 1] - prices[prices.length - period - 1]
    );
    const efficiencyRatio = volatility > 0 ? netChange / volatility : 0;

    // Calculate Smoothing Constant (SC)
    const fastSC = 2 / (fastEMA + 1);
    const slowSC = 2 / (slowEMA + 1);
    const smoothingConstant = Math.pow(
      efficiencyRatio * (fastSC - slowSC) + slowSC,
      2
    );

    // Calculate KAMA
    const currentPrice = prices[prices.length - 1];
    const previousKAMA =
      prices.length > period + 1
        ? this.calculateKaufmanAdaptiveMovingAverage(
            prices.slice(0, -1),
            period,
            fastEMA,
            slowEMA
          )
        : prices[prices.length - 2];

    const kama =
      previousKAMA + smoothingConstant * (currentPrice - previousKAMA);

    return kama;
  }

  // EXPERIMENT #10-4: MESA Sine Wave - Advanced Oscillator
  public calculateMESASineWave(prices: number[], period: number = 20): number {
    if (prices.length < period) return 0;

    // Simplified MESA Sine Wave calculation
    // In practice, this would use Hilbert Transform, but we'll use a simplified approach

    // Calculate phase using price momentum
    const recentPrices = prices.slice(-period);
    const priceChange = recentPrices[recentPrices.length - 1] - recentPrices[0];
    const totalRange = Math.max(...recentPrices) - Math.min(...recentPrices);

    if (totalRange === 0) return 0;

    // Normalize price change to create a phase angle
    const normalizedChange = priceChange / totalRange;
    const phase = Math.atan2(normalizedChange, 1);

    // Calculate sine wave value
    const sineValue = Math.sin(phase);

    return sineValue;
  }

  // EXPERIMENT #10-5: Rainbow Moving Average - Advanced Trend Indicator
  public calculateRainbowMovingAverage(
    prices: number[],
    basePeriod: number = 2
  ): number {
    if (prices.length < basePeriod * 10) return prices[prices.length - 1];

    // Calculate multiple EMAs with different periods
    const periods = [
      basePeriod,
      basePeriod * 2,
      basePeriod * 3,
      basePeriod * 4,
      basePeriod * 5,
      basePeriod * 6,
      basePeriod * 7,
      basePeriod * 8,
      basePeriod * 9,
      basePeriod * 10,
    ];

    const emaValues: number[] = [];

    for (const period of periods) {
      const ema = this.calculateEMA(prices, period);
      emaValues.push(ema);
    }

    // Calculate current price position relative to all EMAs
    const currentPrice = prices[prices.length - 1];
    let aboveCount = 0;
    let belowCount = 0;

    for (const ema of emaValues) {
      if (currentPrice > ema) {
        aboveCount++;
      } else if (currentPrice < ema) {
        belowCount++;
      }
    }

    // Return normalized position (-1 to 1)
    // -1 = below all EMAs (bearish), 1 = above all EMAs (bullish)
    const totalLines = emaValues.length;
    const rainbowPosition = (aboveCount - belowCount) / totalLines;

    return rainbowPosition;
  }
}
