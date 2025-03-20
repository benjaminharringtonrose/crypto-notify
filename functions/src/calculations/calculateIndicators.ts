import { PERIODS } from "../constants";
import { detectDoubleTop } from "./detectDoubleTop";
import { detectHeadAndShoulders } from "./detectHeadAndShoulders";
import { detectTripleTop } from "./detectTripleTop";
import { detectTripleBottom } from "./detectTripleBottom"; // Add this if not already present
import { calculateATR } from "./calculateATR";
import { calculateEMA } from "./calculateEMA";
import { calculateFibonacciLevels } from "./calculateFibonacciLevels";
import { calculateRSI } from "./calculateRSI";
import { calculateSMA } from "./calculateSMA";
import { calculateStdDev } from "./calculateStdDev";
import { calculateStochRSI } from "./calculateStochRSI";
import { calculateVWAP } from "./calculateVWAP";

const calculateSlice = (data: number[], periods: number, offset = 0) =>
  data.slice(-periods - offset, offset === 0 ? undefined : -offset);

export const calculateIndicators = ({
  prices,
  volumes,
  currentPrice,
}: {
  prices: number[];
  volumes: number[];
  currentPrice: number;
}) => {
  const rsi = calculateRSI(calculateSlice(prices, PERIODS.RSI + 1));
  const prevRsi = calculateRSI(calculateSlice(prices, PERIODS.RSI + 1, 1));
  const sma7 = calculateSMA(calculateSlice(prices, PERIODS.SMA_SHORT));
  const sma21 = calculateSMA(calculateSlice(prices, PERIODS.SMA_LONG));
  const prevSma7 = calculateSMA(calculateSlice(prices, PERIODS.SMA_SHORT, 1));
  const prevSma21 = calculateSMA(calculateSlice(prices, PERIODS.SMA_LONG, 1));

  const ema12 = calculateEMA(
    calculateSlice(prices, PERIODS.EMA_SHORT),
    PERIODS.EMA_SHORT
  );
  const ema26 = calculateEMA(
    calculateSlice(prices, PERIODS.EMA_LONG),
    PERIODS.EMA_LONG
  );
  const macdLine = ema12 - ema26;
  const prevEma12 = calculateEMA(
    calculateSlice(prices, PERIODS.EMA_SHORT, 1),
    PERIODS.EMA_SHORT
  );
  const prevEma26 = calculateEMA(
    calculateSlice(prices, PERIODS.EMA_LONG, 1),
    PERIODS.EMA_LONG
  );
  const prevMacdLine = prevEma12 - prevEma26;
  const signalLine = calculateEMA(
    prices.slice(-PERIODS.MACD_SIGNAL).map((_, i) => {
      const slice = calculateSlice(prices, PERIODS.EMA_LONG + i, i);
      return (
        calculateEMA(slice.slice(-PERIODS.EMA_SHORT), PERIODS.EMA_SHORT) -
        calculateEMA(slice, PERIODS.EMA_LONG)
      );
    }),
    PERIODS.MACD_SIGNAL
  );

  const sma20 = calculateSMA(calculateSlice(prices, PERIODS.SMA_MEDIUM));
  const stdDev20 = calculateStdDev(
    calculateSlice(prices, PERIODS.SMA_MEDIUM),
    sma20
  );
  const upperBand = sma20 + 2 * stdDev20;
  const lowerBand = sma20 - 2 * stdDev20;

  const obvValues = [0];
  let obv = 0;
  for (let i = 1; i < prices.length; i++) {
    const priceChange = prices[i] - prices[i - 1];
    obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
    obvValues.push(obv);
  }

  const atr = calculateATR(prices, PERIODS.ATR);
  const atrBaseline = calculateATR(prices.slice(0, -1), PERIODS.ATR);
  const zScore = (currentPrice - sma20) / stdDev20;
  const vwap = calculateVWAP(prices, volumes, PERIODS.VWAP);
  const { stochRsi, signal: stochRsiSignal } = calculateStochRSI(
    prices,
    PERIODS.STOCH_RSI,
    PERIODS.STOCH_RSI,
    PERIODS.STOCH_SMOOTH
  );
  const prevStochRsi = calculateStochRSI(
    prices.slice(0, -1),
    PERIODS.STOCH_RSI,
    PERIODS.STOCH_RSI,
    PERIODS.STOCH_SMOOTH
  ).stochRsi;
  const { levels } = calculateFibonacciLevels(prices, PERIODS.FIBONACCI);
  const fib61_8 = levels[3] || 0;

  const volSmaShort = calculateSMA(
    calculateSlice(volumes, PERIODS.VOL_SMA_SHORT)
  );
  const volSmaLong = calculateSMA(
    calculateSlice(volumes, PERIODS.VOL_SMA_LONG)
  );
  const volumeOscillator =
    volSmaLong !== 0 ? ((volSmaShort - volSmaLong) / volSmaLong) * 100 : 0;
  const prevVolSmaShort = calculateSMA(
    calculateSlice(volumes, PERIODS.VOL_SMA_SHORT, 1)
  );
  const prevVolSmaLong = calculateSMA(
    calculateSlice(volumes, PERIODS.VOL_SMA_LONG, 1)
  );
  const prevVolumeOscillator =
    prevVolSmaLong !== 0
      ? ((prevVolSmaShort - prevVolSmaLong) / prevVolSmaLong) * 100
      : 0;
  const currentVolume = volumes[volumes.length - 1];
  const volSma5 = calculateSMA(calculateSlice(volumes, PERIODS.VOL_SMA_SHORT));
  const isVolumeSpike = currentVolume > volSma5 * 2;

  const isDoubleTop = detectDoubleTop(prices, volumes, currentPrice);
  const isHeadAndShoulders = detectHeadAndShoulders(
    prices,
    volumes,
    currentPrice
  );
  const isTripleTop = detectTripleTop(prices, volumes, currentPrice);
  const isTripleBottom = detectTripleBottom(prices, volumes, currentPrice);

  const momentum =
    prices.length >= PERIODS.MOMENTUM
      ? currentPrice - prices[prices.length - PERIODS.MOMENTUM]
      : 0;
  const priceChangePct =
    prices.length >= 2
      ? ((currentPrice - prices[prices.length - 2]) /
          prices[prices.length - 2]) *
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
};
