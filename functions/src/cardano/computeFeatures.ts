import { calculateATR } from "../calculations/calculateATR";
import { calculateEMA } from "../calculations/calculateEMA";
import { calculateFibonacciLevels } from "../calculations/calculateFibonacciLevels";
import { calculateIfEnoughData } from "../calculations/calculateIfEnoughData";
import { calculateRSI } from "../calculations/calculateRSI";
import { calculateSMA } from "../calculations/calculateSMA";
import { calculateStdDev } from "../calculations/calculateStdDev";
import { calculateStochRSI } from "../calculations/calculateStochRSI";
import { calculateVWAP } from "../calculations/calculateVWAP";
import { PERIODS } from "../constants";
import { detectDoubleTop } from "../calculations/detectDoubleTop";
import { detectHeadAndShoulders } from "../calculations/detectHeadAndShoulders";
import { detectTripleTop } from "../calculations/detectTripleTop";
import { MarketData } from "../types";

export const computeFeatures = ({
  prices,
  volumes,
  dayIndex,
  currentPrice,
}: MarketData) => {
  // Price-based indicators
  const rsi = calculateIfEnoughData(PERIODS.RSI, dayIndex, () =>
    calculateRSI(prices.slice(dayIndex - PERIODS.RSI + 1, dayIndex + 1))
  );
  const prevRsi = calculateIfEnoughData(PERIODS.RSI + 1, dayIndex, () =>
    calculateRSI(prices.slice(dayIndex - PERIODS.RSI, dayIndex))
  );

  // Moving averages
  const smaShort = calculateIfEnoughData(PERIODS.SMA_SHORT, dayIndex, () =>
    calculateSMA(prices.slice(dayIndex - PERIODS.SMA_SHORT + 1, dayIndex + 1))
  );
  const smaLong = calculateIfEnoughData(PERIODS.SMA_LONG, dayIndex, () =>
    calculateSMA(prices.slice(dayIndex - PERIODS.SMA_LONG + 1, dayIndex + 1))
  );
  const prevSmaShort = calculateIfEnoughData(
    PERIODS.SMA_SHORT + 1,
    dayIndex,
    () => calculateSMA(prices.slice(dayIndex - PERIODS.SMA_SHORT, dayIndex))
  );
  const prevSmaLong = calculateIfEnoughData(
    PERIODS.SMA_LONG + 1,
    dayIndex,
    () => calculateSMA(prices.slice(dayIndex - PERIODS.SMA_LONG, dayIndex))
  );

  // EMA and MACD
  const emaShort = calculateIfEnoughData(PERIODS.EMA_SHORT, dayIndex, () =>
    calculateEMA(
      prices.slice(dayIndex - PERIODS.EMA_SHORT + 1, dayIndex + 1),
      PERIODS.EMA_SHORT
    )
  );
  const emaLong = calculateIfEnoughData(PERIODS.EMA_LONG, dayIndex, () =>
    calculateEMA(
      prices.slice(dayIndex - PERIODS.EMA_LONG + 1, dayIndex + 1),
      PERIODS.EMA_LONG
    )
  );
  const macdLine = emaShort - emaLong;

  const prevEmaShort = calculateIfEnoughData(
    PERIODS.EMA_SHORT + 1,
    dayIndex,
    () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_SHORT, dayIndex),
        PERIODS.EMA_SHORT
      )
  );
  const prevEmaLong = calculateIfEnoughData(
    PERIODS.EMA_LONG + 1,
    dayIndex,
    () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_LONG, dayIndex),
        PERIODS.EMA_LONG
      )
  );
  const prevMacdLine = prevEmaShort - prevEmaLong;

  const signalLine = calculateIfEnoughData(
    PERIODS.EMA_LONG + PERIODS.MACD_SIGNAL,
    dayIndex,
    () => {
      const macdValues = prices
        .slice(dayIndex - PERIODS.MACD_SIGNAL + 1, dayIndex + 1)
        .map((_, i) => {
          const slice = prices.slice(
            dayIndex - PERIODS.EMA_LONG - i,
            dayIndex - PERIODS.EMA_SHORT - i + 1
          );
          return (
            calculateEMA(slice, PERIODS.EMA_SHORT) -
            calculateEMA(slice, PERIODS.EMA_LONG)
          );
        });
      return calculateEMA(macdValues, PERIODS.MACD_SIGNAL);
    }
  );

  // Bollinger Bands
  const smaMedium = calculateIfEnoughData(PERIODS.BOLLINGER, dayIndex, () =>
    calculateSMA(prices.slice(dayIndex - PERIODS.BOLLINGER + 1, dayIndex + 1))
  );
  const stdDev = calculateIfEnoughData(PERIODS.BOLLINGER, dayIndex, () =>
    calculateStdDev(
      prices.slice(dayIndex - PERIODS.BOLLINGER + 1, dayIndex + 1),
      smaMedium
    )
  );
  const upperBand = smaMedium + 2 * stdDev;

  // Volume indicators
  const calculateOBV = (): number => {
    let obv = 0;
    for (let i = 1; i <= dayIndex; i++) {
      const priceChange = prices[i] - prices[i - 1];
      obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
    }
    return obv / 1e6; // Normalize to match training scale
  };
  const normalizedOBV = dayIndex >= 1 ? calculateOBV() : 0;

  const volumeSmaShort = calculateIfEnoughData(
    PERIODS.VOL_SMA_SHORT,
    dayIndex,
    () =>
      calculateSMA(
        volumes.slice(dayIndex - PERIODS.VOL_SMA_SHORT + 1, dayIndex + 1)
      )
  );
  const volumeSmaLong = calculateIfEnoughData(
    PERIODS.VOL_SMA_LONG,
    dayIndex,
    () =>
      calculateSMA(
        volumes.slice(dayIndex - PERIODS.VOL_SMA_LONG + 1, dayIndex + 1)
      )
  );
  const volumeOscillator =
    volumeSmaLong !== 0
      ? ((volumeSmaShort - volumeSmaLong) / volumeSmaLong) * 100
      : 0;

  const prevVolumeSmaShort = calculateIfEnoughData(
    PERIODS.VOL_SMA_SHORT + 1,
    dayIndex,
    () =>
      calculateSMA(volumes.slice(dayIndex - PERIODS.VOL_SMA_SHORT, dayIndex))
  );
  const prevVolumeSmaLong = calculateIfEnoughData(
    PERIODS.VOL_SMA_LONG + 1,
    dayIndex,
    () => calculateSMA(volumes.slice(dayIndex - PERIODS.VOL_SMA_LONG, dayIndex))
  );
  const prevVolumeOscillator =
    prevVolumeSmaLong !== 0
      ? ((prevVolumeSmaShort - prevVolumeSmaLong) / prevVolumeSmaLong) * 100
      : 0;

  // Other technical indicators
  const atr = calculateIfEnoughData(PERIODS.ATR, dayIndex, () =>
    calculateATR(prices.slice(0, dayIndex + 1), PERIODS.ATR)
  );
  const atrBaseline = calculateIfEnoughData(PERIODS.ATR + 1, dayIndex, () =>
    calculateATR(prices.slice(0, dayIndex), PERIODS.ATR)
  );
  const zScore = calculateIfEnoughData(
    PERIODS.BOLLINGER,
    dayIndex,
    () => (currentPrice - smaMedium) / stdDev
  );
  const vwap = calculateIfEnoughData(PERIODS.VWAP, dayIndex, () =>
    calculateVWAP(
      prices.slice(0, dayIndex + 1),
      volumes.slice(0, dayIndex + 1),
      PERIODS.VWAP
    )
  );

  const { stochRsi } = calculateIfEnoughData(PERIODS.STOCH_RSI, dayIndex, () =>
    calculateStochRSI(
      prices.slice(0, dayIndex + 1),
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_SMOOTH
    )
  ) as { stochRsi: number };
  const { stochRsi: prevStochRsi } = calculateIfEnoughData(
    PERIODS.STOCH_RSI + 1,
    dayIndex,
    () =>
      calculateStochRSI(
        prices.slice(0, dayIndex),
        PERIODS.STOCH_RSI,
        PERIODS.STOCH_RSI,
        PERIODS.STOCH_SMOOTH
      )
  ) as { stochRsi: number };

  const { levels } = calculateIfEnoughData(PERIODS.FIBONACCI, dayIndex, () =>
    calculateFibonacciLevels(
      prices.slice(dayIndex - PERIODS.FIBONACCI + 1, dayIndex + 1),
      PERIODS.FIBONACCI
    )
  ) as { levels: number[] };
  const fib618 = levels[3] || 0;

  // Pattern detection
  const isDoubleTop = detectDoubleTop(
    prices.slice(0, dayIndex + 1),
    volumes.slice(0, dayIndex + 1),
    currentPrice
  );
  const isHeadAndShoulders = detectHeadAndShoulders(
    prices.slice(0, dayIndex + 1),
    volumes.slice(0, dayIndex + 1),
    currentPrice
  );
  const isTripleTop = detectTripleTop(
    prices.slice(0, dayIndex + 1),
    volumes.slice(0, dayIndex + 1),
    currentPrice
  );

  // Volume spike detection
  const currentVolume = volumes[dayIndex];
  const isVolumeSpike = currentVolume > volumeSmaShort * 2;

  // Momentum indicators
  const momentum = calculateIfEnoughData(
    PERIODS.MOMENTUM,
    dayIndex,
    () => currentPrice - prices[dayIndex - PERIODS.MOMENTUM + 1]
  );
  const priceChangePct =
    dayIndex >= 1
      ? ((currentPrice - prices[dayIndex - 1]) / prices[dayIndex - 1]) * 100
      : 0;

  // Compile features array
  return [
    rsi ?? 0,
    prevRsi ?? 0,
    smaShort,
    smaLong,
    prevSmaShort,
    prevSmaLong,
    macdLine,
    signalLine,
    currentPrice,
    upperBand,
    normalizedOBV,
    atr,
    atrBaseline,
    zScore,
    vwap,
    stochRsi,
    prevStochRsi,
    fib618,
    prices[dayIndex - 1] || prices[0],
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
};
