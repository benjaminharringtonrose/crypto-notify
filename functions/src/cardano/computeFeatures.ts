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
import { detectTripleBottom } from "../calculations/detectTripleBottom";
import { MarketData } from "../types";

export const computeFeatures = ({
  prices,
  volumes,
  dayIndex,
  currentPrice,
  isBTC = false, // Add flag to differentiate BTC
}: MarketData & { isBTC?: boolean }) => {
  if (!prices || !volumes || dayIndex < 0 || dayIndex >= prices.length) {
    return Array(isBTC ? 28 : 29).fill(0); // 28 for BTC, 29 for ADA
  }

  const rsi =
    calculateIfEnoughData(PERIODS.RSI, dayIndex, () =>
      calculateRSI(prices.slice(dayIndex - PERIODS.RSI + 1, dayIndex + 1))
    ) ?? 0;
  const prevRsi =
    calculateIfEnoughData(PERIODS.RSI + 1, dayIndex, () =>
      calculateRSI(prices.slice(dayIndex - PERIODS.RSI, dayIndex))
    ) ?? 0;

  const smaShort =
    calculateIfEnoughData(PERIODS.SMA_SHORT, dayIndex, () =>
      calculateSMA(prices.slice(dayIndex - PERIODS.SMA_SHORT + 1, dayIndex + 1))
    ) ?? prices[dayIndex];
  const smaLong =
    calculateIfEnoughData(PERIODS.SMA_LONG, dayIndex, () =>
      calculateSMA(prices.slice(dayIndex - PERIODS.SMA_LONG + 1, dayIndex + 1))
    ) ?? prices[dayIndex];
  const prevSmaShort =
    calculateIfEnoughData(PERIODS.SMA_SHORT + 1, dayIndex, () =>
      calculateSMA(prices.slice(dayIndex - PERIODS.SMA_SHORT, dayIndex))
    ) ??
    prices[dayIndex - 1] ??
    prices[dayIndex];
  const prevSmaLong =
    calculateIfEnoughData(PERIODS.SMA_LONG + 1, dayIndex, () =>
      calculateSMA(prices.slice(dayIndex - PERIODS.SMA_LONG, dayIndex))
    ) ??
    prices[dayIndex - 1] ??
    prices[dayIndex];

  const emaShort =
    calculateIfEnoughData(PERIODS.EMA_SHORT, dayIndex, () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_SHORT + 1, dayIndex + 1),
        PERIODS.EMA_SHORT
      )
    ) ?? prices[dayIndex];
  const emaLong =
    calculateIfEnoughData(PERIODS.EMA_LONG, dayIndex, () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_LONG + 1, dayIndex + 1),
        PERIODS.EMA_LONG
      )
    ) ?? prices[dayIndex];
  const macdLine = emaShort - emaLong;

  const prevEmaShort =
    calculateIfEnoughData(PERIODS.EMA_SHORT + 1, dayIndex, () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_SHORT, dayIndex),
        PERIODS.EMA_SHORT
      )
    ) ??
    prices[dayIndex - 1] ??
    prices[dayIndex];
  const prevEmaLong =
    calculateIfEnoughData(PERIODS.EMA_LONG + 1, dayIndex, () =>
      calculateEMA(
        prices.slice(dayIndex - PERIODS.EMA_LONG, dayIndex),
        PERIODS.EMA_LONG
      )
    ) ??
    prices[dayIndex - 1] ??
    prices[dayIndex];
  const prevMacdLine = prevEmaShort - prevEmaLong;

  const signalLine =
    calculateIfEnoughData(
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
    ) ?? 0;

  const smaMedium =
    calculateIfEnoughData(PERIODS.BOLLINGER, dayIndex, () =>
      calculateSMA(prices.slice(dayIndex - PERIODS.BOLLINGER + 1, dayIndex + 1))
    ) ?? prices[dayIndex];
  const stdDev =
    calculateIfEnoughData(PERIODS.BOLLINGER, dayIndex, () =>
      calculateStdDev(
        prices.slice(dayIndex - PERIODS.BOLLINGER + 1, dayIndex + 1),
        smaMedium
      )
    ) ?? 0;
  const upperBand = smaMedium + 2 * stdDev;

  const calculateOBV = (): number => {
    let obv = 0;
    for (let i = 1; i <= dayIndex; i++) {
      const priceChange = prices[i] - prices[i - 1];
      obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
    }
    return obv / 1e6;
  };
  const normalizedOBV = dayIndex >= 1 ? calculateOBV() : 0;

  const volumeSmaShort =
    calculateIfEnoughData(PERIODS.VOL_SMA_SHORT, dayIndex, () =>
      calculateSMA(
        volumes.slice(dayIndex - PERIODS.VOL_SMA_SHORT + 1, dayIndex + 1)
      )
    ) ?? volumes[dayIndex];
  const volumeSmaLong =
    calculateIfEnoughData(PERIODS.VOL_SMA_LONG, dayIndex, () =>
      calculateSMA(
        volumes.slice(dayIndex - PERIODS.VOL_SMA_LONG + 1, dayIndex + 1)
      )
    ) ?? volumes[dayIndex];
  const volumeOscillator =
    volumeSmaLong !== 0
      ? ((volumeSmaShort - volumeSmaLong) / volumeSmaLong) * 100
      : 0;

  const prevVolumeSmaShort =
    calculateIfEnoughData(PERIODS.VOL_SMA_SHORT + 1, dayIndex, () =>
      calculateSMA(volumes.slice(dayIndex - PERIODS.VOL_SMA_SHORT, dayIndex))
    ) ??
    volumes[dayIndex - 1] ??
    volumes[dayIndex];
  const prevVolumeSmaLong =
    calculateIfEnoughData(PERIODS.VOL_SMA_LONG + 1, dayIndex, () =>
      calculateSMA(volumes.slice(dayIndex - PERIODS.VOL_SMA_LONG, dayIndex))
    ) ??
    volumes[dayIndex - 1] ??
    volumes[dayIndex];
  const prevVolumeOscillator =
    prevVolumeSmaLong !== 0
      ? ((prevVolumeSmaShort - prevVolumeSmaLong) / prevVolumeSmaLong) * 100
      : 0;

  const atr =
    calculateIfEnoughData(PERIODS.ATR, dayIndex, () =>
      calculateATR(prices.slice(0, dayIndex + 1), PERIODS.ATR)
    ) ?? 0;
  const atrBaseline =
    calculateIfEnoughData(PERIODS.ATR + 1, dayIndex, () =>
      calculateATR(prices.slice(0, dayIndex), PERIODS.ATR)
    ) ?? 0;
  const zScore =
    calculateIfEnoughData(
      PERIODS.BOLLINGER,
      dayIndex,
      () => (currentPrice - smaMedium) / stdDev
    ) ?? 0;
  const vwap =
    calculateIfEnoughData(PERIODS.VWAP, dayIndex, () =>
      calculateVWAP(
        prices.slice(0, dayIndex + 1),
        volumes.slice(0, dayIndex + 1),
        PERIODS.VWAP
      )
    ) ?? prices[dayIndex];

  const { stochRsi } = (calculateIfEnoughData(PERIODS.STOCH_RSI, dayIndex, () =>
    calculateStochRSI(
      prices.slice(0, dayIndex + 1),
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_RSI,
      PERIODS.STOCH_SMOOTH
    )
  ) ?? { stochRsi: 0 }) as { stochRsi: number };
  const { stochRsi: prevStochRsi } = (calculateIfEnoughData(
    PERIODS.STOCH_RSI + 1,
    dayIndex,
    () =>
      calculateStochRSI(
        prices.slice(0, dayIndex),
        PERIODS.STOCH_RSI,
        PERIODS.STOCH_RSI,
        PERIODS.STOCH_SMOOTH
      )
  ) ?? { stochRsi: 0 }) as { stochRsi: number };

  const { levels } = (calculateIfEnoughData(PERIODS.FIBONACCI, dayIndex, () =>
    calculateFibonacciLevels(
      prices.slice(dayIndex - PERIODS.FIBONACCI + 1, dayIndex + 1),
      PERIODS.FIBONACCI
    )
  ) ?? { levels: [0, 0, 0, 0] }) as { levels: number[] };
  const fib618 = levels[3] || 0;

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

  const currentVolume = volumes[dayIndex] ?? 0;
  const isVolumeSpike = currentVolume > volumeSmaShort * 2;

  const momentum =
    calculateIfEnoughData(
      PERIODS.MOMENTUM,
      dayIndex,
      () => currentPrice - prices[dayIndex - PERIODS.MOMENTUM + 1]
    ) ?? 0;
  const priceChangePct =
    dayIndex >= 1
      ? ((currentPrice - prices[dayIndex - 1]) / prices[dayIndex - 1]) * 100
      : 0;

  const baseFeatures = [
    rsi,
    prevRsi,
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

  if (isBTC) return baseFeatures; // 28 features for BTC
  const isTripleBottom = detectTripleBottom(
    prices.slice(0, dayIndex + 1),
    volumes.slice(0, dayIndex + 1),
    currentPrice
  );
  return [...baseFeatures, isTripleBottom ? 1 : 0]; // 29 features for ADA
};
