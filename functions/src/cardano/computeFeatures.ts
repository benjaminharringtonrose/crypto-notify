import { calculateATR } from "../calculations/calculateATR";
import { calculateEMA } from "../calculations/calculateEMA";
import { calculateFibonacciLevels } from "../calculations/calculateFibonacciLevels";
import { calculateRSI } from "../calculations/calculateRSI";
import { calculateSMA } from "../calculations/calculateSMA";
import { calculateStdDev } from "../calculations/calculateStdDev";
import { calculateStochRSI } from "../calculations/calculateStochRSI";
import { calculateVWAP } from "../calculations/calculateVWAP";
import { detectDoubleTop } from "../detections/detectDoubleTop";
import { detectHeadAndShoulders } from "../detections/detectHeadAndShoulders";
import { detectTripleTop } from "../detections/detectTripleTop";

export function computeFeatures(
  prices: number[],
  volumes: number[],
  dayIndex: number,
  currentPrice: number
): number[] {
  const rsi =
    dayIndex >= 14
      ? calculateRSI(prices.slice(dayIndex - 14, dayIndex + 1)) || 0
      : 0;
  const prevRsi =
    dayIndex >= 15
      ? calculateRSI(prices.slice(dayIndex - 15, dayIndex)) || 0
      : 0;
  const sma7 =
    dayIndex >= 6 ? calculateSMA(prices.slice(dayIndex - 6, dayIndex + 1)) : 0;
  const sma21 =
    dayIndex >= 20
      ? calculateSMA(prices.slice(dayIndex - 20, dayIndex + 1))
      : 0;
  const prevSma7 =
    dayIndex >= 7 ? calculateSMA(prices.slice(dayIndex - 7, dayIndex)) : 0;
  const prevSma21 =
    dayIndex >= 21 ? calculateSMA(prices.slice(dayIndex - 21, dayIndex)) : 0;
  const ema12 =
    dayIndex >= 11
      ? calculateEMA(prices.slice(dayIndex - 11, dayIndex + 1), 12)
      : 0;
  const ema26 =
    dayIndex >= 25
      ? calculateEMA(prices.slice(dayIndex - 25, dayIndex + 1), 26)
      : 0;
  const macdLine = ema12 - ema26;
  const signalLine =
    dayIndex >= 34
      ? calculateEMA(
          prices.slice(dayIndex - 8, dayIndex + 1).map((_, i) => {
            const slice = prices.slice(
              dayIndex - 25 - i,
              dayIndex - 13 - i + 1
            );
            return calculateEMA(slice, 12) - calculateEMA(slice, 26);
          }),
          9
        )
      : 0;
  const prevEma12 =
    dayIndex >= 12
      ? calculateEMA(prices.slice(dayIndex - 12, dayIndex), 12)
      : 0;
  const prevEma26 =
    dayIndex >= 26
      ? calculateEMA(prices.slice(dayIndex - 26, dayIndex), 26)
      : 0;
  const prevMacdLine = prevEma12 - prevEma26;
  const sma20 =
    dayIndex >= 19
      ? calculateSMA(prices.slice(dayIndex - 19, dayIndex + 1))
      : 0;
  const stdDev20 =
    dayIndex >= 19
      ? calculateStdDev(prices.slice(dayIndex - 19, dayIndex + 1), sma20)
      : 0;
  const upperBand = sma20 + 2 * stdDev20;

  let obv = 0;
  const obvValues = [0];
  for (let i = 1; i <= dayIndex; i++) {
    const priceChange = prices[i] - prices[i - 1];
    obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
    obvValues.push(obv);
  }
  const normalizedObv = obvValues[obvValues.length - 1] / 1e6; // Scale OBV to match training

  const atr =
    dayIndex >= 13 ? calculateATR(prices.slice(0, dayIndex + 1), 14) : 0;
  const atrBaseline =
    dayIndex >= 14 ? calculateATR(prices.slice(0, dayIndex), 14) : 0;
  const zScore = dayIndex >= 19 ? (currentPrice - sma20) / stdDev20 : 0;
  const vwap =
    dayIndex >= 6
      ? calculateVWAP(
          prices.slice(0, dayIndex + 1),
          volumes.slice(0, dayIndex + 1),
          7
        )
      : 0;
  const { stochRsi } =
    dayIndex >= 13
      ? calculateStochRSI(prices.slice(0, dayIndex + 1), 14, 14, 3)
      : { stochRsi: 0 };
  const { stochRsi: prevStochRsi } =
    dayIndex >= 14
      ? calculateStochRSI(prices.slice(0, dayIndex), 14, 14, 3)
      : { stochRsi: 0 };
  const { levels } =
    dayIndex >= 29
      ? calculateFibonacciLevels(prices.slice(dayIndex - 29, dayIndex + 1), 30)
      : { levels: [0, 0, 0, 0, 0] };
  const fib61_8 = levels[3];
  const volSmaShort =
    dayIndex >= 4 ? calculateSMA(volumes.slice(dayIndex - 4, dayIndex + 1)) : 0;
  const volSmaLong =
    dayIndex >= 13
      ? calculateSMA(volumes.slice(dayIndex - 13, dayIndex + 1))
      : 0;
  const volumeOscillator =
    volSmaLong !== 0 ? ((volSmaShort - volSmaLong) / volSmaLong) * 100 : 0;
  const prevVolSmaShort =
    dayIndex >= 5 ? calculateSMA(volumes.slice(dayIndex - 5, dayIndex)) : 0;
  const prevVolSmaLong =
    dayIndex >= 14 ? calculateSMA(volumes.slice(dayIndex - 14, dayIndex)) : 0;
  const prevVolumeOscillator =
    prevVolSmaLong !== 0
      ? ((prevVolSmaShort - prevVolSmaLong) / prevVolSmaLong) * 100
      : 0;
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
  const currentVolume = volumes[dayIndex];
  const volSma5 =
    dayIndex >= 4 ? calculateSMA(volumes.slice(dayIndex - 4, dayIndex + 1)) : 0;
  const isVolumeSpike = currentVolume > volSma5 * 2;

  const momentum = dayIndex >= 9 ? currentPrice - prices[dayIndex - 9] : 0;
  const priceChangePct =
    dayIndex >= 1
      ? ((currentPrice - prices[dayIndex - 1]) / prices[dayIndex - 1]) * 100
      : 0;

  const features = [
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
    normalizedObv,
    atr,
    atrBaseline,
    zScore,
    vwap,
    stochRsi,
    prevStochRsi,
    fib61_8,
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

  return features;
}
