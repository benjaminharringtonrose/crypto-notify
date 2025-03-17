import dotenv from "dotenv";
import * as admin from "firebase-admin";
import axios, { AxiosResponse } from "axios";
import { AVERAGE_BUY_PRICE, COINGECKO_API_URL } from "../constants";
import { calculateRSI } from "../calculations/calculateRSI";
import { calculateSMA } from "../calculations/calculateSMA";
import { calculateEMA } from "../calculations/calculateEMA";
import { calculateStdDev } from "../calculations/calculateStdDev";
import { calculateATR } from "../calculations/calculateATR";
import { calculateVWAP } from "../calculations/calculateVWAP";
import { calculateStochRSI } from "../calculations/calculateStochRSI";
import { calculateFibonacciLevels } from "../calculations/calculateFibonacciLevels";
import { detectDoubleTop } from "../detections/detectDoubleTop";
import { detectHeadAndShoulders } from "../detections/detectHeadAndShoulders";
import { detectTripleTop } from "../detections/detectTripleTop";
import { predictTrade } from "./predictTrade";
import {
  CoinGeckoMarketChartResponse,
  Recommendation,
  TradeDecision,
} from "../types";

dotenv.config();

export const determineTrade = async (
  cryptoSymbol: string
): Promise<TradeDecision> => {
  try {
    const priceResponse = await axios.get(
      `${COINGECKO_API_URL}/simple/price?ids=${cryptoSymbol.toLowerCase()}&vs_currencies=usd`
    );
    const currentPrice = priceResponse.data[cryptoSymbol.toLowerCase()].usd;

    const historicalResponse: AxiosResponse<CoinGeckoMarketChartResponse> =
      await axios.get(
        `${COINGECKO_API_URL}/coins/${cryptoSymbol.toLowerCase()}/market_chart?vs_currency=usd&days=30`
      );
    const prices = historicalResponse.data.prices.map(([_, price]) => price);
    const volumes = historicalResponse.data.total_volumes.map(
      ([_, vol]) => vol
    );

    const rsi = calculateRSI(prices.slice(-15));
    const prevRsi = calculateRSI(prices.slice(-16, -1));
    const sma7 = calculateSMA(prices.slice(-7));
    const sma21 = calculateSMA(prices.slice(-21));
    const prevSma7 = calculateSMA(prices.slice(-8, -1));
    const prevSma21 = calculateSMA(prices.slice(-22, -1));
    const ema12 = calculateEMA(prices.slice(-12), 12);
    const ema26 = calculateEMA(prices.slice(-26), 26);
    const macdLine = ema12 - ema26;
    const signalLine = calculateEMA(
      prices.slice(-9).map((_, i) => {
        const slice = prices.slice(-26 - i, -14 - i);
        return calculateEMA(slice.slice(-12), 12) - calculateEMA(slice, 26);
      }),
      9
    );
    const prevEma12 = calculateEMA(prices.slice(-13, -1), 12);
    const prevEma26 = calculateEMA(prices.slice(-27, -1), 26);
    const prevMacdLine = prevEma12 - prevEma26;
    const sma20 = calculateSMA(prices.slice(-20));
    const stdDev20 = calculateStdDev(prices.slice(-20), sma20);
    const upperBand = sma20 + 2 * stdDev20;
    const lowerBand = sma20 - 2 * stdDev20;

    let obv = 0;
    const obvValues = [0];
    for (let i = 1; i < prices.length; i++) {
      const priceChange = prices[i] - prices[i - 1];
      obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
      obvValues.push(obv);
    }

    const atr = calculateATR(prices, 14);
    const atrBaseline = calculateATR(prices.slice(0, -1), 14);
    const zScore = (currentPrice - sma20) / stdDev20;
    const vwap = calculateVWAP(prices, volumes, 7);
    const { stochRsi, signal: stochRsiSignal } = calculateStochRSI(
      prices,
      14,
      14,
      3
    );
    const prevStochRsi = calculateStochRSI(
      prices.slice(0, -1),
      14,
      14,
      3
    ).stochRsi;
    const { levels: fibLevels } = calculateFibonacciLevels(prices, 30);
    const fib61_8 = fibLevels[3];
    const volSmaShort = calculateSMA(volumes.slice(-5));
    const volSmaLong = calculateSMA(volumes.slice(-14));
    const volumeOscillator = ((volSmaShort - volSmaLong) / volSmaLong) * 100;
    const prevVolSmaShort = calculateSMA(volumes.slice(-6, -1));
    const prevVolSmaLong = calculateSMA(volumes.slice(-15, -1));
    const prevVolumeOscillator =
      ((prevVolSmaShort - prevVolSmaLong) / prevVolSmaLong) * 100;
    const currentVolume = volumes[volumes.length - 1];
    const volSma5 = calculateSMA(volumes.slice(-5));
    const isVolumeSpike = currentVolume > volSma5 * 2;
    const isDoubleTop = detectDoubleTop(prices, volumes, currentPrice);
    const isHeadAndShoulders = detectHeadAndShoulders(
      prices,
      volumes,
      currentPrice
    );
    const isTripleTop = detectTripleTop(prices, volumes, currentPrice);
    const momentum =
      prices.length >= 10 ? currentPrice - prices[prices.length - 10] : 0;
    const priceChangePct =
      prices.length >= 2
        ? ((currentPrice - prices[prices.length - 2]) /
            prices[prices.length - 2]) *
          100
        : 0;

    const { metConditions, probabilities, recommendation } = await predictTrade(
      {
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
        obvValues,
        atr,
        atrBaseline,
        zScore,
        vwap,
        stochRsi,
        prevStochRsi,
        fib61_8,
        prices,
        volumeOscillator,
        prevVolumeOscillator,
        isDoubleTop,
        isHeadAndShoulders,
        prevMacdLine,
        isTripleTop,
        isVolumeSpike,
        momentum,
        priceChangePct,
      }
    );

    const recommendationBasedOnBuyPrice =
      currentPrice < AVERAGE_BUY_PRICE && recommendation === Recommendation.Sell
        ? Recommendation.HoldBasedOnBuyPrice
        : recommendation;

    return {
      cryptoSymbol,
      currentPrice,
      rsi: rsi?.toFixed(2),
      sma7: sma7.toFixed(2),
      sma21: sma21.toFixed(2),
      macdLine: macdLine.toFixed(2),
      signalLine: signalLine.toFixed(2),
      sma20: sma20.toFixed(2),
      upperBand: upperBand.toFixed(2),
      lowerBand: lowerBand.toFixed(2),
      obv: obv.toFixed(0),
      atr: atr.toFixed(2),
      zScore: zScore.toFixed(2),
      vwap: vwap.toFixed(2),
      stochRsi: stochRsi.toFixed(2),
      stochRsiSignal: stochRsiSignal.toFixed(2),
      fib61_8: fib61_8.toFixed(2),
      volumeOscillator: volumeOscillator.toFixed(2),
      metConditions,
      probabilities,
      recommendation: recommendationBasedOnBuyPrice,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    };
  } catch (error: any) {
    console.error("Error in determineTradeADA:", error.message);
    throw error;
  }
};
