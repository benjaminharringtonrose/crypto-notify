import * as admin from "firebase-admin";
import axios, { AxiosResponse } from "axios";
import { COINGECKO_API_URL } from "../constants";
import { calculateRSI } from "./calculateRSI";
import { calculateSMA } from "./calculateSMA";
import { calculateEMA } from "./calculateEMA";
import { calculateStdDev } from "./calculateStdDev";
import { calculateATR } from "./calculateATR";
import { calculateVWAP } from "./calculateVWAP";
import { calculateStochRSI } from "./calculateStochRSI";
import { calculateFibonacciLevels } from "./calculateFibonacciLevels";
import { CoinGeckoMarketChartResponse, SellDecision } from "./types";

/**
 * Determines whether to sell a cryptocurrency using RSI, SMA, MACD, Bollinger Bands, OBV, divergence, ATR, Z-Score, VWAP, StochRSI, and Fibonacci levels.
 *
 * Combines multiple indicators for a robust sell decision and returns which conditions were met.
 *
 * @param {string} cryptoSymbol - The CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
 * @returns {Promise<SellDecision | Error>} Trading decision details, including met conditions, or an error.
 */
export const calculateSellDecision = async (
  cryptoSymbol: string
): Promise<SellDecision | Error> => {
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

    // RSI
    const rsi = calculateRSI(prices.slice(-15));
    const prevRsi = calculateRSI(prices.slice(-16, -1));

    // SMA
    const sma7 = calculateSMA(prices.slice(-7));
    const sma21 = calculateSMA(prices.slice(-21));
    const prevSma7 = calculateSMA(prices.slice(-8, -1));
    const prevSma21 = calculateSMA(prices.slice(-22, -1));

    // MACD
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

    // Bollinger Bands (and Z-Score prep)
    const sma20 = calculateSMA(prices.slice(-20));
    const stdDev20 = calculateStdDev(prices.slice(-20), sma20);
    const upperBand = sma20 + 2 * stdDev20;

    // OBV
    let obv = 0;
    const obvValues = [0];
    for (let i = 1; i < prices.length; i++) {
      const priceChange = prices[i] - prices[i - 1];
      obv += priceChange > 0 ? volumes[i] : priceChange < 0 ? -volumes[i] : 0;
      obvValues.push(obv);
    }

    // ATR
    const atr = calculateATR(prices, 14);
    const atrBaseline = calculateATR(prices.slice(0, -1), 14);

    // Z-Score
    const zScore = (currentPrice - sma20) / stdDev20;

    // VWAP
    const vwap = calculateVWAP(prices, volumes, 7);

    // StochRSI
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

    // Fibonacci Levels
    const { levels: fibLevels } = calculateFibonacciLevels(prices, 30);
    const fib61_8 = fibLevels[3]; // 61.8% level

    // Decision logic with named conditions
    const conditions: { name: string; met: boolean }[] = [
      { name: "RSI Overbought", met: !!rsi && rsi > 70 },
      { name: "SMA Death Cross", met: sma7 < sma21 && prevSma7 >= prevSma21 },
      { name: "MACD Bearish", met: macdLine < signalLine },
      { name: "Above Bollinger Upper", met: currentPrice > upperBand },
      {
        name: "OBV Dropping",
        met: obvValues[obvValues.length - 1] < obvValues[obvValues.length - 2],
      },
      {
        name: "Bearish RSI Divergence",
        met:
          currentPrice > prices[prices.length - 2] &&
          !!rsi &&
          !!prevRsi &&
          rsi < prevRsi,
      },
      { name: "High ATR Volatility", met: atr > atrBaseline * 2 },
      { name: "Z-Score High", met: zScore > 2 },
      { name: "Above VWAP", met: currentPrice > vwap * 1.05 },
      {
        name: "StochRSI Overbought",
        met: stochRsi > 80 && stochRsi < prevStochRsi,
      },
      {
        name: "Near Fibonacci 61.8%",
        met: currentPrice >= fib61_8 * 0.99 && currentPrice <= fib61_8 * 1.01,
      },
    ];

    const metConditions = conditions
      .filter((cond) => cond.met)
      .map((cond) => cond.name);
    const recommendation = metConditions.length >= 6 ? "sell" : "hold";

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
      lowerBand: (sma20 - 2 * stdDev20).toFixed(2),
      obv: obv.toFixed(0),
      atr: atr.toFixed(2),
      zScore: zScore.toFixed(2),
      vwap: vwap.toFixed(2),
      stochRsi: stochRsi.toFixed(2),
      stochRsiSignal: stochRsiSignal.toFixed(2),
      fib61_8: fib61_8.toFixed(2),
      conditionsMet: metConditions, // New: Array of met condition names
      recommendation,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    };
  } catch (error: any) {
    console.error("Error in shouldSell:", error.message);
    return error;
  }
};
