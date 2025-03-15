import * as admin from "firebase-admin";
import axios, { AxiosResponse } from "axios";
import { COINGECKO_API_URL } from "../constants";
import { calculateRSI } from "./calculateRSI";
import { calculateSMA } from "./calculateSMA";
import { CoinGeckoMarketChartResponse, SellDecision } from "./types";
import { calculateEMA } from "./calculateEMA";
import { calculateStdDev } from "./calculateStdDev";

/**
 * Determines whether to sell a cryptocurrency based on RSI, SMA, MACD, and Bollinger Bands.
 *
 * Fetches current and historical price data from CoinGecko, calculates multiple indicators,
 * and recommends "sell" if RSI > 70 (overbought), SMA shows a death cross, MACD shows a bearish crossover,
 * and price exceeds the upper Bollinger Band.
 *
 * @param {string} cryptoSymbol - The CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
 * @returns {Promise<SellDecision | Error>} Trading decision details or an error.
 */

export const shouldSell = async (
  cryptoSymbol: string
): Promise<SellDecision | Error> => {
  try {
    // Fetch current price
    const priceResponse = await axios.get(
      `${COINGECKO_API_URL}/simple/price?ids=${cryptoSymbol.toLowerCase()}&vs_currencies=usd`
    );
    const currentPrice = priceResponse.data[cryptoSymbol.toLowerCase()].usd;

    // Fetch 30-day historical data (for RSI 14 and 21-day SMA)
    const historicalResponse: AxiosResponse<CoinGeckoMarketChartResponse> =
      await axios.get(
        `${COINGECKO_API_URL}/coins/${cryptoSymbol.toLowerCase()}/market_chart?vs_currency=usd&days=30`
      );
    const prices = historicalResponse.data.prices.map(([_, price]) => price);

    // RSI
    const rsi = calculateRSI(prices.slice(-15));

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

    // Bollinger Bands (20-day)
    const sma20 = calculateSMA(prices.slice(-20));
    const stdDev20 = calculateStdDev(prices.slice(-20), sma20);
    const upperBand = sma20 + 2 * stdDev20;
    const lowerBand = sma20 - 2 * stdDev20;

    // Decision logic
    const isOverbought = rsi && rsi > 70;
    const isDeathCross = sma7 < sma21 && prevSma7 >= prevSma21;
    const isMacdBearish = macdLine < signalLine;
    const isAboveUpperBand = currentPrice > upperBand;
    const recommendation =
      isOverbought && isDeathCross && isMacdBearish && isAboveUpperBand
        ? "sell"
        : "hold";

    return {
      cryptoSymbol,
      currentPrice,
      rsi: rsi?.toFixed(2),
      sma7: sma7.toFixed(2),
      sma21: sma21.toFixed(2),
      macdLine: macdLine.toFixed(2),
      signalLine: signalLine.toFixed(2),
      sma20: sma20.toFixed(2), // Bollinger Bands
      upperBand: upperBand.toFixed(2),
      lowerBand: lowerBand.toFixed(2),
      recommendation,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    };
  } catch (error: any) {
    console.error("Error in shouldSellCrypto:", error.message);
    return error;
  }
};
