import { https } from "firebase-functions";
import {
  CoinbaseProductIds,
  CryptoIds,
  Granularity,
  Recommendation,
} from "../types";
import { formatAnalysisResults, sendSMS } from "../utils";
import { TradingStrategy } from "../cardano/TradingStrategy";
import { TradeExecutor } from "../cardano/TradeExecutor";
import { COINBASE_CONSTANTS, MODEL_CONSTANTS } from "../constants";

const strategy = new TradingStrategy();

const trader = new TradeExecutor({
  apiKey: process.env.COINBASE_API_KEY,
  apiSecret: process.env.COINBASE_API_SECRET,
});

const CONFIG: https.HttpsOptions = {
  memory: "512MiB",
};

const TIMESTEPS_IN_SECONDS =
  MODEL_CONSTANTS.TIMESTEPS * COINBASE_CONSTANTS.SECONDS_PER_DAY;

export const receiveTextADA = https.onRequest(CONFIG, async (_, response) => {
  try {
    const now = Math.floor(Date.now() / 1000);
    const start = now - TIMESTEPS_IN_SECONDS;

    const currentPrice = await trader.getCurrentPrice(CoinbaseProductIds.ADA);

    const adaData = await trader.getMarketData({
      product_id: CoinbaseProductIds.ADA,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    const btcData = await trader.getMarketData({
      product_id: CoinbaseProductIds.BTC,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    const balances = await trader.getAccountBalances();
    const capital = parseFloat(balances.usd?.available_balance.value || "0");
    const holdings = parseFloat(balances.ada?.available_balance.value || "0");

    if (adaData.prices.length < 30 || btcData.prices.length < 30) {
      console.log("Insufficient historical data for prediction");
      throw new Error("Insufficient historical data for prediction");
    }

    const { trade, buyProb, sellProb } = await strategy.decideTrade({
      adaPrices: adaData.prices,
      adaVolumes: adaData.volumes,
      btcPrices: btcData.prices,
      btcVolumes: btcData.volumes,
      capital,
      holdings,
      lastBuyPrice: undefined, // Persist this in Firestore if needed
      peakPrice: undefined,
      buyTimestamp: undefined,
      currentTimestamp: new Date().toISOString(),
    });

    const smsMessage = formatAnalysisResults({
      cryptoSymbol: CryptoIds.Cardano,
      currentPrice,
      probabilities: {
        buy: buyProb,
        sell: sellProb,
        hold: 1 - Math.max(buyProb, sellProb),
      },
      recommendation: trade?.type ?? Recommendation.Hold,
    });

    await sendSMS(smsMessage);
    response.status(200).send("Reply received and logged!");
  } catch (error: any) {
    await sendSMS("Incorrect name, please try again.");
    console.log("Error:::", JSON.stringify(error));
    response.status(500).send("Error logging reply: " + error.message);
  }
});
