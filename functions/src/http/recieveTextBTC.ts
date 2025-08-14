import { https } from "firebase-functions";
import {
  CoinbaseProductIds,
  CryptoIds,
  Granularity,
  Recommendation,
} from "../types";
import {
  formatAnalysisResults,
  formatPredictionExplanation,
  sendSMS,
} from "../utils";
import { TradeModelPredictor } from "../bitcoin/TradeModelPredictor";
import { TIME_CONVERSIONS } from "../constants";
import { CoinbaseService } from "../api/CoinbaseService";

const CONFIG: https.HttpsOptions = {
  memory: "512MiB",
};

export const receiveTextBTC = https.onRequest(CONFIG, async (_, response) => {
  try {
    const coinbase = new CoinbaseService({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });

    const predictor = new TradeModelPredictor();

    const now = Math.floor(Date.now() / 1000);
    const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

    const currentPrice = await coinbase.getCurrentPrice(CoinbaseProductIds.BTC);

    const btcData = await coinbase.getPricesAndVolumes({
      product_id: CoinbaseProductIds.BTC,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    if (btcData.prices.length < TIME_CONVERSIONS.ONE_MONTH_IN_DAYS) {
      console.log("Insufficient historical data for prediction");
      throw new Error("Insufficient historical data for prediction");
    }

    const prediction = await predictor.predict(btcData.prices, btcData.volumes);

    const { buyProb, sellProb, confidence, momentum, trendSlope, atr } =
      prediction;

    let recommendation = null;
    const threshold = 0.73;

    if (buyProb > sellProb && buyProb > threshold) {
      recommendation = Recommendation.Buy;
    } else if (sellProb > buyProb && sellProb > threshold) {
      recommendation = Recommendation.Sell;
    } else {
      recommendation = Recommendation.Hold;
    }

    const analysisMessage = formatAnalysisResults({
      cryptoSymbol: CryptoIds.Bitcoin,
      currentPrice,
      probabilities: {
        buy: buyProb,
        sell: sellProb,
        hold: 1 - Math.max(buyProb, sellProb),
      },
      recommendation,
    });

    const explanationMessage = formatPredictionExplanation({
      recommendation,
      buyProb,
      sellProb,
      confidence,
      momentum,
      trendSlope,
      atr,
    });

    const smsMessage = `${analysisMessage}\n\n${explanationMessage}`;

    await sendSMS(smsMessage);
    response.status(200).send("Reply received and logged!");
  } catch (error: any) {
    await sendSMS("Incorrect name, please try again.");
    console.log("Error:::", JSON.stringify(error));
    response.status(500).send("Error logging reply: " + error.message);
  }
});
