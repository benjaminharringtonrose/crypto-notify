import { https } from "firebase-functions";
import { CryptoIds, Currencies, Recommendation } from "../types";
import { formatAnalysisResults, sendSMS } from "../utils";
import { TradeModelPredictor } from "../cardano/TradeModelPredictor";
import { getHistoricalData } from "../api/getHistoricalData";
import { getCurrentPrice } from "../api/getCurrentPrice";

export const receiveTextADA = https.onRequest(
  { memory: "512MiB" },
  async (_, response) => {
    try {
      const predictor = new TradeModelPredictor();

      const days = 30;
      const adaData = await getHistoricalData("ADA", days);
      const btcData = await getHistoricalData("BTC", days);
      const currentPrice = await getCurrentPrice({
        id: CryptoIds.Cardano,
        currency: Currencies.USD,
      });

      if (adaData.prices.length < 30 || btcData.prices.length < 30) {
        throw new Error("Insufficient historical data for prediction");
      }

      const { buyProb, sellProb, confidence } = await predictor.predict(
        adaData.prices,
        adaData.volumes,
        btcData.prices,
        btcData.volumes
      );

      const probabilities = {
        buy: buyProb,
        sell: sellProb,
        hold: 1 - Math.max(buyProb, sellProb),
      };

      let recommendation: Recommendation;
      if (buyProb > sellProb && confidence >= 0.7) {
        recommendation = Recommendation.Buy;
      } else if (sellProb > buyProb && confidence >= 0.7) {
        recommendation = Recommendation.Sell;
      } else {
        recommendation = Recommendation.Hold;
      }

      const smsMessage = formatAnalysisResults({
        cryptoSymbol: CryptoIds.Cardano,
        currentPrice,
        probabilities,
        recommendation,
      });

      await sendSMS(smsMessage);
      response.status(200).send("Reply received and logged!");
    } catch (error: any) {
      await sendSMS("Incorrect name, please try again.");
      response.status(500).send("Error logging reply: " + error.message);
    }
  }
);
