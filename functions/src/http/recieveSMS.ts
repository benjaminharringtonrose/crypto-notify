import { https } from "firebase-functions";
import { CryptoIds, RecieveSMSRequest, Recommendation } from "../types";
import { formatAnalysisResults, sendSMS } from "../utils";
import { TradeModelPredictor } from "../cardano/TradeModelPredictor";
import { MEMORY } from "../constants";
import { getHistoricalData } from "../api/getHistoricalData";

export const receiveSMS = https.onRequest(
  { memory: MEMORY },
  async (request: RecieveSMSRequest, response) => {
    try {
      const replyText = request.body.text || "No message";

      if (replyText.trim().toLowerCase() !== "cardano") {
        return;
      }

      const predictor = new TradeModelPredictor();

      // Fetch historical data for the last 31 days (timesteps + 1 for current price)
      const days = 31;
      const adaData = await getHistoricalData("ADA", days);
      const btcData = await getHistoricalData("BTC", days);

      if (adaData.prices.length < 30 || btcData.prices.length < 30) {
        throw new Error("Insufficient historical data for prediction");
      }

      const { buyProb, sellProb, confidence } = await predictor.predict(
        adaData.prices,
        adaData.volumes,
        btcData.prices,
        btcData.volumes
      );

      const currentPrice = adaData.prices[adaData.prices.length - 1];
      const probabilities = {
        buy: buyProb,
        sell: sellProb,
        hold: 1 - Math.max(buyProb, sellProb), // Simple heuristic for hold probability
      };

      // Determine recommendation based on thresholds
      const buyThreshold = 0.85; // Match V3 backtester
      const sellThreshold = 0.75; // Match V3 backtester
      let recommendation: Recommendation;
      if (buyProb >= buyThreshold && confidence >= 0.8) {
        recommendation = Recommendation.Buy;
      } else if (sellProb >= sellThreshold && confidence >= 0.8) {
        recommendation = Recommendation.Sell;
      } else {
        recommendation = Recommendation.Hold; // Default to Hold if no strong signal
      }

      const smsMessage = formatAnalysisResults({
        cryptoSymbol: CryptoIds.Cardano,
        currentPrice,
        probabilities,
        recommendation,
        metConditions: [],
      });

      await sendSMS(smsMessage);
      response.status(200).send("Reply received and logged!");
    } catch (error: any) {
      await sendSMS("Incorrect name, please try again.");
      response.status(500).send("Error logging reply: " + error.message);
    }
  }
);
