import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { RUN_TRADE_MODEL_CONFIG } from "../constants";
import {
  Collections,
  CryptoIds,
  Currencies,
  Docs,
  Recommendation,
  TradeRecommendation,
} from "../types";
import { sendSMS, formatAnalysisResults } from "../utils";
import { getFirestore } from "firebase-admin/firestore";
import { TradeModelPredictor } from "../cardano/TradeModelPredictor";
import { getHistoricalData } from "../api/getHistoricalData";
import { getCurrentPrice } from "../api/getCurrentPrice";

dotenv.config();

/**
 * Scheduled function that runs a trading model for Cardano cryptocurrency every 5 minutes.
 * Analyzes trading conditions, generates recommendations, and sends SMS notifications
 * when trading conditions change or probabilities increase.
 */
export const runTradeModel = onSchedule(RUN_TRADE_MODEL_CONFIG, async () => {
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

    // Predict using the new method
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

    const db = getFirestore();
    const recommendationRef = db
      .collection(Collections.TradeRecommendations)
      .doc(Docs.Cardano);

    const previousDoc = await recommendationRef.get();
    const previous = (previousDoc.exists ? previousDoc.data() : undefined) as
      | TradeRecommendation
      | undefined;

    const analysisResults = formatAnalysisResults({
      cryptoSymbol: CryptoIds.Cardano,
      currentPrice,
      probabilities,
      recommendation,
    });

    console.log(analysisResults);

    const sameRecommendation = recommendation === previous?.recommendation;

    if (!previous) {
      switch (recommendation) {
        case Recommendation.Buy:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation,
            probability: probabilities.buy,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Sell:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation,
            probability: probabilities.sell,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Hold:
          await recommendationRef.set({
            recommendation,
            probability: probabilities.hold,
            timestamp: new Date().toISOString(),
          });
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }

    if (!sameRecommendation) {
      switch (recommendation) {
        case Recommendation.Buy:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation,
            probability: probabilities.buy,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Sell:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation,
            probability: probabilities.sell,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Hold:
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }

    if (sameRecommendation) {
      switch (recommendation) {
        case Recommendation.Buy:
          if (
            previous?.probability &&
            probabilities.buy > previous.probability
          ) {
            await sendSMS(`Buy probability increased: ${probabilities.buy}`);
            await recommendationRef.set({
              recommendation,
              probability: probabilities.buy,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        case Recommendation.Sell:
          if (
            previous?.probability &&
            probabilities.sell > previous.probability
          ) {
            await sendSMS(`Sell probability increased: ${probabilities.sell}`);
            await recommendationRef.set({
              recommendation,
              probability: probabilities.sell,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        case Recommendation.Hold:
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }
  } catch (error) {
    console.log("runTradeModel Error:", JSON.stringify(error));
  }
});
