import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTrade } from "../cardano/determineTrade";
import { sendSMS } from "../notifications/sendSMS";
import { EVERY_MIN, RUNNING_ANALYSIS_MODEL_MESSAGE } from "../constants";
import { Collections, CryptoIds, Docs, Recommendation } from "../types";
import { formatAnalysisResults } from "../utils";
// Add Firestore import
import { getFirestore } from "firebase-admin/firestore";

dotenv.config();

// Initialize Firestore
const db = getFirestore();

export const runTradeModel = onSchedule(EVERY_MIN, async () => {
  console.log(RUNNING_ANALYSIS_MODEL_MESSAGE);

  const {
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  } = await determineTrade(CryptoIds.Cardano);

  const lastRecommendationRef = db
    .collection(Collections.TradeRecommendations)
    .doc(Docs.Cardano);

  const lastRecommendationDoc = await lastRecommendationRef.get();
  const previousRecommendation = lastRecommendationDoc.exists
    ? lastRecommendationDoc.data()?.recommendation
    : null;
  const previousProbability = lastRecommendationDoc.exists
    ? lastRecommendationDoc.data()?.probability
    : null;

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  });

  const isBuy = recommendation === Recommendation.Buy;
  const isSell = recommendation === Recommendation.Sell;
  const sameAsPrevious = recommendation === previousRecommendation;
  const differentFromPrevious = recommendation !== previousRecommendation;

  if (sameAsPrevious) {
    if (isBuy && probabilities.buy > previousProbability) {
      await sendSMS(`Buy probability increased: ${probabilities.buy}`);
      await lastRecommendationRef.set({
        recommendation,
        probability: probabilities.buy,
        timestamp: new Date().toISOString(),
      });
    }
    if (isSell && probabilities.sell > previousProbability) {
      await sendSMS(`Sell probability increased: ${probabilities.sell}`);
      await lastRecommendationRef.set({
        recommendation,
        probability: probabilities.sell,
        timestamp: new Date().toISOString(),
      });
    }
    return;
  }

  if (isBuy && differentFromPrevious) {
    await sendSMS(smsMessage);
    await lastRecommendationRef.set({
      recommendation,
      probability: probabilities.buy,
      timestamp: new Date().toISOString(),
    });
    return;
  }

  if (isSell && differentFromPrevious) {
    await sendSMS(smsMessage);
    await lastRecommendationRef.set({
      recommendation,
      probability: probabilities.sell,
      timestamp: new Date().toISOString(),
    });
    return;
  }
});
