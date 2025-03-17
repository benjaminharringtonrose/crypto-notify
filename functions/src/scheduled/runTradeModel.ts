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

  // Reference to the document storing the last recommendation
  const lastRecommendationRef = db
    .collection(Collections.TradeRecommendations)
    .doc(Docs.Cardano);

  // Get the previous recommendation
  const lastRecommendationDoc = await lastRecommendationRef.get();
  const previousRecommendation = lastRecommendationDoc.exists
    ? lastRecommendationDoc.data()?.recommendation
    : null;

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  });

  // Only send SMS if it's buy or sell and not the same as the previous
  if (
    recommendation !== Recommendation.Hold &&
    recommendation !== Recommendation.HoldBasedOnBuyPrice &&
    recommendation !== previousRecommendation
  ) {
    await sendSMS(smsMessage);
  }

  // Store the current recommendation
  await lastRecommendationRef.set({
    recommendation,
    timestamp: new Date().toISOString(),
  });
});
