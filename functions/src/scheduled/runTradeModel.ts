import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTrade } from "../cardano/determineTrade";
import { sendSMS } from "../notifications/sendSMS";
import { EVERY_MIN } from "../constants";
import {
  Collections,
  CryptoIds,
  Docs,
  Recommendation,
  TradeRecommendation,
} from "../types";
import { formatAnalysisResults } from "../utils";
import { getFirestore } from "firebase-admin/firestore";

const db = getFirestore();

/**
 * Scheduled function that runs a trading model for Cardano cryptocurrency every minute.
 * Analyzes trading conditions, generates recommendations, and sends SMS notifications
 * when trading conditions change or probabilities increase.
 */

export const runTradeModel = onSchedule(EVERY_MIN, async () => {
  const {
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  } = await determineTrade(CryptoIds.Cardano);

  const recommendationRef = db
    .collection(Collections.TradeRecommendations)
    .doc(Docs.Cardano);

  const previousDoc = await recommendationRef.get();

  const previous = previousDoc.data() as TradeRecommendation | undefined;

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  });

  const sameRecommendation = recommendation === previous?.recommendation;

  if (!sameRecommendation) {
    switch (recommendation) {
      case Recommendation.Buy:
        await sendSMS(smsMessage);
        await recommendationRef.set({
          recommendation,
          probability: probabilities.buy,
          timestamp: new Date().toISOString(),
        });
        return;
      case Recommendation.Sell:
        await sendSMS(smsMessage);
        await recommendationRef.set({
          recommendation,
          probability: probabilities.sell,
          timestamp: new Date().toISOString(),
        });
        return;
      case Recommendation.Hold:
      case Recommendation.HoldBasedOnBuyPrice:
        return;
    }
  }

  if (sameRecommendation) {
    switch (recommendation) {
      case Recommendation.Buy:
        if (previous?.probability && probabilities.buy > previous.probability) {
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
      case Recommendation.HoldBasedOnBuyPrice:
        return;
    }
  }
});
