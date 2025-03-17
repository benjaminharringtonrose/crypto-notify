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

  const {
    recommendation: previousRecommendation,
    probability: previousProbability,
  } = previousDoc.data() as TradeRecommendation;

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
      await recommendationRef.set({
        recommendation,
        probability: probabilities.buy,
        timestamp: new Date().toISOString(),
      });
    }
    if (isSell && probabilities.sell > previousProbability) {
      await sendSMS(`Sell probability increased: ${probabilities.sell}`);
      await recommendationRef.set({
        recommendation,
        probability: probabilities.sell,
        timestamp: new Date().toISOString(),
      });
    }
    return;
  }

  if (differentFromPrevious) {
    await sendSMS(smsMessage);
    if (isBuy) {
      await recommendationRef.set({
        recommendation,
        probability: probabilities.buy,
        timestamp: new Date().toISOString(),
      });
      return;
    }
    if (isSell) {
      await recommendationRef.set({
        recommendation,
        probability: probabilities.sell,
        timestamp: new Date().toISOString(),
      });
      return;
    }
  }
});
