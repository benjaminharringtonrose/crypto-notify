import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTradeADA } from "../cardano/determineTradeADA";
import { sendSMS } from "../notifications/sendSMS";
import { EVERY_MIN, RUNNING_ANALYSIS_MODEL_MESSAGE } from "../constants";
import { CryptoIds, Recommendation } from "../types";
import { formatAnalysisResults } from "../utils";

dotenv.config();

export const runTradeModel = onSchedule(EVERY_MIN, async () => {
  console.log(RUNNING_ANALYSIS_MODEL_MESSAGE);

  const {
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  } = await determineTradeADA(CryptoIds.Cardano);

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  });

  if (recommendation === Recommendation.Sell) {
    await sendSMS(smsMessage);
  }
});
