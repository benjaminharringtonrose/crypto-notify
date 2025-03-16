import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTradeActionBTC } from "../machineLearning/determineTradeActionBTC";
import { sendSMS } from "../notifications/sendSMS";
import { EVERY_MIN, RUNNING_ANALYSIS_MODEL_MESSAGE } from "../constants";
import { CryptoIds, Recommendation } from "../types";
import { formatAnalysisResults } from "../utils";

dotenv.config();

export const runSellModel = onSchedule(EVERY_MIN, async () => {
  console.log(RUNNING_ANALYSIS_MODEL_MESSAGE);

  const {
    cryptoSymbol,
    currentPrice,
    probability,
    recommendation,
    metConditions,
  } = await determineTradeActionBTC(CryptoIds.Bitcoin);

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probability,
    recommendation,
    metConditions,
  });

  if (recommendation === Recommendation.Sell) {
    await sendSMS(smsMessage);
  }
});
