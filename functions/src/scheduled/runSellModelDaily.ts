import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTradeActionBTC } from "../machineLearning/determineTradeActionBTC";
import { sendSMS } from "../notifications/sendSMS";
import {
  ANALYSIS_SCHEDULE,
  RUNNING_ANALYSIS_MODEL_MESSAGE,
} from "../constants";
import { CryptoIds } from "../types";
import { formatAnalysisResults } from "../utils";

dotenv.config();

export const runSellModelDaily = onSchedule(ANALYSIS_SCHEDULE, async () => {
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

  await sendSMS(smsMessage);
});
