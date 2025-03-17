import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTradeADA } from "../cnn/determineTradeADA";
import { sendSMS } from "../notifications/sendSMS";
import {
  ANALYSIS_SCHEDULE,
  RUNNING_ANALYSIS_MODEL_MESSAGE,
} from "../constants";
import { CryptoIds } from "../types";
import { formatAnalysisResults } from "../utils";

dotenv.config();

export const runTradeModelDaily = onSchedule(ANALYSIS_SCHEDULE, async () => {
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

  await sendSMS(smsMessage);
});
