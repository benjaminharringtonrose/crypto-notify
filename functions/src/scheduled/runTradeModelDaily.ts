import { onSchedule } from "firebase-functions/v2/scheduler";
import { determineTrade } from "../cardano/determineTrade";
import { sendSMS } from "../notifications/sendSMS";
import { ANALYSIS_SCHEDULE } from "../constants";
import { CryptoIds } from "../types";
import { formatAnalysisResults } from "../utils";

export const runTradeModelDaily = onSchedule(ANALYSIS_SCHEDULE, async () => {
  const {
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  } = await determineTrade(CryptoIds.Cardano);

  const smsMessage = formatAnalysisResults({
    cryptoSymbol,
    currentPrice,
    probabilities,
    recommendation,
    metConditions,
  });

  await sendSMS(smsMessage);
});
