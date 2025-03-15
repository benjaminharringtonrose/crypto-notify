import { onSchedule } from "firebase-functions/scheduler";
import { ANALYSIS_SCHEDULE } from "./constants";
import { calculateSellDecision } from "./calculations/calculateSellDecision";
import { sendSmsNotification } from "./notifications/sendSmsNotification";

export const runAnalysisModel = onSchedule(ANALYSIS_SCHEDULE, async () => {
  const value = await calculateSellDecision("cardano");
  console.log(value);
  await sendSmsNotification(JSON.stringify(value));
});
