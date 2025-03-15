import "firebase-admin";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { ANALYSIS_SCHEDULE } from "./constants";
import { calculateSellDecision } from "./calculations/calculateSellDecision";
import { sendSmsNotification } from "./notifications/sendSmsNotification";
import { trainSellModel } from "./machineLearning/trainSellModel";
import dotenv from "dotenv";

dotenv.config();

export const runAnalysisModel = onSchedule(ANALYSIS_SCHEDULE, async () => {
  await trainSellModel();
  const value = await calculateSellDecision("cardano");
  await sendSmsNotification(JSON.stringify(value));
  console.log(value);
});
