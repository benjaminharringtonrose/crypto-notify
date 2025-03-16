import "firebase-admin";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { ANALYSIS_SCHEDULE, TIME_ZONE } from "./constants";
import { calculateSellDecision } from "./calculations/calculateSellDecision";
import { sendSMS } from "./notifications/sendSMS";
import dotenv from "dotenv";

dotenv.config();
export const runAnalysisModel = onSchedule(
  {
    schedule: ANALYSIS_SCHEDULE,
    timeZone: TIME_ZONE,
  },
  async () => {
    console.log("Running analytics model...");
    const value = await calculateSellDecision("cardano");

    const symbol = value.cryptoSymbol.toUpperCase();
    const price = `$${value.currentPrice}`;
    const prob = `${Math.round(parseFloat(value.probability) * 100)}%`;
    const rec =
      value.recommendation.charAt(0).toUpperCase() +
      value.recommendation.slice(1);
    const conditions = value.metConditions.join(", ");
    const smsMessage = `${symbol}: ${price}\nProb: ${prob}\nRec: ${rec}\nConditions: ${conditions}\nReply with a crypto to check again.`;

    await sendSMS(smsMessage);
    console.log("SMS Message:", smsMessage);
    console.log("Full Result:", value);
  }
);
