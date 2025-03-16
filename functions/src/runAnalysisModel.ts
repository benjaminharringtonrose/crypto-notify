import "firebase-admin";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { EVERY_MIN } from "./constants";
import { calculateSellDecision } from "./calculations/calculateSellDecision";
import { sendSMS } from "./notifications/sendSMS";
import dotenv from "dotenv";

dotenv.config();

const formatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

export const runAnalysisModel = onSchedule(EVERY_MIN, async () => {
  console.log("Running analytics model...");

  const {
    cryptoSymbol,
    currentPrice,
    probability,
    recommendation,
    metConditions,
  } = await calculateSellDecision("cardano");

  const symbol = cryptoSymbol.toUpperCase();
  const price = formatter.format(currentPrice);
  const prob = `${(Number(probability) * 100).toFixed(3)}%`;
  const rec = recommendation.charAt(0).toUpperCase() + recommendation.slice(1);
  const conditions = metConditions.join(", ");
  const smsMessage = `${symbol}: ${price}\n\nProbability: ${prob}\n\nRecommendation: ${rec}\n\nSell conditions met: ${conditions}\n\nReply with a cryptocurrency to run the analysis again`;

  if (rec === "Sell") {
    await sendSMS(smsMessage);
  }
});
