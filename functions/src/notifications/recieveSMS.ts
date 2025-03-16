import { https } from "firebase-functions";
import { calculateSellDecision } from "../calculations/calculateSellDecision";
import { sendSMS } from "./sendSMS";

export const receiveSMS = https.onRequest(async (req, res) => {
  // Log the reply to Firebase Realtime Database or Firestore
  try {
    const replyText = req.body.text || "No message";

    const value = await calculateSellDecision(replyText.toLowerCase());

    const symbol = value.cryptoSymbol.toUpperCase();
    const price = `$${value.currentPrice}`;
    const prob = `${Number(value.probability) * 100}%`;
    const rec =
      value.recommendation.charAt(0).toUpperCase() +
      value.recommendation.slice(1);
    const conditions = value.metConditions.join(", ");
    const smsMessage = `${symbol}: ${price}\nProb: ${prob}\nRec: ${rec}\nConditions: ${conditions}\nReply with a crypto to check again.`;

    await sendSMS(smsMessage);
    res.status(200).send("Reply received and logged!");
  } catch (error: any) {
    await sendSMS("Incorrect name, please try again.");
    res.status(500).send("Error logging reply: " + error.message);
  }
});
