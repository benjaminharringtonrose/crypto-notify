import { https } from "firebase-functions";
import { determineTrade } from "../cardano/determineTrade";
import { CryptoIds, RecieveSMSRequest } from "../types";
import { formatAnalysisResults } from "../utils";
import { sendSMS } from "./sendSMS";

export const receiveSMS = https.onRequest(
  async (request: RecieveSMSRequest, response) => {
    try {
      const replyText = request.body.text || "No message";

      const { currentPrice, probabilities, recommendation, metConditions } =
        await determineTrade();

      if (replyText.trim().toLowerCase() !== "cardano") {
        return;
      }

      const smsMessage = formatAnalysisResults({
        cryptoSymbol: CryptoIds.Cardano,
        currentPrice,
        probabilities,
        recommendation,
        metConditions,
      });

      await sendSMS(smsMessage);
      response.status(200).send("Reply received and logged!");
    } catch (error: any) {
      await sendSMS("Incorrect name, please try again.");
      response.status(500).send("Error logging reply: " + error.message);
    }
  }
);
