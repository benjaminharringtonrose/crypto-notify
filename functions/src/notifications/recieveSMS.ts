import { https } from "firebase-functions";
import { determineTradeActionBTC } from "../machineLearning/determineTradeActionBTC";
import { RecieveSMSRequest } from "../types";
import { formatAnalysisResults, formatCurrency } from "../utils";
import { sendSMS } from "./sendSMS";

export const receiveSMS = https.onRequest(
  async (request: RecieveSMSRequest, response) => {
    try {
      const replyText = request.body.text || "No message";

      const {
        cryptoSymbol,
        currentPrice,
        probability,
        recommendation,
        metConditions,
      } = await determineTradeActionBTC(replyText.toLowerCase().trim());

      const smsMessage = formatAnalysisResults({
        cryptoSymbol,
        currentPrice,
        probability,
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
