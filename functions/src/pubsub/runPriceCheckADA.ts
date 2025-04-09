import { onSchedule, ScheduleOptions } from "firebase-functions/v2/scheduler";
import { firestore } from "firebase-admin";
import { MERGE_PAYLOAD, PRICES, TIME_CONVERSIONS } from "../constants";
import {
  sendSMS,
  checkCardanoPriceErrorMessage,
  isAboveThreshold,
  priceAlertTextMessage,
} from "../utils";
import { CoinbaseProductIds, Collections, Docs } from "../types";
import { TradeExecutor } from "../cardano/TradeExecutor";

const CONFIG: ScheduleOptions = {
  schedule: "*/10 * * * *",
  memory: "512MiB",
};

export const runPriceCheckADA = onSchedule(CONFIG, async () => {
  try {
    const trader = new TradeExecutor({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });

    const currentPrice = await trader.getCurrentPrice(CoinbaseProductIds.ADA);

    console.log(`Current Cardano price: $${currentPrice}`);

    const db = firestore();
    const configDocRef = db.collection(Collections.Config).doc(Docs.PriceAlert);

    const configDoc = await configDocRef.get();
    const config = configDoc.exists ? configDoc.data() : {};

    const lastNotified = config?.lastNotified?.toDate() || new Date(0);
    const lastNotifiedThreshold = config?.lastNotifiedThreshold || 0;

    const now = new Date();
    const timeSinceLastNotification = now.getTime() - lastNotified.getTime();
    const cooldownActive =
      timeSinceLastNotification <=
      TIME_CONVERSIONS.THIRTY_MINUTES_IN_MILLISECONDS;

    const exceededThreshold = isAboveThreshold({
      prices: PRICES,
      currentPrice,
    });

    if (
      exceededThreshold &&
      exceededThreshold > lastNotifiedThreshold &&
      !cooldownActive
    ) {
      await sendSMS(priceAlertTextMessage(exceededThreshold, currentPrice));

      const lastNotifiedPayload = {
        lastNotified: firestore.FieldValue.serverTimestamp(),
        lastNotifiedThreshold: exceededThreshold,
      };

      await configDocRef.set(lastNotifiedPayload, MERGE_PAYLOAD);
    }
  } catch (error) {
    console.log(checkCardanoPriceErrorMessage(error));
  }
});
