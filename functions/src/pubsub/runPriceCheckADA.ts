import { onSchedule } from "firebase-functions/v2/scheduler";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import { getCurrentPrice } from "../api/getCurrentPrice";
import { MERGE_PAYLOAD, NOTIFICATION_COOLDOWN, PRICES } from "../constants";
import {
  sendSMS,
  checkCardanoPriceErrorMessage,
  isAboveThreshold,
  priceAlertTextMessage,
} from "../utils";
import { Collections, CryptoIds, Currencies, Docs } from "../types";

dotenv.config();

export const runPriceCheckADA = onSchedule(`*/10 * * * *`, async () => {
  try {
    const currentPrice = await getCurrentPrice({
      id: CryptoIds.Cardano,
      currency: Currencies.USD,
    });

    console.log(`Current Cardano price: $${currentPrice}`);

    const db = firestore();
    const configDocRef = db.collection(Collections.Config).doc(Docs.PriceAlert);

    const configDoc = await configDocRef.get();
    const config = configDoc.exists ? configDoc.data() : {};

    const lastNotified = config?.lastNotified?.toDate() || new Date(0);
    const lastNotifiedThreshold = config?.lastNotifiedThreshold || 0;

    const now = new Date();
    const timeSinceLastNotification = now.getTime() - lastNotified.getTime();
    const cooldownActive = timeSinceLastNotification <= NOTIFICATION_COOLDOWN;

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
