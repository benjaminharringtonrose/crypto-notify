import { onSchedule } from "firebase-functions/v2/scheduler";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import {
  Collections,
  CryptoIds,
  Currencies,
  Docs,
  MERGE_PAYLOAD,
  NOTIFICATION_COOLDOWN,
  PRICE_CHECK_SCHEDULE,
  PRICES,
  RUNNING_SCHEDULE_CHECK_MESSAGE,
} from "./constants";
import {
  checkCardanoPriceErrorMessage,
  cooldownMessage,
  currentCardanoPriceMessage,
  isAboveThreshold,
  notExceededMessage,
  notificationSentMessage,
} from "./utils";
import { sendSmsNotification } from "./notifications/sendSmsNotification";
import { getCurrentPrice } from "./api/getCurrentPrice";

dotenv.config();

export const checkCardanoPrice = onSchedule(PRICE_CHECK_SCHEDULE, async () => {
  try {
    console.log(RUNNING_SCHEDULE_CHECK_MESSAGE);

    const currentPrice = await getCurrentPrice({
      id: CryptoIds.Cardano,
      currency: Currencies.USD,
    });

    console.log(currentCardanoPriceMessage(currentPrice));

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
      await sendSmsNotification(currentPrice, exceededThreshold);

      const lastNotifiedPayload = {
        lastNotified: firestore.FieldValue.serverTimestamp(),
        lastNotifiedThreshold: exceededThreshold,
      };

      await configDocRef.set(lastNotifiedPayload, MERGE_PAYLOAD);

      console.log(notificationSentMessage(currentPrice, exceededThreshold));
    } else if (exceededThreshold && cooldownActive) {
      console.log(
        cooldownMessage({
          exceededThreshold,
          notificationCooldown: NOTIFICATION_COOLDOWN,
          timeSinceLastNotification,
        })
      );
    } else {
      console.log(notExceededMessage(currentPrice));
    }
  } catch (error) {
    console.log(checkCardanoPriceErrorMessage(error));
  }
});
