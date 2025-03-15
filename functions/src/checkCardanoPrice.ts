import { onSchedule } from "firebase-functions/v2/scheduler";
import { logger } from "firebase-functions";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import axios from "axios";
import {
  COINGECKO_API_URL,
  Collections,
  CryptoIds,
  Currencies,
  Docs,
  MERGE,
  RUNNING_SCHEDULE_CHECK_MESSAGE,
} from "./constants";
import {
  checkCardanoPriceErrorMessage,
  cooldownMessage,
  currentCardanoPriceMessage,
  fetchCardanoPriceErrorMessage,
  isAboveThreshold,
  notExceededMessage,
  notificationSentMessage,
} from "./utils";
import { sendSmsNotification } from "./notifications/sendSmsNotification";

dotenv.config();

const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
const PRICE_CHECK_SCHEDULE = `*/1 * * * *`; // every 1 min
const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

const db = firestore();
const configDocRef = db.collection(Collections.Config).doc(Docs.PriceAlert);

export const checkCardanoPrice = onSchedule(PRICE_CHECK_SCHEDULE, async () => {
  try {
    console.log(RUNNING_SCHEDULE_CHECK_MESSAGE);

    const currentPrice = await getCardanoPrice();

    console.log(currentCardanoPriceMessage(currentPrice));

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

      await configDocRef.set(lastNotifiedPayload, MERGE);

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

async function getCardanoPrice(): Promise<number> {
  try {
    const params = {
      ids: CryptoIds.Cardano,
      vs_currencies: Currencies.USD,
    };

    const response = await axios.get(`${COINGECKO_API_URL}/simple/price`, {
      params,
    });

    return response.data.cardano.usd;
  } catch (error) {
    logger.error(fetchCardanoPriceErrorMessage(error));
    throw error;
  }
}
