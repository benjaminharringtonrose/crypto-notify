import { onSchedule } from "firebase-functions/v2/scheduler";
import { logger } from "firebase-functions";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import axios from "axios";
import {
  COIN_GEKO_BASE_URL,
  Collections,
  CryptoIds,
  Currencies,
  Docs,
  MERGE,
  RUNNING_SCHEDULE_CHECK_MESSAGE,
  TEXTBELT_BASE_URL,
} from "./constants";
import {
  checkCardanoPriceErrorMessage,
  cooldownMessage,
  currentCardanoPriceMessage,
  fetchCardanoPriceErrorMessage,
  isAboveThreshold,
  notExceededMessage,
  notificationSentMessage,
  sendSmsErrorMessage,
  textMessage,
} from "./utils";

dotenv.config();

const PRICES = [0.8, 0.9, 1.0];
const SCHEDULE = `*/1 * * * *`; // every 1 min
const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

const db = firestore();
const configDocRef = db.collection(Collections.Config).doc(Docs.PriceAlert);

export const checkCardanoPrice = onSchedule(SCHEDULE, async () => {
  console.log("Checking Cardano price...");

  try {
    logger.info(RUNNING_SCHEDULE_CHECK_MESSAGE);

    const currentPrice = await getCardanoPrice();
    logger.info(currentCardanoPriceMessage(currentPrice));

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

      logger.info(notificationSentMessage(currentPrice, exceededThreshold));
    } else if (exceededThreshold && cooldownActive) {
      logger.info(
        cooldownMessage({
          exceededThreshold,
          notificationCooldown: NOTIFICATION_COOLDOWN,
          timeSinceLastNotification,
        })
      );
    } else {
      logger.info(notExceededMessage(currentPrice));
    }
  } catch (error) {
    logger.error(checkCardanoPriceErrorMessage(error));
  }
});

async function getCardanoPrice(): Promise<number> {
  try {
    const params = {
      ids: CryptoIds.Cardano,
      vs_currencies: Currencies.USD,
    };

    const response = await axios.get(`${COIN_GEKO_BASE_URL}/price`, {
      params,
    });

    return response.data.cardano.usd;
  } catch (error) {
    logger.error(fetchCardanoPriceErrorMessage(error));
    throw error;
  }
}

async function sendSmsNotification(
  currentPrice: number,
  threshold: number
): Promise<void> {
  try {
    await axios.post(`${TEXTBELT_BASE_URL}/text`, {
      phone: process.env.PHONE_NUMBER,
      message: textMessage(threshold, currentPrice),
      key: process.env.TEXTBELT_API_KEY,
    });
  } catch (error) {
    logger.error(sendSmsErrorMessage(error));
    throw error;
  }
}
