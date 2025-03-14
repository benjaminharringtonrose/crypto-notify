import { onSchedule } from "firebase-functions/v2/scheduler";
import { logger } from "firebase-functions";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import axios from "axios";
import { COIN_GEKO_BASE_URL, MERGE, TEXTBELT_BASE_URL } from "./constants";
import { isAboveThreshold } from "./utils";

dotenv.config();

const PRICES = [0.8, 0.9, 1.0];
const SCHEDULE = `*/1 * * * *`; // every 1 min
const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

const db = firestore();
const configDocRef = db.collection("config").doc("priceAlert");

export const checkCardanoPrice = onSchedule(SCHEDULE, async () => {
  try {
    logger.info("Running scheduled Cardano price check");

    const currentPrice = await getCardanoPrice();
    logger.info(`Current Cardano price: $${currentPrice}`);

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

      logger.info(
        `Notification sent for price $${currentPrice} exceeding threshold $${exceededThreshold}`
      );
    } else if (exceededThreshold && cooldownActive) {
      logger.info(
        `Price threshold $${exceededThreshold} exceeded but notification cooldown active. ${Math.floor(
          (NOTIFICATION_COOLDOWN - timeSinceLastNotification) / 60000
        )} minutes remaining.`
      );
    } else {
      logger.info(`No price thresholds exceeded. Current: $${currentPrice}`);
    }
  } catch (error) {
    logger.error("Error in checkCardanoPrice function:", error);
  }
});

async function getCardanoPrice(): Promise<number> {
  try {
    const params = {
      ids: "cardano",
      vs_currencies: "usd",
    };

    const response = await axios.get(`${COIN_GEKO_BASE_URL}/price`, {
      params,
    });

    return response.data.cardano.usd;
  } catch (error) {
    logger.error("Error fetching Cardano price:", error);
    throw error;
  }
}

async function sendSmsNotification(
  currentPrice: number,
  threshold: number
): Promise<void> {
  try {
    logger.info(`Sending SMS message to ${process.env.PHONE_NUMBER}`);
    const message = `CARDANO ALERT: ADA has risen above $${threshold} and is now at $${currentPrice}`;

    await axios.post(`${TEXTBELT_BASE_URL}/text`, {
      phone: process.env.PHONE_NUMBER,
      message,
      key: process.env.TEXTBELT_API_KEY,
    });
  } catch (error) {
    logger.error("Error sending SMS notification:", error);
    throw error;
  }
}
