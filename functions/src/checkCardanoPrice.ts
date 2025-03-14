import { onSchedule } from "firebase-functions/v2/scheduler";
import { logger } from "firebase-functions";
import { firestore } from "firebase-admin";
import dotenv from "dotenv";
import axios from "axios";
import { TEXTBELT_BASE_URL } from "./constants";

dotenv.config();

const ABOVE_TARGET_PRICES = [0.8, 0.9, 1.0];
const CHECK_INTERVAL = "1";
const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 MINS

const db = firestore();
const configDocRef = db.collection("config").doc("priceAlert");

export const checkCardanoPrice = onSchedule(
  `*/${CHECK_INTERVAL} * * * *`,
  async (_event) => {
    try {
      logger.info("Running scheduled Cardano price check");

      const currentPrice = await getCardanoPrice();
      logger.info(`Current Cardano price: $${currentPrice}`);

      const configDoc = await configDocRef.get();
      const config = configDoc.exists ? configDoc.data() : {};

      // Get last notified time and the last notified price threshold
      const lastNotified = config?.lastNotified?.toDate() || new Date(0);
      const lastNotifiedThreshold = config?.lastNotifiedThreshold || 0;

      const now = new Date();
      const timeSinceLastNotification = now.getTime() - lastNotified.getTime();
      const cooldownActive = timeSinceLastNotification <= NOTIFICATION_COOLDOWN;

      // Find the first price threshold that has been exceeded
      const exceededThreshold = ABOVE_TARGET_PRICES.find(
        (threshold) => currentPrice >= threshold
      );

      // Only notify if we found an exceeded threshold, it's higher than the last one we notified about,
      // and the cooldown is not active
      if (
        exceededThreshold &&
        exceededThreshold > lastNotifiedThreshold &&
        !cooldownActive
      ) {
        await sendSmsNotification(currentPrice, exceededThreshold);

        await configDocRef.set(
          {
            lastNotified: firestore.FieldValue.serverTimestamp(),
            lastNotifiedThreshold: exceededThreshold,
          },
          { merge: true }
        );

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
  }
);

async function getCardanoPrice(): Promise<number> {
  try {
    const response = await axios.get(
      "https://api.coingecko.com/api/v3/simple/price",
      {
        params: {
          ids: "cardano",
          vs_currencies: "usd",
        },
      }
    );

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
      phone: process.env.PHONE_NUMBER || "9102289283",
      message,
      key:
        process.env.TEXTBELT_API_KEY ||
        "37cc7e109746163ca34a9cdaf2ab78a9024b0574vnvhnU6EN4wc7zrG6eJnJgaEK",
    });
  } catch (error) {
    logger.error("Error sending SMS notification:", error);
    throw error;
  }
}
