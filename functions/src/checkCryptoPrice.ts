import { onSchedule } from "firebase-functions/v2/scheduler";
import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import dotenv from "dotenv";
import axios from "axios";
import { TEXTBELT_BASE_URL } from "./constants";

dotenv.config();

const TARGET_PRICE = 0.8;
const CHECK_INTERVAL = "1";
const NOTIFICATION_COOLDOWN = 60 * 60 * 1000; // 1 HR

const db = admin.firestore();
const configDocRef = db.collection("config").doc("priceAlert");

export const checkCryptoPrice = onSchedule(
  `*/${CHECK_INTERVAL} * * * *`,
  async (_event) => {
    try {
      console.log("Running scheduled Cardano price check");

      const currentPrice = await getCardanoPrice();
      console.log(`Current Cardano price: $${currentPrice}`);

      const configDoc = await configDocRef.get();

      const lastNotified = configDoc.exists
        ? configDoc.data()?.lastNotified?.toDate() || new Date(0)
        : new Date(0);

      const now = new Date();
      const timeSinceLastNotification = now.getTime() - lastNotified.getTime();

      // Check if price is below target and we haven't notified recently
      if (
        currentPrice <= TARGET_PRICE &&
        timeSinceLastNotification > NOTIFICATION_COOLDOWN
      ) {
        await sendSmsNotification(currentPrice);

        await configDocRef.set(
          {
            lastNotified: admin.firestore.FieldValue.serverTimestamp(),
          },
          { merge: true }
        );
      } else if (currentPrice <= TARGET_PRICE) {
        console.log(
          `Price is below target but notification cooldown active. ${Math.floor(
            (NOTIFICATION_COOLDOWN - timeSinceLastNotification) / 60000
          )} minutes remaining.`
        );
      } else {
        console.log(
          `Price above target. Current: $${currentPrice}, Target: $${TARGET_PRICE}`
        );
      }
    } catch (error) {
      console.error("Error in checkCardanoPrice function:", error);
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
    console.error("Error fetching Cardano price:", error);
    throw error;
  }
}

async function sendSmsNotification(currentPrice: number): Promise<void> {
  try {
    const message = `CARDANO ALERT: ADA is now at $${currentPrice} (Target: $${TARGET_PRICE})`;
    await axios.post(`${TEXTBELT_BASE_URL}/text`, {
      phone: process.env.PHONE_NUMBER,
      message: message,
      key: process.env.TEXTBELT_API_KEY,
    });
  } catch (error) {
    functions.logger.error("Error sending SMS notification:", error);
    throw error;
  }
}
