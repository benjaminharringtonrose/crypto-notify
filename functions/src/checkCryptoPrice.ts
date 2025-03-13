import { onSchedule } from "firebase-functions/v2/scheduler";
import * as functions from "firebase-functions";
// import * as admin from "firebase-admin";
import * as nodemailer from "nodemailer";
import dotenv from "dotenv";
import axios from "axios";

dotenv.config();

// Hardcoded configuration
const PHONE_NUMBER = process.env.PHONE_NUMBER; // Replace with your actual phone number
const TARGET_PRICE = 0.8; // Your target price in USD
const CHECK_INTERVAL = "1"; // How often to check the price
// const NOTIFICATION_COOLDOWN = 60 * 60 * 1000; // 1 hour in milliseconds - to avoid spam

// const config = functions.config();

const smtpEmail = process.env.EMAIL;
const smtpPassword = process.env.PASSWORD;

const transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  port: 587,
  secure: false,
  auth: {
    user: smtpEmail,
    pass: smtpPassword,
  },
});

const carrierGateway = "vtext.com";

// Reference to Firestore for storing last notification time
// const db = admin.firestore();
// const configDocRef = db.collection("config").doc("priceAlert");

export const checkCryptoPrice = onSchedule(
  `*/${CHECK_INTERVAL} * * * *`,
  async (_event) => {
    try {
      console.log("Running scheduled Cardano price check");

      functions.logger.info("smtpEmail", smtpEmail);
      functions.logger.info("smtpPassword", smtpPassword);

      // Get current Cardano price
      const currentPrice = await getCardanoPrice();
      console.log(`Current Cardano price: $${currentPrice}`);

      // Get last notification timestamp
      //   const configDoc = await configDocRef.get();
      //   const lastNotified = configDoc.exists
      //     ? configDoc.data()?.lastNotified?.toDate() || new Date(0)
      //     : new Date(0);

      //   const now = new Date();
      //   const timeSinceLastNotification = now.getTime() - lastNotified.getTime();

      // Check if price is below target and we haven't notified recently
      if (
        currentPrice <= TARGET_PRICE
        // && timeSinceLastNotification > NOTIFICATION_COOLDOWN
      ) {
        // Send SMS notification
        await sendSmsNotification(currentPrice);

        // Update last notified timestamp
        // await configDocRef.set(
        //   {
        //     lastNotified: admin.firestore.FieldValue.serverTimestamp(),
        //   },
        //   { merge: true }
        // );

        console.log(`SMS notification sent to ${PHONE_NUMBER}`);
      } else if (currentPrice <= TARGET_PRICE) {
        // console.log(
        //   `Price is below target but notification cooldown active. ${Math.floor(
        //     (NOTIFICATION_COOLDOWN - timeSinceLastNotification) / 60000
        //   )} minutes remaining.`
        // );
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

/**
 * Fetch the current price of Cardano (ADA) in USD
 */
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

/**
 * Send SMS notification using Firebase Authentication
 */
async function sendSmsNotification(currentPrice: number): Promise<void> {
  try {
    // For simplicity we'll use the Firebase Authentication custom SMS verification
    const message = `CARDANO ALERT: ADA is now at $${currentPrice} (Target: $${TARGET_PRICE})`;

    const smsEmail = `${PHONE_NUMBER}@${carrierGateway}`;

    const mailOptions = {
      from: smtpEmail,
      to: smsEmail,
      subject: "", // Leave blank for SMS
      text: message, // Message body (keep under 160 characters for SMS)
    };

    await transporter.sendMail(mailOptions);

    console.log(`SMS notification sent to ${PHONE_NUMBER}`);
  } catch (error) {
    console.error("Error sending SMS notification:", error);
    throw error;
  }
}
