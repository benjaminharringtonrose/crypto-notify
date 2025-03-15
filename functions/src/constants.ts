import { firestore } from "firebase-admin";

export const TEXTBELT_API_URL = "https://textbelt.com";
export const COINGECKO_API_URL = "https://api.coingecko.com/api/v3";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
export const PRICE_CHECK_SCHEDULE = `*/1 * * * *`; // every 1 min
export const ANALYSIS_SCHEDULE = `25 15 * * *`; // every day at 2:30 PM ET
export const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

export const RUNNING_SCHEDULE_CHECK_MESSAGE =
  "Running scheduled Cardano price check";

export const MERGE_PAYLOAD: firestore.SetOptions = {
  merge: true,
};

export enum CryptoIds {
  Cardano = "cardano",
}

export enum Currencies {
  USD = "usd",
}

export enum Collections {
  Config = "config",
}

export enum Docs {
  PriceAlert = "priceAlert",
}
