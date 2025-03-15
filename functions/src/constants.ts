import { firestore } from "firebase-admin";

export const TEXTBELT_API_URL = "https://textbelt.com";

export const COINGECKO_API_URL = "https://api.coingecko.com/api/v3";

export const RUNNING_SCHEDULE_CHECK_MESSAGE =
  "Running scheduled Cardano price check";

export const MERGE: firestore.SetOptions = {
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
