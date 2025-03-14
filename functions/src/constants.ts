import { firestore } from "firebase-admin";

export const TEXTBELT_BASE_URL = "https://textbelt.com";

export const COIN_GEKO_BASE_URL = "https://api.coingecko.com/api/v3/simple";

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
