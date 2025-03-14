import { firestore } from "firebase-admin";

export const TEXTBELT_BASE_URL = "https://textbelt.com";

export const COIN_GEKO_BASE_URL = "https://api.coingecko.com/api/v3/simple";

export const MERGE: firestore.SetOptions = {
  merge: true,
};
