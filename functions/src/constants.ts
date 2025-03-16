import { firestore } from "firebase-admin";

export const TEXTBELT_API_URL = "https://textbelt.com";
export const COINGECKO_API_URL = "https://api.coingecko.com/api/v3";
export const CRYPTOCOMPARE_API_URL =
  "https://min-api.cryptocompare.com/data/v2";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
export const EVERY_MIN = `*/1 * * * *`; // every 1 min
export const EVERYDAY_AT = `25 12 * * *`; // every day at 11:40 AM ET
export const TIME_ZONE = "America/New_York";
export const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

export const FIVE_YEARS_IN_DAYS = 1826;

export const ANALYSIS_SCHEDULE = {
  schedule: EVERYDAY_AT,
  timeZone: TIME_ZONE,
};

export const RUNNING_SCHEDULE_CHECK_MESSAGE =
  "Running scheduled Cardano price check";

export const RUNNING_ANALYSIS_MODEL_MESSAGE = "Running analysis model...";

export const MERGE_PAYLOAD: firestore.SetOptions = {
  merge: true,
};
