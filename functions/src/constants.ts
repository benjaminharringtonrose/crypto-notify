import { firestore } from "firebase-admin";
import { ScheduleOptions } from "firebase-functions/scheduler";

export const TEXTBELT_API_URL = "https://textbelt.com";
export const COINGECKO_API_URL = "https://api.coingecko.com/api/v3";
export const CRYPTOCOMPARE_API_URL =
  "https://min-api.cryptocompare.com/data/v2";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
export const EVERY_MIN = `*/1 * * * *`; // every 1 min
export const EVERYDAY_AT = `45 17 * * *`; // every day at 4:45 PM ET
export const TIME_ZONE = "America/New_York";
export const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins
export const AVERAGE_BUY_PRICE = 0.72;

export const FIVE_YEARS_IN_DAYS = 1826;

export const RUN_TRADE_MODEL_CONFIG: ScheduleOptions = {
  schedule: EVERY_MIN,
  memory: "512MiB",
};

export const ANALYSIS_SCHEDULE: ScheduleOptions = {
  schedule: EVERYDAY_AT,
  timeZone: TIME_ZONE,
};

export const RUNNING_ANALYSIS_MODEL_MESSAGE = "Running analysis model...";

export const MERGE_PAYLOAD: firestore.SetOptions = {
  merge: true,
};
