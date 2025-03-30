import { firestore } from "firebase-admin";
import { ScheduleOptions } from "firebase-functions/scheduler";

export const TEXTBELT_API_URL = "https://textbelt.com";
export const COINGECKO_API_URL = "https://api.coingecko.com/api/v3";
export const CRYPTOCOMPARE_API_URL =
  "https://min-api.cryptocompare.com/data/v2";

export const CARDANO_BITCOIN_PRICE_URL = `${COINGECKO_API_URL}/simple/price?ids=cardano,bitcoin&vs_currencies=usd`;
export const CARDANO_30_DAY_HISTORICAL_URL = `${COINGECKO_API_URL}/coins/cardano/market_chart?vs_currency=usd&days=30`;
export const BITCOIN_30_DAY_HISTORICAL_URL = `${COINGECKO_API_URL}/coins/bitcoin/market_chart?vs_currency=usd&days=30`;

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
export const EVERY_TEN_MINS = `*/10 * * * *`; // every 10 mins
export const EVERYDAY_AT_NOON = `05 10 * * *`; // every day at 10:00am ET
export const TIME_ZONE = "America/New_York";
export const LOW_MEMORY = "512MiB";
export const MEDIUM_MEMORY = "1GiB";
export const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins
export const AVERAGE_BUY_PRICE = 0.72;
export const TRADE_PREDICTOR_WEIGHTS = "tradePredictorWeights.json";

export const FIVE_YEARS_IN_DAYS = 1826;

export const TRAINING_CONFIG: ScheduleOptions = {
  schedule: EVERYDAY_AT_NOON,
  memory: LOW_MEMORY,
};

export const RUN_TRADE_MODEL_CONFIG: ScheduleOptions = {
  schedule: EVERY_TEN_MINS,
  memory: LOW_MEMORY,
};

export const RUNNING_ANALYSIS_MODEL_MESSAGE = "Running analysis model...";

export const MERGE_PAYLOAD: firestore.SetOptions = {
  merge: true,
};

export const PERIODS = {
  RSI: 14,
  SMA_SHORT: 7,
  SMA_MEDIUM: 20,
  SMA_LONG: 21,
  EMA_SHORT: 12,
  EMA_LONG: 26,
  MACD_SIGNAL: 9,
  BOLLINGER: 20,
  ATR: 14,
  VWAP: 7,
  STOCH_RSI: 14,
  STOCH_SMOOTH: 3,
  FIBONACCI: 30,
  VOL_SMA_SHORT: 5,
  VOL_SMA_LONG: 14,
  MOMENTUM: 10,
} as const;
