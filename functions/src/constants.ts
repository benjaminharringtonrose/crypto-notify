import { firestore } from "firebase-admin";

export const TEXTBELT_API_URL = "https://textbelt.com";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export const NOTIFICATION_COOLDOWN = 30 * 60 * 1000; // 30 mins

export const TRADE_PREDICTOR_WEIGHTS = "tradePredictorWeights.json";

export const FIVE_YEARS_IN_DAYS = 1826;

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
