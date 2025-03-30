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

export const MODEL_CONSTANTS = {
  TIMESTEPS: 30, // Number of timesteps for the sequence
  FEATURE_COUNT: 61, // Number of features per timestep
  MIN_CONFIDENCE_DEFAULT: 0.6, // Increased from 0.5 for stronger signals
  PROFIT_TAKE_MULTIPLIER_DEFAULT: 3.0, // Default profit take multiplier
  BASE_POSITION_SIZE_DEFAULT: 0.08, // Default base position size
  SLIPPAGE: 0.001, // Slippage factor
  COMMISSION: 0.1, // Commission per trade
  STOP_LOSS_MULTIPLIER_DEFAULT: 3.0, // Tightened from 4.0 for quicker exits
  TRAILING_STOP_DEFAULT: 0.06, // Default trailing stop percentage
  MIN_HOLD_DAYS_DEFAULT: 4, // Minimum holding period in days
  LOGIT_THRESHOLD_DEFAULT: 0.05, // Default logit threshold for trade decisions
  BUY_PROB_THRESHOLD_DEFAULT: 0.5, // Default buy probability threshold
  SELL_PROB_THRESHOLD_DEFAULT: 0.33, // Default sell probability threshold
  MOMENTUM_WINDOW_THRESHOLD: 0.01, // ATR threshold for switching momentum window
};
