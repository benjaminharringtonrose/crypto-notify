import { firestore } from "firebase-admin";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export const MERGE_PAYLOAD: firestore.SetOptions = {
  merge: true,
};

export const FILE_NAMES = {
  WEIGHTS: "tradePredictorWeights.json",
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
  ATR_BREAKOUT: 14,
} as const;

export const STRATEGY_CONFIG = {
  MIN_CONFIDENCE_DEFAULT: 0.4,
  PROFIT_TAKE_MULTIPLIER_DEFAULT: 3.5,
  BASE_POSITION_SIZE_DEFAULT: 0.1,
  SLIPPAGE: 0.001,
  COMMISSION: 0.1,
  STOP_LOSS_MULTIPLIER_DEFAULT: 4, // Increased from 3.5 for more room
  TRAILING_STOP_DEFAULT: 0.07, // Increased from 0.05 for more flexibility
  MIN_HOLD_DAYS_DEFAULT: 5, // Increased from 3 for longer holds
  BUY_PROB_THRESHOLD_DEFAULT: 0.45,
  SELL_PROB_THRESHOLD_DEFAULT: 0.3,
  MOMENTUM_WINDOW_THRESHOLD: 0.01,
  MAX_ATR_THRESHOLD: 0.15,
  MIN_PROFIT_THRESHOLD: 0.005,
  MOMENTUM_THRESHOLD: 0.015,
  VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD: 0.15,
  TREND_STRENGTH_THRESHOLD: 0.003,
  DEVIATION_THRESHOLD: 0.008,
  VOLUME_MULTIPLIER: 1.2,
  TREND_SLOPE_THRESHOLD: 0.008,
  SHORT_MOMENTUM_THRESHOLD: 0.008,
  DAYS_SINCE_TRADE_THRESHOLD: 15,
  DYNAMIC_BREAKOUT_THRESHOLD: 0.3,
  HIGH_CONFIDENCE_THRESHOLD: 0.7,
  MOMENTUM_MULTIPLIER: 0.05,
  MAX_PROFIT_TAKE: 4.0,
  NEGATIVE_MOMENTUM_THRESHOLD: -0.04,
  NEGATIVE_SHORT_MOMENTUM_THRESHOLD: -0.015,
  MEAN_REVERSION_THRESHOLD: 0.01,
  NEGATIVE_SHORT_MOMENTUM_MIN: -0.01,
  MOMENTUM_MAX: 0.025,
  POSITION_SIZE_MAX: 0.25,
  POSITION_SIZE_MAX_HIGH_ATR: 0.2,
  POSITION_SIZE_MAX_HIGH_CONFIDENCE: 0.35,
  CONFIDENCE_BOOST_MULTIPLIER: 1.2,
  TREND_SLOPE_BOOST_THRESHOLD: 0.02,
  TREND_SLOPE_POSITION_BOOST: 1.2,
  ATR_POSITION_THRESHOLD: 0.05,
  BUY_PROB_MAX_MULTIPLIER: 2.0,
  VOLUME_BOOST_THRESHOLD: 1.2,
  NEGATIVE_DEVIATION_THRESHOLD: -0.015,
  TREND_STRENGTH_REVERSAL_THRESHOLD: -0.01,
  CONSECUTIVE_BUYS_MAX: 4,
  CONSECUTIVE_BUY_CONFIDENCE_LEVELS: [0.25, 0.3, 0.35, 0.4],
  STRATEGY_PERSISTENCE_TRADES: 3, // New: Min trades to persist strategy
  STRATEGY_PERSISTENCE_DAYS: 5, // New: Min days to persist strategy
  STRATEGY_OVERRIDE_CONFIDENCE: 0.8, // New: Confidence threshold to override persistence
};

const MODEL_CONFIG_BASE = {
  CONV1D_FILTERS_1: 12,
  CONV1D_FILTERS_2: 24,
  CONV1D_KERNEL_SIZE_1: 5,
  CONV1D_KERNEL_SIZE_2: 3,
  LSTM_UNITS_1: 64,
  LSTM_UNITS_2: 32,
  LSTM_UNITS_3: 8,
  TIME_DISTRIBUTED_DENSE_UNITS: 16,
  DENSE_UNITS_1: 24,
  OUTPUT_UNITS: 2,
  L2_REGULARIZATION: 0.01,
  DROPOUT_RATE: 0.5,
  TIMESTEPS_AFTER_CONV: 24,
  TIMESTEPS: 30,
  ADA_FEATURE_COUNT: 32,
  BTC_FEATURE_COUNT: 29,
  FEATURE_COUNT: 61,
};

export const MODEL_CONFIG = {
  ...MODEL_CONFIG_BASE,
  LSTM1_INPUT_FEATURES: MODEL_CONFIG_BASE.CONV1D_FILTERS_2,
  CONV1D_1_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.CONV1D_KERNEL_SIZE_1,
    MODEL_CONFIG_BASE.FEATURE_COUNT,
    MODEL_CONFIG_BASE.CONV1D_FILTERS_1,
  ] as [number, number, number],
  CONV1D_2_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.CONV1D_KERNEL_SIZE_2,
    MODEL_CONFIG_BASE.CONV1D_FILTERS_1,
    MODEL_CONFIG_BASE.CONV1D_FILTERS_2,
  ] as [number, number, number],
  LSTM1_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.CONV1D_FILTERS_2,
    MODEL_CONFIG_BASE.LSTM_UNITS_1 * 4,
  ] as [number, number],
  LSTM1_RECURRENT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_1,
    MODEL_CONFIG_BASE.LSTM_UNITS_1 * 4,
  ] as [number, number],
  LSTM2_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_1,
    MODEL_CONFIG_BASE.LSTM_UNITS_2 * 4,
  ] as [number, number],
  LSTM2_RECURRENT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_2,
    MODEL_CONFIG_BASE.LSTM_UNITS_2 * 4,
  ] as [number, number],
  LSTM3_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_2,
    MODEL_CONFIG_BASE.LSTM_UNITS_3 * 4,
  ] as [number, number],
  LSTM3_RECURRENT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_3,
    MODEL_CONFIG_BASE.LSTM_UNITS_3 * 4,
  ] as [number, number],
  TIME_DISTRIBUTED_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.LSTM_UNITS_3,
    MODEL_CONFIG_BASE.TIME_DISTRIBUTED_DENSE_UNITS,
  ] as [number, number],
  DENSE_1_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.TIMESTEPS_AFTER_CONV *
      MODEL_CONFIG_BASE.TIME_DISTRIBUTED_DENSE_UNITS,
    MODEL_CONFIG_BASE.DENSE_UNITS_1,
  ] as [number, number],
  DENSE_2_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.DENSE_UNITS_1,
    MODEL_CONFIG_BASE.OUTPUT_UNITS,
  ] as [number, number],
};

export const TRAINING_CONFIG = {
  EPOCHS: 100,
  BATCH_SIZE: 128,
  SHUFFLE_CHUNK_SIZE: 10,
  INITIAL_LEARNING_RATE: 0.0008,
  MIN_LEARNING_RATE: 0.00005,
  CYCLIC_LR_STEP_SIZE: 10,
  OUTPUT_CLASSES: 2,
  LOOKBACK_DAYS: 450,
  PREDICTION_DAYS: 365,
  TRAIN_SPLIT: 0.7,
  PREFETCH_BUFFER: 2,
  PATIENCE: 40,
  BYTES_TO_MB: 1024 * 1024,
  MS_TO_SECONDS: 1000,
  GAMMA: 2.0,
  ALPHA: [0.6, 0.4] as [number, number],
};

export const TIME_CONVERSIONS = {
  ONE_SECOND_IN_MILLISECONDS: 1000,
  THIRTY_MINUTES_IN_MILLISECONDS: 1800000,
  ONE_MINUTE_IN_SECONDS: 60,
  ONE_HOUR_IN_SECONDS: 3600,
  ONE_DAY_IN_MILLISECONDS: 86400000,
  ONE_DAY_IN_SECONDS: 86400,
  TIMESTEP_IN_SECONDS: MODEL_CONFIG_BASE.TIMESTEPS * 86400,
  ONE_MONTH_IN_DAYS: 30,
};
