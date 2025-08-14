import { firestore } from "firebase-admin";

export const PRICES = [0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
export const MERGE_PAYLOAD: firestore.SetOptions = { merge: true };
export const FILE_NAMES = { WEIGHTS: "tradePredictorWeights.json" };

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
  SMA_50: 50,
  SMA_200: 200,
  ADX: 14,
} as const;

export const STRATEGY_CONFIG = {
  MIN_CONFIDENCE_DEFAULT: 0.16, // Back to successful setting
  PROFIT_TAKE_MULTIPLIER_DEFAULT: 2.0, // Back to successful setting
  BASE_POSITION_SIZE_DEFAULT: 0.085, // Back to successful setting
  SLIPPAGE: 0.001,
  COMMISSION: 0.005, // Fixed: Reduced from 0.1 (10%) to 0.005 (0.5%)
  STOP_LOSS_MULTIPLIER_DEFAULT: 5.2, // Back to successful setting
  TRAILING_STOP_DEFAULT: 0.12, // Back to successful setting
  MIN_HOLD_DAYS_DEFAULT: 2.5, // Back to successful setting
  BUY_PROB_THRESHOLD_DEFAULT: 0.105, // Back to successful setting
  SELL_PROB_THRESHOLD_DEFAULT: 0.185, // Back to successful setting
  MOMENTUM_WINDOW_THRESHOLD: 0.01,
  MAX_ATR_THRESHOLD: 0.15,
  MIN_PROFIT_THRESHOLD: 0.0012, // Back to successful setting
  MOMENTUM_THRESHOLD: 0.0016, // Back to successful setting
  VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD: 0.016, // Back to successful setting
  TREND_STRENGTH_THRESHOLD: 0.0035, // Back to successful setting
  TREND_SLOPE_THRESHOLD: 0.0035, // Back to successful setting
  SHORT_MOMENTUM_THRESHOLD: 0.0035, // Back to successful setting
  DAYS_SINCE_TRADE_THRESHOLD: 5, // Back to successful setting
  DYNAMIC_BREAKOUT_THRESHOLD: 0.2, // Back to successful setting
  HIGH_CONFIDENCE_THRESHOLD: 0.46, // Back to successful setting
  MOMENTUM_MULTIPLIER: 0.05,
  MAX_PROFIT_TAKE: 2.0, // Back to successful setting
  NEGATIVE_MOMENTUM_THRESHOLD: -0.032,
  NEGATIVE_SHORT_MOMENTUM_THRESHOLD: -0.011,
  MEAN_REVERSION_THRESHOLD: 0.012, // Back to successful setting
  NEGATIVE_SHORT_MOMENTUM_MIN: -0.0085,
  MOMENTUM_MAX: 0.021,
  POSITION_SIZE_MAX: 0.3, // Back to successful setting
  POSITION_SIZE_MAX_HIGH_ATR: 0.35, // Back to successful setting
  POSITION_SIZE_MAX_HIGH_CONFIDENCE: 0.4, // Back to successful setting
  CONFIDENCE_BOOST_MULTIPLIER: 1.75, // Back to successful setting
  TREND_SLOPE_BOOST_THRESHOLD: 0.016,
  TREND_SLOPE_POSITION_BOOST: 1.28,
  VOLUME_BOOST_THRESHOLD: 1.2,
  NEGATIVE_DEVIATION_THRESHOLD: -0.015,
  VOLUME_MULTIPLIER: 1.5,
  TREND_STRENGTH_REVERSAL_THRESHOLD: -0.01,
  ATR_POSITION_THRESHOLD: 0.05,
  BUY_PROB_MAX_MULTIPLIER: 2.0,
  CONSECUTIVE_BUYS_MAX: 4,
  CONSECUTIVE_BUY_CONFIDENCE_LEVELS: [0.25, 0.3, 0.35, 0.4],
};

const MODEL_CONFIG_BASE = {
  CONV1D_FILTERS_1: 12, // Increased from 8 for better feature extraction
  CONV1D_FILTERS_2: 24, // Increased from 16 for better feature extraction
  CONV1D_KERNEL_SIZE_1: 5,
  CONV1D_KERNEL_SIZE_2: 3,
  LSTM_UNITS_1: 48, // Increased from 32 for better temporal modeling
  LSTM_UNITS_2: 24, // Increased from 16 for better temporal modeling
  LSTM_UNITS_3: 12, // Increased from 8 for better temporal modeling
  TIME_DISTRIBUTED_DENSE_UNITS: 16, // Increased from 12 for better feature learning
  DENSE_UNITS_1: 24, // Increased from 16 for better feature learning
  OUTPUT_UNITS: 2,
  L2_REGULARIZATION: 0.008, // Reduced from 0.01 for better generalization
  DROPOUT_RATE: 0.35, // Reduced from 0.4 for better training stability
  ATTENTION_UNITS_1: 16, // New: units for first attention layer
  ATTENTION_UNITS_2: 12, // New: units for second attention layer
  RESIDUAL_FILTERS: 8, // New: filters for residual connections
  TIMESTEPS_AFTER_CONV: 24,
  TIMESTEPS: 30,
  BTC_FEATURE_COUNT: 62, // Bitcoin features only
  FEATURE_COUNT: 62, // Total features
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
    MODEL_CONFIG_BASE.LSTM_UNITS_3,
    MODEL_CONFIG_BASE.DENSE_UNITS_1,
  ] as [number, number],
  DENSE_2_WEIGHT_SHAPE: [
    MODEL_CONFIG_BASE.DENSE_UNITS_1,
    Math.floor(MODEL_CONFIG_BASE.DENSE_UNITS_1 / 2),
  ] as [number, number],
  OUTPUT_WEIGHT_SHAPE: [
    Math.floor(MODEL_CONFIG_BASE.DENSE_UNITS_1 / 2),
    MODEL_CONFIG_BASE.OUTPUT_UNITS,
  ] as [number, number],
};

export const TRAINING_CONFIG = {
  EPOCHS: 120, // Increased from 100 for better convergence
  BATCH_SIZE: 32, // Reduced from 64 for better generalization with complex model
  SHUFFLE_CHUNK_SIZE: 10,
  INITIAL_LEARNING_RATE: 0.0008, // Reduced from 0.001 for more stable training
  MIN_LEARNING_RATE: 0.000005, // Reduced from 0.00001 for better fine-tuning
  CYCLIC_LR_STEP_SIZE: 12, // Increased from 8 for longer cycles
  OUTPUT_CLASSES: 2,
  START_DAYS_AGO: 1200, // Increased from 1000 for more training data
  TRAIN_SPLIT: 0.85, // Increased from 0.8 for more training data
  PREFETCH_BUFFER: 4, // Increased from 2 for better data loading
  PATIENCE: 20, // Increased from 15 for more stable training
  BYTES_TO_MB: 1024 * 1024,
  MS_TO_SECONDS: 1000,
  GAMMA: 1.5, // Reduced from 2.0 for less aggressive focal loss
  ALPHA: [0.35, 0.65] as [number, number], // Moderate buy emphasis (more conservative)
  GRADIENT_CLIP_NORM: 1.0, // Reduced from 1.5 for more stable gradients
  LR_DECAY_RATE: 0.92, // Reduced from 0.95 for slower decay
  WARMUP_EPOCHS: 8, // Increased from 5 for better initialization
  WARMUP_INITIAL_LR: 0.00005, // Reduced from 0.0001 for gentler warmup
  ATTENTION_DROPOUT: 0.1, // New: dropout for attention mechanisms
  RESIDUAL_DROPOUT: 0.15, // New: dropout for residual connections
  LAYER_NORM_EPSILON: 1e-6, // New: epsilon for layer normalization
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
