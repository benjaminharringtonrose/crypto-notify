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
  MIN_CONFIDENCE_DEFAULT: 0.15, // Increased for better trade quality
  PROFIT_TAKE_MULTIPLIER_DEFAULT: 2.5, // Increased for better profit capture
  BASE_POSITION_SIZE_DEFAULT: 0.08, // Slightly reduced for better risk management
  SLIPPAGE: 0.001,
  COMMISSION: 0.005,
  STOP_LOSS_MULTIPLIER_DEFAULT: 4.0, // Reduced for tighter risk control
  TRAILING_STOP_DEFAULT: 0.1, // Reduced for better profit protection
  MIN_HOLD_DAYS_DEFAULT: 3, // Increased for better trend following
  BUY_PROB_THRESHOLD_DEFAULT: 0.08, // Increased for better signal quality
  SELL_PROB_THRESHOLD_DEFAULT: 0.15, // Increased for better signal quality
  MOMENTUM_WINDOW_THRESHOLD: 0.015, // Increased for better momentum detection
  MAX_ATR_THRESHOLD: 1.8, // Reduced for better volatility control
  MIN_PROFIT_THRESHOLD: 0.0015, // Increased for better profit targets
  MOMENTUM_THRESHOLD: 0.002, // Increased for better momentum signals
  VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD: 0.012, // Increased for better signal quality
  TREND_STRENGTH_THRESHOLD: 0.002, // Increased for better trend signals
  TREND_SLOPE_THRESHOLD: 0.002, // Increased for better trend signals
  SHORT_MOMENTUM_THRESHOLD: 0.002, // Increased for better momentum signals
  DAYS_SINCE_TRADE_THRESHOLD: 4, // Reduced for more frequent trading
  DYNAMIC_BREAKOUT_THRESHOLD: 0.18, // Increased for better breakout signals
  HIGH_CONFIDENCE_THRESHOLD: 0.5, // Increased for better confidence threshold
  MOMENTUM_MULTIPLIER: 0.06, // Increased for better momentum detection
  MAX_PROFIT_TAKE: 2.5, // Increased for better profit capture
  NEGATIVE_MOMENTUM_THRESHOLD: -0.025, // Reduced for better momentum control
  NEGATIVE_SHORT_MOMENTUM_THRESHOLD: -0.008, // Reduced for better momentum control
  MEAN_REVERSION_THRESHOLD: 0.015, // Increased for better mean reversion signals
  NEGATIVE_SHORT_MOMENTUM_MIN: -0.006, // Reduced for better momentum control
  MOMENTUM_MAX: 0.025, // Increased for better momentum signals
  POSITION_SIZE_MAX: 0.25, // Reduced for better risk management
  POSITION_SIZE_MAX_HIGH_ATR: 0.3, // Reduced for better risk management
  POSITION_SIZE_MAX_HIGH_CONFIDENCE: 0.35, // Reduced for better risk management
  CONFIDENCE_BOOST_MULTIPLIER: 1.5, // Reduced for more conservative approach
  TREND_SLOPE_BOOST_THRESHOLD: 0.02, // Increased for better trend signals
  TREND_SLOPE_POSITION_BOOST: 1.2, // Reduced for more conservative approach
  VOLUME_BOOST_THRESHOLD: 1.3, // Increased for better volume signals
  NEGATIVE_DEVIATION_THRESHOLD: -0.02, // Increased for better mean reversion
  VOLUME_MULTIPLIER: 1.4, // Increased for better volume signals
  TREND_STRENGTH_REVERSAL_THRESHOLD: -0.015, // Increased for better trend signals
  ATR_POSITION_THRESHOLD: 0.06, // Increased for better volatility control
  BUY_PROB_MAX_MULTIPLIER: 1.8, // Reduced for more conservative approach
  CONSECUTIVE_BUYS_MAX: 3, // Reduced for better risk management
  CONSECUTIVE_BUY_CONFIDENCE_LEVELS: [0.3, 0.35, 0.4], // Increased confidence levels
};

const MODEL_CONFIG_BASE = {
  CONV1D_FILTERS_1: 24, // Increased for better feature extraction
  CONV1D_FILTERS_2: 48, // Increased for better feature extraction
  CONV1D_KERNEL_SIZE_1: 5,
  CONV1D_KERNEL_SIZE_2: 3,
  LSTM_UNITS_1: 96, // Increased for better temporal modeling
  LSTM_UNITS_2: 48, // Increased for better temporal modeling
  LSTM_UNITS_3: 24, // Increased for better temporal modeling
  TIME_DISTRIBUTED_DENSE_UNITS: 32, // Increased for better feature learning
  DENSE_UNITS_1: 48, // Increased for better feature learning
  OUTPUT_UNITS: 2,
  L2_REGULARIZATION: 0.002, // Reduced for less regularization
  DROPOUT_RATE: 0.25, // Reduced dropout for better training
  ATTENTION_UNITS_1: 32, // Increased for better attention mechanism
  ATTENTION_UNITS_2: 24, // Increased for better attention mechanism
  RESIDUAL_FILTERS: 16, // Increased for better residual connections
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
  EPOCHS: 200, // Increased for better convergence
  BATCH_SIZE: 32, // Reduced for better gradient updates
  SHUFFLE_CHUNK_SIZE: 10,
  INITIAL_LEARNING_RATE: 0.001, // Increased for faster learning
  MIN_LEARNING_RATE: 0.00001, // Increased minimum learning rate
  CYCLIC_LR_STEP_SIZE: 20, // Increased for better learning rate cycling
  OUTPUT_CLASSES: 2,
  START_DAYS_AGO: 1200,
  TRAIN_SPLIT: 0.8, // Reduced to prevent overfitting
  PREFETCH_BUFFER: 4,
  PATIENCE: 25, // Increased patience for better convergence
  BYTES_TO_MB: 1024 * 1024,
  MS_TO_SECONDS: 1000,
  GAMMA: 1.5, // Reduced focal loss gamma for better balance
  ALPHA: [0.5, 0.5] as [number, number], // Balanced alpha for better class balance
  GRADIENT_CLIP_NORM: 1.0,
  LR_DECAY_RATE: 0.95, // Slower decay for better convergence
  WARMUP_EPOCHS: 3, // Reduced warmup
  WARMUP_INITIAL_LR: 0.0001, // Increased warmup learning rate
  ATTENTION_DROPOUT: 0.2, // Increased dropout for better regularization
  RESIDUAL_DROPOUT: 0.2, // Increased dropout for better regularization
  L2_REGULARIZATION: 0.001, // Reduced L2 for less regularization
  BATCH_NORMALIZATION: true,
  USE_ATTENTION: true,
  USE_RESIDUAL_CONNECTIONS: true,
  USE_GRADIENT_CLIPPING: true,
  USE_LEARNING_RATE_SCHEDULER: true,
  USE_WARMUP: true,
  USE_DROPOUT: true,
  USE_L2_REGULARIZATION: true,
  USE_BATCH_NORMALIZATION: true,
  VERBOSE: 1,
  CALLBACKS: [
    "earlyStopping",
    "reduceLROnPlateau",
    "modelCheckpoint",
    "tensorBoard",
    "csvLogger",
    "gradientClipping",
    "curriculumLearning",
    "exponentialDecayLR",
    "trainingLogger",
    "predictionLogger",
  ],
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
