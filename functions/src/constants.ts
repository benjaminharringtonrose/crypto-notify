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
  MIN_CONFIDENCE_DEFAULT: 0.05, // Back to permissive to allow trades
  PROFIT_TAKE_MULTIPLIER_DEFAULT: 2.0, // Balanced profit taking
  BASE_POSITION_SIZE_DEFAULT: 0.1, // Balanced position sizing
  SLIPPAGE: 0.001,
  COMMISSION: 0.005,
  STOP_LOSS_MULTIPLIER_DEFAULT: 3.0, // Balanced stop loss
  TRAILING_STOP_DEFAULT: 0.08, // Balanced trailing stop
  MIN_HOLD_DAYS_DEFAULT: 2, // Balanced holding period
  BUY_PROB_THRESHOLD_DEFAULT: 0.05, // Back to permissive
  SELL_PROB_THRESHOLD_DEFAULT: 0.05, // Back to permissive
  MOMENTUM_WINDOW_THRESHOLD: 0.015, // Back to balanced
  MAX_ATR_THRESHOLD: 2.0, // Back to balanced
  MIN_PROFIT_THRESHOLD: 0.002, // Back to balanced
  MOMENTUM_THRESHOLD: 0.003, // Back to balanced
  VOLATILITY_ADJUSTED_MOMENTUM_THRESHOLD: -0.1, // Back to permissive
  TREND_STRENGTH_THRESHOLD: -0.1, // Back to permissive
  TREND_SLOPE_THRESHOLD: -0.1, // Back to permissive
  SHORT_MOMENTUM_THRESHOLD: -0.05, // Back to permissive
  DAYS_SINCE_TRADE_THRESHOLD: 3, // Back to balanced
  DYNAMIC_BREAKOUT_THRESHOLD: 0.1, // Back to balanced
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
  VOLUME_BOOST_THRESHOLD: 0.1, // Much more permissive for volume
  NEGATIVE_DEVIATION_THRESHOLD: -0.1, // Much more permissive for mean reversion
  VOLUME_MULTIPLIER: 0.1, // Much more permissive for volume
  TREND_STRENGTH_REVERSAL_THRESHOLD: -0.015, // Increased for better trend signals
  ATR_POSITION_THRESHOLD: 0.06, // Increased for better volatility control
};

export const MODEL_CONFIG = {
  CONV1D_FILTERS_1: 32, // Much smaller for faster training
  CONV1D_FILTERS_2: 64, // Much smaller for faster training
  CONV1D_KERNEL_SIZE_1: 5, // Smaller kernel for faster training
  CONV1D_KERNEL_SIZE_2: 3, // Smaller kernel for faster training
  LSTM_UNITS_1: 64, // Much smaller for faster training
  LSTM_UNITS_2: 32, // Much smaller for faster training
  LSTM_UNITS_3: 16, // Much smaller for faster training
  TIME_DISTRIBUTED_DENSE_UNITS: 32, // Much smaller for faster training
  DENSE_UNITS_1: 64, // Much smaller for faster training
  OUTPUT_UNITS: 2,
  L2_REGULARIZATION: 0.001, // More regularization for smaller model
  DROPOUT_RATE: 0.2, // More dropout for smaller model
  ATTENTION_UNITS_1: 32, // Much smaller for faster training
  ATTENTION_UNITS_2: 16, // Much smaller for faster training
  RESIDUAL_UNITS_1: 32, // Much smaller for faster training
  RESIDUAL_UNITS_2: 16, // Much smaller for faster training
  TIMESTEPS: 35, // v1.3.0: Increased for monthly cycle capture in 7-day predictions
};

export const TRAINING_CONFIG = {
  EPOCHS: 35, // PROVEN OPTIMAL: 30 epochs for faster iteration, 50+ caused issues
  BATCH_SIZE: 16, // PROVEN OPTIMAL: Best balance for gradient updates
  SHUFFLE_CHUNK_SIZE: 10,
  INITIAL_LEARNING_RATE: 0.0005, // PROVEN OPTIMAL: 0.0008 showed promise but 0.0005 is baseline
  MIN_LEARNING_RATE: 0.00005, // 7DAY-2: Higher floor for better 7-day convergence
  CYCLIC_LR_STEP_SIZE: 15, // PROVEN OPTIMAL: 10 showed minor promise but 15 is baseline
  OUTPUT_CLASSES: 2,
  START_DAYS_AGO: 730, // PROVEN OPTIMAL: 600 days documented in experiments
  TRAIN_SPLIT: 0.8, // PROVEN OPTIMAL: 80% train / 20% validation split
  PREFETCH_BUFFER: 4,
  PATIENCE: 10, // PROVEN OPTIMAL: Early stopping patience
  TRAINING_VERBOSE: 0, // 0=silent, 1=progress bar, 2=one line per epoch
  BYTES_TO_MB: 1024 * 1024,
  MS_TO_SECONDS: 1000,
  GAMMA: 1.5, // PROVEN OPTIMAL: Found in experiments, prevents class collapse
  ALPHA: [0.45, 0.55] as [number, number], // 7DAY-1: More balanced for 7-day predictions
  GRADIENT_CLIP_NORM: 1.0, // REDUCED: 5.0 → 1.0 for natural gradients
  LR_DECAY_RATE: 0.8, // REDUCED: was 0.92, now 0.8 for more aggressive decay
  WARMUP_EPOCHS: 2, // REDUCED: was 5, now 2 for faster warmup
  WARMUP_INITIAL_LR: 0.0001, // INCREASED: was 0.00005, now 0.0001
};

export const TIME_CONVERSIONS = {
  ONE_SECOND_IN_MILLISECONDS: 1000,
  THIRTY_MINUTES_IN_MILLISECONDS: 1800000,
  ONE_HOUR_IN_SECONDS: 3600,
  ONE_DAY_IN_MILLISECONDS: 86400000,
  ONE_DAY_IN_SECONDS: 86400,
  TIMESTEP_IN_SECONDS: MODEL_CONFIG.TIMESTEPS * 86400, // v1.3.0: Updated with 35 timesteps
  ONE_MONTH_IN_DAYS: 30,
};
