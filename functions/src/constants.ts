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
  BUY_PROB_MAX_MULTIPLIER: 1.8, // Reduced for more conservative approach
  CONSECUTIVE_BUYS_MAX: 3, // Reduced for better risk management
  CONSECUTIVE_BUY_CONFIDENCE_LEVELS: [0.3, 0.35, 0.4], // Increased confidence levels
};

const MODEL_CONFIG_BASE = {
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
  FEATURE_COUNT: 25, // EXPERIMENT NEW-6: Reduced core feature set
  BTC_FEATURE_COUNT: 25, // EXPERIMENT NEW-6: Core indicators only
  TIMESTEPS: 30, // EXPERIMENT NEXT-A: Increased for weekly pattern capture
  BATCH_NORM_MOMENTUM: 0.99,
  BATCH_NORM_EPSILON: 0.001,
  ACTIVATION: "relu" as const,
  RECURRENT_ACTIVATION: "sigmoid" as const,
  KERNEL_INITIALIZER: "glorotNormal" as const,
  RECURRENT_INITIALIZER: "orthogonal" as const,
  BIAS_INITIALIZER: "zeros" as const,
  KERNEL_REGULARIZER: "l2" as const,
  RECURRENT_REGULARIZER: "l2" as const,
  BIAS_REGULARIZER: "l2" as const,
  ACTIVITY_REGULARIZER: "l2" as const,
  KERNEL_CONSTRAINT: "maxNorm" as const,
  RECURRENT_CONSTRAINT: "maxNorm" as const,
  BIAS_CONSTRAINT: "maxNorm" as const,
  DROPOUT_IMPLEMENTATION: 1,
  UNROLL: false,
  TIME_MAJOR: false,
  GO_BACKWARDS: false,
  STATEFUL: false,
  RETURN_SEQUENCES: true,
  RETURN_STATE: false,
  IMPLEMENTATION: 1,
  RESET_AFTER: true,
  USE_BIAS: true,
  TRAINABLE: true,
  DYNAMIC: false,
  INPUT_SHAPE: [30, 25] as [number, number], // EXPERIMENT NEXT-A: 30 timesteps, 25 features
  OUTPUT_SHAPE: [2] as [number],
  LOSS: "focalLoss" as const,
  OPTIMIZER: "adam" as const,
  METRICS: ["binaryAccuracy", "customF1Buy", "customF1Sell"] as const,
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
  ] as const,
  COMPILE_OPTIONS: {
    loss: "focalLoss",
    optimizer: "adam",
    metrics: ["binaryAccuracy", "customF1Buy", "customF1Sell"],
  },
  FIT_OPTIONS: {
    epochs: 100,
    batchSize: 64,
    validationSplit: 0.15,
    shuffle: true,
    verbose: 1,
    callbacks: [
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
  },
  PREDICT_OPTIONS: {
    batchSize: 32,
    verbose: 0,
  },
  EVALUATE_OPTIONS: {
    batchSize: 32,
    verbose: 0,
  },
  SAVE_OPTIONS: {
    includeOptimizer: false,
    saveFormat: "tfjs" as const,
  },
  LOAD_OPTIONS: {
    customObjects: {},
  },
  SUMMARY_OPTIONS: {
    lineLength: 80,
    positions: [0.44, 0.62, 0.75, 1.0] as [number, number, number, number],
  },
  PLOT_OPTIONS: {
    showShapes: true,
    showLayerNames: true,
    showLayerActivations: true,
    showLayerWeights: true,
    showLayerGradients: true,
    showLayerOutputs: true,
    showLayerInputs: true,
    showLayerConfigs: true,
    showLayerSummaries: true,
    showLayerMetrics: true,
  },
  DEBUG_OPTIONS: {
    verbose: true,
    logLevel: "info" as const,
    logFormat: "detailed" as const,
    logTiming: true,
    logMemory: true,
    logGradients: true,
    logWeights: true,
    logActivations: true,
    logPredictions: true,
    logMetrics: true,
    logCallbacks: true,
    logTraining: true,
    logValidation: true,
    logTesting: true,
    logInference: true,
    logPerformance: true,
    logOptimization: true,
    logRegularization: true,
    logNormalization: true,
    logAugmentation: true,
    logBalancing: true,
  },
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
  // RECOVERY-1: Revert to PROVEN optimal training configuration
  EPOCHS: 30, // PROVEN OPTIMAL: 30 epochs for faster iteration, 50+ caused issues
  BATCH_SIZE: 16, // PROVEN OPTIMAL: Best balance for gradient updates
  SHUFFLE_CHUNK_SIZE: 10,
  INITIAL_LEARNING_RATE: 0.0005, // PROVEN OPTIMAL: 0.0008 showed promise but 0.0005 is baseline
  MIN_LEARNING_RATE: 0.00005, // 7DAY-2: Higher floor for better 7-day convergence
  CYCLIC_LR_STEP_SIZE: 15, // PROVEN OPTIMAL: 10 showed minor promise but 15 is baseline
  OUTPUT_CLASSES: 2,
  START_DAYS_AGO: 600, // PROVEN OPTIMAL: 600 days documented in experiments
  TRAIN_SPLIT: 0.8, // PROVEN OPTIMAL: 80% train / 20% validation split
  PREFETCH_BUFFER: 4,
  PATIENCE: 10, // PROVEN OPTIMAL: Early stopping patience
  BYTES_TO_MB: 1024 * 1024,
  MS_TO_SECONDS: 1000,
  GAMMA: 1.5, // PROVEN OPTIMAL: Found in experiments, prevents class collapse
  ALPHA: [0.45, 0.55] as [number, number], // 7DAY-1: More balanced for 7-day predictions
  GRADIENT_CLIP_NORM: 1.0, // REDUCED: 5.0 → 1.0 for natural gradients
  LR_DECAY_RATE: 0.8, // REDUCED: was 0.92, now 0.8 for more aggressive decay
  WARMUP_EPOCHS: 2, // REDUCED: was 5, now 2 for faster warmup
  WARMUP_INITIAL_LR: 0.0001, // INCREASED: was 0.00005, now 0.0001
  ATTENTION_DROPOUT: 0.1, // REDUCED: was 0.15, now 0.1
  RESIDUAL_DROPOUT: 0.1, // REDUCED: was 0.15, now 0.1
  L2_REGULARIZATION: 0.0001, // REDUCED: was 0.0008, now 0.0001 for less constraint
  BATCH_NORMALIZATION: false, // DISABLED: was true, now false for speed
  USE_ATTENTION: false, // DISABLED: was true, now false for speed
  USE_RESIDUAL_CONNECTIONS: false, // DISABLED: was true, now false for speed
  USE_GRADIENT_CLIPPING: true,
  USE_LEARNING_RATE_SCHEDULER: true,
  USE_WARMUP: true,
  USE_DROPOUT: true,
  USE_L2_REGULARIZATION: true,
  USE_BATCH_NORMALIZATION: false, // DISABLED: was true, now false for speed
  VERBOSE: 1,
  // CRITICAL FIX: Use stratified split instead of time-based split
  TIME_BASED_SPLIT: false, // CHANGED: was true, now false for better learning
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
  TIMESTEP_IN_SECONDS: MODEL_CONFIG_BASE.TIMESTEPS * 86400, // Updated with new timesteps
  ONE_MONTH_IN_DAYS: 30,
};
