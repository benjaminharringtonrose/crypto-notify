import { RLEnvironmentConfig, RLAgentConfig } from "./RLTradingEnvironment";

/**
 * IMPROVED RL CONFIGURATIONS
 *
 * These configurations address the key issues observed in training:
 * 1. Poor reward function leading to consistent losses
 * 2. Insufficient exploration (epsilon stuck at 0.08)
 * 3. Overly complex state representation
 * 4. Inadequate risk management
 * 5. Poor learning rate and network architecture
 */

/**
 * IMPROVED ENVIRONMENT CONFIGURATION
 * Focus on simple, profitable patterns first
 */
export const ImprovedRLEnvironmentConfig: RLEnvironmentConfig = {
  // Basic trading parameters
  initialCapital: 10000,
  commission: 0.001, // Reduced from 0.0025 - lower transaction costs
  slippage: 0.0005, // Reduced from 0.001 - more realistic
  timesteps: 20, // Reduced from 30 - faster training
  maxPositionSize: 0.8, // Reduced from 1.0 - better risk management
  minPositionSize: 0.1, // Increased from 0.05 - more meaningful positions

  // IMPROVED REWARD SCALING
  rewardScaling: 500, // Increased from 200 - stronger reward signals
  riskFreeRate: 0.02, // 2% annual risk-free rate
  transactionCostPenalty: 0.1, // Reduced from 0.5 - less penalty for trading
  holdingPenalty: 0, // Removed - allow holding profitable positions
  volatilityPenalty: 0.05, // Reduced from 0.2 - less penalty for volatility

  // ENABLED FEATURES - Focus on core functionality
  enableMarketRegimeDetection: true,
  enableSyntheticData: false, // Disabled initially - focus on real data
  enableHybridApproach: false, // Disabled initially - simplify learning
  enableDynamicPositionSizing: true,

  // IMPROVED REGIME THRESHOLDS - More sensitive detection
  regimeThresholds: {
    volatilityThreshold: 0.015, // Reduced from 0.02 - more sensitive
    trendStrengthThreshold: 0.0002, // Reduced from 0.0005 - detect weaker trends
    momentumThreshold: 0.001, // Reduced from 0.003 - detect weaker momentum
  },

  // IMPROVED RISK MANAGEMENT
  riskManagement: {
    maxDrawdownThreshold: 0.15, // Reduced from 0.25 - stricter risk control
    positionSizeMultiplier: 1.2, // Reduced from 1.5 - more conservative
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },

  // SYNTHETIC DATA CONFIG (disabled for now)
  syntheticDataConfig: {
    enableTrendingMarkets: false,
    enableRangingMarkets: false,
    enableVolatileMarkets: false,
    noiseLevel: 0.01,
    trendStrength: 0.002,
  },
};

/**
 * IMPROVED AGENT CONFIGURATION
 * Focus on better learning dynamics
 */
export const ImprovedRLAgentConfig: RLAgentConfig = {
  // IMPROVED LEARNING PARAMETERS
  learningRate: 0.001, // Reduced from 0.003 - more stable learning
  discountFactor: 0.95, // Increased from 0.90 - better long-term planning
  epsilon: 0.8, // Increased from 0.5 - much more exploration
  epsilonDecay: 0.995, // Slower decay - maintain exploration longer
  epsilonMin: 0.15, // Increased from 0.08 - maintain some exploration

  // IMPROVED TRAINING PARAMETERS
  batchSize: 64, // Reduced from 128 - more frequent updates
  memorySize: 10000, // Reduced from 20000 - faster learning from recent experiences
  targetUpdateFrequency: 10, // Reduced from 25 - more frequent target updates

  // SIMPLIFIED NETWORK ARCHITECTURE
  hiddenLayers: [256, 128, 64], // Reduced complexity - faster training
  activationFunction: "relu",
  optimizer: "adam",
  lossFunction: "meanSquaredError",
  gradientClipping: 0.5, // Increased from 0.3 - more stable gradients

  // ENABLED ADVANCED FEATURES
  experienceReplay: true,
  prioritizedReplay: true,
  doubleDQN: true,
  duelingDQN: true,
};

/**
 * CONSERVATIVE CONFIGURATION
 * For more risk-averse trading
 */
export const ConservativeRLConfig: RLEnvironmentConfig = {
  ...ImprovedRLEnvironmentConfig,
  maxPositionSize: 0.5, // Even more conservative
  minPositionSize: 0.05,
  rewardScaling: 300, // Lower reward scaling
  regimeThresholds: {
    volatilityThreshold: 0.02, // Higher threshold - less sensitive
    trendStrengthThreshold: 0.0005, // Higher threshold
    momentumThreshold: 0.002, // Higher threshold
  },
  riskManagement: {
    maxDrawdownThreshold: 0.1, // Very strict risk control
    positionSizeMultiplier: 1.0, // No multiplier
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * AGGRESSIVE CONFIGURATION
 * For higher returns (more risk)
 */
export const AggressiveRLConfig: RLEnvironmentConfig = {
  ...ImprovedRLEnvironmentConfig,
  maxPositionSize: 1.0, // Full position size
  minPositionSize: 0.2,
  rewardScaling: 800, // Higher reward scaling
  regimeThresholds: {
    volatilityThreshold: 0.01, // More sensitive
    trendStrengthThreshold: 0.0001, // More sensitive
    momentumThreshold: 0.0005, // More sensitive
  },
  riskManagement: {
    maxDrawdownThreshold: 0.2, // Higher risk tolerance
    positionSizeMultiplier: 1.5, // Larger positions
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * LEARNING-FOCUSED CONFIGURATION
 * Optimized for training and exploration
 */
export const LearningFocusedConfig: RLEnvironmentConfig = {
  ...ImprovedRLEnvironmentConfig,
  rewardScaling: 1000, // Very high reward scaling for clear signals
  transactionCostPenalty: 0.05, // Minimal transaction costs
  volatilityPenalty: 0.02, // Minimal volatility penalty
  regimeThresholds: {
    volatilityThreshold: 0.008, // Very sensitive
    trendStrengthThreshold: 0.00005, // Very sensitive
    momentumThreshold: 0.0002, // Very sensitive
  },
  riskManagement: {
    maxDrawdownThreshold: 0.25, // Higher tolerance for learning
    positionSizeMultiplier: 1.8, // Larger positions for learning
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * QUICK TRAINING CONFIGURATION
 * For rapid testing and iteration
 */
export const QuickTrainingConfig: RLEnvironmentConfig = {
  ...ImprovedRLEnvironmentConfig,
  timesteps: 10, // Very short episodes
  rewardScaling: 2000, // Very high rewards for quick learning
  transactionCostPenalty: 0.02, // Minimal costs
  volatilityPenalty: 0.01, // Minimal penalty
  regimeThresholds: {
    volatilityThreshold: 0.005, // Very sensitive
    trendStrengthThreshold: 0.00002, // Very sensitive
    momentumThreshold: 0.0001, // Very sensitive
  },
  riskManagement: {
    maxDrawdownThreshold: 0.3, // High tolerance
    positionSizeMultiplier: 2.0, // Large positions
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * UTILITY FUNCTIONS
 */
export function createCustomRLConfig(
  baseConfig: RLEnvironmentConfig,
  overrides: Partial<RLEnvironmentConfig>
): RLEnvironmentConfig {
  return { ...baseConfig, ...overrides };
}

export function validateRLConfig(config: RLEnvironmentConfig): boolean {
  return (
    config.initialCapital > 0 &&
    config.commission >= 0 &&
    config.slippage >= 0 &&
    config.timesteps > 0 &&
    config.maxPositionSize > 0 &&
    config.minPositionSize > 0 &&
    config.rewardScaling > 0
  );
}

/**
 * CONFIGURATION PRESETS FOR DIFFERENT MARKET CONDITIONS
 */
export const MarketConditionConfigs = {
  bull: createCustomRLConfig(ImprovedRLEnvironmentConfig, {
    maxPositionSize: 1.0,
    rewardScaling: 600,
    regimeThresholds: {
      volatilityThreshold: 0.012,
      trendStrengthThreshold: 0.0001,
      momentumThreshold: 0.0005,
    },
  }),
  bear: createCustomRLConfig(ImprovedRLEnvironmentConfig, {
    maxPositionSize: 0.3,
    rewardScaling: 400,
    regimeThresholds: {
      volatilityThreshold: 0.025,
      trendStrengthThreshold: 0.0008,
      momentumThreshold: 0.003,
    },
  }),
  sideways: createCustomRLConfig(ImprovedRLEnvironmentConfig, {
    maxPositionSize: 0.6,
    rewardScaling: 300,
    regimeThresholds: {
      volatilityThreshold: 0.018,
      trendStrengthThreshold: 0.0004,
      momentumThreshold: 0.002,
    },
  }),
};
