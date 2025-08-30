import { RLEnvironmentConfig, RLAgentConfig } from "./RLTradingEnvironment";

/**
 * Enhanced RL Configuration with all new features enabled
 * This configuration demonstrates the implementation of:
 * 1. Market regime detection - only trade in favorable conditions
 * 2. Synthetic data - create easier market scenarios for learning
 * 3. Hybrid approach - combine RL with rule-based strategies
 * 4. Sophisticated risk management - dynamic position sizing
 */
export const EnhancedRLEnvironmentConfig: RLEnvironmentConfig = {
  // Basic trading parameters
  initialCapital: 10000,
  commission: 0.001, // 0.1% commission
  slippage: 0.0005, // 0.05% slippage
  timesteps: 50, // Lookback period for features
  maxPositionSize: 1.0, // Maximum position size (100%)
  minPositionSize: 0.1, // Minimum position size (10%)
  rewardScaling: 100, // Scale rewards for better learning
  riskFreeRate: 0.02, // 2% annual risk-free rate
  transactionCostPenalty: 0.01, // Penalty for frequent trading
  holdingPenalty: 0.001, // Small penalty for holding positions
  volatilityPenalty: 0.05, // Penalty for high volatility

  // NEW: Enable enhanced features
  enableMarketRegimeDetection: true,
  enableSyntheticData: true,
  enableHybridApproach: true,
  enableDynamicPositionSizing: true,

  // Market regime detection thresholds
  regimeThresholds: {
    volatilityThreshold: 0.03, // 3% volatility threshold for volatile regime
    trendStrengthThreshold: 0.001, // Trend strength threshold
    momentumThreshold: 0.005, // Momentum threshold for bullish/bearish
  },

  // Risk management configuration
  riskManagement: {
    maxDrawdownThreshold: 0.2, // 20% max drawdown
    positionSizeMultiplier: 1.0, // Base position size multiplier
    volatilityScaling: true, // Scale position size by volatility
    regimeBasedPositionSizing: true, // Use regime-based position sizing
  },

  // Synthetic data configuration
  syntheticDataConfig: {
    enableTrendingMarkets: true, // Create trending market scenarios
    enableRangingMarkets: true, // Create ranging market scenarios
    enableVolatileMarkets: true, // Create volatile market scenarios
    noiseLevel: 0.01, // 1% noise level for synthetic data
    trendStrength: 0.002, // 0.2% daily trend strength
  },
};

/**
 * Enhanced RL Agent Configuration
 */
export const EnhancedRLAgentConfig: RLAgentConfig = {
  // Learning parameters
  learningRate: 0.001,
  discountFactor: 0.95, // Future reward discount
  epsilon: 0.3, // Initial exploration rate
  epsilonDecay: 0.995, // Exploration decay rate
  epsilonMin: 0.01, // Minimum exploration rate

  // Training parameters
  batchSize: 32,
  memorySize: 10000, // Experience replay memory size
  targetUpdateFrequency: 100, // Update target network every 100 steps

  // Network architecture
  hiddenLayers: [128, 64, 32], // Deeper network for complex patterns
  activationFunction: "relu",
  optimizer: "adam",
  lossFunction: "meanSquaredError",
  gradientClipping: 1.0,

  // Advanced features
  experienceReplay: true,
  prioritizedReplay: true, // Prioritize important experiences
  doubleDQN: true, // Use Double DQN for better stability
  duelingDQN: true, // Use Dueling DQN for better value estimation
};

/**
 * Conservative RL Configuration - Focus on risk management
 */
export const ConservativeRLConfig: RLEnvironmentConfig = {
  ...EnhancedRLEnvironmentConfig,
  maxPositionSize: 0.5, // More conservative position sizing
  minPositionSize: 0.05, // Smaller minimum position
  regimeThresholds: {
    volatilityThreshold: 0.02, // Lower volatility threshold
    trendStrengthThreshold: 0.002, // Higher trend strength requirement
    momentumThreshold: 0.01, // Higher momentum requirement
  },
  riskManagement: {
    maxDrawdownThreshold: 0.15, // Lower drawdown threshold
    positionSizeMultiplier: 0.7, // More conservative position sizing
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * Aggressive RL Configuration - Focus on returns
 */
export const AggressiveRLConfig: RLEnvironmentConfig = {
  ...EnhancedRLEnvironmentConfig,
  maxPositionSize: 1.0, // Full position sizing
  minPositionSize: 0.2, // Higher minimum position
  regimeThresholds: {
    volatilityThreshold: 0.05, // Higher volatility tolerance
    trendStrengthThreshold: 0.0005, // Lower trend strength requirement
    momentumThreshold: 0.003, // Lower momentum requirement
  },
  riskManagement: {
    maxDrawdownThreshold: 0.25, // Higher drawdown tolerance
    positionSizeMultiplier: 1.2, // More aggressive position sizing
    volatilityScaling: false, // Disable volatility scaling
    regimeBasedPositionSizing: true,
  },
};

/**
 * Hybrid-Focused Configuration - Emphasize rule-based integration
 */
export const HybridFocusedConfig: RLEnvironmentConfig = {
  ...EnhancedRLEnvironmentConfig,
  enableHybridApproach: true,
  enableMarketRegimeDetection: true,
  enableDynamicPositionSizing: true,
  enableSyntheticData: false, // Focus on real data for hybrid approach

  regimeThresholds: {
    volatilityThreshold: 0.025,
    trendStrengthThreshold: 0.0015,
    momentumThreshold: 0.007,
  },

  riskManagement: {
    maxDrawdownThreshold: 0.18,
    positionSizeMultiplier: 0.9,
    volatilityScaling: true,
    regimeBasedPositionSizing: true,
  },
};

/**
 * Learning-Focused Configuration - Emphasize synthetic data for training
 */
export const LearningFocusedConfig: RLEnvironmentConfig = {
  ...EnhancedRLEnvironmentConfig,
  enableSyntheticData: true,
  enableMarketRegimeDetection: true,
  enableHybridApproach: false, // Focus on pure RL learning
  enableDynamicPositionSizing: true,

  syntheticDataConfig: {
    enableTrendingMarkets: true,
    enableRangingMarkets: true,
    enableVolatileMarkets: true,
    noiseLevel: 0.015, // Higher noise for more realistic learning
    trendStrength: 0.003, // Stronger trends for easier learning
  },

  regimeThresholds: {
    volatilityThreshold: 0.035,
    trendStrengthThreshold: 0.0008,
    momentumThreshold: 0.004,
  },
};

/**
 * Configuration presets for different market conditions
 */
export const MarketConditionConfigs = {
  // Bull market configuration
  bullMarket: {
    ...EnhancedRLEnvironmentConfig,
    regimeThresholds: {
      volatilityThreshold: 0.04,
      trendStrengthThreshold: 0.0005, // Lower threshold to catch trends early
      momentumThreshold: 0.003,
    },
    riskManagement: {
      maxDrawdownThreshold: 0.25,
      positionSizeMultiplier: 1.1, // Slightly more aggressive
      volatilityScaling: false, // Don't scale down in bull markets
      regimeBasedPositionSizing: true,
    },
  },

  // Bear market configuration
  bearMarket: {
    ...EnhancedRLEnvironmentConfig,
    regimeThresholds: {
      volatilityThreshold: 0.02, // Lower threshold for volatile conditions
      trendStrengthThreshold: 0.002, // Higher threshold for stronger trends
      momentumThreshold: 0.008,
    },
    riskManagement: {
      maxDrawdownThreshold: 0.15, // More conservative
      positionSizeMultiplier: 0.6, // Much more conservative
      volatilityScaling: true,
      regimeBasedPositionSizing: true,
    },
  },

  // Sideways market configuration
  sidewaysMarket: {
    ...EnhancedRLEnvironmentConfig,
    regimeThresholds: {
      volatilityThreshold: 0.015, // Lower volatility threshold
      trendStrengthThreshold: 0.003, // Higher trend strength requirement
      momentumThreshold: 0.01,
    },
    riskManagement: {
      maxDrawdownThreshold: 0.12, // Very conservative
      positionSizeMultiplier: 0.5, // Small positions
      volatilityScaling: true,
      regimeBasedPositionSizing: true,
    },
  },
};

/**
 * Utility function to create custom configuration
 */
export function createCustomRLConfig(
  overrides: Partial<RLEnvironmentConfig>
): RLEnvironmentConfig {
  return {
    ...EnhancedRLEnvironmentConfig,
    ...overrides,
  };
}

/**
 * Utility function to validate configuration
 */
export function validateRLConfig(config: RLEnvironmentConfig): string[] {
  const errors: string[] = [];

  // Validate basic parameters
  if (config.initialCapital <= 0) {
    errors.push("Initial capital must be positive");
  }
  if (config.maxPositionSize <= 0 || config.maxPositionSize > 1) {
    errors.push("Max position size must be between 0 and 1");
  }
  if (
    config.minPositionSize < 0 ||
    config.minPositionSize > config.maxPositionSize
  ) {
    errors.push("Min position size must be between 0 and max position size");
  }

  // Validate regime thresholds
  if (config.regimeThresholds.volatilityThreshold <= 0) {
    errors.push("Volatility threshold must be positive");
  }
  if (config.regimeThresholds.trendStrengthThreshold <= 0) {
    errors.push("Trend strength threshold must be positive");
  }
  if (config.regimeThresholds.momentumThreshold <= 0) {
    errors.push("Momentum threshold must be positive");
  }

  // Validate risk management
  if (
    config.riskManagement.maxDrawdownThreshold <= 0 ||
    config.riskManagement.maxDrawdownThreshold > 1
  ) {
    errors.push("Max drawdown threshold must be between 0 and 1");
  }
  if (config.riskManagement.positionSizeMultiplier <= 0) {
    errors.push("Position size multiplier must be positive");
  }

  // Validate synthetic data config
  if (config.syntheticDataConfig.noiseLevel < 0) {
    errors.push("Noise level must be non-negative");
  }
  if (config.syntheticDataConfig.trendStrength < 0) {
    errors.push("Trend strength must be non-negative");
  }

  return errors;
}
