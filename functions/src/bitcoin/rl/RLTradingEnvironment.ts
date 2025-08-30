import FeatureCalculator from "../shared/FeatureCalculator";

// Enhanced RL Environment Configuration
export interface RLEnvironmentConfig {
  initialCapital: number;
  commission: number;
  slippage: number;
  timesteps: number;
  maxPositionSize: number;
  minPositionSize: number;
  rewardScaling: number;
  riskFreeRate: number;
  transactionCostPenalty: number;
  holdingPenalty: number;
  volatilityPenalty: number;
  // New configuration parameters
  enableMarketRegimeDetection: boolean;
  enableSyntheticData: boolean;
  enableHybridApproach: boolean;
  enableDynamicPositionSizing: boolean;
  regimeThresholds: {
    volatilityThreshold: number;
    trendStrengthThreshold: number;
    momentumThreshold: number;
  };
  riskManagement: {
    maxDrawdownThreshold: number;
    positionSizeMultiplier: number;
    volatilityScaling: boolean;
    regimeBasedPositionSizing: boolean;
  };
  syntheticDataConfig: {
    enableTrendingMarkets: boolean;
    enableRangingMarkets: boolean;
    enableVolatileMarkets: boolean;
    noiseLevel: number;
    trendStrength: number;
  };
}

// RL Agent Configuration
export interface RLAgentConfig {
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFrequency: number;
  hiddenLayers: number[];
  activationFunction: string;
  optimizer: string;
  lossFunction: string;
  gradientClipping: number;
  experienceReplay: boolean;
  prioritizedReplay: boolean;
  doubleDQN: boolean;
  duelingDQN: boolean;
}

// Enhanced State representation for the RL agent
export interface RLState {
  features: number[]; // Technical indicators and market features
  position: number; // Current position (-1 to 1, where 0 = no position)
  capital: number; // Available capital
  portfolioValue: number; // Total portfolio value
  priceHistory: number[]; // Recent price history
  volumeHistory: number[]; // Recent volume history
  timeStep: number; // Current time step
  marketRegime: string; // Current market regime (trending, ranging, volatile, bearish, bullish)
  volatility: number; // Current market volatility
  momentum: number; // Current price momentum
  trendStrength: number; // Current trend strength
  // New state variables
  regimeConfidence: number; // Confidence in regime classification (0-1)
  riskScore: number; // Current risk score (0-1)
  optimalPositionSize: number; // Optimal position size based on risk management
  marketCondition: string; // Favorable, neutral, unfavorable
  syntheticDataFlag: boolean; // Whether current data is synthetic
  ruleBasedSignal: number; // Rule-based trading signal (-1 to 1)
  hybridWeight: number; // Weight for hybrid approach (0-1)
}

// Enhanced Action space for the RL agent
export enum RLAction {
  HOLD = 0,
  HOLD_CASH = 1, // Explicitly hold cash (no position)
  BUY_SMALL = 2,
  BUY_MEDIUM = 3,
  BUY_LARGE = 4,
  SELL_SMALL = 5,
  SELL_MEDIUM = 6,
  SELL_LARGE = 7,
  // New actions for hybrid approach
  RULE_BASED_BUY = 8,
  RULE_BASED_SELL = 9,
  HYBRID_BUY = 10,
  HYBRID_SELL = 11,
}

// Enhanced Reward structure for the RL agent
export interface RLReward {
  total: number;
  components: {
    returns: number;
    profitIncentive: number;
    lossPenalty: number;
    stabilityBonus: number;
    survivalIncentive: number;
    regimeIncentive: number;
    riskBonus: number;
    transactionCosts: number;
    drawdownPenalty: number;
  };
}

// Market Regime Classification
export interface MarketRegime {
  type: string; // trending, ranging, volatile, bearish, bullish
  confidence: number; // 0-1 confidence in classification
  characteristics: {
    volatility: number;
    trendStrength: number;
    momentum: number;
    volumeProfile: string;
  };
  tradingConditions: {
    isFavorable: boolean;
    recommendedPositionSize: number;
    riskLevel: string; // low, medium, high
  };
}

// Enhanced Episode result tracking
export interface RLEpisodeResult {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  averageHoldingPeriod: number;
  finalPortfolioValue: number;
  actions: RLAction[];
  rewards: RLReward[];
  states: RLState[];
  // New metrics
  regimePerformance: { [regime: string]: number }; // Performance by regime
  hybridPerformance: number; // Performance of hybrid decisions
  riskManagementScore: number; // Risk management effectiveness
  syntheticDataPerformance: number; // Performance on synthetic data
  favorableConditionTrades: number; // Trades in favorable conditions
}

export class RLTradingEnvironment {
  private config: RLEnvironmentConfig;
  private featureCalculator: FeatureCalculator;
  private prices: number[];
  private volumes: number[];
  private currentIndex!: number;
  private initialCapital!: number;
  private currentCapital!: number;
  private currentPosition!: number;
  private entryPrice!: number;
  private peakValue!: number;
  private episodeResults!: RLEpisodeResult;
  private actionHistory!: RLAction[];
  private rewardHistory!: RLReward[];
  private stateHistory!: RLState[];
  private regimeHistory!: MarketRegime[];
  private syntheticDataIndices!: Set<number>;

  constructor(
    prices: number[],
    volumes: number[],
    config: RLEnvironmentConfig
  ) {
    this.config = config;
    this.prices = prices;
    this.volumes = volumes;
    this.featureCalculator = new FeatureCalculator();

    // Generate synthetic data if enabled
    if (this.config.enableSyntheticData) {
      this.generateSyntheticData();
    }

    this.reset();
  }

  /**
   * Generate synthetic data for easier learning scenarios
   */
  private generateSyntheticData(): void {
    this.syntheticDataIndices = new Set();
    const startIndex = Math.floor(this.prices.length * 0.7);

    for (let i = startIndex; i < this.prices.length; i++) {
      this.syntheticDataIndices.add(i);

      if (
        this.config.syntheticDataConfig.enableTrendingMarkets &&
        i % 3 === 0
      ) {
        // Create trending market
        const trend = this.config.syntheticDataConfig.trendStrength;
        const noise =
          (Math.random() - 0.5) * this.config.syntheticDataConfig.noiseLevel;
        this.prices[i] = this.prices[i - 1] * (1 + trend + noise);
      } else if (
        this.config.syntheticDataConfig.enableRangingMarkets &&
        i % 3 === 1
      ) {
        // Create ranging market
        const range = 0.02; // 2% range
        const noise =
          (Math.random() - 0.5) * this.config.syntheticDataConfig.noiseLevel;
        this.prices[i] =
          this.prices[i - 1] * (1 + (Math.random() - 0.5) * range + noise);
      } else if (
        this.config.syntheticDataConfig.enableVolatileMarkets &&
        i % 3 === 2
      ) {
        // Create volatile market
        const volatility = 0.05; // 5% volatility
        const noise = (Math.random() - 0.5) * volatility;
        this.prices[i] = this.prices[i - 1] * (1 + noise);
      }

      // Ensure price doesn't go negative
      this.prices[i] = Math.max(this.prices[i], 0.01);
    }
  }

  /**
   * Enhanced market regime detection
   */
  private detectMarketRegime(): MarketRegime {
    const volatility = this.calculateVolatility();
    const trendStrength = this.calculateTrendStrength();
    const momentum = this.calculateMomentum();
    const volumeProfile = this.analyzeVolumeProfile();

    let regimeType = "ranging";
    let confidence = 0.5;
    let isFavorable = false;
    let recommendedPositionSize = this.config.maxPositionSize * 0.5;
    let riskLevel = "medium";

    // Enhanced regime classification
    if (volatility > this.config.regimeThresholds.volatilityThreshold) {
      regimeType = "volatile";
      confidence = Math.min(0.9, volatility / 0.1);
      isFavorable = false;
      recommendedPositionSize = this.config.maxPositionSize * 0.2;
      riskLevel = "high";
    } else if (
      Math.abs(trendStrength) >
      this.config.regimeThresholds.trendStrengthThreshold
    ) {
      if (
        trendStrength > 0 &&
        momentum > this.config.regimeThresholds.momentumThreshold
      ) {
        regimeType = "bullish";
        confidence = Math.min(0.9, (trendStrength + momentum) / 0.1);
        isFavorable = true;
        recommendedPositionSize = this.config.maxPositionSize * 0.8;
        riskLevel = "low";
      } else if (
        trendStrength < 0 &&
        momentum < -this.config.regimeThresholds.momentumThreshold
      ) {
        regimeType = "bearish";
        confidence = Math.min(
          0.9,
          (Math.abs(trendStrength) + Math.abs(momentum)) / 0.1
        );
        isFavorable = false;
        recommendedPositionSize = this.config.maxPositionSize * 0.1;
        riskLevel = "high";
      } else {
        regimeType = "trending";
        confidence = Math.min(0.8, Math.abs(trendStrength) / 0.05);
        isFavorable = Math.abs(momentum) > 0.01;
        recommendedPositionSize = this.config.maxPositionSize * 0.6;
        riskLevel = "medium";
      }
    } else {
      regimeType = "ranging";
      confidence = 0.7;
      isFavorable = volatility < 0.02; // Low volatility is favorable for ranging
      recommendedPositionSize = this.config.maxPositionSize * 0.4;
      riskLevel = "low";
    }

    return {
      type: regimeType,
      confidence,
      characteristics: {
        volatility,
        trendStrength,
        momentum,
        volumeProfile,
      },
      tradingConditions: {
        isFavorable,
        recommendedPositionSize,
        riskLevel,
      },
    };
  }

  /**
   * Analyze volume profile for regime detection
   */
  private analyzeVolumeProfile(): string {
    const lookback = Math.min(20, this.currentIndex);
    const recentVolumes = this.volumes.slice(
      this.currentIndex - lookback,
      this.currentIndex + 1
    );
    const avgVolume =
      recentVolumes.reduce((sum, vol) => sum + vol, 0) / recentVolumes.length;
    const currentVolume = this.volumes[this.currentIndex];

    if (currentVolume > avgVolume * 1.5) return "high";
    if (currentVolume < avgVolume * 0.5) return "low";
    return "normal";
  }

  /**
   * Calculate optimal position size based on risk management
   */
  private calculateOptimalPositionSize(): number {
    if (!this.config.enableDynamicPositionSizing) {
      return this.config.maxPositionSize;
    }

    const regime = this.regimeHistory[this.regimeHistory.length - 1];
    const volatility = this.calculateVolatility();
    const riskScore = this.calculateRiskScore();

    let positionSize = this.config.maxPositionSize;

    // Regime-based position sizing
    if (this.config.riskManagement.regimeBasedPositionSizing) {
      positionSize = regime.tradingConditions.recommendedPositionSize;
    }

    // Volatility scaling
    if (this.config.riskManagement.volatilityScaling) {
      const volatilityFactor = Math.max(0.1, 1 - volatility * 10);
      positionSize *= volatilityFactor;
    }

    // Risk score adjustment
    const riskFactor = Math.max(0.1, 1 - riskScore);
    positionSize *= riskFactor;

    // Ensure within bounds
    return Math.max(
      this.config.minPositionSize,
      Math.min(this.config.maxPositionSize, positionSize)
    );
  }

  /**
   * Calculate comprehensive risk score
   */
  private calculateRiskScore(): number {
    const volatility = this.calculateVolatility();
    const drawdown = this.calculateCurrentDrawdown();
    const positionRisk = Math.abs(this.currentPosition);
    const regime = this.regimeHistory[this.regimeHistory.length - 1];

    // Combine risk factors
    const volatilityRisk = Math.min(1, volatility * 20);
    const drawdownRisk = Math.min(1, drawdown * 2);
    const positionRiskScore = positionRisk;
    const regimeRisk =
      regime.tradingConditions.riskLevel === "high"
        ? 0.8
        : regime.tradingConditions.riskLevel === "medium"
        ? 0.5
        : 0.2;

    return (volatilityRisk + drawdownRisk + positionRiskScore + regimeRisk) / 4;
  }

  /**
   * Calculate current drawdown
   */
  private calculateCurrentDrawdown(): number {
    const currentValue = this.getPortfolioValue();
    this.peakValue = Math.max(this.peakValue, currentValue);

    if (this.peakValue <= 0) return 0;

    return (this.peakValue - currentValue) / this.peakValue;
  }

  /**
   * Generate rule-based trading signal
   */
  private generateRuleBasedSignal(): number {
    const regime = this.regimeHistory[this.regimeHistory.length - 1];
    const momentum = this.calculateMomentum();
    const trendStrength = this.calculateTrendStrength();
    const volatility = this.calculateVolatility();

    let signal = 0;

    // Simple rule-based strategy
    if (regime.type === "bullish" && momentum > 0.01 && trendStrength > 0.001) {
      signal = 0.8; // Strong buy
    } else if (
      regime.type === "bearish" &&
      momentum < -0.01 &&
      trendStrength < -0.001
    ) {
      signal = -0.8; // Strong sell
    } else if (regime.type === "trending" && Math.abs(momentum) > 0.005) {
      signal = momentum > 0 ? 0.5 : -0.5; // Moderate buy/sell
    } else if (regime.type === "ranging" && volatility < 0.02) {
      signal = 0.2; // Small buy for ranging markets
    } else if (regime.type === "volatile") {
      signal = 0; // No position in volatile markets
    }

    return signal;
  }

  /**
   * Calculate hybrid weight based on regime confidence and performance
   */
  private calculateHybridWeight(): number {
    const regime = this.regimeHistory[this.regimeHistory.length - 1];
    const confidence = regime.confidence;

    // Higher confidence in regime = higher weight for rule-based
    return Math.min(0.8, confidence * 0.8);
  }

  /**
   * Reset the environment for a new episode
   */
  public reset(): RLState {
    this.currentIndex = this.config.timesteps;
    this.initialCapital = this.config.initialCapital;
    this.currentCapital = this.config.initialCapital;
    this.currentPosition = 0;
    this.entryPrice = 0;
    this.peakValue = this.config.initialCapital;
    this.actionHistory = [];
    this.rewardHistory = [];
    this.stateHistory = [];
    this.regimeHistory = [];

    // Initialize episode results
    this.episodeResults = {
      totalReturn: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      totalTrades: 0,
      averageHoldingPeriod: 0,
      finalPortfolioValue: this.config.initialCapital,
      actions: [],
      rewards: [],
      states: [],
      regimePerformance: {},
      hybridPerformance: 0,
      riskManagementScore: 0,
      syntheticDataPerformance: 0,
      favorableConditionTrades: 0,
    };

    return this.getCurrentState();
  }

  /**
   * Execute an action and return the new state, reward, and done flag
   */
  public step(action: RLAction): {
    state: RLState;
    reward: RLReward;
    done: boolean;
    info: any;
  } {
    const currentPrice = this.prices[this.currentIndex];
    const previousPortfolioValue = this.getPortfolioValue();
    const previousPosition = this.currentPosition;

    // Execute the action with enhanced logic
    this.executeAction(action, currentPrice);

    // Move to next time step
    this.currentIndex++;

    // Calculate reward with enhanced components
    const reward = this.calculateReward(
      previousPortfolioValue,
      previousPosition
    );

    // Check if episode is done
    const done = this.currentIndex >= this.prices.length - 1;

    // Enhanced stop-loss with regime consideration
    const currentValue = this.getPortfolioValue();
    const regime = this.regimeHistory[this.regimeHistory.length - 1];
    const stopLossThreshold =
      regime?.tradingConditions.riskLevel === "high" ? 0.15 : 0.2;
    const stopLossTriggered =
      currentValue < this.config.initialCapital * stopLossThreshold;
    const finalDone = done || stopLossTriggered;

    // Get new state
    const state = this.getCurrentState();

    // Update episode tracking
    this.actionHistory.push(action);
    this.rewardHistory.push(reward);
    this.stateHistory.push(state);

    // Update episode results if done
    if (finalDone) {
      this.calculateEpisodeResults();
    }

    // Log progress every 100 steps
    if (this.currentIndex % 100 === 0) {
      console.log(
        `        📊 Step ${this.currentIndex}: Action=${action}, Regime=${
          regime?.type
        }, Price=${currentPrice.toFixed(
          2
        )}, Portfolio=${this.getPortfolioValue().toFixed(2)}, Done=${finalDone}`
      );
    }

    return {
      state,
      reward,
      done: finalDone,
      info: {
        portfolioValue: this.getPortfolioValue(),
        position: this.currentPosition,
        currentPrice,
        timeStep: this.currentIndex,
        stopLossTriggered,
        regime: regime?.type,
        riskScore: this.calculateRiskScore(),
      },
    };
  }

  /**
   * Enhanced action execution with hybrid approach
   */
  private executeAction(action: RLAction, currentPrice: number): void {
    const slippage = this.config.slippage;
    const commission = this.config.commission;
    const optimalPositionSize = this.calculateOptimalPositionSize();

    switch (action) {
      case RLAction.HOLD:
        // No action taken
        break;

      case RLAction.HOLD_CASH:
        // Explicitly hold cash - close any existing position
        if (this.currentPosition !== 0) {
          this.executeSell(1.0, currentPrice, slippage, commission);
        }
        break;

      case RLAction.BUY_SMALL:
        this.executeBuy(
          0.25 * optimalPositionSize,
          currentPrice,
          slippage,
          commission
        );
        break;

      case RLAction.BUY_MEDIUM:
        this.executeBuy(
          0.5 * optimalPositionSize,
          currentPrice,
          slippage,
          commission
        );
        break;

      case RLAction.BUY_LARGE:
        this.executeBuy(
          optimalPositionSize,
          currentPrice,
          slippage,
          commission
        );
        break;

      case RLAction.SELL_SMALL:
        this.executeSell(0.25, currentPrice, slippage, commission);
        break;

      case RLAction.SELL_MEDIUM:
        this.executeSell(0.5, currentPrice, slippage, commission);
        break;

      case RLAction.SELL_LARGE:
        this.executeSell(1.0, currentPrice, slippage, commission);
        break;

      case RLAction.RULE_BASED_BUY:
        // Execute rule-based buy signal
        const ruleSignal = this.generateRuleBasedSignal();
        if (ruleSignal > 0) {
          this.executeBuy(
            ruleSignal * optimalPositionSize,
            currentPrice,
            slippage,
            commission
          );
        }
        break;

      case RLAction.RULE_BASED_SELL:
        // Execute rule-based sell signal
        const ruleSellSignal = this.generateRuleBasedSignal();
        if (ruleSellSignal < 0) {
          this.executeSell(
            Math.abs(ruleSellSignal),
            currentPrice,
            slippage,
            commission
          );
        }
        break;

      case RLAction.HYBRID_BUY:
        // Hybrid approach: combine RL and rule-based
        const hybridWeight = this.calculateHybridWeight();
        const ruleSignalHybrid = this.generateRuleBasedSignal();
        const hybridSize =
          (0.5 + hybridWeight * ruleSignalHybrid) * optimalPositionSize;
        if (hybridSize > 0) {
          this.executeBuy(hybridSize, currentPrice, slippage, commission);
        }
        break;

      case RLAction.HYBRID_SELL:
        // Hybrid approach: combine RL and rule-based
        const hybridSellWeight = this.calculateHybridWeight();
        const ruleSellSignalHybrid = this.generateRuleBasedSignal();
        const hybridSellSize =
          0.5 + hybridSellWeight * Math.abs(ruleSellSignalHybrid);
        if (hybridSellSize > 0) {
          this.executeSell(hybridSellSize, currentPrice, slippage, commission);
        }
        break;
    }
  }

  /**
   * Execute a buy order
   */
  private executeBuy(
    positionSize: number,
    price: number,
    slippage: number,
    commission: number
  ): void {
    if (this.currentPosition >= 1.0) {
      return; // Already fully positioned
    }

    const maxPositionIncrease = 1.0 - this.currentPosition;
    const actualPositionIncrease = Math.min(positionSize, maxPositionIncrease);
    const availableCapital = this.currentCapital * 0.95; // Keep 5% buffer

    if (availableCapital <= 0) {
      return; // No capital available
    }

    const executionPrice = price * (1 + slippage);
    const positionValue = availableCapital * actualPositionIncrease;
    const transactionCost = positionValue * commission;

    this.currentCapital -= positionValue + transactionCost;
    this.currentPosition += actualPositionIncrease;

    if (this.currentPosition > 0 && this.entryPrice === 0) {
      this.entryPrice = executionPrice;
    }

    this.episodeResults.totalTrades++;
  }

  /**
   * Execute a sell order
   */
  private executeSell(
    positionSize: number,
    price: number,
    slippage: number,
    commission: number
  ): void {
    if (this.currentPosition <= 0) {
      return; // No position to sell
    }

    const actualPositionDecrease = Math.min(positionSize, this.currentPosition);
    const positionValue = this.getPortfolioValue() * actualPositionDecrease;
    const transactionCost = positionValue * commission;

    this.currentCapital += positionValue - transactionCost;
    this.currentPosition -= actualPositionDecrease;

    if (this.currentPosition <= 0) {
      this.entryPrice = 0; // Reset entry price
    }

    this.episodeResults.totalTrades++;
  }

  /**
   * Calculate the reward for the current action - ENHANCED VERSION
   */
  private calculateReward(
    previousPortfolioValue: number,
    previousPosition: number
  ): RLReward {
    const currentPortfolioValue = this.getPortfolioValue();

    // Handle division by zero or negative values
    let returns = 0;
    if (previousPortfolioValue > 0) {
      returns =
        (currentPortfolioValue - previousPortfolioValue) /
        previousPortfolioValue;
    } else if (currentPortfolioValue > 0) {
      returns = 1.0; // 100% return if starting from zero
    }

    // Calculate risk-adjusted returns
    const riskFreeReturn = this.config.riskFreeRate / 252; // Daily risk-free rate
    const excessReturn = returns - riskFreeReturn;

    // Calculate transaction cost penalty (much reduced)
    const positionChange = Math.abs(this.currentPosition - previousPosition);
    const transactionCostPenalty =
      positionChange * this.config.transactionCostPenalty;

    // Calculate drawdown penalty (much reduced)
    const drawdownPenalty = this.calculateDrawdownPenalty() * 0.1; // Reduced by 90%

    // IMPROVED REWARD: Simplified and more effective
    // Focus on core profitability with clear signals
    const baseReward = excessReturn * this.config.rewardScaling;

    // Strong profit incentives
    const profitIncentive = returns > 0 ? returns * 20 : 0; // Strong positive reinforcement
    const lossPenalty = returns < 0 ? returns * 10 : 0; // Moderate negative reinforcement

    // Survival and stability bonuses
    const stabilityBonus = this.currentPosition === 0 ? 0.1 : 0; // Small bonus for cash
    const survivalIncentive =
      currentPortfolioValue > this.config.initialCapital * 0.8 ? 0.5 : 0;

    // Regime-based bonuses (simplified)
    const regimeIncentive =
      this.getCurrentState().marketRegime === "bullish" && returns > 0
        ? 1.0
        : 0;

    // Risk management bonuses
    const riskBonus = this.calculateRiskManagementBonus();

    // IMPROVED TOTAL REWARD: Clear, simple, effective
    const totalReward =
      baseReward +
      profitIncentive +
      lossPenalty +
      stabilityBonus +
      survivalIncentive +
      regimeIncentive +
      riskBonus -
      transactionCostPenalty * 0.5 - // Reduced transaction cost penalty
      drawdownPenalty * 0.3; // Reduced drawdown penalty

    return {
      total: totalReward,
      components: {
        returns: baseReward,
        profitIncentive: profitIncentive,
        lossPenalty: lossPenalty,
        stabilityBonus: stabilityBonus,
        survivalIncentive: survivalIncentive,
        regimeIncentive: regimeIncentive,
        riskBonus: riskBonus,
        transactionCosts: -transactionCostPenalty * 0.5,
        drawdownPenalty: -drawdownPenalty * 0.3,
      },
    };
  }

  /**
   * Calculate drawdown penalty
   */
  private calculateDrawdownPenalty(): number {
    const currentValue = this.getPortfolioValue();
    this.peakValue = Math.max(this.peakValue, currentValue);

    // Handle division by zero
    if (this.peakValue <= 0) {
      return 0;
    }

    const drawdown = (this.peakValue - currentValue) / this.peakValue;
    return drawdown * 0.05; // Reduced penalty for drawdowns (was 0.1)
  }

  /**
   * Calculate risk management bonus for good risk management
   */
  private calculateRiskManagementBonus(): number {
    const riskScore = this.calculateRiskScore();
    const optimalPositionSize = this.calculateOptimalPositionSize();
    const actualPositionSize = Math.abs(this.currentPosition);

    let bonus = 0;

    // Bonus for maintaining low risk score
    if (riskScore < 0.3) {
      bonus += 0.05;
    } else if (riskScore < 0.5) {
      bonus += 0.02;
    }

    // Bonus for using appropriate position sizing
    const positionSizeRatio = actualPositionSize / optimalPositionSize;
    if (positionSizeRatio > 0.8 && positionSizeRatio < 1.2) {
      bonus += 0.03; // Bonus for using close to optimal position size
    }

    // Bonus for avoiding excessive drawdown
    const drawdown = this.calculateCurrentDrawdown();
    if (drawdown < 0.1) {
      bonus += 0.04; // Bonus for keeping drawdown low
    }

    return bonus;
  }

  /**
   * Calculate performance by market regime
   */
  private calculateRegimePerformance(): void {
    const regimePerformance: { [regime: string]: number } = {};
    const regimeReturns: { [regime: string]: number[] } = {};

    // Group returns by regime
    for (
      let i = 0;
      i < this.regimeHistory.length && i < this.rewardHistory.length;
      i++
    ) {
      const regime = this.regimeHistory[i];
      const reward = this.rewardHistory[i];

      if (!regimeReturns[regime.type]) {
        regimeReturns[regime.type] = [];
      }
      regimeReturns[regime.type].push(reward.total);
    }

    // Calculate average return for each regime
    for (const [regime, returns] of Object.entries(regimeReturns)) {
      const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
      regimePerformance[regime] = avgReturn;
    }

    this.episodeResults.regimePerformance = regimePerformance;
  }

  /**
   * Calculate hybrid approach performance
   */
  private calculateHybridPerformance(): void {
    let hybridReturns = 0;
    let hybridCount = 0;

    for (
      let i = 0;
      i < this.actionHistory.length && i < this.rewardHistory.length;
      i++
    ) {
      const action = this.actionHistory[i];
      const reward = this.rewardHistory[i];

      if (
        action === RLAction.HYBRID_BUY ||
        action === RLAction.HYBRID_SELL ||
        action === RLAction.RULE_BASED_BUY ||
        action === RLAction.RULE_BASED_SELL
      ) {
        hybridReturns += reward.total;
        hybridCount++;
      }
    }

    this.episodeResults.hybridPerformance =
      hybridCount > 0 ? hybridReturns / hybridCount : 0;
  }

  /**
   * Calculate risk management effectiveness score
   */
  private calculateRiskManagementScore(): void {
    let totalRiskScore = 0;
    let lowRiskCount = 0;
    let totalDrawdown = 0;

    for (const state of this.stateHistory) {
      totalRiskScore += state.riskScore;
      if (state.riskScore < 0.3) {
        lowRiskCount++;
      }
    }

    // Calculate average drawdown
    for (let i = 1; i < this.stateHistory.length; i++) {
      const currentValue = this.stateHistory[i].portfolioValue;
      const previousValue = this.stateHistory[i - 1].portfolioValue;
      if (previousValue > 0) {
        const drawdown = (previousValue - currentValue) / previousValue;
        totalDrawdown += Math.max(0, drawdown);
      }
    }

    const avgRiskScore = totalRiskScore / this.stateHistory.length;
    const riskManagementRatio = lowRiskCount / this.stateHistory.length;
    const avgDrawdown = totalDrawdown / this.stateHistory.length;

    // Combine metrics into overall score
    this.episodeResults.riskManagementScore =
      (1 - avgRiskScore) * 0.4 +
      riskManagementRatio * 0.4 +
      (1 - avgDrawdown) * 0.2;
  }

  /**
   * Calculate performance on synthetic data
   */
  private calculateSyntheticDataPerformance(): void {
    let syntheticReturns = 0;
    let syntheticCount = 0;

    for (
      let i = 0;
      i < this.stateHistory.length && i < this.rewardHistory.length;
      i++
    ) {
      const state = this.stateHistory[i];
      const reward = this.rewardHistory[i];

      if (state.syntheticDataFlag) {
        syntheticReturns += reward.total;
        syntheticCount++;
      }
    }

    this.episodeResults.syntheticDataPerformance =
      syntheticCount > 0 ? syntheticReturns / syntheticCount : 0;
  }

  /**
   * Calculate number of trades in favorable conditions
   */
  private calculateFavorableConditionTrades(): void {
    let favorableTrades = 0;

    for (const state of this.stateHistory) {
      if (
        state.marketCondition === "favorable" &&
        Math.abs(state.position) > 0.1
      ) {
        favorableTrades++;
      }
    }

    this.episodeResults.favorableConditionTrades = favorableTrades;
  }

  /**
   * Calculate current market volatility
   */
  private calculateVolatility(): number {
    const lookback = Math.min(20, this.currentIndex);
    const returns = [];

    for (
      let i = this.currentIndex - lookback + 1;
      i <= this.currentIndex;
      i++
    ) {
      if (i > 0 && this.prices[i - 1] > 0) {
        const return_val =
          (this.prices[i] - this.prices[i - 1]) / this.prices[i - 1];
        returns.push(return_val);
      }
    }

    if (returns.length === 0) return 0;

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance =
      returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
      returns.length;

    // Ensure variance is not negative (shouldn't happen, but safety check)
    return Math.sqrt(Math.max(variance, 0));
  }

  /**
   * Get the current portfolio value
   */
  private getPortfolioValue(): number {
    const currentPrice = this.prices[this.currentIndex];

    // Handle case where no position has been taken (entryPrice = 0)
    if (this.entryPrice <= 0 || this.currentPosition <= 0) {
      return this.currentCapital;
    }

    const positionValue =
      this.currentCapital *
      this.currentPosition *
      (currentPrice / this.entryPrice);

    // Ensure portfolio value is never negative
    const totalValue = this.currentCapital + positionValue;
    return Math.max(totalValue, 0);
  }

  /**
   * Get the current state representation
   */
  private getCurrentState(): RLState {
    const currentPrice = this.prices[this.currentIndex];

    // Calculate technical features
    const features = this.featureCalculator.compute({
      prices: this.prices,
      volumes: this.volumes,
      dayIndex: this.currentIndex,
      currentPrice,
    });

    // Detect market regime
    const regime = this.detectMarketRegime();
    this.regimeHistory.push(regime);

    // Calculate market metrics
    const volatility = this.calculateVolatility();
    const trendStrength = this.calculateTrendStrength();
    const riskScore = this.calculateRiskScore();
    const optimalPositionSize = this.calculateOptimalPositionSize();
    const ruleBasedSignal = this.generateRuleBasedSignal();
    const hybridWeight = this.calculateHybridWeight();

    // Determine market condition
    let marketCondition = "neutral";
    if (regime.tradingConditions.isFavorable) {
      marketCondition = "favorable";
    } else if (regime.tradingConditions.riskLevel === "high") {
      marketCondition = "unfavorable";
    }

    // Check if current data is synthetic
    const syntheticDataFlag =
      this.syntheticDataIndices?.has(this.currentIndex) || false;

    // Get recent price and volume history
    const priceHistory = this.prices.slice(
      Math.max(0, this.currentIndex - this.config.timesteps),
      this.currentIndex + 1
    );
    const volumeHistory = this.volumes.slice(
      Math.max(0, this.currentIndex - this.config.timesteps),
      this.currentIndex + 1
    );

    return {
      features,
      position: this.currentPosition,
      capital: this.currentCapital,
      portfolioValue: this.getPortfolioValue(),
      priceHistory,
      volumeHistory,
      timeStep: this.currentIndex,
      marketRegime: regime.type,
      volatility,
      momentum: this.calculateMomentum(),
      trendStrength,
      regimeConfidence: regime.confidence,
      riskScore,
      optimalPositionSize,
      marketCondition,
      syntheticDataFlag,
      ruleBasedSignal,
      hybridWeight,
    };
  }

  /**
   * Calculate price momentum
   */
  private calculateMomentum(): number {
    const lookback = Math.min(10, this.currentIndex);
    if (lookback === 0) return 0;

    const currentPrice = this.prices[this.currentIndex];
    const pastPrice = this.prices[this.currentIndex - lookback];

    // Handle division by zero
    if (pastPrice <= 0) return 0;

    return (currentPrice - pastPrice) / pastPrice;
  }

  /**
   * Calculate trend strength
   */
  private calculateTrendStrength(): number {
    const lookback = Math.min(20, this.currentIndex);
    if (lookback < 5) return 0;

    const prices = this.prices.slice(
      this.currentIndex - lookback,
      this.currentIndex + 1
    );
    const linearRegression = this.calculateLinearRegression(prices);
    return linearRegression.slope;
  }

  /**
   * Calculate linear regression slope
   */
  private calculateLinearRegression(prices: number[]): {
    slope: number;
    intercept: number;
  } {
    const n = prices.length;
    const x = Array.from({ length: n }, (_, i) => i);

    const sumX = x.reduce((sum, val) => sum + val, 0);
    const sumY = prices.reduce((sum, val) => sum + val, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * prices[i], 0);
    const sumXX = x.reduce((sum, val) => sum + val * val, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return { slope, intercept };
  }

  /**
   * Calculate final episode results
   */
  private calculateEpisodeResults(): void {
    const finalValue = this.getPortfolioValue();
    this.episodeResults.finalPortfolioValue = finalValue;
    this.episodeResults.totalReturn =
      (finalValue - this.initialCapital) / this.initialCapital;

    // Calculate Sharpe ratio
    const returns = this.rewardHistory.map((r) => r.total);
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance =
      returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) /
      returns.length;
    this.episodeResults.sharpeRatio =
      variance > 0 ? meanReturn / Math.sqrt(variance) : 0;

    // Calculate max drawdown
    let peak = this.initialCapital;
    let maxDrawdown = 0;
    const portfolioValues = this.stateHistory.map((s) => s.portfolioValue);

    for (const value of portfolioValues) {
      peak = Math.max(peak, value);
      const drawdown = (peak - value) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    this.episodeResults.maxDrawdown = maxDrawdown;

    // Calculate win rate
    const positiveReturns = returns.filter((r) => r > 0).length;
    this.episodeResults.winRate =
      returns.length > 0 ? positiveReturns / returns.length : 0;

    // Calculate average holding period
    const holdingPeriods = this.calculateHoldingPeriods();
    this.episodeResults.averageHoldingPeriod =
      holdingPeriods.length > 0
        ? holdingPeriods.reduce((sum, period) => sum + period, 0) /
          holdingPeriods.length
        : 0;

    // NEW: Calculate regime performance
    this.calculateRegimePerformance();

    // NEW: Calculate hybrid performance
    this.calculateHybridPerformance();

    // NEW: Calculate risk management score
    this.calculateRiskManagementScore();

    // NEW: Calculate synthetic data performance
    this.calculateSyntheticDataPerformance();

    // NEW: Calculate favorable condition trades
    this.calculateFavorableConditionTrades();

    // Copy arrays
    this.episodeResults.actions = [...this.actionHistory];
    this.episodeResults.rewards = [...this.rewardHistory];
    this.episodeResults.states = [...this.stateHistory];
  }

  /**
   * Calculate holding periods for positions
   */
  private calculateHoldingPeriods(): number[] {
    const holdingPeriods: number[] = [];
    let holdingStart = -1;

    for (let i = 0; i < this.actionHistory.length; i++) {
      const hasPosition = this.stateHistory[i].position > 0;

      if (hasPosition && holdingStart === -1) {
        holdingStart = i;
      } else if (!hasPosition && holdingStart !== -1) {
        holdingPeriods.push(i - holdingStart);
        holdingStart = -1;
      }
    }

    // Add final holding period if still holding
    if (holdingStart !== -1) {
      holdingPeriods.push(this.actionHistory.length - holdingStart);
    }

    return holdingPeriods;
  }

  /**
   * Get episode results
   */
  public getEpisodeResults(): RLEpisodeResult {
    return this.episodeResults;
  }

  /**
   * Get action space size
   */
  public getActionSpaceSize(): number {
    return 12; // Total number of actions including new hybrid actions
  }

  /**
   * Get state space size
   */
  public getStateSpaceSize(): number {
    const state = this.getCurrentState();
    return state.features.length + 17; // Features + additional state variables (increased for new variables)
  }

  /**
   * Get current market data
   */
  public getCurrentMarketData(): {
    price: number;
    volume: number;
    index: number;
  } {
    return {
      price: this.prices[this.currentIndex],
      volume: this.volumes[this.currentIndex],
      index: this.currentIndex,
    };
  }

  /**
   * Get environment configuration
   */
  public getConfig(): RLEnvironmentConfig {
    return this.config;
  }
}
