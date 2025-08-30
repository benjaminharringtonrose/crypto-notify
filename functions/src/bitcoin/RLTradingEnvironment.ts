import FeatureCalculator from "./FeatureCalculator";

// RL Environment Configuration
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
}

// State representation for the RL agent
export interface RLState {
  features: number[]; // Technical indicators and market features
  position: number; // Current position (-1 to 1, where 0 = no position)
  capital: number; // Available capital
  portfolioValue: number; // Total portfolio value
  priceHistory: number[]; // Recent price history
  volumeHistory: number[]; // Recent volume history
  timeStep: number; // Current time step
  marketRegime: string; // Current market regime (trending, ranging, volatile)
  volatility: number; // Current market volatility
  momentum: number; // Current price momentum
  trendStrength: number; // Current trend strength
}

// Action space for the RL agent
export enum RLAction {
  HOLD = 0,
  BUY_SMALL = 1,
  BUY_MEDIUM = 2,
  BUY_LARGE = 3,
  SELL_SMALL = 4,
  SELL_MEDIUM = 5,
  SELL_LARGE = 6,
}

// Reward structure for the RL agent
export interface RLReward {
  total: number;
  components: {
    returns: number;
    transactionCosts: number;
    riskPenalty: number;
    volatilityPenalty: number;
    holdingPenalty: number;
    drawdownPenalty: number;
  };
}

// Episode result tracking
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

  constructor(
    prices: number[],
    volumes: number[],
    config: RLEnvironmentConfig
  ) {
    this.config = config;
    this.prices = prices;
    this.volumes = volumes;
    this.featureCalculator = new FeatureCalculator();
    this.reset();
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

    // Execute the action
    this.executeAction(action, currentPrice);

    // Move to next time step
    this.currentIndex++;

    // Calculate reward
    const reward = this.calculateReward(
      previousPortfolioValue,
      previousPosition
    );

    // Check if episode is done
    const done = this.currentIndex >= this.prices.length - 1;

    // Get new state
    const state = this.getCurrentState();

    // Update episode tracking
    this.actionHistory.push(action);
    this.rewardHistory.push(reward);
    this.stateHistory.push(state);

    // Update episode results if done
    if (done) {
      this.calculateEpisodeResults();
    }

    return {
      state,
      reward,
      done,
      info: {
        portfolioValue: this.getPortfolioValue(),
        position: this.currentPosition,
        currentPrice,
        timeStep: this.currentIndex,
      },
    };
  }

  /**
   * Execute a trading action
   */
  private executeAction(action: RLAction, currentPrice: number): void {
    const slippage = this.config.slippage;
    const commission = this.config.commission;

    switch (action) {
      case RLAction.HOLD:
        // No action taken
        break;

      case RLAction.BUY_SMALL:
        this.executeBuy(0.25, currentPrice, slippage, commission);
        break;

      case RLAction.BUY_MEDIUM:
        this.executeBuy(0.5, currentPrice, slippage, commission);
        break;

      case RLAction.BUY_LARGE:
        this.executeBuy(1.0, currentPrice, slippage, commission);
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
   * Calculate the reward for the current action
   */
  private calculateReward(
    previousPortfolioValue: number,
    previousPosition: number
  ): RLReward {
    const currentPortfolioValue = this.getPortfolioValue();
    const returns =
      (currentPortfolioValue - previousPortfolioValue) / previousPortfolioValue;

    // Calculate risk-adjusted returns
    const riskFreeReturn = this.config.riskFreeRate / 252; // Daily risk-free rate
    const excessReturn = returns - riskFreeReturn;

    // Calculate volatility penalty
    const priceChange =
      (this.prices[this.currentIndex] - this.prices[this.currentIndex - 1]) /
      this.prices[this.currentIndex - 1];
    const volatilityPenalty =
      Math.abs(priceChange) * this.config.volatilityPenalty;

    // Calculate transaction cost penalty
    const positionChange = Math.abs(this.currentPosition - previousPosition);
    const transactionCostPenalty =
      positionChange * this.config.transactionCostPenalty;

    // Calculate holding penalty (encourage active trading)
    const holdingPenalty =
      this.currentPosition > 0 ? this.config.holdingPenalty : 0;

    // Calculate drawdown penalty
    const drawdownPenalty = this.calculateDrawdownPenalty();

    // Calculate risk penalty (penalize excessive risk)
    const riskPenalty = this.calculateRiskPenalty();

    // Combine all components
    const totalReward =
      excessReturn * this.config.rewardScaling -
      transactionCostPenalty -
      volatilityPenalty -
      holdingPenalty -
      drawdownPenalty -
      riskPenalty;

    return {
      total: totalReward,
      components: {
        returns: excessReturn * this.config.rewardScaling,
        transactionCosts: -transactionCostPenalty,
        riskPenalty: -riskPenalty,
        volatilityPenalty: -volatilityPenalty,
        holdingPenalty: -holdingPenalty,
        drawdownPenalty: -drawdownPenalty,
      },
    };
  }

  /**
   * Calculate drawdown penalty
   */
  private calculateDrawdownPenalty(): number {
    const currentValue = this.getPortfolioValue();
    this.peakValue = Math.max(this.peakValue, currentValue);
    const drawdown = (this.peakValue - currentValue) / this.peakValue;
    return drawdown * 0.1; // Penalty for drawdowns
  }

  /**
   * Calculate risk penalty based on position size and volatility
   */
  private calculateRiskPenalty(): number {
    const volatility = this.calculateVolatility();
    const positionRisk = Math.abs(this.currentPosition) * volatility;
    return positionRisk * 0.05; // Penalty for high risk positions
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
      if (i > 0) {
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
    return Math.sqrt(variance);
  }

  /**
   * Get the current portfolio value
   */
  private getPortfolioValue(): number {
    const currentPrice = this.prices[this.currentIndex];
    const positionValue =
      this.currentCapital *
      this.currentPosition *
      (currentPrice / this.entryPrice);
    return this.currentCapital + positionValue;
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

    // Calculate market metrics
    const volatility = this.calculateVolatility();
    const trendStrength = this.calculateTrendStrength();
    const marketRegime = this.determineMarketRegime();

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
      marketRegime,
      volatility,
      momentum: this.calculateMomentum(),
      trendStrength,
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
   * Determine current market regime
   */
  private determineMarketRegime(): string {
    const volatility = this.calculateVolatility();
    const trendStrength = this.calculateTrendStrength();

    if (volatility > 0.05) {
      return "volatile";
    } else if (Math.abs(trendStrength) > 0.001) {
      return "trending";
    } else {
      return "ranging";
    }
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
    return Object.keys(RLAction).length / 2; // Exclude enum reverse mappings
  }

  /**
   * Get state space size
   */
  public getStateSpaceSize(): number {
    const state = this.getCurrentState();
    return state.features.length + 10; // Features + additional state variables
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
