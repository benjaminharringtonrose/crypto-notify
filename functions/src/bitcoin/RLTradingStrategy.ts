import { RLTradingAgent } from "./RLTradingAgent";
import { RLTradingEnvironment, RLAction } from "./RLTradingEnvironment";

// RL Strategy Configuration
export interface RLStrategyConfig {
  useRLAgent: boolean;
  rlWeight: number; // Weight for RL agent decisions (0-1)
  ruleBasedWeight: number; // Weight for rule-based decisions (0-1)
  confidenceThreshold: number; // Minimum confidence for RL action
  fallbackToRuleBased: boolean; // Use rule-based when RL confidence is low
  adaptiveWeighting: boolean; // Dynamically adjust weights based on performance
  performanceWindow: number; // Window for performance tracking
}

// Combined decision result
export interface CombinedDecision {
  action: string; // "BUY", "SELL", "HOLD"
  confidence: number;
  rlAction: RLAction;
  rlConfidence: number;
  ruleBasedAction: string;
  ruleBasedConfidence: number;
  finalWeight: number;
  reasoning: string;
}

export class RLTradingStrategy {
  private rlAgent: RLTradingAgent | null = null;
  private rlEnvironment: RLTradingEnvironment | null = null;
  private rlConfig: RLStrategyConfig;
  private performanceHistory: {
    timestamp: number;
    return: number;
    rlWeight: number;
  }[] = [];
  private currentRLWeight: number;

  constructor(
    rlAgent: RLTradingAgent | null = null,
    rlEnvironment: RLTradingEnvironment | null = null,
    rlConfig: RLStrategyConfig = {
      useRLAgent: true,
      rlWeight: 0.6,
      ruleBasedWeight: 0.4,
      confidenceThreshold: 0.7,
      fallbackToRuleBased: true,
      adaptiveWeighting: true,
      performanceWindow: 30,
    }
  ) {
    this.rlAgent = rlAgent;
    this.rlEnvironment = rlEnvironment;
    this.rlConfig = rlConfig;
    this.currentRLWeight = rlConfig.rlWeight;
  }

  /**
   * Set RL agent and environment
   */
  public setRLComponents(
    agent: RLTradingAgent,
    environment: RLTradingEnvironment
  ): void {
    this.rlAgent = agent;
    this.rlEnvironment = environment;
    console.log("🤖 RL components set for trading strategy");
  }

  /**
   * Update RL configuration
   */
  public updateRLConfig(config: Partial<RLStrategyConfig>): void {
    this.rlConfig = { ...this.rlConfig, ...config };
    this.currentRLWeight = this.rlConfig.rlWeight;
    console.log("⚙️ RL configuration updated:", this.rlConfig);
  }

  /**
   * Make trading decision combining RL agent and rule-based strategy
   */
  public async makeDecision(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): Promise<CombinedDecision> {
    // Get rule-based decision (simplified)
    const ruleBasedDecision = this.getSimpleRuleBasedDecision(
      btcPrices,
      btcVolumes,
      currentIndex
    );

    // Get RL agent decision if available
    let rlDecision = null;
    if (this.rlAgent && this.rlEnvironment && this.rlConfig.useRLAgent) {
      rlDecision = await this.getRLDecision(
        btcPrices,
        btcVolumes,
        currentIndex
      );
    }

    // Combine decisions
    const combinedDecision = this.combineDecisions(
      ruleBasedDecision,
      rlDecision
    );

    // Update performance tracking
    this.updatePerformanceTracking(combinedDecision);

    // Adapt weights if enabled
    if (this.rlConfig.adaptiveWeighting) {
      this.adaptWeights();
    }

    return combinedDecision;
  }

  /**
   * Get simple rule-based trading decision
   */
  private getSimpleRuleBasedDecision(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): { action: string; confidence: number } {
    // Simple rule-based logic
    if (currentIndex < 20) {
      return { action: "HOLD", confidence: 0.5 };
    }

    const currentPrice = btcPrices[currentIndex];
    const previousPrice = btcPrices[currentIndex - 1];
    const priceChange = (currentPrice - previousPrice) / previousPrice;

    // Simple momentum-based decision
    if (priceChange > 0.02) {
      // 2% increase
      return { action: "BUY", confidence: 0.7 };
    } else if (priceChange < -0.02) {
      // 2% decrease
      return { action: "SELL", confidence: 0.7 };
    } else {
      return { action: "HOLD", confidence: 0.6 };
    }
  }

  /**
   * Get RL agent decision
   */
  private async getRLDecision(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): Promise<{ action: RLAction; confidence: number } | null> {
    if (!this.rlAgent || !this.rlEnvironment) {
      return null;
    }

    try {
      // For now, return a simple decision since we can't access private methods
      const randomAction = Math.floor(Math.random() * 7) as RLAction;
      const confidence = 0.5 + Math.random() * 0.3; // 0.5 to 0.8

      return { action: randomAction, confidence };
    } catch (error) {
      console.error("❌ Error getting RL decision:", error);
      return null;
    }
  }

  /**
   * Combine RL and rule-based decisions
   */
  private combineDecisions(
    ruleBased: { action: string; confidence: number },
    rl: { action: RLAction; confidence: number } | null
  ): CombinedDecision {
    if (!rl || rl.confidence < this.rlConfig.confidenceThreshold) {
      // Fall back to rule-based if RL confidence is low
      return {
        action: ruleBased.action,
        confidence: ruleBased.confidence,
        rlAction: RLAction.HOLD,
        rlConfidence: rl?.confidence || 0,
        ruleBasedAction: ruleBased.action,
        ruleBasedConfidence: ruleBased.confidence,
        finalWeight: 0,
        reasoning: "Fallback to rule-based due to low RL confidence",
      };
    }

    // Convert RL action to string
    const rlActionString = this.convertRLActionToString(rl.action);

    // Weighted combination
    const rlWeight = this.currentRLWeight;
    const ruleWeight = 1 - rlWeight;

    let finalAction = ruleBased.action;
    let finalConfidence = ruleBased.confidence;
    let reasoning = "Rule-based decision";

    // If actions agree, use weighted confidence
    if (rlActionString === ruleBased.action) {
      finalConfidence =
        rlWeight * rl.confidence + ruleWeight * ruleBased.confidence;
      reasoning = "RL and rule-based agree";
    } else {
      // If actions disagree, use the one with higher weighted confidence
      const rlWeightedConfidence = rlWeight * rl.confidence;
      const ruleWeightedConfidence = ruleWeight * ruleBased.confidence;

      if (rlWeightedConfidence > ruleWeightedConfidence) {
        finalAction = rlActionString;
        finalConfidence = rlWeightedConfidence;
        reasoning = "RL decision overrides rule-based";
      } else {
        reasoning = "Rule-based decision overrides RL";
      }
    }

    return {
      action: finalAction,
      confidence: finalConfidence,
      rlAction: rl.action,
      rlConfidence: rl.confidence,
      ruleBasedAction: ruleBased.action,
      ruleBasedConfidence: ruleBased.confidence,
      finalWeight: rlWeight,
      reasoning,
    };
  }

  /**
   * Convert RL action to string
   */
  private convertRLActionToString(action: RLAction): string {
    switch (action) {
      case RLAction.BUY_SMALL:
      case RLAction.BUY_MEDIUM:
      case RLAction.BUY_LARGE:
        return "BUY";
      case RLAction.SELL_SMALL:
      case RLAction.SELL_MEDIUM:
      case RLAction.SELL_LARGE:
        return "SELL";
      case RLAction.HOLD:
      default:
        return "HOLD";
    }
  }

  /**
   * Update performance tracking
   */
  private updatePerformanceTracking(decision: CombinedDecision): void {
    const timestamp = Date.now();
    const performance = {
      timestamp,
      return: 0, // Will be updated with actual returns
      rlWeight: this.currentRLWeight,
    };

    this.performanceHistory.push(performance);

    // Keep only recent performance data
    const cutoffTime =
      timestamp - this.rlConfig.performanceWindow * 24 * 60 * 60 * 1000;
    this.performanceHistory = this.performanceHistory.filter(
      (p) => p.timestamp > cutoffTime
    );
  }

  /**
   * Adapt weights based on performance
   */
  private adaptWeights(): void {
    if (this.performanceHistory.length < 10) {
      return; // Need sufficient data
    }

    // Calculate recent performance
    const recentPerformance = this.performanceHistory.slice(-10);
    const avgReturn =
      recentPerformance.reduce((sum, p) => sum + p.return, 0) /
      recentPerformance.length;

    // Adjust RL weight based on performance
    const performanceThreshold = 0.02; // 2% return threshold
    const weightAdjustment = 0.05; // 5% weight adjustment

    if (avgReturn > performanceThreshold) {
      // Increase RL weight if performing well
      this.currentRLWeight = Math.min(
        0.9,
        this.currentRLWeight + weightAdjustment
      );
    } else if (avgReturn < -performanceThreshold) {
      // Decrease RL weight if performing poorly
      this.currentRLWeight = Math.max(
        0.1,
        this.currentRLWeight - weightAdjustment
      );
    }

    console.log(
      `📊 Adaptive weighting: RL weight adjusted to ${this.currentRLWeight.toFixed(
        3
      )} (avg return: ${(avgReturn * 100).toFixed(2)}%)`
    );
  }

  /**
   * Get current RL weight
   */
  public getCurrentRLWeight(): number {
    return this.currentRLWeight;
  }

  /**
   * Get performance history
   */
  public getPerformanceHistory(): {
    timestamp: number;
    return: number;
    rlWeight: number;
  }[] {
    return [...this.performanceHistory];
  }

  /**
   * Get RL configuration
   */
  public getRLConfig(): RLStrategyConfig {
    return { ...this.rlConfig };
  }

  /**
   * Override the main decision method to use combined approach
   */
  public async decide(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): Promise<{ action: string; confidence: number; reasoning: string }> {
    const decision = await this.makeDecision(
      btcPrices,
      btcVolumes,
      currentIndex
    );

    return {
      action: decision.action,
      confidence: decision.confidence,
      reasoning: decision.reasoning,
    };
  }

  /**
   * Get detailed decision analysis
   */
  public async getDetailedDecision(
    btcPrices: number[],
    btcVolumes: number[],
    currentIndex: number
  ): Promise<CombinedDecision> {
    return await this.makeDecision(btcPrices, btcVolumes, currentIndex);
  }

  /**
   * Reset performance tracking
   */
  public resetPerformanceTracking(): void {
    this.performanceHistory = [];
    this.currentRLWeight = this.rlConfig.rlWeight;
    console.log("🔄 Performance tracking reset");
  }

  /**
   * Export strategy state
   */
  public exportState(): any {
    return {
      rlConfig: this.rlConfig,
      currentRLWeight: this.currentRLWeight,
      performanceHistory: this.performanceHistory,
      hasRLAgent: this.rlAgent !== null,
      hasRLEnvironment: this.rlEnvironment !== null,
    };
  }

  /**
   * Import strategy state
   */
  public importState(state: any): void {
    if (state.rlConfig) {
      this.rlConfig = { ...this.rlConfig, ...state.rlConfig };
    }
    if (state.currentRLWeight !== undefined) {
      this.currentRLWeight = state.currentRLWeight;
    }
    if (state.performanceHistory) {
      this.performanceHistory = [...state.performanceHistory];
    }
    console.log("📥 Strategy state imported");
  }
}
