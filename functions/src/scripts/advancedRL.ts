import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import FeatureCalculator from "../bitcoin/FeatureCalculator";

// Initialize Firebase
FirebaseService.getInstance();

// Experience replay buffer
interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

// Advanced RL Environment
class AdvancedRLEnvironment {
  private prices: number[];
  private volumes: number[];
  private currentIndex: number;
  private capital: number;
  private position: number;
  private entryPrice: number;
  private featureCalculator: FeatureCalculator;
  private initialCapital: number;
  private peakValue: number;
  private tradeCount: number;
  private winCount: number;

  constructor(prices: number[], volumes: number[]) {
    this.prices = prices;
    this.volumes = volumes;
    this.currentIndex = 35;
    this.initialCapital = 10000;
    this.capital = this.initialCapital;
    this.position = 0;
    this.entryPrice = 0;
    this.peakValue = this.initialCapital;
    this.tradeCount = 0;
    this.winCount = 0;
    this.featureCalculator = new FeatureCalculator();
  }

  public reset(): number[] {
    this.currentIndex = 35;
    this.capital = this.initialCapital;
    this.position = 0;
    this.entryPrice = 0;
    this.peakValue = this.initialCapital;
    this.tradeCount = 0;
    this.winCount = 0;
    return this.getState();
  }

  public step(action: number): {
    state: number[];
    reward: number;
    done: boolean;
    info: {
      portfolioValue: number;
      position: number;
      tradeCount: number;
      winRate: number;
    };
  } {
    const currentPrice = this.prices[this.currentIndex];
    const previousValue = this.getPortfolioValue();

    // Execute action: 0 = HOLD, 1 = BUY, 2 = SELL
    if (action === 1 && this.position === 0) {
      // Buy
      this.position = 1;
      this.entryPrice = currentPrice;
      this.tradeCount++;
    } else if (action === 2 && this.position === 1) {
      // Sell
      const tradeReturn = (currentPrice - this.entryPrice) / this.entryPrice;
      if (tradeReturn > 0) this.winCount++;

      this.position = 0;
      this.entryPrice = 0;
      this.tradeCount++;
    }

    // Move to next step
    this.currentIndex++;

    // Calculate reward with risk adjustment
    const currentValue = this.getPortfolioValue();
    const rawReward = (currentValue - previousValue) / previousValue;

    // Risk-adjusted reward
    const drawdown = (this.peakValue - currentValue) / this.peakValue;
    const riskPenalty = drawdown * 0.1; // Penalty for drawdowns

    // Transaction cost penalty
    const transactionPenalty = action !== 0 ? 0.001 : 0; // 0.1% transaction cost

    const reward = rawReward - riskPenalty - transactionPenalty;

    // Update peak value
    this.peakValue = Math.max(this.peakValue, currentValue);

    // Check if done
    const done = this.currentIndex >= this.prices.length - 1;

    return {
      state: this.getState(),
      reward,
      done,
      info: {
        portfolioValue: currentValue,
        position: this.position,
        tradeCount: this.tradeCount,
        winRate: this.tradeCount > 0 ? this.winCount / this.tradeCount : 0,
      },
    };
  }

  private getState(): number[] {
    if (this.currentIndex >= this.prices.length) {
      return Array(15).fill(0);
    }

    const features = this.featureCalculator.compute({
      prices: this.prices,
      volumes: this.volumes,
      dayIndex: this.currentIndex,
      currentPrice: this.prices[this.currentIndex],
    });

    // Use first 15 features and add position info
    const stateFeatures = features.slice(0, 15);
    const positionInfo = [
      this.position,
      this.capital / this.initialCapital, // Normalized capital
      this.getPortfolioValue() / this.initialCapital, // Normalized portfolio value
    ];

    return [...stateFeatures, ...positionInfo];
  }

  private getPortfolioValue(): number {
    if (this.position === 0) {
      return this.capital;
    }
    const currentPrice = this.prices[this.currentIndex];
    return this.capital * (currentPrice / this.entryPrice);
  }

  public getCurrentIndex(): number {
    return this.currentIndex;
  }

  public getMetrics(): {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    tradeCount: number;
  } {
    const finalValue = this.getPortfolioValue();
    const totalReturn =
      (finalValue - this.initialCapital) / this.initialCapital;
    const maxDrawdown =
      (this.peakValue - Math.min(this.peakValue, finalValue)) / this.peakValue;
    const winRate = this.tradeCount > 0 ? this.winCount / this.tradeCount : 0;

    // Simple Sharpe ratio approximation
    const sharpeRatio = totalReturn / (maxDrawdown + 0.01); // Add small constant to avoid division by zero

    return {
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      winRate,
      tradeCount: this.tradeCount,
    };
  }
}

// Advanced Q-Learning Agent with Experience Replay
class AdvancedQLAgent {
  private qTable: Map<string, number[]>;
  private experienceBuffer: Experience[];
  private learningRate: number;
  private discountFactor: number;
  private epsilon: number;
  private epsilonDecay: number;
  private epsilonMin: number;
  private batchSize: number;
  private bufferSize: number;

  constructor() {
    this.qTable = new Map();
    this.experienceBuffer = [];
    this.learningRate = 0.1;
    this.discountFactor = 0.95;
    this.epsilon = 1.0;
    this.epsilonDecay = 0.995;
    this.epsilonMin = 0.01;
    this.batchSize = 32;
    this.bufferSize = 10000;
  }

  public chooseAction(state: number[]): number {
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * 3); // Random action
    }
    return this.getGreedyAction(state);
  }

  private getGreedyAction(state: number[]): number {
    const stateKey = this.stateToKey(state);
    const qValues = this.qTable.get(stateKey) || [0, 0, 0];
    return qValues.indexOf(Math.max(...qValues));
  }

  public storeExperience(experience: Experience): void {
    this.experienceBuffer.push(experience);

    // Remove oldest experience if buffer is full
    if (this.experienceBuffer.length > this.bufferSize) {
      this.experienceBuffer.shift();
    }
  }

  public train(): void {
    if (this.experienceBuffer.length < this.batchSize) {
      return; // Not enough experiences
    }

    // Sample random batch
    const batch = this.sampleBatch();

    for (const experience of batch) {
      this.updateQValue(
        experience.state,
        experience.action,
        experience.reward,
        experience.nextState
      );
    }
  }

  private sampleBatch(): Experience[] {
    const batch: Experience[] = [];
    const bufferSize = this.experienceBuffer.length;

    for (let i = 0; i < this.batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * bufferSize);
      batch.push(this.experienceBuffer[randomIndex]);
    }

    return batch;
  }

  public updateQValue(
    state: number[],
    action: number,
    reward: number,
    nextState: number[]
  ): void {
    const stateKey = this.stateToKey(state);
    const nextStateKey = this.stateToKey(nextState);

    const qValues = this.qTable.get(stateKey) || [0, 0, 0];
    const nextQValues = this.qTable.get(nextStateKey) || [0, 0, 0];

    const maxNextQ = Math.max(...nextQValues);
    const currentQ = qValues[action];

    qValues[action] =
      currentQ +
      this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);

    this.qTable.set(stateKey, qValues);
  }

  private stateToKey(state: number[]): string {
    // Discretize state for Q-table with more precision
    return state.map((val) => Math.round(val * 1000) / 1000).join(",");
  }

  public decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  public getEpsilon(): number {
    return this.epsilon;
  }

  public getQTableSize(): number {
    return this.qTable.size;
  }
}

// Training function
async function trainAdvancedRL(): Promise<void> {
  console.log("🚀 Starting Advanced RL Training...");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 730); // 2 years

    console.log(`📈 Loaded ${btcData.prices.length} days of BTC data`);

    // Create environment and agent
    const environment = new AdvancedRLEnvironment(
      btcData.prices,
      btcData.volumes
    );
    const agent = new AdvancedQLAgent();

    // Training parameters
    const episodes = 200;
    const maxSteps = 200;
    const trainingFrequency = 5; // Train every 5 steps

    console.log("🎯 Training agent...");

    let bestSharpeRatio = -Infinity;
    let bestEpisode = 0;

    for (let episode = 0; episode < episodes; episode++) {
      let state = environment.reset();
      let totalReward = 0;
      let steps = 0;

      while (
        steps < maxSteps &&
        environment.getCurrentIndex() < btcData.prices.length - 1
      ) {
        const action = agent.chooseAction(state);
        const { state: nextState, reward, done } = environment.step(action);

        // Store experience
        agent.storeExperience({
          state,
          action,
          reward,
          nextState,
          done,
        });

        // Train periodically
        if (steps % trainingFrequency === 0) {
          agent.train();
        }

        totalReward += reward;
        state = nextState;
        steps++;

        if (done) break;
      }

      agent.decayEpsilon();

      // Calculate episode metrics
      const metrics = environment.getMetrics();

      // Track best performance
      if (metrics.sharpeRatio > bestSharpeRatio) {
        bestSharpeRatio = metrics.sharpeRatio;
        bestEpisode = episode;
      }

      if (episode % 20 === 0) {
        console.log(
          `Episode ${episode}: ` +
            `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
            `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
            `Drawdown=${(metrics.maxDrawdown * 100).toFixed(2)}%, ` +
            `WinRate=${(metrics.winRate * 100).toFixed(1)}%, ` +
            `Trades=${metrics.tradeCount}, ` +
            `Epsilon=${agent.getEpsilon().toFixed(3)}, ` +
            `QTableSize=${agent.getQTableSize()}`
        );
      }
    }

    console.log("✅ Advanced RL Training completed!");
    console.log(
      `🏆 Best Sharpe ratio: ${bestSharpeRatio.toFixed(
        3
      )} (Episode ${bestEpisode})`
    );

    // Test the trained agent
    console.log("🧪 Testing trained agent...");
    let testState = environment.reset();
    let testSteps = 0;
    const testMaxSteps = 100;

    while (
      testSteps < testMaxSteps &&
      environment.getCurrentIndex() < btcData.prices.length - 1
    ) {
      const action = agent.chooseAction(testState);
      const { state: nextState, done } = environment.step(action);

      testState = nextState;
      testSteps++;

      if (done) break;
    }

    const testMetrics = environment.getMetrics();
    console.log("📊 Test Results:");
    console.log(
      `  Total Return: ${(testMetrics.totalReturn * 100).toFixed(2)}%`
    );
    console.log(`  Sharpe Ratio: ${testMetrics.sharpeRatio.toFixed(3)}`);
    console.log(
      `  Max Drawdown: ${(testMetrics.maxDrawdown * 100).toFixed(2)}%`
    );
    console.log(`  Win Rate: ${(testMetrics.winRate * 100).toFixed(1)}%`);
    console.log(`  Total Trades: ${testMetrics.tradeCount}`);
    console.log(`  Final Q-Table Size: ${agent.getQTableSize()}`);
  } catch (error) {
    console.error("❌ Advanced RL Training failed:", error);
    throw error;
  }
}

// Run training
if (require.main === module) {
  trainAdvancedRL().catch(console.error);
}

export { trainAdvancedRL };
