import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import FeatureCalculator from "../bitcoin/FeatureCalculator";

// Initialize Firebase
FirebaseService.getInstance();

// Simple RL Environment
class SimpleRLEnvironment {
  private prices: number[];
  private volumes: number[];
  private currentIndex: number;
  private capital: number;
  private position: number;
  private entryPrice: number;
  private featureCalculator: FeatureCalculator;

  constructor(prices: number[], volumes: number[]) {
    this.prices = prices;
    this.volumes = volumes;
    this.currentIndex = 35; // Start after enough data for features
    this.capital = 10000;
    this.position = 0;
    this.entryPrice = 0;
    this.featureCalculator = new FeatureCalculator();
  }

  public reset(): number[] {
    this.currentIndex = 35;
    this.capital = 10000;
    this.position = 0;
    this.entryPrice = 0;
    return this.getState();
  }

  public step(action: number): {
    state: number[];
    reward: number;
    done: boolean;
  } {
    const currentPrice = this.prices[this.currentIndex];
    const previousValue = this.getPortfolioValue();

    // Execute action: 0 = HOLD, 1 = BUY, 2 = SELL
    if (action === 1 && this.position === 0) {
      // Buy
      this.position = 1;
      this.entryPrice = currentPrice;
    } else if (action === 2 && this.position === 1) {
      // Sell
      this.position = 0;
      this.entryPrice = 0;
    }

    // Move to next step
    this.currentIndex++;

    // Calculate reward
    const currentValue = this.getPortfolioValue();
    const reward = (currentValue - previousValue) / previousValue;

    // Check if done
    const done = this.currentIndex >= this.prices.length - 1;

    return {
      state: this.getState(),
      reward,
      done,
    };
  }

  private getState(): number[] {
    if (this.currentIndex >= this.prices.length) {
      return Array(10).fill(0);
    }

    const features = this.featureCalculator.compute({
      prices: this.prices,
      volumes: this.volumes,
      dayIndex: this.currentIndex,
      currentPrice: this.prices[this.currentIndex],
    });

    // Use first 10 features for simplicity
    return features.slice(0, 10);
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
}

// Simple Q-Learning Agent
class SimpleQLAgent {
  private qTable: Map<string, number[]>;
  private learningRate: number;
  private discountFactor: number;
  private epsilon: number;
  private epsilonDecay: number;
  private epsilonMin: number;

  constructor() {
    this.qTable = new Map();
    this.learningRate = 0.1;
    this.discountFactor = 0.95;
    this.epsilon = 1.0;
    this.epsilonDecay = 0.995;
    this.epsilonMin = 0.01;
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
    // Discretize state for Q-table
    return state.map((val) => Math.round(val * 100) / 100).join(",");
  }

  public decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  public getEpsilon(): number {
    return this.epsilon;
  }
}

// Training function
async function trainSimpleRL(): Promise<void> {
  console.log("🚀 Starting Simple RL Training...");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 365); // 1 year

    console.log(`📈 Loaded ${btcData.prices.length} days of BTC data`);

    // Create environment and agent
    const environment = new SimpleRLEnvironment(
      btcData.prices,
      btcData.volumes
    );
    const agent = new SimpleQLAgent();

    // Training parameters
    const episodes = 100;
    const maxSteps = 100;

    console.log("🎯 Training agent...");

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

        agent.updateQValue(state, action, reward, nextState);

        totalReward += reward;
        state = nextState;
        steps++;

        if (done) break;
      }

      agent.decayEpsilon();

      if (episode % 10 === 0) {
        console.log(
          `Episode ${episode}: Total Reward=${totalReward.toFixed(4)}, ` +
            `Steps=${steps}, Epsilon=${agent.getEpsilon().toFixed(3)}`
        );
      }
    }

    console.log("✅ Simple RL Training completed!");

    // Test the trained agent
    console.log("🧪 Testing trained agent...");
    let testState = environment.reset();
    let testReward = 0;
    let testSteps = 0;

    while (
      testSteps < 50 &&
      environment.getCurrentIndex() < btcData.prices.length - 1
    ) {
      const action = agent.chooseAction(testState);
      const { state: nextState, reward, done } = environment.step(action);

      testReward += reward;
      testState = nextState;
      testSteps++;

      if (done) break;
    }

    console.log(
      `📊 Test Results: Total Reward=${testReward.toFixed(
        4
      )}, Steps=${testSteps}`
    );
  } catch (error) {
    console.error("❌ Simple RL Training failed:", error);
    throw error;
  }
}

// Run training
if (require.main === module) {
  trainSimpleRL().catch(console.error);
}

export { trainSimpleRL };
