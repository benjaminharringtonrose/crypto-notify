import {
  RLTradingEnvironment,
  RLEnvironmentConfig,
} from "../bitcoin/RLTradingEnvironment";
import { RLTradingAgent, RLAgentConfig } from "../bitcoin/RLTradingAgent";
import { RLTradingStrategy } from "../bitcoin/RLTradingStrategy";
import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";

// Initialize Firebase
FirebaseService.getInstance();

// Example configurations
const EXAMPLE_ENV_CONFIG: RLEnvironmentConfig = {
  initialCapital: 10000,
  commission: 0.005,
  slippage: 0.001,
  timesteps: 35,
  maxPositionSize: 1.0,
  minPositionSize: 0.25,
  rewardScaling: 100,
  riskFreeRate: 0.02,
  transactionCostPenalty: 0.01,
  holdingPenalty: 0.001,
  volatilityPenalty: 0.1,
};

const EXAMPLE_AGENT_CONFIG: RLAgentConfig = {
  learningRate: 0.0001,
  discountFactor: 0.99,
  epsilon: 1.0,
  epsilonDecay: 0.995,
  epsilonMin: 0.01,
  batchSize: 32,
  memorySize: 10000,
  targetUpdateFrequency: 100,
  hiddenLayers: [128, 64, 32],
  activationFunction: "relu",
  optimizer: "adam",
  lossFunction: "meanSquaredError",
  gradientClipping: 1.0,
  experienceReplay: true,
  prioritizedReplay: true,
  doubleDQN: true,
  duelingDQN: false,
};

/**
 * Example 1: Basic RL Training
 */
export async function exampleBasicTraining(): Promise<void> {
  console.log("🚀 Example 1: Basic RL Training");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 730); // 2 years

    console.log(`📈 Loaded ${btcData.prices.length} days of BTC data`);

    // Create environment
    const environment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      EXAMPLE_ENV_CONFIG
    );

    // Create agent
    const agent = new RLTradingAgent(environment, EXAMPLE_AGENT_CONFIG);

    // Train for a few episodes
    console.log("🎯 Training agent...");
    const results = await agent.trainForEpisodes(50, (metrics) => {
      if (metrics.episode % 10 === 0) {
        console.log(
          `Episode ${metrics.episode}: Return=${(
            metrics.totalReturn * 100
          ).toFixed(2)}%, ` +
            `Sharpe=${metrics.sharpeRatio.toFixed(
              3
            )}, Epsilon=${metrics.epsilon.toFixed(3)}`
        );
      }
    });

    console.log("✅ Training completed!");
    console.log(
      `🏆 Best Sharpe ratio: ${Math.max(
        ...results.map((r) => r.sharpeRatio)
      ).toFixed(3)}`
    );
  } catch (error) {
    console.error("❌ Training failed:", error);
  }
}

/**
 * Example 2: RL Strategy Integration
 */
export async function exampleRLStrategyIntegration(): Promise<void> {
  console.log("🚀 Example 2: RL Strategy Integration");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 730);

    // Create environment and agent
    const environment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      EXAMPLE_ENV_CONFIG
    );

    const agent = new RLTradingAgent(environment, EXAMPLE_AGENT_CONFIG);

    // Train agent briefly
    console.log("🎯 Training agent for integration...");
    await agent.trainForEpisodes(20);

    // Create RL-enhanced trading strategy
    const rlStrategy = new RLTradingStrategy(agent, environment, {
      useRLAgent: true,
      rlWeight: 0.6,
      ruleBasedWeight: 0.4,
      confidenceThreshold: 0.7,
      fallbackToRuleBased: true,
      adaptiveWeighting: true,
      performanceWindow: 30,
    });

    // Test decision making
    console.log("🤖 Testing RL-enhanced decision making...");

    for (let i = 100; i < 110; i++) {
      const decision = await rlStrategy.makeDecision(
        btcData.prices,
        btcData.volumes,
        i
      );

      console.log(`Day ${i}:`);
      console.log(`  Action: ${decision.action}`);
      console.log(`  Confidence: ${decision.confidence.toFixed(3)}`);
      console.log(`  RL Action: ${decision.rlAction}`);
      console.log(`  RL Confidence: ${decision.rlConfidence.toFixed(3)}`);
      console.log(`  Reasoning: ${decision.reasoning}`);
      console.log(
        `  Current RL Weight: ${rlStrategy.getCurrentRLWeight().toFixed(3)}`
      );
      console.log("");
    }

    console.log("✅ RL strategy integration completed!");
  } catch (error) {
    console.error("❌ Integration failed:", error);
  }
}

/**
 * Example 3: Performance Comparison
 */
export async function examplePerformanceComparison(): Promise<void> {
  console.log("🚀 Example 3: Performance Comparison");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 730);

    // Split data for comparison
    const trainData = {
      prices: btcData.prices.slice(0, Math.floor(btcData.prices.length * 0.8)),
      volumes: btcData.volumes.slice(
        0,
        Math.floor(btcData.volumes.length * 0.8)
      ),
    };

    const testData = {
      prices: btcData.prices.slice(Math.floor(btcData.prices.length * 0.8)),
      volumes: btcData.volumes.slice(Math.floor(btcData.volumes.length * 0.8)),
    };

    // Create training environment and agent
    const trainEnvironment = new RLTradingEnvironment(
      trainData.prices,
      trainData.volumes,
      EXAMPLE_ENV_CONFIG
    );

    const agent = new RLTradingAgent(trainEnvironment, EXAMPLE_AGENT_CONFIG);

    // Train agent
    console.log("🎯 Training agent...");
    await agent.trainForEpisodes(100);

    // Create test environment
    const testEnvironment = new RLTradingEnvironment(
      testData.prices,
      testData.volumes,
      EXAMPLE_ENV_CONFIG
    );

    // Evaluate RL agent
    console.log("🔍 Evaluating RL agent...");
    const rlResults = await agent.evaluate(testEnvironment, 5);

    // Create rule-based strategy for comparison
    const ruleBasedStrategy = new RLTradingStrategy();

    // Simulate rule-based performance
    console.log("🔍 Evaluating rule-based strategy...");
    const ruleBasedResults = await simulateRuleBasedPerformance(
      testData.prices,
      testData.volumes,
      ruleBasedStrategy
    );

    // Compare results
    console.log("📊 Performance Comparison:");
    console.log("");
    console.log("RL Agent Results:");
    console.log(
      `  Average Return: ${(
        (rlResults.reduce((sum, r) => sum + r.totalReturn, 0) /
          rlResults.length) *
        100
      ).toFixed(2)}%`
    );
    console.log(
      `  Average Sharpe: ${(
        rlResults.reduce((sum, r) => sum + r.sharpeRatio, 0) / rlResults.length
      ).toFixed(3)}`
    );
    console.log(
      `  Average Max Drawdown: ${(
        (rlResults.reduce((sum, r) => sum + r.maxDrawdown, 0) /
          rlResults.length) *
        100
      ).toFixed(2)}%`
    );
    console.log(
      `  Average Win Rate: ${(
        (rlResults.reduce((sum, r) => sum + r.winRate, 0) / rlResults.length) *
        100
      ).toFixed(2)}%`
    );
    console.log("");
    console.log("Rule-Based Strategy Results:");
    console.log(
      `  Average Return: ${(
        (ruleBasedResults.reduce((sum, r) => sum + r.totalReturn, 0) /
          ruleBasedResults.length) *
        100
      ).toFixed(2)}%`
    );
    console.log(
      `  Average Sharpe: ${(
        ruleBasedResults.reduce((sum, r) => sum + r.sharpeRatio, 0) /
        ruleBasedResults.length
      ).toFixed(3)}`
    );
    console.log(
      `  Average Max Drawdown: ${(
        (ruleBasedResults.reduce((sum, r) => sum + r.maxDrawdown, 0) /
          ruleBasedResults.length) *
        100
      ).toFixed(2)}%`
    );
    console.log(
      `  Average Win Rate: ${(
        (ruleBasedResults.reduce((sum, r) => sum + r.winRate, 0) /
          ruleBasedResults.length) *
        100
      ).toFixed(2)}%`
    );

    console.log("✅ Performance comparison completed!");
  } catch (error) {
    console.error("❌ Performance comparison failed:", error);
  }
}

/**
 * Simulate rule-based strategy performance
 */
async function simulateRuleBasedPerformance(
  prices: number[],
  volumes: number[],
  strategy: RLTradingStrategy
): Promise<any[]> {
  const results = [];

  for (let episode = 0; episode < 5; episode++) {
    let capital = EXAMPLE_ENV_CONFIG.initialCapital;
    let position = 0;
    let entryPrice = 0;
    let peakValue = capital;
    let trades = 0;
    let wins = 0;
    let returns = [];

    for (let i = 35; i < prices.length - 1; i++) {
      try {
        const decision = await strategy.decide(prices, volumes, i);

        if (decision.action === "BUY" && position === 0) {
          position = 1;
          entryPrice = prices[i];
          trades++;
        } else if (decision.action === "SELL" && position === 1) {
          const exitPrice = prices[i];
          const tradeReturn = (exitPrice - entryPrice) / entryPrice;
          returns.push(tradeReturn);

          if (tradeReturn > 0) wins++;

          position = 0;
          entryPrice = 0;
        }

        // Calculate current portfolio value
        const currentValue =
          position === 1
            ? capital * (1 + (prices[i] - entryPrice) / entryPrice)
            : capital;

        peakValue = Math.max(peakValue, currentValue);
      } catch (error) {
        // Continue on error
      }
    }

    const totalReturn =
      returns.length > 0 ? returns.reduce((sum, r) => sum + r, 0) : 0;

    const sharpeRatio =
      returns.length > 0
        ? totalReturn /
          Math.sqrt(
            returns.reduce(
              (sum, r) => sum + Math.pow(r - totalReturn / returns.length, 2),
              0
            ) / returns.length
          )
        : 0;

    const maxDrawdown =
      (peakValue -
        Math.min(
          ...returns.map(
            (_, i) =>
              capital *
              (1 + returns.slice(0, i + 1).reduce((sum, r) => sum + r, 0))
          )
        )) /
      peakValue;

    const winRate = trades > 0 ? wins / trades : 0;

    results.push({
      episode,
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      winRate,
      totalTrades: trades,
    });
  }

  return results;
}

/**
 * Example 4: Model Persistence
 */
export async function exampleModelPersistence(): Promise<void> {
  console.log("🚀 Example 4: Model Persistence");

  try {
    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 730);

    // Create environment and agent
    const environment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      EXAMPLE_ENV_CONFIG
    );

    const agent = new RLTradingAgent(environment, EXAMPLE_AGENT_CONFIG);

    // Train agent
    console.log("🎯 Training agent...");
    await agent.trainForEpisodes(50);

    // Save model
    console.log("💾 Saving model...");
    await agent.saveModel("/tmp/rl_example_model");

    // Create new agent and load model
    console.log("📥 Loading model...");
    const newAgent = new RLTradingAgent(environment, EXAMPLE_AGENT_CONFIG);
    await newAgent.loadModel("/tmp/rl_example_model");

    // Test loaded model
    console.log("🧪 Testing loaded model...");
    const testResults = await newAgent.evaluate(environment, 3);

    console.log("✅ Model persistence test completed!");
    console.log(
      `📊 Loaded model performance: ${(
        (testResults.reduce((sum, r) => sum + r.totalReturn, 0) /
          testResults.length) *
        100
      ).toFixed(2)}% average return`
    );
  } catch (error) {
    console.error("❌ Model persistence failed:", error);
  }
}

/**
 * Main function to run all examples
 */
export async function runAllExamples(): Promise<void> {
  console.log("🎯 Running RL Integration Examples");
  console.log("==================================");

  try {
    await exampleBasicTraining();
    console.log("");

    await exampleRLStrategyIntegration();
    console.log("");

    await examplePerformanceComparison();
    console.log("");

    await exampleModelPersistence();
    console.log("");

    console.log("🎉 All examples completed successfully!");
  } catch (error) {
    console.error("❌ Examples failed:", error);
  }
}

// Run if called directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}
