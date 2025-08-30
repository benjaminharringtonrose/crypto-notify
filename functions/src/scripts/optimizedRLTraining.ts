import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import { RLTradingEnvironment } from "../bitcoin/rl/RLTradingEnvironment";
import { EnhancedRLAgent } from "../bitcoin/rl/EnhancedRLAgent";
import { FeatureDetector } from "../bitcoin/shared/FeatureDetector";
import {
  HighReturnRLEnvironmentConfig,
  MomentumFocusedConfig,
  VolatilityHarvestingConfig,
  MultiTimeframeConfig,
  ReturnOptimizationStrategies,
} from "../bitcoin/rl/HighReturnRLConfig";

// Initialize Firebase
FirebaseService.getInstance();

/**
 * Parse command line arguments
 */
function parseArgs(): {
  episodes: number;
  strategy: string;
  dataDays: number;
  quickMode: boolean;
} {
  const args = process.argv.slice(2);
  let episodes = 150;
  let strategy = "high-return";
  let dataDays = 365;
  let quickMode = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--episodes" && i + 1 < args.length) {
      episodes = parseInt(args[i + 1]);
      i++;
    } else if (arg === "--strategy" && i + 1 < args.length) {
      strategy = args[i + 1];
      i++;
    } else if (arg === "--data-days" && i + 1 < args.length) {
      dataDays = parseInt(args[i + 1]);
      i++;
    } else if (arg === "--quick") {
      quickMode = true;
    }
  }

  return { episodes, strategy, dataDays, quickMode };
}

/**
 * Calculate baseline performance (buy and hold)
 */
function calculateBaselinePerformance(prices: number[]): number {
  const initialPrice = prices[0];
  const finalPrice = prices[prices.length - 1];
  return (finalPrice - initialPrice) / initialPrice;
}

/**
 * Get configuration based on strategy
 */
function getStrategyConfig(strategy: string) {
  switch (strategy) {
    case "momentum":
      return MomentumFocusedConfig;
    case "volatility":
      return VolatilityHarvestingConfig;
    case "multi-timeframe":
      return MultiTimeframeConfig;
    case "high-return":
    default:
      return HighReturnRLEnvironmentConfig;
  }
}

/**
 * Optimized RL Agent Configuration
 */
const OPTIMIZED_AGENT_CONFIG = {
  // Aggressive learning parameters for higher returns
  learningRate: 0.003, // Higher learning rate
  discountFactor: 0.9, // Lower discount factor for more immediate rewards
  epsilon: 0.5, // Higher initial exploration
  epsilonDecay: 0.999, // Slower decay to maintain exploration
  epsilonMin: 0.08, // Higher minimum exploration

  // Training parameters
  batchSize: 128, // Larger batch size for better gradients
  memorySize: 20000, // Larger memory for more diverse experiences
  targetUpdateFrequency: 25, // More frequent target updates

  // Network architecture optimized for returns
  hiddenLayers: [512, 256, 128], // Larger network for complex patterns
  activationFunction: "relu",
  optimizer: "adam",
  lossFunction: "meanSquaredError",
  gradientClipping: 0.3,

  // Advanced features
  experienceReplay: true,
  prioritizedReplay: true,
  doubleDQN: true,
  duelingDQN: true,
  useTradeModelFactory: true,
  timesteps: 30,
  features: 36,
};

/**
 * Optimized RL Training with Multiple Strategies
 */
async function trainOptimizedRL(): Promise<void> {
  const { episodes, strategy, dataDays, quickMode } = parseArgs();

  console.log("🚀 Starting Optimized RL Training...");
  console.log(
    `📊 Configuration: ${strategy} strategy, ${episodes} episodes, ${dataDays} days of data`
  );
  console.log(`⚡ Quick Mode: ${quickMode ? "Enabled" : "Disabled"}`);

  try {
    // Detect feature count dynamically
    const featureCount = await FeatureDetector.detectFeatureCount();
    console.log(`🔍 Detected ${featureCount} features for RL training`);

    // Load data
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", dataDays);

    console.log(`📈 Loaded ${btcData.prices.length} days of BTC data`);

    // Get strategy configuration
    const envConfig = getStrategyConfig(strategy);

    // Apply quick mode optimizations if enabled
    if (quickMode) {
      envConfig.rewardScaling = 300; // Higher reward scaling
      envConfig.riskManagement.positionSizeMultiplier = 1.5; // More aggressive
      envConfig.regimeThresholds.trendStrengthThreshold = 0.0003; // Lower threshold
      envConfig.regimeThresholds.momentumThreshold = 0.002; // Lower threshold
    }

    console.log(`🎯 Strategy: ${strategy.toUpperCase()}`);
    console.log(`📊 Environment Config:`);
    console.log(`  - Reward Scaling: ${envConfig.rewardScaling}`);
    console.log(
      `  - Position Size Multiplier: ${envConfig.riskManagement.positionSizeMultiplier}`
    );
    console.log(
      `  - Regime Thresholds: ${JSON.stringify(envConfig.regimeThresholds)}`
    );

    // Create environment
    const environment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      envConfig
    );

    // Create dynamic agent configuration with detected feature count
    const dynamicAgentConfig = {
      ...OPTIMIZED_AGENT_CONFIG,
      features: featureCount,
    };

    // Create agent
    const agent = new EnhancedRLAgent(environment, dynamicAgentConfig);

    console.log(`🤖 Agent Config:`);
    console.log(`  - Learning Rate: ${dynamicAgentConfig.learningRate}`);
    console.log(`  - Network: ${dynamicAgentConfig.hiddenLayers.join(" → ")}`);
    console.log(`  - Memory Size: ${dynamicAgentConfig.memorySize}`);
    console.log(`  - Features: ${dynamicAgentConfig.features}`);

    // Training with progress tracking
    const results = await agent.trainForEpisodes(episodes, (metrics) => {
      if (metrics.episode % 25 === 0 || metrics.episode < 10) {
        console.log(
          `Episode ${metrics.episode}/${episodes}: ` +
            `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
            `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
            `Epsilon=${metrics.epsilon.toFixed(3)}, ` +
            `Trades=${metrics.totalTrades}`
        );
      }
    });

    // Calculate baseline performance
    console.log("\n📊 Calculating baseline performance...");
    const baselineReturn = calculateBaselinePerformance(btcData.prices);
    console.log(
      `📈 Buy & Hold Baseline: ${(baselineReturn * 100).toFixed(2)}% return`
    );

    // Analyze results
    console.log("\n📊 Training Results Analysis:");
    console.log("");

    const avgReturn =
      results.reduce((sum, r) => sum + r.totalReturn, 0) / results.length;
    const avgSharpe =
      results.reduce((sum, r) => sum + r.sharpeRatio, 0) / results.length;
    const bestSharpe = Math.max(...results.map((r) => r.sharpeRatio));
    const avgTrades =
      results.reduce((sum, r) => sum + r.totalTrades, 0) / results.length;
    const positiveEpisodes = results.filter((r) => r.totalReturn > 0).length;
    const winRate = (positiveEpisodes / results.length) * 100;

    // Performance metrics
    console.log(`🎯 ${strategy.toUpperCase()} Strategy Results:`);
    console.log(`  Average Return: ${(avgReturn * 100).toFixed(2)}%`);
    console.log(`  Average Sharpe: ${avgSharpe.toFixed(3)}`);
    console.log(`  Best Sharpe: ${bestSharpe.toFixed(3)}`);
    console.log(`  Average Trades: ${avgTrades.toFixed(1)}`);
    console.log(`  Win Rate: ${winRate.toFixed(1)}%`);
    console.log(
      `  vs Baseline: ${((avgReturn - baselineReturn) * 100).toFixed(2)}%`
    );
    console.log("");

    // Performance analysis
    const improvement =
      ((avgReturn - baselineReturn) / Math.abs(baselineReturn)) * 100;
    console.log("📈 Performance Analysis:");
    console.log(
      `  Strategy Performance: ${
        avgReturn > 0 ? "✅ Profitable" : "❌ Loss-making"
      }`
    );
    console.log(
      `  Risk-Adjusted Performance: ${
        avgSharpe > 0.5
          ? "✅ Good"
          : avgSharpe > 0.2
          ? "⚠️ Moderate"
          : "❌ Poor"
      }`
    );
    console.log(
      `  Trading Activity: ${
        avgTrades > 50 ? "✅ High" : avgTrades > 20 ? "⚠️ Moderate" : "❌ Low"
      }`
    );
    console.log(
      `  Consistency: ${
        winRate > 60 ? "✅ Good" : winRate > 40 ? "⚠️ Moderate" : "❌ Poor"
      }`
    );
    console.log("");

    // Strategy recommendations
    console.log("💡 Strategy Recommendations for Higher Returns:");
    console.log("");

    for (const [, strategy] of Object.entries(ReturnOptimizationStrategies)) {
      console.log(`📈 ${strategy.description}`);
      console.log(`   Implementation: ${strategy.implementation}`);
      console.log(`   Expected Return: ${strategy.expectedReturn}`);
      console.log("");
    }

    // Final summary
    console.log("🏆 Training Summary:");
    console.log(`  Strategy: ${strategy.toUpperCase()}`);
    console.log(`  Episodes: ${episodes}`);
    console.log(`  Data Period: ${dataDays} days`);
    console.log(
      `  Best Performance: ${(
        Math.max(...results.map((r) => r.totalReturn)) * 100
      ).toFixed(2)}%`
    );
    console.log(`  Average Performance: ${(avgReturn * 100).toFixed(2)}%`);
    console.log(
      `  Risk-Adjusted Performance: ${bestSharpe.toFixed(3)} Sharpe ratio`
    );
    console.log(
      `  vs Buy & Hold: ${improvement > 0 ? "+" : ""}${improvement.toFixed(2)}%`
    );

    console.log("\n✅ Optimized RL Training completed successfully!");
  } catch (error) {
    console.error("❌ Optimized RL Training failed:", error);
    throw error;
  }
}

// Run training
if (require.main === module) {
  trainOptimizedRL().catch(console.error);
}

export { trainOptimizedRL };
