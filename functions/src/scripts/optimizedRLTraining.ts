import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import { RLTradingEnvironment } from "../bitcoin/rl/RLTradingEnvironment";
import { EnhancedRLAgent } from "../bitcoin/rl/EnhancedRLAgent";
import { FeatureDetector } from "../bitcoin/shared/FeatureDetector";
import {
  ImprovedRLEnvironmentConfig,
  ImprovedRLAgentConfig,
  ConservativeRLConfig,
  AggressiveRLConfig,
  LearningFocusedConfig,
  QuickTrainingConfig,
  MarketConditionConfigs,
} from "../bitcoin/rl/ImprovedRLConfig";

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
  let episodes = 100;
  let strategy = "improved";
  let dataDays = 180; // Reduced from 365 for faster training
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
    case "conservative":
      return ConservativeRLConfig;
    case "aggressive":
      return AggressiveRLConfig;
    case "learning":
      return LearningFocusedConfig;
    case "quick":
      return QuickTrainingConfig;
    case "bull":
      return MarketConditionConfigs.bull;
    case "bear":
      return MarketConditionConfigs.bear;
    case "sideways":
      return MarketConditionConfigs.sideways;
    case "improved":
    default:
      return ImprovedRLEnvironmentConfig;
  }
}

/**
 * IMPROVED RL Agent Configuration
 * Addresses the learning issues we observed
 */
const OPTIMIZED_AGENT_CONFIG = {
  ...ImprovedRLAgentConfig,
  // Additional improvements based on training observations
  epsilon: 0.9, // Even higher initial exploration
  epsilonDecay: 0.997, // Even slower decay
  epsilonMin: 0.2, // Higher minimum exploration
  learningRate: 0.0005, // Lower learning rate for stability
  batchSize: 32, // Smaller batches for more frequent updates
  memorySize: 5000, // Smaller memory for faster learning from recent experiences
  targetUpdateFrequency: 5, // Very frequent target updates
  hiddenLayers: [128, 64, 32], // Even simpler network
  useTradeModelFactory: true,
  timesteps: 20, // Reduced for faster training
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
    console.log("📈 Use 'learning' config for initial training");
    console.log("📈 Use 'aggressive' config for higher returns");
    console.log("📈 Use 'conservative' config for risk management");
    console.log("📈 Use 'quick' config for rapid testing");
    console.log("");

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
