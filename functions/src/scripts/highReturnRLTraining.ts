import { CryptoCompareService } from "../api/CryptoCompareService";
import { FirebaseService } from "../api/FirebaseService";
import { RLTradingEnvironment } from "../bitcoin/rl/RLTradingEnvironment";
import { EnhancedRLAgent } from "../bitcoin/rl/EnhancedRLAgent";
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
 * Calculate baseline performance (buy and hold)
 */
function calculateBaselinePerformance(prices: number[]): number {
  const initialPrice = prices[0];
  const finalPrice = prices[prices.length - 1];
  return (finalPrice - initialPrice) / initialPrice;
}

/**
 * High-Return RL Training with Multiple Strategies
 */
async function trainHighReturnRL(): Promise<void> {
  console.log(
    "🚀 Starting High-Return RL Training with Multiple Strategies..."
  );

  try {
    // Load data - use longer period for better learning
    const cryptoCompare = new CryptoCompareService();
    const btcData = await cryptoCompare.getHistoricalData("BTC", 365); // 1 year

    console.log(
      `📈 Loaded ${btcData.prices.length} days of BTC data for high-return training`
    );

    // Strategy 1: High-Return Base Configuration
    console.log("🎯 Strategy 1: High-Return Base Configuration");
    const highReturnEnvironment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      HighReturnRLEnvironmentConfig
    );

    const highReturnAgentConfig = {
      learningRate: 0.002,
      discountFactor: 0.92,
      epsilon: 0.4,
      epsilonDecay: 0.998,
      epsilonMin: 0.05,
      batchSize: 64,
      memorySize: 15000,
      targetUpdateFrequency: 50,
      hiddenLayers: [256, 128, 64],
      activationFunction: "relu",
      optimizer: "adam",
      lossFunction: "meanSquaredError",
      gradientClipping: 0.5,
      experienceReplay: true,
      prioritizedReplay: true,
      doubleDQN: true,
      duelingDQN: true,
      useTradeModelFactory: true,
      timesteps: 30,
      features: 36,
    };

    const highReturnAgent = new EnhancedRLAgent(
      highReturnEnvironment,
      highReturnAgentConfig
    );

    console.log("📊 High-Return Configuration:");
    console.log(
      `  - Reward Scaling: ${HighReturnRLEnvironmentConfig.rewardScaling}`
    );
    console.log(
      `  - Position Size Multiplier: ${HighReturnRLEnvironmentConfig.riskManagement.positionSizeMultiplier}`
    );
    console.log(
      `  - Regime Thresholds: ${JSON.stringify(
        HighReturnRLEnvironmentConfig.regimeThresholds
      )}`
    );

    const highReturnResults = await highReturnAgent.trainForEpisodes(
      150, // More episodes for better learning
      (metrics) => {
        if (metrics.episode % 25 === 0) {
          console.log(
            `High-Return Episode ${metrics.episode}: ` +
              `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
              `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
              `Epsilon=${metrics.epsilon.toFixed(3)}, ` +
              `Trades=${metrics.totalTrades}`
          );
        }
      }
    );

    // Strategy 2: Momentum-Focused Configuration
    console.log("\n🎯 Strategy 2: Momentum-Focused Configuration");
    const momentumEnvironment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      MomentumFocusedConfig
    );

    const momentumAgent = new EnhancedRLAgent(
      momentumEnvironment,
      highReturnAgentConfig
    );

    console.log("📊 Momentum Configuration:");
    console.log(`  - Reward Scaling: ${MomentumFocusedConfig.rewardScaling}`);
    console.log(
      `  - Position Size Multiplier: ${MomentumFocusedConfig.riskManagement.positionSizeMultiplier}`
    );
    console.log(
      `  - Trend Threshold: ${MomentumFocusedConfig.regimeThresholds.trendStrengthThreshold}`
    );

    const momentumResults = await momentumAgent.trainForEpisodes(
      100,
      (metrics) => {
        if (metrics.episode % 25 === 0) {
          console.log(
            `Momentum Episode ${metrics.episode}: ` +
              `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
              `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
              `Epsilon=${metrics.epsilon.toFixed(3)}, ` +
              `Trades=${metrics.totalTrades}`
          );
        }
      }
    );

    // Strategy 3: Volatility-Harvesting Configuration
    console.log("\n🎯 Strategy 3: Volatility-Harvesting Configuration");
    const volatilityEnvironment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      VolatilityHarvestingConfig
    );

    const volatilityAgent = new EnhancedRLAgent(
      volatilityEnvironment,
      highReturnAgentConfig
    );

    console.log("📊 Volatility-Harvesting Configuration:");
    console.log(
      `  - Volatility Threshold: ${VolatilityHarvestingConfig.regimeThresholds.volatilityThreshold}`
    );
    console.log(
      `  - Position Size Multiplier: ${VolatilityHarvestingConfig.riskManagement.positionSizeMultiplier}`
    );
    console.log(
      `  - Volatility Scaling: ${VolatilityHarvestingConfig.riskManagement.volatilityScaling}`
    );

    const volatilityResults = await volatilityAgent.trainForEpisodes(
      100,
      (metrics) => {
        if (metrics.episode % 25 === 0) {
          console.log(
            `Volatility Episode ${metrics.episode}: ` +
              `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
              `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
              `Epsilon=${metrics.epsilon.toFixed(3)}, ` +
              `Trades=${metrics.totalTrades}`
          );
        }
      }
    );

    // Strategy 4: Multi-Timeframe Configuration
    console.log("\n🎯 Strategy 4: Multi-Timeframe Configuration");
    const multiTimeframeEnvironment = new RLTradingEnvironment(
      btcData.prices,
      btcData.volumes,
      MultiTimeframeConfig
    );

    const multiTimeframeAgent = new EnhancedRLAgent(
      multiTimeframeEnvironment,
      highReturnAgentConfig
    );

    console.log("📊 Multi-Timeframe Configuration:");
    console.log(`  - Timesteps: ${MultiTimeframeConfig.timesteps}`);
    console.log(
      `  - Position Size Multiplier: ${MultiTimeframeConfig.riskManagement.positionSizeMultiplier}`
    );

    const multiTimeframeResults = await multiTimeframeAgent.trainForEpisodes(
      100,
      (metrics) => {
        if (metrics.episode % 25 === 0) {
          console.log(
            `Multi-Timeframe Episode ${metrics.episode}: ` +
              `Return=${(metrics.totalReturn * 100).toFixed(2)}%, ` +
              `Sharpe=${metrics.sharpeRatio.toFixed(3)}, ` +
              `Epsilon=${metrics.epsilon.toFixed(3)}, ` +
              `Trades=${metrics.totalTrades}`
          );
        }
      }
    );

    // Calculate baseline performance
    console.log("\n📊 Calculating baseline performance...");
    const baselineReturn = calculateBaselinePerformance(btcData.prices);
    console.log(
      `📈 Buy & Hold Baseline: ${(baselineReturn * 100).toFixed(2)}% return`
    );

    // Compare all strategies
    console.log("\n📊 Performance Comparison - All Strategies:");
    console.log("");

    const strategies = [
      { name: "High-Return Base", results: highReturnResults },
      { name: "Momentum-Focused", results: momentumResults },
      { name: "Volatility-Harvesting", results: volatilityResults },
      { name: "Multi-Timeframe", results: multiTimeframeResults },
    ];

    let bestStrategy = "";
    let bestReturn = -Infinity;
    let bestSharpe = -Infinity;

    for (const strategy of strategies) {
      const avgReturn =
        strategy.results.reduce((sum, r) => sum + r.totalReturn, 0) /
        strategy.results.length;
      const avgSharpe =
        strategy.results.reduce((sum, r) => sum + r.sharpeRatio, 0) /
        strategy.results.length;
      const bestSharpeRatio = Math.max(
        ...strategy.results.map((r) => r.sharpeRatio)
      );
      const avgTrades =
        strategy.results.reduce((sum, r) => sum + r.totalTrades, 0) /
        strategy.results.length;

      console.log(`🎯 ${strategy.name}:`);
      console.log(`  Average Return: ${(avgReturn * 100).toFixed(2)}%`);
      console.log(`  Average Sharpe: ${avgSharpe.toFixed(3)}`);
      console.log(`  Best Sharpe: ${bestSharpeRatio.toFixed(3)}`);
      console.log(`  Average Trades: ${avgTrades.toFixed(1)}`);
      console.log(
        `  vs Baseline: ${((avgReturn - baselineReturn) * 100).toFixed(2)}%`
      );
      console.log("");

      if (avgReturn > bestReturn) {
        bestReturn = avgReturn;
        bestStrategy = strategy.name;
      }
      if (avgSharpe > bestSharpe) {
        bestSharpe = avgSharpe;
      }
    }

    console.log("🏆 Best Performance Summary:");
    console.log(`  Best Strategy: ${bestStrategy}`);
    console.log(`  Best Average Return: ${(bestReturn * 100).toFixed(2)}%`);
    console.log(`  Best Average Sharpe: ${bestSharpe.toFixed(3)}`);
    console.log(`  Baseline Return: ${(baselineReturn * 100).toFixed(2)}%`);
    console.log(
      `  Total Improvement: ${((bestReturn - baselineReturn) * 100).toFixed(
        2
      )}%`
    );

    // Strategy recommendations
    console.log("\n💡 Strategy Recommendations for Higher Returns:");
    console.log("");

    for (const [, strategy] of Object.entries(ReturnOptimizationStrategies)) {
      console.log(`📈 ${strategy.description}`);
      console.log(`   Implementation: ${strategy.implementation}`);
      console.log(`   Expected Return: ${strategy.expectedReturn}`);
      console.log("");
    }

    console.log("✅ High-Return RL Training completed successfully!");
  } catch (error) {
    console.error("❌ High-Return RL Training failed:", error);
    throw error;
  }
}

// Run training
if (require.main === module) {
  trainHighReturnRL().catch(console.error);
}

export { trainHighReturnRL };
