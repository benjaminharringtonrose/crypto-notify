import {
  RLTradingEnvironment,
  RLEnvironmentConfig,
} from "../bitcoin/RLTradingEnvironment";
import { RLTradingAgent, RLAgentConfig } from "../bitcoin/RLTradingAgent";
import { FirebaseService } from "../api/FirebaseService";
import { CryptoCompareService } from "../api/CryptoCompareService";
import { TRAINING_CONFIG, MODEL_CONFIG } from "../constants";
import * as admin from "firebase-admin";

// Initialize Firebase
FirebaseService.getInstance();

// RL Environment Configuration
const RL_ENV_CONFIG: RLEnvironmentConfig = {
  initialCapital: 10000, // $10,000 starting capital
  commission: 0.005, // 0.5% commission
  slippage: 0.001, // 0.1% slippage
  timesteps: MODEL_CONFIG.TIMESTEPS,
  maxPositionSize: 1.0, // 100% position size
  minPositionSize: 0.25, // 25% minimum position size
  rewardScaling: 100, // Scale rewards for better learning
  riskFreeRate: 0.02, // 2% annual risk-free rate
  transactionCostPenalty: 0.01, // Penalty for frequent trading
  holdingPenalty: 0.001, // Small penalty for holding positions
  volatilityPenalty: 0.1, // Penalty for high volatility periods
};

// RL Agent Configuration
const RL_AGENT_CONFIG: RLAgentConfig = {
  learningRate: 0.0001, // Conservative learning rate
  discountFactor: 0.99, // High discount factor for long-term planning
  epsilon: 1.0, // Start with 100% exploration
  epsilonDecay: 0.995, // Slow decay for stable learning
  epsilonMin: 0.01, // Minimum 1% exploration
  batchSize: 32, // Batch size for training
  memorySize: 10000, // Experience replay memory size
  targetUpdateFrequency: 100, // Update target network every 100 steps
  hiddenLayers: [128, 64, 32], // Neural network architecture
  activationFunction: "relu",
  optimizer: "adam",
  lossFunction: "meanSquaredError",
  gradientClipping: 1.0,
  experienceReplay: true,
  prioritizedReplay: true,
  doubleDQN: true,
  duelingDQN: false, // Can be enabled for more advanced architecture
};

// Training Configuration
const RL_TRAINING_CONFIG = {
  totalEpisodes: 1000,
  evaluationEpisodes: 10,
  saveFrequency: 100, // Save model every 100 episodes
  logFrequency: 10, // Log metrics every 10 episodes
  earlyStoppingPatience: 50, // Stop if no improvement for 50 episodes
  minImprovement: 0.01, // Minimum improvement threshold
};

export class RLTrainer {
  private cryptoCompare: CryptoCompareService;
  private environment!: RLTradingEnvironment;
  private agent!: RLTradingAgent;
  private bucket: any;
  private bestSharpeRatio: number = -Infinity;
  private patienceCounter: number = 0;
  private trainingHistory: any[] = [];

  constructor(seed?: number) {
    this.cryptoCompare = new CryptoCompareService();
    this.bucket = admin.storage().bucket();
    console.log("RL Trainer initialized");
  }

  /**
   * Train the RL agent
   */
  public async train(): Promise<void> {
    console.log("🚀 Starting RL Agent Training...");
    console.log(
      "📊 Environment Config:",
      JSON.stringify(RL_ENV_CONFIG, null, 2)
    );
    console.log("🤖 Agent Config:", JSON.stringify(RL_AGENT_CONFIG, null, 2));

    try {
      // Load and prepare data
      const { btcPrices, btcVolumes } = await this.loadData();
      console.log(`📈 Loaded ${btcPrices.length} days of BTC data`);

      // Create environment
      this.environment = new RLTradingEnvironment(
        btcPrices,
        btcVolumes,
        RL_ENV_CONFIG
      );

      // Create agent
      this.agent = new RLTradingAgent(this.environment, RL_AGENT_CONFIG);
      console.log("🤖 RL Agent created successfully");

      // Train the agent
      await this.trainAgent();

      // Evaluate final performance
      await this.evaluateAgent();

      // Save final model
      await this.saveModel("final");

      console.log("✅ RL Training completed successfully!");
    } catch (error) {
      console.error("❌ RL Training failed:", error);
      throw error;
    }
  }

  /**
   * Load historical data
   */
  private async loadData(): Promise<{
    btcPrices: number[];
    btcVolumes: number[];
  }> {
    console.log("📥 Loading historical BTC data...");

    const btcData = await this.cryptoCompare.getHistoricalData(
      "BTC",
      TRAINING_CONFIG.START_DAYS_AGO
    );

    if (!btcData.prices || !btcData.volumes) {
      throw new Error("Failed to load BTC data");
    }

    console.log(`✅ Loaded ${btcData.prices.length} days of BTC data`);
    return {
      btcPrices: btcData.prices,
      btcVolumes: btcData.volumes,
    };
  }

  /**
   * Train the agent for multiple episodes
   */
  private async trainAgent(): Promise<void> {
    console.log(
      `🎯 Training for ${RL_TRAINING_CONFIG.totalEpisodes} episodes...`
    );

    const startTime = Date.now();
    let bestEpisode = 0;

    for (
      let episode = 0;
      episode < RL_TRAINING_CONFIG.totalEpisodes;
      episode++
    ) {
      // Run episode
      const metrics = await this.agent.runEpisode();

      // Store training history
      this.trainingHistory.push({
        timestamp: new Date().toISOString(),
        ...metrics,
      });

      // Log progress
      if (episode % RL_TRAINING_CONFIG.logFrequency === 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const episodesPerSecond = (episode + 1) / elapsed;
        const eta =
          (RL_TRAINING_CONFIG.totalEpisodes - episode - 1) / episodesPerSecond;

        console.log(
          `📊 Episode ${episode + 1}/${RL_TRAINING_CONFIG.totalEpisodes} ` +
            `(${episodesPerSecond.toFixed(2)} eps/s, ETA: ${eta.toFixed(
              0
            )}s) - ` +
            `Return: ${(metrics.totalReturn * 100).toFixed(2)}% - ` +
            `Sharpe: ${metrics.sharpeRatio.toFixed(3)} - ` +
            `Epsilon: ${metrics.epsilon.toFixed(3)} - ` +
            `Trades: ${metrics.totalTrades}`
        );
      }

      // Check for improvement
      if (
        metrics.sharpeRatio >
        this.bestSharpeRatio + RL_TRAINING_CONFIG.minImprovement
      ) {
        this.bestSharpeRatio = metrics.sharpeRatio;
        bestEpisode = episode;
        this.patienceCounter = 0;

        console.log(
          `🏆 New best Sharpe ratio: ${metrics.sharpeRatio.toFixed(3)} ` +
            `(Episode ${episode + 1})`
        );

        // Save best model
        await this.saveModel("best");
      } else {
        this.patienceCounter++;
      }

      // Save model periodically
      if (episode % RL_TRAINING_CONFIG.saveFrequency === 0 && episode > 0) {
        await this.saveModel(`checkpoint_${episode}`);
      }

      // Early stopping
      if (this.patienceCounter >= RL_TRAINING_CONFIG.earlyStoppingPatience) {
        console.log(
          `⏹️ Early stopping triggered after ${episode + 1} episodes ` +
            `(no improvement for ${RL_TRAINING_CONFIG.earlyStoppingPatience} episodes)`
        );
        break;
      }
    }

    const totalTime = (Date.now() - startTime) / 1000;
    console.log(
      `✅ Training completed in ${totalTime.toFixed(2)}s ` +
        `(${(RL_TRAINING_CONFIG.totalEpisodes / totalTime).toFixed(
          2
        )} episodes/s)`
    );
    console.log(
      `🏆 Best Sharpe ratio: ${this.bestSharpeRatio.toFixed(3)} (Episode ${
        bestEpisode + 1
      })`
    );
  }

  /**
   * Evaluate the trained agent
   */
  private async evaluateAgent(): Promise<void> {
    console.log("🔍 Evaluating trained agent...");

    // Create evaluation environment with different data split
    const { btcPrices, btcVolumes } = await this.loadData();

    // Use last 20% of data for evaluation
    const evalStartIndex = Math.floor(btcPrices.length * 0.8);
    const evalPrices = btcPrices.slice(evalStartIndex);
    const evalVolumes = btcVolumes.slice(evalStartIndex);

    const evalEnvironment = new RLTradingEnvironment(
      evalPrices,
      evalVolumes,
      RL_ENV_CONFIG
    );

    // Evaluate agent
    const evalResults = await this.agent.evaluate(
      evalEnvironment,
      RL_TRAINING_CONFIG.evaluationEpisodes
    );

    // Calculate average metrics
    const avgMetrics = this.calculateAverageMetrics(evalResults);

    console.log("📊 Evaluation Results:");
    console.log(
      `  Average Return: ${(avgMetrics.totalReturn * 100).toFixed(2)}%`
    );
    console.log(`  Average Sharpe Ratio: ${avgMetrics.sharpeRatio.toFixed(3)}`);
    console.log(
      `  Average Max Drawdown: ${(avgMetrics.maxDrawdown * 100).toFixed(2)}%`
    );
    console.log(
      `  Average Win Rate: ${(avgMetrics.winRate * 100).toFixed(2)}%`
    );
    console.log(`  Average Total Trades: ${avgMetrics.totalTrades.toFixed(1)}`);

    // Save evaluation results
    await this.saveEvaluationResults(evalResults, avgMetrics);
  }

  /**
   * Calculate average metrics from evaluation results
   */
  private calculateAverageMetrics(results: any[]): any {
    const metrics = {
      totalReturn: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      totalTrades: 0,
      totalReward: 0,
      averageReward: 0,
    };

    results.forEach((result) => {
      metrics.totalReturn += result.totalReturn;
      metrics.sharpeRatio += result.sharpeRatio;
      metrics.maxDrawdown += result.maxDrawdown;
      metrics.winRate += result.winRate;
      metrics.totalTrades += result.totalTrades;
      metrics.totalReward += result.totalReward;
      metrics.averageReward += result.averageReward;
    });

    const count = results.length;
    metrics.totalReturn /= count;
    metrics.sharpeRatio /= count;
    metrics.maxDrawdown /= count;
    metrics.winRate /= count;
    metrics.totalTrades /= count;
    metrics.totalReward /= count;
    metrics.averageReward /= count;

    return metrics;
  }

  /**
   * Save model to Firebase Storage
   */
  private async saveModel(suffix: string): Promise<void> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const modelPath = `models/rl_agent_${suffix}_${timestamp}`;

      // Save model locally first
      const localPath = `/tmp/rl_model_${suffix}`;
      await this.agent.saveModel(localPath);

      // Upload to Firebase Storage
      const files = [`${localPath}/model.json`, `${localPath}/weights.bin`];

      for (const file of files) {
        const fileName = file.split("/").pop();
        const destination = `${modelPath}/${fileName}`;

        await this.bucket.upload(file, {
          destination,
          metadata: {
            contentType: fileName?.endsWith(".json")
              ? "application/json"
              : "application/octet-stream",
          },
        });
      }

      console.log(`💾 Model saved: ${modelPath}`);

      // Save training metadata
      const metadata = {
        timestamp: new Date().toISOString(),
        modelPath,
        environmentConfig: RL_ENV_CONFIG,
        agentConfig: RL_AGENT_CONFIG,
        trainingMetrics: this.agent.getTrainingMetrics(),
        bestSharpeRatio: this.bestSharpeRatio,
        trainingHistory: this.trainingHistory,
      };

      const metadataPath = `models/rl_agent_${suffix}_metadata_${timestamp}.json`;
      const metadataFile = this.bucket.file(metadataPath);

      await metadataFile.save(JSON.stringify(metadata, null, 2), {
        metadata: {
          contentType: "application/json",
        },
      });

      console.log(`📋 Metadata saved: ${metadataPath}`);
    } catch (error) {
      console.error("❌ Failed to save model:", error);
    }
  }

  /**
   * Save evaluation results
   */
  private async saveEvaluationResults(
    evalResults: any[],
    avgMetrics: any
  ): Promise<void> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const resultsPath = `evaluation/rl_agent_evaluation_${timestamp}.json`;

      const evaluationData = {
        timestamp: new Date().toISOString(),
        evaluationEpisodes: RL_TRAINING_CONFIG.evaluationEpisodes,
        averageMetrics: avgMetrics,
        individualResults: evalResults,
        environmentConfig: RL_ENV_CONFIG,
        agentConfig: RL_AGENT_CONFIG,
      };

      const resultsFile = this.bucket.file(resultsPath);
      await resultsFile.save(JSON.stringify(evaluationData, null, 2), {
        metadata: {
          contentType: "application/json",
        },
      });

      console.log(`📊 Evaluation results saved: ${resultsPath}`);
    } catch (error) {
      console.error("❌ Failed to save evaluation results:", error);
    }
  }

  /**
   * Get training summary
   */
  public getTrainingSummary(): any {
    const metrics = this.agent.getTrainingMetrics();
    const latestMetrics = metrics[metrics.length - 1];

    return {
      totalEpisodes: metrics.length,
      bestSharpeRatio: this.bestSharpeRatio,
      finalEpsilon: this.agent.getEpsilon(),
      latestMetrics,
      trainingHistory: this.trainingHistory,
    };
  }

  /**
   * Clean up resources
   */
  public dispose(): void {
    if (this.agent) {
      this.agent.dispose();
    }
  }
}

// Main execution function
export async function main(): Promise<void> {
  const trainer = new RLTrainer();

  try {
    await trainer.train();
    const summary = trainer.getTrainingSummary();
    console.log("🎉 Training Summary:", JSON.stringify(summary, null, 2));
  } catch (error) {
    console.error("❌ Training failed:", error);
    throw error;
  } finally {
    trainer.dispose();
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}
