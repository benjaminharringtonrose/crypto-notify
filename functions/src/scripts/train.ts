import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import * as tf from "@tensorflow/tfjs-node";

interface TrainingResult {
  run: number;
  seed: number;
  balancedAccuracy: number;
  buyF1: number;
  sellF1: number;
  combinedF1: number;
  matthewsCorrelation: number;
  bestThreshold: number;
  trainingTime: number;
  totalEpochs: number;
}

interface TrainingStats {
  metric: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  confidenceInterval: [number, number];
}

function calculateStats(values: number[]): TrainingStats {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...values);
  const max = Math.max(...values);

  // 95% confidence interval assuming normal distribution
  const marginOfError = 1.96 * (std / Math.sqrt(values.length));
  const confidenceInterval: [number, number] = [
    mean - marginOfError,
    mean + marginOfError,
  ];

  return {
    metric: "",
    mean,
    std,
    min,
    max,
    confidenceInterval,
  };
}

function printStats(label: string, stats: TrainingStats, unit: string = "") {
  console.log(`${label}:`);
  console.log(
    `  Mean: ${stats.mean.toFixed(4)}${unit} ± ${stats.std.toFixed(4)}`
  );
  console.log(
    `  Range: [${stats.min.toFixed(4)}, ${stats.max.toFixed(4)}]${unit}`
  );
  console.log(
    `  95% CI: [${stats.confidenceInterval[0].toFixed(
      4
    )}, ${stats.confidenceInterval[1].toFixed(4)}]${unit}`
  );
}

async function runMultipleTrainingSessions() {
  const NUM_RUNS = 5; // Number of training runs for statistical analysis
  const RANDOM_SEEDS = [42, 123, 456, 789, 999]; // Fixed seeds for reproducibility
  const results: TrainingResult[] = [];

  try {
    console.log("\n🎯 MULTI-RUN TRAINING ANALYSIS");
    console.log(
      "🔬 STATISTICAL VALIDATION: Running multiple training sessions"
    );
    console.log(
      `📊 CONFIGURATION: ${NUM_RUNS} runs with different random seeds`
    );
    console.log("🎲 SEEDS:", RANDOM_SEEDS.join(", "));
    console.log(
      "⚙️ PURPOSE: Calculate mean ± std dev for robust performance assessment"
    );
    console.log(
      "📈 ARCHITECTURE: Conv1D(48,3) → BN → LSTM(64) → Dense(32) → Output(2)"
    );
    console.log("🔮 FEATURES: 36 advanced market microstructure indicators");
    console.log("📅 DATA: 730 days (2 years), 35 epochs training\n");

    const overallStartTime = Date.now();

    for (let i = 0; i < NUM_RUNS; i++) {
      const seed = RANDOM_SEEDS[i];

      console.log(`\n${"=".repeat(60)}`);
      console.log(`🏃‍♂️ TRAINING RUN ${i + 1}/${NUM_RUNS} (Seed: ${seed})`);
      console.log(`${"=".repeat(60)}`);

      // Set deterministic seed for this run
      tf.randomUniform([1, 1], 0, 1, "float32", seed);

      const trainer = new TradeModelTrainer(seed);
      const startTime = Date.now();

      try {
        await trainer.train();

        const endTime = Date.now();
        const trainingTime = (endTime - startTime) / 1000;

        // Extract metrics from trainer
        const result: TrainingResult = {
          run: i + 1,
          seed: seed,
          balancedAccuracy: trainer.getBalancedAccuracy(),
          buyF1: trainer.getBuyF1(),
          sellF1: trainer.getSellF1(),
          combinedF1: trainer.getCombinedF1(),
          matthewsCorrelation: trainer.getMatthewsCorrelation(),
          bestThreshold: trainer.getBestThreshold(),
          trainingTime: trainingTime,
          totalEpochs: trainer.getFinalMetrics().finalEpoch,
        };

        results.push(result);

        console.log(`✅ Run ${i + 1} completed in ${trainingTime.toFixed(2)}s`);
        console.log(`🎯 Threshold: ${result.bestThreshold.toFixed(4)}`);
      } catch (error) {
        console.error(`❌ Run ${i + 1} failed:`, error);
        // Continue with remaining runs
      }

      // Small delay between runs
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    const overallEndTime = Date.now();
    const totalTime = (overallEndTime - overallStartTime) / 1000;

    // Calculate and display statistics
    console.log(`\n${"=".repeat(80)}`);
    console.log("📊 STATISTICAL ANALYSIS RESULTS");
    console.log(`${"=".repeat(80)}`);

    if (results.length === 0) {
      console.log("❌ No successful training runs to analyze");
      return;
    }

    console.log(`✅ Successful runs: ${results.length}/${NUM_RUNS}`);
    console.log(`⏱️  Total time: ${(totalTime / 60).toFixed(1)} minutes\n`);

    // Calculate statistics for each key metric
    console.log("🔍 PERFORMANCE METRICS ANALYSIS:");
    printStats(
      "📊 Balanced Accuracy",
      calculateStats(results.map((r) => r.balancedAccuracy)),
      "%"
    );
    printStats("🟢 Buy F1 Score", calculateStats(results.map((r) => r.buyF1)));
    printStats(
      "🔴 Sell F1 Score",
      calculateStats(results.map((r) => r.sellF1))
    );
    printStats(
      "⚖️  Combined F1 Score",
      calculateStats(results.map((r) => r.combinedF1))
    );
    printStats(
      "📈 Matthews Correlation",
      calculateStats(results.map((r) => r.matthewsCorrelation))
    );

    console.log("\n🔧 TRAINING PARAMETERS:");
    printStats(
      "🎯 Best Threshold",
      calculateStats(results.map((r) => r.bestThreshold))
    );
    printStats(
      "⏱️  Training Time",
      calculateStats(results.map((r) => r.trainingTime)),
      "s"
    );
    printStats(
      "🔄 Total Epochs",
      calculateStats(results.map((r) => r.totalEpochs))
    );

    console.log(`\n📋 DETAILED RESULTS TABLE:`);
    console.log(
      "Run | Seed | Bal.Acc | Buy F1 | Sell F1 | Comb F1 | MCC    | Thresh | Epochs | Time(s)"
    );
    console.log("-".repeat(90));
    results.forEach((r) => {
      console.log(
        `${r.run.toString().padStart(3)} | ` +
          `${r.seed.toString().padStart(4)} | ` +
          `${(r.balancedAccuracy * 100).toFixed(1).padStart(6)}% | ` +
          `${r.buyF1.toFixed(3).padStart(6)} | ` +
          `${r.sellF1.toFixed(3).padStart(7)} | ` +
          `${r.combinedF1.toFixed(3).padStart(7)} | ` +
          `${r.matthewsCorrelation.toFixed(3).padStart(6)} | ` +
          `${r.bestThreshold.toFixed(4).padStart(6)} | ` +
          `${r.totalEpochs.toString().padStart(6)} | ` +
          `${r.trainingTime.toFixed(1).padStart(6)}`
      );
    });

    console.log(`\n🎯 TRADING CONFIDENCE ASSESSMENT:`);
    const thresholdStats = calculateStats(results.map((r) => r.bestThreshold));
    const timeStats = calculateStats(results.map((r) => r.trainingTime));

    if (thresholdStats.std < 0.05) {
      console.log(
        "✅ HIGH CONFIDENCE: Low threshold variance indicates stable model"
      );
    } else if (thresholdStats.std < 0.1) {
      console.log("⚠️  MEDIUM CONFIDENCE: Moderate threshold variance");
    } else {
      console.log(
        "❌ LOW CONFIDENCE: High threshold variance indicates unstable model"
      );
    }

    // Enhanced confidence assessment with all metrics
    const balAccStats = calculateStats(results.map((r) => r.balancedAccuracy));
    const buyF1Stats = calculateStats(results.map((r) => r.buyF1));
    const sellF1Stats = calculateStats(results.map((r) => r.sellF1));
    const mccStats = calculateStats(results.map((r) => r.matthewsCorrelation));

    console.log(`\n💡 PRODUCTION RECOMMENDATIONS:`);
    console.log(`🎯 Trading Configuration:`);
    console.log(
      `   - Threshold: ${thresholdStats.mean.toFixed(
        4
      )} ± ${thresholdStats.std.toFixed(4)}`
    );
    console.log(
      `   - Expected balanced accuracy: ${(balAccStats.mean * 100).toFixed(
        1
      )}% ± ${(balAccStats.std * 100).toFixed(1)}%`
    );
    console.log(
      `   - Buy F1 performance: ${buyF1Stats.mean.toFixed(
        3
      )} ± ${buyF1Stats.std.toFixed(3)}`
    );
    console.log(
      `   - Sell F1 performance: ${sellF1Stats.mean.toFixed(
        3
      )} ± ${sellF1Stats.std.toFixed(3)}`
    );

    console.log(`\n📊 Model Quality Assessment:`);
    console.log(
      `   - Matthews Correlation: ${mccStats.mean.toFixed(
        3
      )} ± ${mccStats.std.toFixed(3)}`
    );
    console.log(
      `   - Training time: ${timeStats.mean.toFixed(
        1
      )}s ± ${timeStats.std.toFixed(1)}s`
    );

    const isStable = balAccStats.std < 0.03 && buyF1Stats.std < 0.05;
    console.log(
      `   - Model stability: ${
        isStable
          ? "EXCELLENT"
          : balAccStats.std < 0.05
          ? "GOOD"
          : "NEEDS IMPROVEMENT"
      }`
    );

    console.log(`\n🚀 Trading Strategy Implications:`);
    if (balAccStats.mean > 0.65) {
      console.log(
        `   ✅ STRONG MODEL: >65% balanced accuracy suggests profitable trading potential`
      );
    } else if (balAccStats.mean > 0.6) {
      console.log(
        `   ⚠️  MODERATE MODEL: 60-65% balanced accuracy - proceed with caution`
      );
    } else {
      console.log(
        `   ❌ WEAK MODEL: <60% balanced accuracy - not recommended for live trading`
      );
    }
  } catch (error) {
    console.error("\n❌ Multi-run training failed:", error);
    console.error(
      "Stack trace:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    process.exit(1);
  }
}

// Run the multi-session statistical training analysis
runMultipleTrainingSessions();
