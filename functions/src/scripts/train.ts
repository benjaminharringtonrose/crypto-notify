import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🧪 EXPERIMENT NEXT-A: Sequence Length Optimization");
    console.log("🔬 TESTING: Timesteps 24 → 30 (25% increase)");
    console.log("💡 HYPOTHESIS: Weekly crypto patterns need longer sequences");
    console.log("📊 BASELINE: 52.75% accuracy, 0.9830 combined score");
    console.log("🎯 TARGET: >55% accuracy with maintained class balance\n");

    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log(`⏱️  Total training time: ${trainingTime.toFixed(2)} seconds`);

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 EXPERIMENT NEXT-A RESULTS:");
    console.log("Baseline (24 timesteps): 52.75% accuracy, 0.9830 combined score");
    console.log("Current (30 timesteps): Analyzing performance change...");
  } catch (error) {
    console.error("\n❌ Training failed:", error);
    console.error(
      "Stack trace:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    process.exit(1);
  }
}

// Run the enhanced training
runTraining();
