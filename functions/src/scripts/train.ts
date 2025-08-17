import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🧪 EXPERIMENT NEXT-F: Advanced Regularization");
    console.log("🔬 TESTING: L2 → L1+L2 regularization (0.001 + 0.0005)");
    console.log("💡 HYPOTHESIS: Combined regularization reduces overfitting");
    console.log("📊 BASELINE: NEXT-E combined score 1.0176");
    console.log("🎯 TARGET: >1.05 combined score with better generalization\n");

    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log(`⏱️  Total training time: ${trainingTime.toFixed(2)} seconds`);

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 EXPERIMENT NEXT-F RESULTS:");
    console.log("NEXT-E (Batch 32): 1.0176 combined score (SUCCESS! ✅)");
    console.log("NEXT-F (L1+L2 reg): Analyzing regularization improvement...");
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
