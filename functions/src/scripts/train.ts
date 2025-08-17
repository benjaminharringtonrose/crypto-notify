import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🚀 EXPERIMENT 7DAY-1: CLASS BALANCE + LR OPTIMIZATION");
    console.log("🎯 IMPROVING: 7-day baseline with class balance optimization");
    console.log(
      "📊 ARCHITECTURE: Conv1D(48,3) → BN → LSTM(64) → Dense(32) → Output(2)"
    );
    console.log("⚙️ TRAINING: Alpha [0.45,0.55], Min LR 0.00005, 30 epochs");
    console.log(
      "🔮 PREDICTION: 7-day ahead buy/sell signals (established baseline)"
    );
    console.log(
      "🔬 HYPOTHESIS: Better class balance + higher LR floor improves 7-day predictions"
    );
    console.log(
      "🎯 TARGET: Improve balanced accuracy from 53.85% and MCC from 0.0789\n"
    );

    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log(
      `\n⏱️  Training completed in: ${trainingTime.toFixed(2)} seconds`
    );

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 EXPERIMENT 7DAY-1 ANALYSIS:");
    console.log("✅ Compare balanced accuracy vs baseline 53.85%");
    console.log("✅ Monitor MCC improvement from baseline 0.0789");
    console.log("✅ Check F1 score balance (baseline: 0.5555/0.5279)");
    console.log("✅ Verify class balance remains stable");
    console.log("💡 Focal loss Alpha [0.45,0.55] should improve class balance");
    console.log(
      "💡 Higher min LR should prevent over-conservative convergence"
    );
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
