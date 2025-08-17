import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🚀 EXPERIMENT v1.3.0: TIMESTEP OPTIMIZATION");
    console.log(
      "🎯 BUILDING ON: v1.2.0 success (58.97% balanced accuracy, MCC 0.1795)"
    );
    console.log(
      "📊 ARCHITECTURE: Conv1D(48,3) → BN → LSTM(72) → Dense(32) → Output(2)"
    );
    console.log("⚙️ CHANGE: Timesteps 30 → 35 (+16.7% history, ~5 weeks)");
    console.log("🔮 PREDICTION: 7-day ahead buy/sell signals");
    console.log(
      "🔬 HYPOTHESIS: 5 weeks of history captures monthly cycles for better 7-day predictions"
    );
    console.log("🎯 TARGET: Balanced accuracy >60% and MCC >0.18\n");

    const startTime = Date.now();

    await trainer.train();

    const endTime = Date.now();
    const trainingTime = (endTime - startTime) / 1000;

    console.log(
      `\n⏱️  Training completed in: ${trainingTime.toFixed(2)} seconds`
    );

    const bestThreshold = trainer.getBestThreshold();
    console.log(`🎯 Best validation threshold: ${bestThreshold.toFixed(4)}`);

    console.log("\n📈 EXPERIMENT v1.3.0 ANALYSIS:");
    console.log("✅ Compare balanced accuracy vs v1.2.0: 58.97%");
    console.log("✅ Monitor MCC improvement from v1.2.0: 0.1795");
    console.log("✅ Target: Both Buy & Sell F1 >0.58 (balanced excellence)");
    console.log("✅ Watch training time (35 timesteps = +16.7% computation)");
    console.log("💡 5 weeks of history should capture monthly market cycles");
    console.log(
      "💡 Better context for 7-day predictions without overfitting risk"
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
