import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";

async function runTraining() {
  const trainer = new TradeModelTrainer();

  try {
    console.log("\n🚀 EXPERIMENT v1.4.0: ADVANCED FEATURE ENGINEERING");
    console.log(
      "🎯 BUILDING ON: v1.3.0 breakthrough (59.35% balanced accuracy, MCC 0.2242)"
    );
    console.log(
      "📊 ARCHITECTURE: Conv1D(48,3) → BN → LSTM(72) → Dense(32) → Output(2)"
    );
    console.log(
      "⚙️ CHANGE: Enhanced technical indicators (multi-timeframe RSI, volume momentum)"
    );
    console.log("🔮 PREDICTION: 7-day ahead buy/sell signals");
    console.log(
      "🔬 HYPOTHESIS: Advanced features improve sell predictions while maintaining buy excellence"
    );
    console.log(
      "🎯 TARGET: Buy F1 >0.67, Sell F1 >0.55, Balanced accuracy >60%\n"
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

    console.log("\n📈 EXPERIMENT v1.4.0 ANALYSIS:");
    console.log("✅ Compare balanced accuracy vs v1.3.0: 59.35%");
    console.log("✅ Monitor MCC improvement from v1.3.0: 0.2242");
    console.log("✅ Target: Buy F1 maintain >0.67, Sell F1 improve >0.55");
    console.log("✅ Watch for feature dilution with enhanced indicator set");
    console.log(
      "💡 Multi-timeframe RSI should capture different momentum patterns"
    );
    console.log(
      "💡 Volume momentum should help identify sell pressure patterns"
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
