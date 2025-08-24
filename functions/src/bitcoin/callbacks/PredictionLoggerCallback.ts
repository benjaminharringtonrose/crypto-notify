import * as tf from "@tensorflow/tfjs-node";
import { TradeModelTrainer } from "../TradeModelTrainer";

export class PredictionLoggerCallback {
  private model: tf.LayersModel | null = null;
  private X_val: tf.Tensor;
  private y_val: tf.Tensor;
  private epochCount: number = 0;
  private bestBuyF1: number = 0;
  private bestBalancedAccuracy: number = 0;

  constructor(X_val: tf.Tensor, y_val: tf.Tensor, trainer: TradeModelTrainer) {
    this.X_val = X_val;
    this.y_val = y_val;
  }

  setModel(model: tf.LayersModel): void {
    this.model = model;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
    this.epochCount++;

    if (!this.model || !logs) return;

    try {
      // Get predictions
      const predictions = this.model.predict(this.X_val) as tf.Tensor;
      const predArray = (await predictions.array()) as number[][];
      const trueArray = (await this.y_val.array()) as number[][];

      // Calculate detailed metrics
      let buyCorrect = 0;
      let buyTotal = 0;
      let sellCorrect = 0;
      let sellTotal = 0;
      let buyPredictions = 0;
      let sellPredictions = 0;

      for (let i = 0; i < predArray.length; i++) {
        const pred = predArray[i];
        const true_label = trueArray[i];

        const predClass = pred[1] > pred[0] ? 1 : 0; // Buy if Buy prob > Sell prob
        const trueClass = true_label[1] > true_label[0] ? 1 : 0;

        if (trueClass === 1) {
          // Actual Buy
          buyTotal++;
          if (predClass === 1) buyCorrect++;
        } else {
          // Actual Sell
          sellTotal++;
          if (predClass === 0) sellCorrect++;
        }

        if (predClass === 1) buyPredictions++;
        else sellPredictions++;
      }

      // Calculate metrics
      const buyPrecision = buyPredictions > 0 ? buyCorrect / buyPredictions : 0;
      const buyRecall = buyTotal > 0 ? buyCorrect / buyTotal : 0;
      const sellPrecision =
        sellPredictions > 0 ? sellCorrect / sellPredictions : 0;
      const sellRecall = sellTotal > 0 ? sellCorrect / sellTotal : 0;

      const buyF1 =
        buyPrecision + buyRecall > 0
          ? (2 * (buyPrecision * buyRecall)) / (buyPrecision + buyRecall)
          : 0;
      const sellF1 =
        sellPrecision + sellRecall > 0
          ? (2 * (sellPrecision * sellRecall)) / (sellPrecision + sellRecall)
          : 0;

      const balancedAccuracy = (buyRecall + sellRecall) / 2;

      // Track best performance
      if (buyF1 > this.bestBuyF1) {
        this.bestBuyF1 = buyF1;
      }
      if (balancedAccuracy > this.bestBalancedAccuracy) {
        this.bestBalancedAccuracy = balancedAccuracy;
      }

      // Log detailed metrics every 10 epochs or when Buy F1 improves
      if (this.epochCount % 10 === 0 || buyF1 > this.bestBuyF1 * 0.95) {
        console.log(`\n=== Epoch ${epoch} Detailed Metrics ===`);
        console.log(
          `Buy Precision: ${buyPrecision.toFixed(
            4
          )}, Recall: ${buyRecall.toFixed(4)}, F1: ${buyF1.toFixed(4)}`
        );
        console.log(
          `Sell Precision: ${sellPrecision.toFixed(
            4
          )}, Recall: ${sellRecall.toFixed(4)}, F1: ${sellF1.toFixed(4)}`
        );
        console.log(`Balanced Accuracy: ${balancedAccuracy.toFixed(4)}`);
        console.log(
          `Buy Predictions: ${buyPredictions}/${predArray.length} (${(
            (buyPredictions / predArray.length) *
            100
          ).toFixed(1)}%)`
        );
        console.log(
          `Best Buy F1: ${this.bestBuyF1.toFixed(
            4
          )}, Best Balanced Acc: ${this.bestBalancedAccuracy.toFixed(4)}`
        );
        console.log(`Validation Loss: ${logs.val_loss?.toFixed(4)}`);
        console.log(`========================================\n`);
      }

      // Clean up tensors
      predictions.dispose();
    } catch (error) {
      console.warn("Error in PredictionLoggerCallback:", error);
    }
  }
}
