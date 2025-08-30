import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../../constants";

interface EarlyStoppingOptions {
  monitor?:
    | "val_loss"
    | "val_customF1"
    | "val_customF1Buy"
    | "val_customF1Sell";
  patience?: number;
  restoreBestWeights?: boolean;
}

export class EarlyStoppingCallback extends tf.CustomCallback {
  private monitor:
    | "val_loss"
    | "val_customF1"
    | "val_customF1Buy"
    | "val_customF1Sell";
  private patience: number;
  private bestWeights: tf.Tensor[];
  private bestValue: number;
  private wait: number;
  private stoppedEpoch: number;
  private restoreBestWeights: boolean;
  private model?: tf.LayersModel;

  constructor(options: EarlyStoppingOptions = {}) {
    super({});
    this.monitor = options.monitor || "val_customF1Buy"; // Default to buy F1
    this.patience = options.patience || TRAINING_CONFIG.PATIENCE;
    this.restoreBestWeights = options.restoreBestWeights ?? true;
    this.bestWeights = [];
    this.bestValue = this.monitor === "val_loss" ? Infinity : -Infinity;
    this.wait = 0;
    this.stoppedEpoch = 0;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model) throw new Error("Model not set in EarlyStoppingCallback");
    if (!logs) return;

    let currentValue: number | undefined;
    switch (this.monitor) {
      case "val_loss":
        currentValue = logs["val_loss"];
        break;
      case "val_customF1":
        if (
          logs["val_customF1Buy"] !== undefined &&
          logs["val_customF1Sell"] !== undefined
        ) {
          currentValue =
            (logs["val_customF1Buy"] + logs["val_customF1Sell"]) / 2;
        } else {
          console.warn(
            "val_customF1 requires both val_customF1Buy and val_customF1Sell, but they are not found"
          );
        }
        break;
      case "val_customF1Buy":
        currentValue = logs["val_customF1Buy"];
        break;
      case "val_customF1Sell":
        currentValue = logs["val_customF1Sell"];
        break;
    }

    if (currentValue === undefined) {
      console.warn(
        `Metric ${
          this.monitor
        } not found in logs. Available keys: ${Object.keys(logs).join(", ")}`
      );
      return;
    }

    const isBetter =
      this.monitor === "val_loss"
        ? currentValue < this.bestValue
        : currentValue > this.bestValue;

    if (isBetter) {
      this.bestValue = currentValue;
      this.wait = 0;
      if (this.restoreBestWeights) {
        this.bestWeights = this.model.getWeights().map((w) => w.clone());
      }
      console.log(
        `New best ${this.monitor}: ${this.bestValue.toFixed(4)} at epoch ${
          epoch + 1
        }`
      );
    } else {
      this.wait += 1;
      console.log(
        `No improvement in ${this.monitor}. Wait: ${this.wait}/${this.patience}`
      );
      if (this.wait >= this.patience) {
        this.stoppedEpoch = epoch;
        this.model.stopTraining = true;
      }
    }
  }

  async onTrainEnd() {
    if (this.stoppedEpoch > 0 && this.restoreBestWeights && this.model) {
      console.log(
        `Training stopped at epoch ${
          this.stoppedEpoch + 1
        }. Restored weights from best ${this.monitor}: ${this.bestValue.toFixed(
          4
        )}`
      );
      this.model.setWeights(this.bestWeights);
      this.bestWeights.forEach((w) => w.dispose());
    }
  }
}
