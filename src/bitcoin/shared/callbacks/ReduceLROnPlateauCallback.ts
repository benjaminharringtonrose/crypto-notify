import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../../constants";

/**
 * ReduceLROnPlateauCallback
 * - Monitors a validation metric and reduces LR when it plateaus
 * - Default monitor: val_loss (mode: min)
 */
export class ReduceLROnPlateauCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private readonly factor: number;
  private readonly patience: number;
  private readonly minDelta: number;
  private readonly minLr: number;
  private readonly monitor: string;
  private readonly mode: "min" | "max";
  private best: number | null = null;
  private wait: number = 0;
  private lastLr: number;

  constructor(options?: {
    factor?: number; // multiply LR by this factor on plateau
    patience?: number; // epochs to wait for improvement
    minDelta?: number; // minimum change to qualify as improvement
    minLr?: number; // lower bound for LR
    monitor?: string; // metric name to monitor
    mode?: "min" | "max"; // improvement direction
    initialLr?: number; // starting LR for tracking
  }) {
    super({});
    this.factor = options?.factor ?? 0.5;
    this.patience = options?.patience ?? 3;
    this.minDelta = options?.minDelta ?? 1e-4;
    this.minLr = options?.minLr ?? TRAINING_CONFIG.MIN_LEARNING_RATE;
    this.monitor = options?.monitor ?? "val_loss";
    this.mode = options?.mode ?? "min";
    this.lastLr = options?.initialLr ?? TRAINING_CONFIG.INITIAL_LEARNING_RATE;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  private isImproved(current: number, best: number | null): boolean {
    if (best === null) return true;
    if (this.mode === "min") return current < best - this.minDelta;
    return current > best + this.minDelta;
  }

  private getMonitoredValue(logs?: tf.Logs): number | null {
    if (!logs) return null;
    const v = logs[this.monitor];
    return typeof v === "number" ? v : null;
  }

  private recompileWithLr(lr: number): void {
    if (!this.model)
      throw new Error("Model not set in ReduceLROnPlateauCallback");
    const newOptimizer = tf.train.adam(lr);
    this.model.compile({
      optimizer: newOptimizer,
      loss: this.model.loss,
      metrics: this.model.metrics as string[],
    });
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    const current = this.getMonitoredValue(logs);
    if (current === null) return;

    if (this.isImproved(current, this.best)) {
      this.best = current;
      this.wait = 0;
      return;
    }

    this.wait++;
    if (this.wait >= this.patience) {
      const newLr = Math.max(this.minLr, this.lastLr * this.factor);
      if (newLr < this.lastLr - 1e-12) {
        this.lastLr = newLr;
        this.recompileWithLr(newLr);
        // reset wait after reduction
        this.wait = 0;
        console.log(
          `ReduceLROnPlateau: reducing learning rate to ${newLr.toExponential(
            6
          )} at epoch ${epoch + 1}`
        );
      }
    }
  }
}

export default ReduceLROnPlateauCallback;
