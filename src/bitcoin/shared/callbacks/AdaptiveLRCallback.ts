import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../../constants";

/**
 * AdaptiveLRCallback
 * - Adjusts learning rate based on training progress and validation metrics
 * - Increases LR when validation improves, decreases when it plateaus
 * - Includes momentum to smooth out LR changes
 */
export class AdaptiveLRCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private readonly baseLr: number;
  private readonly maxLr: number;
  private readonly minLr: number;
  private readonly increaseFactor: number;
  private readonly decreaseFactor: number;
  private readonly patience: number;
  private readonly minDelta: number;
  private readonly monitor: string;
  private readonly mode: "min" | "max";

  private best: number | null = null;
  private wait: number = 0;
  private currentLr: number;
  private lrMomentum: number = 0.0;

  constructor(options?: {
    baseLr?: number;
    maxLr?: number;
    minLr?: number;
    increaseFactor?: number;
    decreaseFactor?: number;
    patience?: number;
    minDelta?: number;
    monitor?: string;
    mode?: "min" | "max";
  }) {
    super({});
    this.baseLr = options?.baseLr ?? TRAINING_CONFIG.WARMUP_INITIAL_LR;
    this.maxLr = options?.maxLr ?? TRAINING_CONFIG.INITIAL_LEARNING_RATE;
    this.minLr = options?.minLr ?? TRAINING_CONFIG.MIN_LEARNING_RATE;
    this.increaseFactor = options?.increaseFactor ?? 1.1;
    this.decreaseFactor = options?.decreaseFactor ?? 0.8;
    this.patience = options?.patience ?? 3;
    this.minDelta = options?.minDelta ?? 1e-4;
    this.monitor = options?.monitor ?? "val_loss";
    this.mode = options?.mode ?? "min";
    this.currentLr = this.baseLr;
  }

  setModel(model: tf.LayersModel): void {
    this.model = model;
  }

  private updateLearningRate(epoch: number, logs?: tf.Logs): void {
    if (!this.model || !logs) return;

    const current = logs[this.monitor] as number;
    if (typeof current !== "number") return;

    // Initialize best value
    if (this.best === null) {
      this.best = current;
      console.log(`🎯 Initial ${this.monitor}: ${current.toFixed(4)}`);
      return;
    }

    // Check if current is better (use very lenient threshold)
    const improvementThreshold =
      this.mode === "min" ? -this.minDelta : this.minDelta;
    const isImproving =
      this.mode === "min"
        ? current < this.best + improvementThreshold
        : current > this.best + improvementThreshold;

    if (isImproving) {
      this.best = current;
      this.wait = 0;

      // Increase LR more aggressively
      const lrIncrease = this.currentLr * (this.increaseFactor - 1);
      this.currentLr = Math.min(this.maxLr, this.currentLr + lrIncrease);
      this.lrMomentum = Math.min(0.3, this.lrMomentum + 0.05);

      console.log(
        `📈 LR increased to ${this.currentLr.toFixed(6)} (${
          this.monitor
        }: ${current.toFixed(4)} → ${this.best.toFixed(4)})`
      );
    } else {
      this.wait++;

      // Only decrease LR if we've been stuck for a long time AND we're not near minimum
      if (this.wait >= this.patience && this.currentLr > this.minLr * 2) {
        const lrDecrease = this.currentLr * (1 - this.decreaseFactor);
        this.currentLr = Math.max(this.minLr, this.currentLr - lrDecrease);
        this.lrMomentum = Math.max(-0.2, this.lrMomentum - 0.02);
        this.wait = 0;

        console.log(
          `📉 LR decreased to ${this.currentLr.toFixed(6)} (plateau: ${
            this.wait
          } epochs, ${this.monitor}: ${current.toFixed(4)})`
        );
      } else {
        console.log(
          `⏳ Plateau wait: ${this.wait}/${this.patience} (${
            this.monitor
          }: ${current.toFixed(4)}, LR: ${this.currentLr.toFixed(6)})`
        );
      }
    }

    // Apply new learning rate by recompiling optimizer
    const currentOptimizer = this.model.optimizer;
    this.model.compile({
      optimizer: currentOptimizer,
      loss: this.model.loss,
      metrics: this.model.metrics,
    });

    // Set learning rate on the optimizer
    if ("setLearningRate" in currentOptimizer) {
      (currentOptimizer as any).setLearningRate(this.currentLr);
    }
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
    this.updateLearningRate(epoch, logs);
  }

  getCurrentLR(): number {
    return this.currentLr;
  }

  getBestMetric(): number | null {
    return this.best;
  }
}
