import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../../constants";

/**
 * CyclicalLRCallback (triangular policy, epoch-based)
 * LR cycles between baseLr and maxLr over 2*stepSize epochs.
 */
export class CyclicalLRCallback extends tf.CustomCallback {
  private model: tf.LayersModel | null = null;
  private readonly baseLr: number;
  private readonly maxLr: number;
  private readonly stepSize: number;

  constructor(options?: {
    baseLr?: number;
    maxLr?: number;
    stepSize?: number; // epochs from base->max (half-cycle)
  }) {
    super({});
    this.baseLr = options?.baseLr ?? TRAINING_CONFIG.WARMUP_INITIAL_LR;
    this.maxLr = options?.maxLr ?? TRAINING_CONFIG.INITIAL_LEARNING_RATE;
    this.stepSize = Math.max(
      1,
      options?.stepSize ?? TRAINING_CONFIG.CYCLIC_LR_STEP_SIZE
    );
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  private triangularLr(epoch: number): number {
    const cycle = Math.floor(1 + epoch / (2 * this.stepSize));
    const x = Math.abs(epoch / this.stepSize - 2 * cycle + 1);
    const scale = Math.max(0, 1 - x);
    const lr = this.baseLr + (this.maxLr - this.baseLr) * scale;
    return Math.max(TRAINING_CONFIG.MIN_LEARNING_RATE, lr);
  }

  private recompileWithLr(lr: number): void {
    if (!this.model) throw new Error("Model not set in CyclicalLRCallback");
    const newOptimizer = tf.train.adam(lr);
    this.model.compile({
      optimizer: newOptimizer,
      loss: this.model.loss,
      metrics: this.model.metrics as string[],
    });
  }

  async onEpochEnd(epoch: number) {
    const lr = this.triangularLr(epoch + 1); // advance for next epoch
    this.recompileWithLr(lr);
    console.log(
      `CyclicalLR: set learning rate to ${lr.toExponential(6)} for next epoch`
    );
  }
}

export default CyclicalLRCallback;
