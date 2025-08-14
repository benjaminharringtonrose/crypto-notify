import * as tf from "@tensorflow/tfjs-node";
import { TRAINING_CONFIG } from "../../constants";
import { TradeModelTrainer } from "../TradeModelTrainer";
import { Metrics } from "../Metrics";

interface EpochData {
  epoch: number;
  trainAcc: number;
  valAcc: number;
  valLoss: number;
  accSpread: number;
  valF1Buy: number;
  valF1Sell: number;
  combinedScore: number;
  learningRate: number;
  difficulty: number;
  effectiveWeight: number;
  buyCount: number;
  sellCount: number;
  buyRatio: number;
  precisionBuy: number;
  precisionSell: number;
  recallBuy: number;
  recallSell: number;
  gradientNorm?: number;
}

export class TrainingLoggerCallback extends tf.CustomCallback {
  private model?: tf.LayersModel;
  private trainer: TradeModelTrainer;
  private validationFeatures: tf.Tensor;
  private validationLabels: tf.Tensor;
  private epochData: EpochData[] = [];
  private headerPrinted = false;
  private lastGradientNorm?: number;

  constructor(
    validationFeatures: tf.Tensor,
    validationLabels: tf.Tensor,
    trainer: TradeModelTrainer
  ) {
    super({});
    this.validationFeatures = validationFeatures;
    this.validationLabels = validationLabels;
    this.trainer = trainer;
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }

  setGradientNorm(norm: number) {
    this.lastGradientNorm = norm;
  }

  private printHeader() {
    if (this.headerPrinted) return;

    console.log("\n" + "=".repeat(120));
    console.log("TRAINING PROGRESS TABLE");
    console.log("=".repeat(120));
    this.headerPrinted = true;
  }

  private formatNumber(num: number, decimals: number = 4): string {
    return num.toFixed(decimals);
  }

  private async calculateValidationMetrics(): Promise<{
    buyCount: number;
    sellCount: number;
    buyRatio: number;
    precisionBuy: number;
    precisionSell: number;
    recallBuy: number;
    recallSell: number;
    effectiveWeight: number;
  }> {
    if (!this.model) {
      throw new Error("Model not set in TrainingLoggerCallback");
    }

    const predsAllVal = this.model.predict(
      this.validationFeatures
    ) as tf.Tensor;
    const predArray = (await predsAllVal.array()) as number[][];
    const bestThreshold = this.trainer.getBestThreshold();
    const predLabelsAllVal = predArray.map((p) =>
      p[1] > bestThreshold ? 1 : 0
    );
    const yVal = Array.from(
      await this.validationLabels.argMax(-1).data()
    ) as number[];

    const buyCount = predLabelsAllVal.filter((p) => p === 1).length;
    const totalValSamples = predLabelsAllVal.length;
    const sellCount = totalValSamples - buyCount;
    const buyRatio = buyCount / totalValSamples;

    const metrics = Metrics.calculateMetrics(predLabelsAllVal, yVal);

    // Calculate effective weight
    const alpha = tf.tensor1d(TRAINING_CONFIG.ALPHA);
    const yTrue = this.validationLabels;
    const pt = yTrue.mul(predsAllVal).sum(-1).clipByValue(0, 1);
    const focalWeight = tf.pow(tf.sub(1, pt), TRAINING_CONFIG.GAMMA);
    const yTrueIndices = yTrue.argMax(-1);
    const alphaWeighted = tf.gather(alpha, yTrueIndices);
    const effectiveWeight = focalWeight.mul(alphaWeighted).mean().dataSync()[0];

    // Cleanup tensors
    predsAllVal.dispose();
    alpha.dispose();
    pt.dispose();
    focalWeight.dispose();
    alphaWeighted.dispose();

    return {
      buyCount,
      sellCount,
      buyRatio,
      precisionBuy: metrics.precisionBuy,
      precisionSell: metrics.precisionSell,
      recallBuy: metrics.recallBuy,
      recallSell: metrics.recallSell,
      effectiveWeight,
    };
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!logs) return;

    const trainAcc = logs["binaryAccuracy"] || 0;
    const valAcc = logs["val_binaryAccuracy"] || 0;
    const valLoss = logs["val_loss"] || Infinity;
    const accSpread = trainAcc - valAcc;
    const valF1Buy = logs["val_customF1Buy"] || 0;
    const valF1Sell = logs["val_customF1Sell"] || 0;
    const combinedScore = valF1Buy + valF1Sell - valLoss * 0.1;

    // Calculate learning rate
    const lr = this.getLearningRate(epoch);

    // Calculate difficulty (curriculum learning)
    const difficulty = this.getDifficultyLevel(epoch);

    // Calculate validation metrics
    const validationMetrics = await this.calculateValidationMetrics();

    // Store epoch data
    const epochData: EpochData = {
      epoch: epoch + 1,
      trainAcc,
      valAcc,
      valLoss,
      accSpread,
      valF1Buy,
      valF1Sell,
      combinedScore,
      learningRate: lr,
      difficulty,
      effectiveWeight: validationMetrics.effectiveWeight,
      buyCount: validationMetrics.buyCount,
      sellCount: validationMetrics.sellCount,
      buyRatio: validationMetrics.buyRatio,
      precisionBuy: validationMetrics.precisionBuy,
      precisionSell: validationMetrics.precisionSell,
      recallBuy: validationMetrics.recallBuy,
      recallSell: validationMetrics.recallSell,
      gradientNorm: this.lastGradientNorm,
    };

    this.epochData.push(epochData);

    // Print header on first epoch
    this.printHeader();

    // Print epoch metrics in a clean, readable format
    console.log(`\nEpoch ${epochData.epoch}:`);
    console.log("─".repeat(50));

    // Performance Metrics
    console.log(`📊 Performance:`);
    console.log(`   Train Accuracy: ${this.formatNumber(epochData.trainAcc)}`);
    console.log(`   Val Accuracy:   ${this.formatNumber(epochData.valAcc)}`);
    console.log(`   Val Loss:       ${this.formatNumber(epochData.valLoss)}`);
    console.log(
      `   Combined Score: ${this.formatNumber(epochData.combinedScore)}`
    );

    // F1 Scores
    console.log(`\n🎯 F1 Scores:`);
    console.log(`   Buy F1:  ${this.formatNumber(epochData.valF1Buy)}`);
    console.log(`   Sell F1: ${this.formatNumber(epochData.valF1Sell)}`);

    // Precision & Recall
    console.log(`\n📈 Precision/Recall:`);
    console.log(
      `   Buy:  ${this.formatNumber(
        epochData.precisionBuy
      )} / ${this.formatNumber(epochData.recallBuy)}`
    );
    console.log(
      `   Sell: ${this.formatNumber(
        epochData.precisionSell
      )} / ${this.formatNumber(epochData.recallSell)}`
    );

    // Training Info
    console.log(`\n⚙️  Training:`);
    console.log(
      `   Learning Rate: ${this.formatNumber(epochData.learningRate, 6)}`
    );
    console.log(
      `   Difficulty:    ${this.formatNumber(epochData.difficulty * 100, 1)}%`
    );
    console.log(
      `   Buy/Sell:      ${epochData.buyCount}/${epochData.sellCount}`
    );

    if (epochData.gradientNorm) {
      console.log(
        `   Grad Norm:     ${this.formatNumber(epochData.gradientNorm)}`
      );
    }

    console.log("─".repeat(50));

    // Clear gradient norm for next epoch
    this.lastGradientNorm = undefined;
  }

  private getLearningRate(epoch: number): number {
    const initialLR = TRAINING_CONFIG.INITIAL_LEARNING_RATE;
    const decayRate = TRAINING_CONFIG.LR_DECAY_RATE;
    const warmupEpochs = TRAINING_CONFIG.WARMUP_EPOCHS;
    const warmupInitialLR = TRAINING_CONFIG.WARMUP_INITIAL_LR;

    if (epoch < warmupEpochs) {
      const slope = (initialLR - warmupInitialLR) / warmupEpochs;
      return warmupInitialLR + slope * epoch;
    }
    const decayEpochs = epoch - warmupEpochs;
    return initialLR * Math.pow(decayRate, decayEpochs);
  }

  private getDifficultyLevel(epoch: number): number {
    // Curriculum learning difficulty schedule
    if (epoch < 30) return 0.6;
    if (epoch < 60) {
      const progress = (epoch - 30) / 30;
      return 0.6 + progress * 0.2;
    }
    if (epoch < 90) {
      const progress = (epoch - 60) / 30;
      return 0.8 + progress * 0.1;
    }
    return 1.0;
  }

  // Method to get epoch data for external use
  getEpochData(): EpochData[] {
    return this.epochData;
  }

  // Method to print summary at the end of training
  printTrainingSummary() {
    if (this.epochData.length === 0) return;

    const lastEpoch = this.epochData[this.epochData.length - 1];
    const bestEpoch = this.epochData.reduce((best, current) =>
      current.combinedScore > best.combinedScore ? current : best
    );

    console.log("\n" + "=".repeat(80));
    console.log("🎉 TRAINING SUMMARY");
    console.log("=".repeat(80));

    // Print summary in clean format
    console.log(`📊 Training Results:`);
    console.log(`   Total Epochs:        ${this.epochData.length}`);
    console.log(
      `   Best Combined Score: ${this.formatNumber(
        bestEpoch.combinedScore
      )} (Epoch ${bestEpoch.epoch})`
    );
    console.log(
      `   Final Val Loss:      ${this.formatNumber(lastEpoch.valLoss)}`
    );
    console.log(
      `   Final Val Accuracy:  ${this.formatNumber(lastEpoch.valAcc)}`
    );

    console.log(`\n🎯 Final F1 Scores:`);
    console.log(`   Buy F1:  ${this.formatNumber(lastEpoch.valF1Buy)}`);
    console.log(`   Sell F1: ${this.formatNumber(lastEpoch.valF1Sell)}`);

    console.log(`\n📈 Final Precision/Recall:`);
    console.log(
      `   Buy:  ${this.formatNumber(
        lastEpoch.precisionBuy
      )} / ${this.formatNumber(lastEpoch.recallBuy)}`
    );
    console.log(
      `   Sell: ${this.formatNumber(
        lastEpoch.precisionSell
      )} / ${this.formatNumber(lastEpoch.recallSell)}`
    );

    console.log("=".repeat(80));
  }

  // Method to display all epochs in a single table
  printAllEpochsTable() {
    if (this.epochData.length === 0) return;

    console.log("\n" + "=".repeat(120));
    console.log("COMPLETE TRAINING HISTORY");
    console.log("=".repeat(120));

    // Convert epoch data to table format
    const tableData = this.epochData.map((epoch) => ({
      Epoch: epoch.epoch,
      "Train Acc": this.formatNumber(epoch.trainAcc),
      "Val Acc": this.formatNumber(epoch.valAcc),
      "Val Loss": this.formatNumber(epoch.valLoss),
      Spread: this.formatNumber(epoch.accSpread),
      "F1 Buy": this.formatNumber(epoch.valF1Buy),
      "F1 Sell": this.formatNumber(epoch.valF1Sell),
      Combined: this.formatNumber(epoch.combinedScore),
      LR: this.formatNumber(epoch.learningRate, 6),
      "Diff%": this.formatNumber(epoch.difficulty * 100, 1),
      EffWgt: this.formatNumber(epoch.effectiveWeight),
      "Buy/Sell": `${epoch.buyCount}/${epoch.sellCount}`,
      "Prec B/S": `${this.formatNumber(epoch.precisionBuy)}/${this.formatNumber(
        epoch.precisionSell
      )}`,
      "Recall B/S": `${this.formatNumber(epoch.recallBuy)}/${this.formatNumber(
        epoch.recallSell
      )}`,
      "Grad Norm": epoch.gradientNorm
        ? this.formatNumber(epoch.gradientNorm)
        : "N/A",
    }));

    console.table(tableData);
  }
}
