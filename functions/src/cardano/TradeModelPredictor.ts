import * as tf from "@tensorflow/tfjs-node";
import { ModelWeightManager } from "./TradeModelWeightManager";
import TradeModelFactory from "./TradeModelFactory";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { FirebaseService } from "../api/FirebaseService";
import { MODEL_CONSTANTS } from "../constants";

export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps = MODEL_CONSTANTS.TIMESTEPS;
  private model: tf.LayersModel;
  private isWeightsLoaded = false;

  constructor() {
    this.weightManager = new ModelWeightManager();
    this.sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);
    FirebaseService.getInstance();
    const factory = new TradeModelFactory(
      MODEL_CONSTANTS.TIMESTEPS,
      MODEL_CONSTANTS.FEATURE_COUNT
    );
    this.model = factory.createModel();
    this.loadWeightsAsync();
  }

  private async loadWeightsAsync(): Promise<void> {
    try {
      await this.weightManager.loadWeights();
      this.weightManager.setWeights(this.model);
      this.isWeightsLoaded = true;
      console.log("Weights loaded successfully");
    } catch (error) {
      console.error("Failed to load weights:", error);
      throw error;
    }
  }

  public async predict(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[]
  ): Promise<{
    buyLogit: number;
    sellLogit: number;
    buyProb: number;
    sellProb: number;
    confidence: number;
    momentum: number;
    trendSlope: number;
    atr: number;
  }> {
    if (!this.isWeightsLoaded) {
      console.log("Weights not yet loaded, awaiting load...");
      await this.loadWeightsAsync();
    }

    const startTime = performance.now();

    const startIndex = Math.max(0, adaPrices.length - this.timesteps - 1);
    const endIndex = adaPrices.length - 1;
    const sequenceRaw = this.sequenceGenerator.generateSequence(
      adaPrices,
      adaVolumes,
      btcPrices,
      btcVolumes,
      startIndex,
      endIndex
    );

    console.log(
      `Raw sequence length: ${sequenceRaw.length}, features per timestep: ${
        sequenceRaw[0]?.length || 0
      }`
    );
    console.log(
      `Total values in sequence: ${
        sequenceRaw.length * (sequenceRaw[0]?.length || 0)
      }`
    );

    const sequence = sequenceRaw.slice(-this.timesteps);
    if (
      sequence.length !== MODEL_CONSTANTS.TIMESTEPS ||
      sequence[0].length !== MODEL_CONSTANTS.FEATURE_COUNT
    ) {
      throw new Error(
        `Sequence shape mismatch: expected [${MODEL_CONSTANTS.TIMESTEPS}, ${
          MODEL_CONSTANTS.FEATURE_COUNT
        }], got [${sequence.length}, ${sequence[0]?.length || 0}]`
      );
    }

    const features = tf.tensor3d(
      [sequence],
      [1, MODEL_CONSTANTS.TIMESTEPS, MODEL_CONSTANTS.FEATURE_COUNT]
    );
    const means = tf.tensor1d(this.weightManager.getFeatureMeans());
    const stds = tf.tensor1d(this.weightManager.getFeatureStds());
    const featuresNormalized = features.sub(means).div(stds.add(1e-6));

    const logits = this.model.predict(featuresNormalized) as tf.Tensor2D;
    const logitsArray = await logits.data();
    const [sellLogit, buyLogit] = [logitsArray[0], logitsArray[1]];
    const probs = tf.tensor2d([[sellLogit, buyLogit]]).softmax();
    const probArray = await probs.data();
    let [sellProb, buyProb] = [probArray[0], probArray[1]];
    const confidence = Math.max(buyProb, sellProb);

    const atr = sequence[sequence.length - 1][11];
    const momentumWindowSize =
      atr > MODEL_CONSTANTS.MOMENTUM_WINDOW_THRESHOLD ? 5 : 14;
    const momentumWindow = adaPrices.slice(-momentumWindowSize);
    const momentum =
      momentumWindow.length >= 2
        ? (momentumWindow[momentumWindow.length - 1] - momentumWindow[0]) /
          momentumWindow[0]
        : 0;

    const trendWindow = adaPrices.slice(-momentumWindowSize);
    const trendSlope =
      trendWindow.length >= 2
        ? (trendWindow[trendWindow.length - 1] - trendWindow[0]) /
          (trendWindow.length - 1)
        : 0;

    const endTime = performance.now();
    console.log(
      `Prediction executed in ${(endTime - startTime).toFixed(2)} ms`
    );
    console.log(
      `Input stats - ADA: min=${Math.min(...adaPrices).toFixed(
        4
      )}, max=${Math.max(...adaPrices).toFixed(4)}, mean=${(
        adaPrices.reduce((a, b) => a + b, 0) / adaPrices.length
      ).toFixed(4)}`
    );
    console.log(
      `Input stats - BTC: min=${Math.min(...btcPrices).toFixed(
        4
      )}, max=${Math.max(...btcPrices).toFixed(4)}, mean=${(
        btcPrices.reduce((a, b) => a + b, 0) / btcPrices.length
      ).toFixed(4)}`
    );
    console.log(
      `ATR: ${atr.toFixed(4)}, Momentum Window: ${momentumWindowSize}`
    );
    console.log(
      `Logits: [Sell: ${sellLogit.toFixed(4)}, Buy: ${buyLogit.toFixed(4)}]`
    );
    console.log(
      `Probs: [Sell: ${sellProb.toFixed(4)}, Buy: ${buyProb.toFixed(4)}]`
    );
    console.log(
      `Confidence: ${confidence.toFixed(4)}, Momentum: ${momentum.toFixed(
        4
      )}, Trend Slope: ${trendSlope.toFixed(4)}`
    );

    features.dispose();
    featuresNormalized.dispose();
    logits.dispose();
    probs.dispose();
    means.dispose();
    stds.dispose();

    return {
      buyLogit,
      sellLogit,
      buyProb,
      sellProb,
      confidence,
      momentum,
      trendSlope,
      atr,
    };
  }
}
