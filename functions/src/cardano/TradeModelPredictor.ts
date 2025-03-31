import * as tf from "@tensorflow/tfjs-node";
import { ModelWeightManager } from "./TradeModelWeightManager";
import TradeModelFactory from "./TradeModelFactory";
import { FeatureSequenceGenerator } from "./FeatureSequenceGenerator";
import { FirebaseService } from "../api/FirebaseService";
import { MODEL_CONFIG, PERIODS } from "../constants";

export class TradeModelPredictor {
  private weightManager: ModelWeightManager;
  private sequenceGenerator: FeatureSequenceGenerator;
  private timesteps = MODEL_CONFIG.TIMESTEPS;
  private model: tf.LayersModel;
  private isWeightsLoaded = false;

  constructor() {
    this.weightManager = new ModelWeightManager();
    this.sequenceGenerator = new FeatureSequenceGenerator(this.timesteps);
    FirebaseService.getInstance();
    const factory = new TradeModelFactory(
      MODEL_CONFIG.TIMESTEPS,
      MODEL_CONFIG.FEATURE_COUNT
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
    shortMomentum: number;
    trendSlope: number;
    atr: number;
    momentumDivergence: number;
    volatilityAdjustedMomentum: number;
    trendStrength: number;
    atrBreakout: number;
  }> {
    if (!this.isWeightsLoaded) {
      console.log("Weights not yet loaded, awaiting load...");
      await this.loadWeightsAsync();
    }

    const startTime = performance.now();
    const startIndex = Math.max(0, adaPrices.length - this.timesteps - 1);
    const endIndex = adaPrices.length - 1;
    const sequence = this.sequenceGenerator
      .generateSequence(
        adaPrices,
        adaVolumes,
        btcPrices,
        btcVolumes,
        startIndex,
        endIndex
      )
      .slice(-this.timesteps);

    if (
      sequence.length !== MODEL_CONFIG.TIMESTEPS ||
      sequence[0].length !== MODEL_CONFIG.FEATURE_COUNT
    ) {
      throw new Error(
        `Sequence shape mismatch: expected [${MODEL_CONFIG.TIMESTEPS}, ${
          MODEL_CONFIG.FEATURE_COUNT
        }], got [${sequence.length}, ${sequence[0]?.length || 0}]`
      );
    }

    const features = tf.tensor3d(
      [sequence],
      [1, MODEL_CONFIG.TIMESTEPS, MODEL_CONFIG.FEATURE_COUNT]
    );
    const means = tf.tensor1d(this.weightManager.getFeatureMeans());
    const stds = tf.tensor1d(this.weightManager.getFeatureStds());
    const featuresNormalized = features.sub(means).div(stds.add(1e-6));

    const logits = this.model.predict(featuresNormalized) as tf.Tensor2D;
    const logitsArray = await logits.data();
    const [sellLogit, buyLogit] = [logitsArray[0], logitsArray[1]];
    const probs = tf.tensor2d([[sellLogit, buyLogit]]).softmax();
    const probArray = await probs.data();
    const [sellProb, buyProb] = [probArray[0], probArray[1]];
    const confidence = Math.max(buyProb, sellProb);

    const atr = sequence[sequence.length - 1][11];
    const momentumWindowSize =
      atr > MODEL_CONFIG.MOMENTUM_WINDOW_THRESHOLD ? 5 : 14;
    const momentumWindow = adaPrices.slice(-momentumWindowSize);
    const momentum =
      momentumWindow.length >= 2
        ? (momentumWindow[momentumWindow.length - 1] - momentumWindow[0]) /
          momentumWindow[0]
        : 0;

    const shortMomentumWindow = adaPrices.slice(-3);
    const shortMomentum =
      shortMomentumWindow.length >= 2
        ? (shortMomentumWindow[shortMomentumWindow.length - 1] -
            shortMomentumWindow[0]) /
          shortMomentumWindow[0]
        : 0;

    const trendWindow = adaPrices.slice(-momentumWindowSize);
    const trendSlope =
      trendWindow.length >= 2
        ? (trendWindow[trendWindow.length - 1] - trendWindow[0]) /
          (trendWindow.length - 1)
        : 0;

    const momentumDivergence = shortMomentum - momentum;
    const volatilityAdjustedMomentum = momentum / (atr || 0.01);
    const trendStrength = trendSlope * volatilityAdjustedMomentum;

    const atrWindow = sequence.slice(-PERIODS.ATR_BREAKOUT).map((s) => s[11]);
    const atrSma =
      atrWindow.length >= PERIODS.ATR_BREAKOUT
        ? atrWindow.reduce((sum, val) => sum + val, 0) / atrWindow.length
        : atr;
    const atrBreakout = atr / atrSma;

    const endTime = performance.now();
    console.log(
      `Prediction executed in ${(endTime - startTime).toFixed(2)} ms`
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
      )}, Short Momentum: ${shortMomentum.toFixed(
        4
      )}, Trend Slope: ${trendSlope.toFixed(
        4
      )}, Momentum Divergence: ${momentumDivergence.toFixed(
        4
      )}, Vol-Adj Momentum: ${volatilityAdjustedMomentum.toFixed(
        4
      )}, Trend Strength: ${trendStrength.toFixed(
        4
      )}, ATR Breakout: ${atrBreakout.toFixed(4)}`
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
      shortMomentum,
      trendSlope,
      atr,
      momentumDivergence,
      volatilityAdjustedMomentum,
      trendStrength,
      atrBreakout,
    };
  }
}
