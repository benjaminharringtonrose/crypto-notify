import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { Indicators, PredictTrade, Recommendation } from "../types";

dotenv.config();

interface StrategyResult {
  buyProb: number;
  sellProb: number;
  holdProb: number;
  recommendation: Recommendation;
}

class Strategy {
  private name: string;

  constructor(name: string) {
    this.name = name;
  }

  evaluate(indicators: Indicators): StrategyResult {
    const {
      rsi,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
      currentPrice,
      prices,
      atr,
      atrBaseline,
      isVolumeSpike,
      prevMacdLine,
    } = indicators;
    const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const sma50 =
      prices.length >= 50
        ? prices.slice(-50).reduce((a, b) => a + b, 0) / 50
        : sma20;
    const stdDev20 = Math.sqrt(
      prices.slice(-20).reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / 20
    );
    const lowerBand = sma20 - 2 * stdDev20;
    const upperBandBB = sma20 + 2 * stdDev20;
    const macdHistogram = macdLine - signalLine;
    const prevMacdHistogram = prevMacdLine - signalLine;
    const rsiSafe = rsi !== undefined ? rsi : 50;
    const adxProxy = Math.abs((sma7 - sma50) / sma50);
    const volatility = atr / atrBaseline;
    const recentHigh = Math.max(...prices.slice(-10));
    const recentLow = Math.min(...prices.slice(-10));

    let buyProb = 0.3;
    let sellProb = 0.3;
    let holdProb = 0.3;

    switch (this.name) {
      case "TrendFollowing":
        if (sma7 > sma21 && prevSma7 <= prevSma21 && adxProxy > 0.02)
          buyProb += 0.3;
        if (sma7 < sma21 && prevSma7 >= prevSma21 && adxProxy > 0.02)
          sellProb += 0.3;
        if (macdHistogram > 0 && prevMacdHistogram <= 0) buyProb += 0.2;
        if (macdHistogram < 0 && prevMacdHistogram >= 0) sellProb += 0.2;
        if (adxProxy <= 0.02) holdProb += 0.05;
        break;

      case "MeanReversion":
        if (rsiSafe < 30 && currentPrice < lowerBand) buyProb += 0.4;
        if (rsiSafe > 70 && currentPrice > upperBandBB) sellProb += 0.4;
        if (rsiSafe >= 30 && rsiSafe <= 70) holdProb += 0.05;
        break;

      case "Breakout":
        if (currentPrice > recentHigh && isVolumeSpike && volatility > 1.5)
          buyProb += 0.4;
        if (currentPrice < recentLow && isVolumeSpike && volatility > 1.5)
          sellProb += 0.4;
        if (currentPrice <= recentHigh && currentPrice >= recentLow)
          holdProb += 0.05;
        if (volatility <= 1.5) holdProb += 0.03;
        break;
    }

    const total = buyProb + sellProb + holdProb;
    buyProb /= total;
    sellProb /= total;
    holdProb /= total;

    const recommendation =
      rsiSafe < 30 && buyProb > 0.33
        ? Recommendation.Buy
        : adxProxy > 0.02 && sellProb > 0.33
        ? Recommendation.Sell
        : buyProb > 0.35
        ? Recommendation.Buy
        : sellProb > 0.35
        ? Recommendation.Sell
        : Recommendation.Hold;
    return { buyProb, sellProb, holdProb, recommendation };
  }
}

export const predictTrade = async (
  indicators: Indicators
): Promise<PredictTrade> => {
  const bucket = admin.storage().bucket(process.env.STORAGE_BUCKET);
  const file = bucket.file("tradePredictorWeights.json");
  const [weightsData] = await file.download();
  const { weights } = JSON.parse(weightsData.toString());

  const validateArray = (
    arr: number[],
    expectedLength: number,
    name: string
  ) => {
    if (
      !arr ||
      arr.length !== expectedLength ||
      arr.some((v) => !Number.isFinite(v))
    ) {
      throw new Error(
        `Invalid ${name}: length ${
          arr?.length || 0
        }, expected ${expectedLength}`
      );
    }
  };

  validateArray(weights.conv1Weights, 3 * 57 * 64, "conv1Weights");
  validateArray(weights.conv1Bias, 64, "conv1Bias");
  validateArray(weights.conv2Weights, 3 * 64 * 32, "conv2Weights");
  validateArray(weights.conv2Bias, 32, "conv2Bias");
  validateArray(weights.gru1Weights, 32 * 192, "gru1Weights");
  validateArray(weights.gru1RecurrentWeights, 64 * 192, "gru1RecurrentWeights");
  validateArray(weights.gru1Bias, 192, "gru1Bias");
  validateArray(weights.gru2Weights, 64 * 96, "gru2Weights");
  validateArray(weights.gru2RecurrentWeights, 32 * 96, "gru2RecurrentWeights");
  validateArray(weights.gru2Bias, 96, "gru2Bias");
  validateArray(weights.residualWeights, 7 * 64 * 32, "residualWeights");
  validateArray(weights.residualBias, 32, "residualBias");
  validateArray(weights.dense1Weights, 32 * 16, "dense1Weights");
  validateArray(weights.dense1Bias, 16, "dense1Bias");
  validateArray(weights.dense2Weights, 16 * 3, "dense2Weights");
  validateArray(weights.dense2Bias, 3, "dense2Bias");
  validateArray(weights.featureMins, 57, "featureMins");
  validateArray(weights.featureMaxs, 57, "featureMaxs");

  const conv1Weights = tf.tensor3d(weights.conv1Weights, [3, 57, 64]);
  const conv1Bias = tf.tensor1d(weights.conv1Bias);
  const conv2Weights = tf.tensor3d(weights.conv2Weights, [3, 64, 32]);
  const conv2Bias = tf.tensor1d(weights.conv2Bias);
  const gru1Weights = tf.tensor2d(weights.gru1Weights, [32, 192]);
  const gru1RecurrentWeights = tf.tensor2d(
    weights.gru1RecurrentWeights,
    [64, 192]
  );
  const gru1Bias = tf.tensor1d(weights.gru1Bias);
  const gru2Weights = tf.tensor2d(weights.gru2Weights, [64, 96]);
  const gru2RecurrentWeights = tf.tensor2d(
    weights.gru2RecurrentWeights,
    [32, 96]
  );
  const gru2Bias = tf.tensor1d(weights.gru2Bias);
  const residualWeights = tf.tensor2d(weights.residualWeights, [7 * 64, 32]);
  const residualBias = tf.tensor1d(weights.residualBias);
  const dense1Weights = tf.tensor2d(weights.dense1Weights, [32, 16]);
  const dense1Bias = tf.tensor1d(weights.dense1Bias);
  const dense2Weights = tf.tensor2d(weights.dense2Weights, [16, 3]);
  const dense2Bias = tf.tensor1d(weights.dense2Bias);
  const featureMins = tf.tensor1d(weights.featureMins);
  const featureMaxs = tf.tensor1d(weights.featureMaxs);

  const timesteps = 14;
  const featureSequence: number[][] = [];
  const { prices, btcPrices } = indicators;
  for (
    let i = Math.max(0, prices.length - timesteps);
    i < prices.length && i < btcPrices.length;
    i++
  ) {
    const dayFeatures = [
      indicators.rsi || 0,
      indicators.prevRsi || 0,
      indicators.sma7,
      indicators.sma21,
      indicators.prevSma7,
      indicators.prevSma21,
      indicators.macdLine,
      indicators.signalLine,
      prices[i],
      indicators.upperBand,
      indicators.obvValues[indicators.obvValues.length - 1] / 1e6,
      indicators.atr,
      indicators.atrBaseline,
      indicators.zScore,
      indicators.vwap,
      indicators.stochRsi,
      indicators.prevStochRsi,
      indicators.fib61_8,
      i > 0 ? prices[i - 1] : prices[0],
      indicators.volumeOscillator,
      indicators.prevVolumeOscillator,
      indicators.isDoubleTop ? 1 : 0,
      indicators.isHeadAndShoulders ? 1 : 0,
      indicators.prevMacdLine,
      indicators.isTripleTop ? 1 : 0,
      indicators.isVolumeSpike ? 1 : 0,
      indicators.momentum || 0,
      indicators.priceChangePct || 0,
      indicators.isTripleBottom ? 1 : 0,
      indicators.btcRsi || 0,
      indicators.btcPrevRsi || 0,
      indicators.btcSma7,
      indicators.btcSma21,
      indicators.btcPrevSma7,
      indicators.btcPrevSma21,
      indicators.btcMacdLine,
      indicators.btcSignalLine,
      btcPrices[i],
      indicators.btcUpperBand,
      indicators.btcObvValues[indicators.btcObvValues.length - 1] / 1e6,
      indicators.btcAtr,
      indicators.btcAtrBaseline,
      indicators.btcZScore,
      indicators.btcVwap,
      indicators.btcStochRsi,
      indicators.btcPrevStochRsi,
      indicators.btcFib61_8,
      i > 0 ? btcPrices[i - 1] : btcPrices[0],
      indicators.btcVolumeOscillator,
      indicators.btcPrevVolumeOscillator,
      indicators.btcIsDoubleTop ? 1 : 0,
      indicators.btcIsHeadAndShoulders ? 1 : 0,
      indicators.btcPrevMacdLine,
      indicators.btcIsTripleTop ? 1 : 0,
      indicators.isVolumeSpike ? 1 : 0,
      indicators.btcMomentum || 0,
      indicators.btcPriceChangePct || 0,
    ];
    featureSequence.push(dayFeatures);
  }
  while (featureSequence.length < timesteps)
    featureSequence.unshift(featureSequence[0] || Array(57).fill(0));

  const featuresRaw = tf.tensor2d(featureSequence, [timesteps, 57]);
  const featuresNormalized = featuresRaw
    .sub(featureMins)
    .div(featureMaxs.sub(featureMins).add(1e-6))
    .clipByValue(0, 1)
    .reshape([1, timesteps, 57]) as tf.Tensor3D;

  let conv1Output = tf
    .conv1d(featuresNormalized, conv1Weights, 1, "same")
    .add(conv1Bias) as tf.Tensor3D;
  conv1Output = tf.relu(conv1Output);
  conv1Output = tf
    .maxPool(
      conv1Output.reshape([1, 14, 1, 64]) as tf.Tensor3D,
      [2, 1],
      [2, 1],
      "valid"
    )
    .reshape([1, 7, 64]) as tf.Tensor3D;

  let conv2Output = tf
    .conv1d(conv1Output, conv2Weights, 1, "same")
    .add(conv2Bias) as tf.Tensor3D;
  conv2Output = tf.relu(conv2Output);

  const gru1Layer = tf.layers.gru({
    units: 64,
    returnSequences: true,
    weights: [gru1Weights, gru1RecurrentWeights, gru1Bias],
  });
  const gru1Output = gru1Layer.apply(conv2Output) as tf.Tensor3D;

  const gru2Layer = tf.layers.gru({
    units: 32,
    returnSequences: false,
    weights: [gru2Weights, gru2RecurrentWeights, gru2Bias],
  });
  const gru2Output = gru2Layer.apply(gru1Output) as tf.Tensor2D;

  const flattenedGru1 = tf.layers.flatten().apply(gru1Output) as tf.Tensor2D;
  const residualOutput = tf
    .matMul(flattenedGru1, residualWeights)
    .add(residualBias) as tf.Tensor2D;

  const combinedOutput = tf.add(gru2Output, residualOutput) as tf.Tensor2D;
  const dropoutOutput = combinedOutput
    .mul(tf.randomUniform([1, 32]).greater(0.3).cast("float32"))
    .div(0.7) as tf.Tensor2D;

  const dense1Output = tf.layers
    .dense({
      units: 16,
      activation: "relu",
      weights: [dense1Weights, dense1Bias],
    })
    .apply(dropoutOutput) as tf.Tensor2D;
  const logits = tf.layers
    .dense({ units: 3, weights: [dense2Weights, dense2Bias] })
    .apply(dense1Output) as tf.Tensor2D;

  const probabilityTensor = tf.softmax(logits);
  const probabilities = await probabilityTensor.data();
  let [sellProbBase, holdProbBase, buyProbBase] = probabilities;

  const actionSum = buyProbBase + sellProbBase;
  const sma50 =
    prices.length >= 50
      ? prices.slice(-50).reduce((a, b) => a + b, 0) / 50
      : prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const isBearish = indicators.sma7 < sma50;
  if (actionSum > 0) {
    buyProbBase = (buyProbBase / actionSum) * 0.8;
    sellProbBase = (sellProbBase / actionSum) * 0.8;
    holdProbBase = isBearish ? 0.15 : 0.2; // Lower in bearish conditions
  }

  const adxProxy = Math.abs((indicators.sma7 - sma50) / sma50);
  const volatility = indicators.atr / indicators.atrBaseline;
  const rsiSafe = indicators.rsi !== undefined ? indicators.rsi : 50;

  const trendStrategy = new Strategy("TrendFollowing");
  const meanRevStrategy = new Strategy("MeanReversion");
  const breakoutStrategy = new Strategy("Breakout");

  const trendResult = trendStrategy.evaluate(indicators);
  const meanRevResult = meanRevStrategy.evaluate(indicators);
  const breakoutResult = breakoutStrategy.evaluate(indicators);

  const trendWeight = adxProxy > 0.02 ? 0.5 : 0.2;
  const meanRevWeight = rsiSafe < 30 || rsiSafe > 70 ? 0.5 : 0.3; // Increased from 0.4
  const breakoutWeight =
    volatility > 1.5 || indicators.isVolumeSpike ? 0.3 : 0.2;

  let buyProb =
    buyProbBase +
    trendWeight * trendResult.buyProb +
    meanRevWeight * meanRevResult.buyProb +
    breakoutWeight * breakoutResult.buyProb;
  let sellProb =
    sellProbBase +
    trendWeight * trendResult.sellProb +
    meanRevWeight * meanRevResult.sellProb +
    breakoutWeight * breakoutResult.sellProb;
  let holdProb =
    holdProbBase +
    trendWeight * trendResult.holdProb +
    meanRevWeight * meanRevResult.holdProb +
    breakoutWeight * breakoutResult.holdProb;

  const total = buyProb + sellProb + holdProb;
  buyProb /= total;
  sellProb /= total;
  holdProb /= total;

  const profitPotential =
    (indicators.currentPrice -
      prices.slice(-5).reduce((a, b) => a + b, 0) / 5) /
    (prices.slice(-5).reduce((a, b) => a + b, 0) / 5);
  const recommendation =
    rsiSafe < 30 && buyProb > 0.33 && profitPotential > -0.03
      ? Recommendation.Buy
      : adxProxy > 0.02 &&
        sellProb > 0.33 &&
        (profitPotential > 0.05 || profitPotential < 0)
      ? Recommendation.Sell
      : buyProb > 0.35 && profitPotential > -0.03
      ? Recommendation.Buy
      : sellProb > 0.35 &&
        (profitPotential > 0.05 || (adxProxy < 0.02 && profitPotential > -0.05))
      ? Recommendation.Sell
      : Recommendation.Hold;

  console.log(
    `Adjusted Probabilities - Buy: ${buyProb.toFixed(
      3
    )}, Sell: ${sellProb.toFixed(3)}, Hold: ${holdProb.toFixed(3)}`
  );
  console.log(`Recommendation: ${recommendation}`);
  console.log(
    `Profit Potential: ${(profitPotential * 100).toFixed(2)}%, Volatility: ${
      volatility > 2 ? "High" : "Normal"
    }, Trend: ${adxProxy > 0.02 ? "Strong" : "Weak"}, RSI: ${rsiSafe.toFixed(
      2
    )}`
  );

  [
    conv1Weights,
    conv1Bias,
    conv2Weights,
    conv2Bias,
    gru1Weights,
    gru1RecurrentWeights,
    gru1Bias,
    gru2Weights,
    gru2RecurrentWeights,
    gru2Bias,
    residualWeights,
    residualBias,
    dense1Weights,
    dense1Bias,
    dense2Weights,
    dense2Bias,
    featureMins,
    featureMaxs,
    featuresRaw,
    featuresNormalized,
    conv1Output,
    conv2Output,
    gru1Output,
    gru2Output,
    flattenedGru1,
    residualOutput,
    combinedOutput,
    dropoutOutput,
    dense1Output,
    logits,
    probabilityTensor,
  ].forEach((tensor) => tensor.dispose());

  return {
    metConditions: [],
    probabilities: { buy: buyProb, hold: holdProb, sell: sellProb },
    recommendation,
  };
};
