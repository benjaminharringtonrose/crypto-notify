import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { Condition, Indicators, PredictTrade, Recommendation } from "../types";

dotenv.config();

export const predictTrade = async ({
  rsi,
  prevRsi,
  sma7,
  sma21,
  prevSma7,
  prevSma21,
  macdLine,
  signalLine,
  currentPrice,
  upperBand,
  obvValues,
  atr,
  atrBaseline,
  zScore,
  vwap,
  stochRsi,
  prevStochRsi,
  fib61_8,
  prices,
  volumeOscillator,
  prevVolumeOscillator,
  isDoubleTop,
  isHeadAndShoulders,
  prevMacdLine,
  isTripleTop,
  isVolumeSpike,
  momentum,
  priceChangePct,
  btcRsi,
  btcPrevRsi,
  btcSma7,
  btcSma21,
  btcPrevSma7,
  btcPrevSma21,
  btcMacdLine,
  btcSignalLine,
  btcCurrentPrice,
  btcUpperBand,
  btcObvValues,
  btcAtr,
  btcAtrBaseline,
  btcZScore,
  btcVwap,
  btcStochRsi,
  btcPrevStochRsi,
  btcFib61_8,
  btcPrices,
  btcVolumeOscillator,
  btcPrevVolumeOscillator,
  btcIsDoubleTop,
  btcIsHeadAndShoulders,
  btcPrevMacdLine,
  btcIsTripleTop,
  btcIsVolumeSpike,
  btcMomentum,
  btcPriceChangePct,
}: Indicators): Promise<PredictTrade> => {
  const bucket = admin.storage().bucket(process.env.STORAGE_BUCKET);
  const file = bucket.file("tradePredictorWeights.json");
  const [weightsData] = await file.download();
  const modelData = JSON.parse(weightsData.toString());
  const parsedWeights = modelData.weights;

  const validateArray = (
    arr: number[],
    expectedLength: number,
    name: string
  ) => {
    if (!arr || arr.length !== expectedLength) {
      throw new Error(
        `Invalid ${name} length: ${
          arr?.length || 0
        }, expected ${expectedLength}`
      );
    }
    if (arr.some((v) => !Number.isFinite(v))) {
      throw new Error(`${name} contains NaN or Infinity`);
    }
  };

  // Validate weights for Version 18 architecture
  validateArray(parsedWeights.conv1Weights, 2 * 56 * 64, "conv1Weights"); // [2, 56, 64] = 7168
  validateArray(parsedWeights.conv1Bias, 64, "conv1Bias");
  validateArray(parsedWeights.conv2Weights, 2 * 64 * 32, "conv2Weights"); // [2, 64, 32] = 4096
  validateArray(parsedWeights.conv2Bias, 32, "conv2Bias");
  validateArray(parsedWeights.gru1Weights, 32 * 96, "gru1Weights"); // [32, 32*3] = 3072
  validateArray(
    parsedWeights.gru1RecurrentWeights,
    32 * 96,
    "gru1RecurrentWeights"
  ); // [32, 32*3] = 3072
  validateArray(parsedWeights.gru1Bias, 96, "gru1Bias");
  validateArray(parsedWeights.gru2Weights, 32 * 48, "gru2Weights"); // [32, 16*3] = 1536
  validateArray(
    parsedWeights.gru2RecurrentWeights,
    16 * 48,
    "gru2RecurrentWeights"
  ); // [16, 16*3] = 768
  validateArray(parsedWeights.gru2Bias, 48, "gru2Bias");
  validateArray(parsedWeights.residualWeights, 96 * 16, "residualWeights"); // [3*32, 16] = 1536
  validateArray(parsedWeights.residualBias, 16, "residualBias");
  validateArray(parsedWeights.dense1Weights, 16 * 8, "dense1Weights"); // [16, 8] = 128
  validateArray(parsedWeights.dense1Bias, 8, "dense1Bias");
  validateArray(parsedWeights.dense2Weights, 8 * 3, "dense2Weights"); // [8, 3] = 24
  validateArray(parsedWeights.dense2Bias, 3, "dense2Bias");
  validateArray(parsedWeights.featureMins, 56, "featureMins"); // 28 ADA + 28 BTC
  validateArray(parsedWeights.featureMaxs, 56, "featureMaxs");

  // Load tensors
  const conv1Weights = tf.tensor3d(parsedWeights.conv1Weights, [2, 56, 64]);
  const conv1Bias = tf.tensor1d(parsedWeights.conv1Bias);
  const conv2Weights = tf.tensor3d(parsedWeights.conv2Weights, [2, 64, 32]);
  const conv2Bias = tf.tensor1d(parsedWeights.conv2Bias);
  const gru1Weights = tf.tensor2d(parsedWeights.gru1Weights, [32, 96]);
  const gru1RecurrentWeights = tf.tensor2d(
    parsedWeights.gru1RecurrentWeights,
    [32, 96]
  );
  const gru1Bias = tf.tensor1d(parsedWeights.gru1Bias);
  const gru2Weights = tf.tensor2d(parsedWeights.gru2Weights, [32, 48]);
  const gru2RecurrentWeights = tf.tensor2d(
    parsedWeights.gru2RecurrentWeights,
    [16, 48]
  );
  const gru2Bias = tf.tensor1d(parsedWeights.gru2Bias);
  const residualWeights = tf.tensor2d(parsedWeights.residualWeights, [96, 16]);
  const residualBias = tf.tensor1d(parsedWeights.residualBias);
  const dense1Weights = tf.tensor2d(parsedWeights.dense1Weights, [16, 8]);
  const dense1Bias = tf.tensor1d(parsedWeights.dense1Bias);
  const dense2Weights = tf.tensor2d(parsedWeights.dense2Weights, [8, 3]);
  const dense2Bias = tf.tensor1d(parsedWeights.dense2Bias);
  const featureMins = tf.tensor1d(parsedWeights.featureMins);
  const featureMaxs = tf.tensor1d(parsedWeights.featureMaxs);

  // Feature preprocessing
  const timesteps = 7;
  const featureSequence: number[][] = [];
  for (
    let i = Math.max(0, prices.length - timesteps);
    i < prices.length && i < btcPrices.length;
    i++
  ) {
    const dayFeatures = [
      rsi || 0,
      prevRsi || 0,
      sma7,
      sma21,
      prevSma7,
      prevSma21,
      macdLine,
      signalLine,
      prices[i],
      upperBand,
      obvValues[obvValues.length - 1] / 1e6,
      atr,
      atrBaseline,
      zScore,
      vwap,
      stochRsi,
      prevStochRsi,
      fib61_8,
      i > 0 ? prices[i - 1] : prices[0],
      volumeOscillator,
      prevVolumeOscillator,
      isDoubleTop ? 1 : 0,
      isHeadAndShoulders ? 1 : 0,
      prevMacdLine,
      isTripleTop ? 1 : 0,
      isVolumeSpike ? 1 : 0,
      momentum || 0,
      priceChangePct || 0,
      btcRsi || 0,
      btcPrevRsi || 0,
      btcSma7,
      btcSma21,
      btcPrevSma7,
      btcPrevSma21,
      btcMacdLine,
      btcSignalLine,
      btcPrices[i],
      btcUpperBand,
      btcObvValues[btcObvValues.length - 1] / 1e6,
      btcAtr,
      btcAtrBaseline,
      btcZScore,
      btcVwap,
      btcStochRsi,
      btcPrevStochRsi,
      btcFib61_8,
      i > 0 ? btcPrices[i - 1] : btcPrices[0],
      btcVolumeOscillator,
      btcPrevVolumeOscillator,
      btcIsDoubleTop ? 1 : 0,
      btcIsHeadAndShoulders ? 1 : 0,
      btcPrevMacdLine,
      btcIsTripleTop ? 1 : 0,
      btcIsVolumeSpike ? 1 : 0,
      btcMomentum || 0,
      btcPriceChangePct || 0,
    ];
    featureSequence.push(dayFeatures);
  }

  while (featureSequence.length < timesteps) {
    featureSequence.unshift(featureSequence[0] || Array(56).fill(0));
  }

  const featuresRaw = tf.tensor2d(featureSequence, [timesteps, 56]);
  const featuresNormalized = featuresRaw
    .sub(featureMins)
    .div(featureMaxs.sub(featureMins).add(1e-6))
    .clipByValue(0, 1)
    .reshape([1, timesteps, 56]) as tf.Tensor3D;

  // Model inference
  let conv1Output = tf
    .conv1d(featuresNormalized, conv1Weights, 1, "same")
    .add(conv1Bias) as tf.Tensor3D; // [1, 7, 64]
  conv1Output = tf.relu(conv1Output);
  conv1Output = tf
    .maxPool(
      conv1Output.reshape([1, 7, 1, 64]) as tf.Tensor3D,
      [2, 1],
      [2, 1],
      "valid"
    )
    .reshape([1, 3, 64]) as tf.Tensor3D; // [1, 3, 64]

  let conv2Output = tf
    .conv1d(conv1Output, conv2Weights, 1, "same")
    .add(conv2Bias) as tf.Tensor3D; // [1, 3, 32]
  conv2Output = tf.relu(conv2Output);

  const gru1Layer = tf.layers.gru({
    units: 32,
    returnSequences: true,
    weights: [gru1Weights, gru1RecurrentWeights, gru1Bias],
  });
  const gru1Output = gru1Layer.apply(conv2Output) as tf.Tensor3D; // [1, 3, 32]

  const gru2Layer = tf.layers.gru({
    units: 16,
    returnSequences: false,
    weights: [gru2Weights, gru2RecurrentWeights, gru2Bias],
  });
  const gru2Output = gru2Layer.apply(gru1Output) as tf.Tensor2D; // [1, 16]

  const flattenedGru1 = tf.layers.flatten().apply(gru1Output) as tf.Tensor2D; // [1, 96]
  const residualOutput = tf.layers
    .dense({
      units: 16,
      weights: [residualWeights, residualBias],
    })
    .apply(flattenedGru1) as tf.Tensor2D; // [1, 16]

  const combinedOutput = tf.add(gru2Output, residualOutput) as tf.Tensor2D; // [1, 16]
  const dropoutOutput = combinedOutput
    .mul(tf.randomUniform([1, 16]).greater(0.25).cast("float32"))
    .div(0.75) as tf.Tensor2D; // [1, 16]

  const dense1Output = tf.layers
    .dense({
      units: 8,
      activation: "relu",
      weights: [dense1Weights, dense1Bias],
    })
    .apply(dropoutOutput) as tf.Tensor2D; // [1, 8]

  const logits = tf.layers
    .dense({
      units: 3,
      weights: [dense2Weights, dense2Bias],
    })
    .apply(dense1Output) as tf.Tensor2D; // [1, 3]

  const probabilityTensor = tf.softmax(logits);
  const probabilities = await probabilityTensor.data();
  const [sellProb, holdProb, buyProb] = probabilities;

  // Conditions and recommendation
  const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const stdDev20 = Math.sqrt(
    prices.slice(-20).reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / 20
  );
  const lowerBand = sma20 - 2 * stdDev20;

  const conditions: Condition[] = [
    { name: "RSI Overbought", met: !!rsi && rsi > 70, weight: 0.1 },
    {
      name: "SMA Death Cross",
      met: sma7 < sma21 && prevSma7 >= prevSma21,
      weight: 0.07,
    },
    { name: "MACD Bearish", met: macdLine < signalLine, weight: 0.1 },
    {
      name: "Above Bollinger Upper",
      met: currentPrice > upperBand,
      weight: 0.06,
    },
    {
      name: "OBV Dropping",
      met: obvValues[obvValues.length - 1] < obvValues[obvValues.length - 2],
      weight: 0.06,
    },
    {
      name: "Bearish RSI Divergence",
      met:
        currentPrice > prices[prices.length - 2] &&
        !!rsi &&
        !!prevRsi &&
        rsi < prevRsi,
      weight: 0.07,
    },
    { name: "High ATR Volatility", met: atr > atrBaseline * 2, weight: 0.05 },
    { name: "Z-Score High", met: zScore > 2, weight: 0.07 },
    { name: "Above VWAP", met: currentPrice > vwap * 1.05, weight: 0.06 },
    {
      name: "StochRSI Overbought",
      met: stochRsi > 80 && stochRsi < prevStochRsi,
      weight: 0.07,
    },
    {
      name: "Near Fibonacci 61.8%",
      met: currentPrice >= fib61_8 * 0.99 && currentPrice <= fib61_8 * 1.01,
      weight: 0.06,
    },
    {
      name: "Bearish MACD Divergence",
      met: currentPrice > prices[prices.length - 2] && macdLine < prevMacdLine,
      weight: 0.07,
    },
    {
      name: "Volume Oscillator Declining",
      met: volumeOscillator < 0 && volumeOscillator < prevVolumeOscillator,
      weight: 0.06,
    },
    { name: "Double Top Pattern", met: isDoubleTop, weight: 0.08 },
    {
      name: "Head and Shoulders Pattern",
      met: isHeadAndShoulders,
      weight: 0.08,
    },
    { name: "Triple Top Pattern", met: isTripleTop, weight: 0.08 },
    { name: "Volume Spike", met: isVolumeSpike, weight: 0.06 },
    { name: "Negative Momentum", met: momentum < 0, weight: 0.07 },
    { name: "RSI Oversold", met: !!rsi && rsi < 30, weight: 0.1 },
    {
      name: "SMA Golden Cross",
      met: sma7 > sma21 && prevSma7 <= prevSma21,
      weight: 0.07,
    },
    { name: "MACD Bullish", met: macdLine > signalLine, weight: 0.1 },
    {
      name: "Below Bollinger Lower",
      met: currentPrice < lowerBand,
      weight: 0.06,
    },
    {
      name: "OBV Rising",
      met: obvValues[obvValues.length - 1] > obvValues[obvValues.length - 2],
      weight: 0.06,
    },
    {
      name: "Bullish RSI Divergence",
      met:
        currentPrice < prices[prices.length - 2] &&
        !!rsi &&
        !!prevRsi &&
        rsi > prevRsi,
      weight: 0.07,
    },
    { name: "Low ATR Volatility", met: atr < atrBaseline * 0.5, weight: 0.05 },
    { name: "Z-Score Low", met: zScore < -2, weight: 0.07 },
    { name: "Below VWAP", met: currentPrice < vwap * 0.95, weight: 0.06 },
    {
      name: "StochRSI Oversold",
      met: stochRsi < 20 && stochRsi > prevStochRsi,
      weight: 0.07,
    },
    {
      name: "Bullish MACD Divergence",
      met: currentPrice < prices[prices.length - 2] && macdLine > prevMacdLine,
      weight: 0.07,
    },
    {
      name: "Volume Oscillator Rising",
      met: volumeOscillator > 0 && volumeOscillator > prevVolumeOscillator,
      weight: 0.06,
    },
    { name: "Positive Momentum", met: momentum > 0, weight: 0.07 },
  ];

  const metConditions = conditions
    .filter((cond) => cond.met)
    .map((cond) => cond.name);

  const maxProb = Math.max(buyProb, holdProb, sellProb);
  let recommendation: Recommendation = Recommendation.Hold;
  if (maxProb === buyProb && buyProb >= 0.45)
    recommendation = Recommendation.Buy;
  else if (maxProb === sellProb && sellProb >= 0.45)
    recommendation = Recommendation.Sell;

  // Clean up
  conv1Weights.dispose();
  conv1Bias.dispose();
  conv2Weights.dispose();
  conv2Bias.dispose();
  gru1Weights.dispose();
  gru1RecurrentWeights.dispose();
  gru1Bias.dispose();
  gru2Weights.dispose();
  gru2RecurrentWeights.dispose();
  gru2Bias.dispose();
  residualWeights.dispose();
  residualBias.dispose();
  dense1Weights.dispose();
  dense1Bias.dispose();
  dense2Weights.dispose();
  dense2Bias.dispose();
  featureMins.dispose();
  featureMaxs.dispose();
  featuresRaw.dispose();
  featuresNormalized.dispose();
  conv1Output.dispose();
  conv2Output.dispose();
  gru1Output.dispose();
  gru2Output.dispose();
  flattenedGru1.dispose();
  residualOutput.dispose();
  combinedOutput.dispose();
  dropoutOutput.dispose();
  dense1Output.dispose();
  logits.dispose();
  probabilityTensor.dispose();

  return {
    metConditions,
    probabilities: { buy: buyProb, hold: holdProb, sell: sellProb },
    recommendation,
  };
};
