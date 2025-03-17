import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { Condition, Indicators, PredictTrade, Recommendation } from "../types";

dotenv.config();

export const predictTradeADA = async ({
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

  validateArray(parsedWeights.conv1Weights, 2 * 28 * 64, "conv1Weights");
  validateArray(parsedWeights.conv1Bias, 64, "conv1Bias");
  validateArray(parsedWeights.conv2Weights, 2 * 64 * 32, "conv2Weights");
  validateArray(parsedWeights.conv2Bias, 32, "conv2Bias");
  validateArray(parsedWeights.conv3Weights, 2 * 32 * 16, "conv3Weights");
  validateArray(parsedWeights.conv3Bias, 16, "conv3Bias");
  validateArray(parsedWeights.denseWeights, 16 * 3, "denseWeights");
  validateArray(parsedWeights.denseBias, 3, "denseBias");
  validateArray(parsedWeights.featureMins, 28, "featureMins");
  validateArray(parsedWeights.featureMaxs, 28, "featureMaxs");

  const conv1Weights = tf.tensor3d(parsedWeights.conv1Weights, [2, 28, 64]);
  const conv1Bias = tf.tensor1d(parsedWeights.conv1Bias);
  const conv2Weights = tf.tensor3d(parsedWeights.conv2Weights, [2, 64, 32]);
  const conv2Bias = tf.tensor1d(parsedWeights.conv2Bias);
  const conv3Weights = tf.tensor3d(parsedWeights.conv3Weights, [2, 32, 16]);
  const conv3Bias = tf.tensor1d(parsedWeights.conv3Bias);
  const denseWeights = tf.tensor2d(parsedWeights.denseWeights, [16, 3]);
  const denseBias = tf.tensor1d(parsedWeights.denseBias);
  const featureMins = tf.tensor1d(parsedWeights.featureMins);
  const featureMaxs = tf.tensor1d(parsedWeights.featureMaxs);

  console.log("conv1Weights shape:", conv1Weights.shape);
  console.log(
    "conv1Weights sample:",
    Array.from(await conv1Weights.data()).slice(0, 5)
  );
  console.log(
    "conv1Bias sample:",
    Array.from(await conv1Bias.data()).slice(0, 5)
  );
  console.log("conv2Weights shape:", conv2Weights.shape);
  console.log("conv3Weights shape:", conv3Weights.shape);
  console.log("denseWeights shape:", denseWeights.shape);

  const obvWindow = obvValues.slice(-30);
  const obvMin = Math.min(...obvWindow);
  const obvMax = Math.max(...obvWindow);
  const normalizedObv =
    obvMax !== obvMin
      ? (obvValues[obvValues.length - 1] - obvMin) / (obvMax - obvMin)
      : 0;

  const timesteps = 7;
  const featureSequence: number[][] = [];

  if (!prices || prices.length === 0) {
    throw new Error("Prices array is empty or undefined");
  }

  for (let i = Math.max(0, prices.length - timesteps); i < prices.length; i++) {
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
      normalizedObv,
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
    ];
    featureSequence.push(dayFeatures);
  }

  while (featureSequence.length < timesteps) {
    if (featureSequence.length === 0) {
      const defaultRow = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        currentPrice || 0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        prices.length > 0 ? prices[0] : 0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      ];
      featureSequence.push(defaultRow);
    } else {
      featureSequence.unshift(featureSequence[0]);
    }
  }

  if (
    featureSequence.length !== timesteps ||
    featureSequence[0].length !== 28
  ) {
    throw new Error(
      `featureSequence has invalid shape: [${featureSequence.length}, ${
        featureSequence[0]?.length || 0
      }] (expected [${timesteps}, 28])`
    );
  }

  const featuresRaw = tf.tensor2d(featureSequence, [timesteps, 28]); // [7, 28]
  const featuresNormalized = featuresRaw
    .sub(featureMins)
    .div(featureMaxs.sub(featureMins).add(1e-6));
  const features = featuresNormalized
    .clipByValue(0, 1)
    .reshape([1, timesteps, 28]) as tf.Tensor3D; // [1, 7, 28]
  console.log("features shape:", features.shape);
  console.log(
    "features sample:",
    Array.from(await features.data()).slice(0, 5)
  );

  let conv1Output = tf
    .conv1d(features, conv1Weights, 1, "same")
    .add(conv1Bias) as tf.Tensor3D;
  conv1Output = tf.relu(conv1Output);
  console.log("conv1PrePool shape:", conv1Output.shape);
  console.log(
    "conv1PrePool sample:",
    Array.from(await conv1Output.data()).slice(0, 5)
  );
  // Reshape to 4D for maxPool: [batch, timesteps, 1, channels]
  const conv1Reshaped = conv1Output.reshape([1, 7, 1, 64]) as tf.Tensor4D;
  conv1Output = tf
    .maxPool(conv1Reshaped, [2, 1], [2, 1], "valid")
    .reshape([1, 3, 64]); // [1, 3, 64]
  if (conv1Output.shape[0] === 0) {
    throw new Error("conv1Output batch size is 0 after pooling");
  }
  console.log("conv1Output shape:", conv1Output.shape);
  console.log(
    "conv1Output sample:",
    Array.from(await conv1Output.data()).slice(0, 5)
  );

  let conv2Output = tf
    .conv1d(conv1Output, conv2Weights, 1, "same")
    .add(conv2Bias) as tf.Tensor3D;
  conv2Output = tf.relu(conv2Output);
  const conv2Reshaped = conv2Output.reshape([1, 3, 1, 32]) as tf.Tensor4D;
  conv2Output = tf
    .maxPool(conv2Reshaped, [2, 1], [2, 1], "valid")
    .reshape([1, 1, 32]); // [1, 1, 32]
  console.log("conv2Output shape:", conv2Output.shape);

  let conv3Output = tf
    .conv1d(conv2Output, conv3Weights, 1, "same")
    .add(conv3Bias) as tf.Tensor3D;
  conv3Output = tf.relu(conv3Output); // [1, 1, 16]
  console.log("conv3Output shape:", conv3Output.shape);

  const flatOutput = conv3Output.reshape([1, -1]) as tf.Tensor2D; // [1, 16]
  console.log("flatOutput shape:", flatOutput.shape);

  const dropoutMask = tf
    .randomUniform([1, flatOutput.shape[1]])
    .greater(0.4)
    .cast("float32");
  const droppedOutput = flatOutput.mul(dropoutMask).div(0.6); // [1, 16]
  console.log("droppedOutput shape:", droppedOutput.shape);

  const logits = droppedOutput.matMul(denseWeights).add(denseBias); // [1, 16] × [16, 3] = [1, 3]
  const probabilityTensor = tf.softmax(logits);
  const probabilities = await probabilityTensor.data();
  const [sellProb, holdProb, buyProb] = probabilities;

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

  conv1Weights.dispose();
  conv1Bias.dispose();
  conv2Weights.dispose();
  conv2Bias.dispose();
  conv3Weights.dispose();
  conv3Bias.dispose();
  denseWeights.dispose();
  denseBias.dispose();
  featureMins.dispose();
  featureMaxs.dispose();
  featuresRaw.dispose();
  features.dispose();
  conv1Output.dispose();
  conv2Output.dispose();
  conv3Output.dispose();
  flatOutput.dispose();
  dropoutMask.dispose();
  droppedOutput.dispose();
  logits.dispose();
  probabilityTensor.dispose();

  return {
    metConditions,
    probabilities: { buy: buyProb, hold: holdProb, sell: sellProb },
    recommendation,
  };
};
