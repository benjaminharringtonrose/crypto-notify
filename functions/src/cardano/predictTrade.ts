import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { Condition, Indicators, PredictTrade, Recommendation } from "../types";
import { calculateFibonacciLevels } from "../calculations/calculateFibonacciLevels";

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
  isTripleBottom,
  btcIsTripleBottom,
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
    if (!arr || arr.length !== expectedLength)
      throw new Error(
        `Invalid ${name} length: ${
          arr?.length || 0
        }, expected ${expectedLength}`
      );
    if (arr.some((v) => !Number.isFinite(v)))
      throw new Error(`${name} contains NaN or Infinity`);
  };

  validateArray(parsedWeights.conv1Weights, 3 * 57 * 64, "conv1Weights");
  validateArray(parsedWeights.conv1Bias, 64, "conv1Bias");
  validateArray(parsedWeights.conv2Weights, 3 * 64 * 32, "conv2Weights");
  validateArray(parsedWeights.conv2Bias, 32, "conv2Bias");
  validateArray(parsedWeights.gru1Weights, 32 * 192, "gru1Weights");
  validateArray(
    parsedWeights.gru1RecurrentWeights,
    64 * 192,
    "gru1RecurrentWeights"
  );
  validateArray(parsedWeights.gru1Bias, 192, "gru1Bias");
  validateArray(parsedWeights.gru2Weights, 64 * 96, "gru2Weights");
  validateArray(
    parsedWeights.gru2RecurrentWeights,
    32 * 96,
    "gru2RecurrentWeights"
  );
  validateArray(parsedWeights.gru2Bias, 96, "gru2Bias");
  validateArray(parsedWeights.residualWeights, 7 * 64 * 32, "residualWeights");
  validateArray(parsedWeights.residualBias, 32, "residualBias");
  validateArray(parsedWeights.dense1Weights, 32 * 16, "dense1Weights");
  validateArray(parsedWeights.dense1Bias, 16, "dense1Bias");
  validateArray(parsedWeights.dense2Weights, 16 * 3, "dense2Weights");
  validateArray(parsedWeights.dense2Bias, 3, "dense2Bias");
  validateArray(parsedWeights.featureMins, 57, "featureMins");
  validateArray(parsedWeights.featureMaxs, 57, "featureMaxs");

  const conv1Weights = tf.tensor3d(parsedWeights.conv1Weights, [3, 57, 64]);
  const conv1Bias = tf.tensor1d(parsedWeights.conv1Bias);
  const conv2Weights = tf.tensor3d(parsedWeights.conv2Weights, [3, 64, 32]);
  const conv2Bias = tf.tensor1d(parsedWeights.conv2Bias);
  const gru1Weights = tf.tensor2d(parsedWeights.gru1Weights, [32, 192]);
  const gru1RecurrentWeights = tf.tensor2d(
    parsedWeights.gru1RecurrentWeights,
    [64, 192]
  );
  const gru1Bias = tf.tensor1d(parsedWeights.gru1Bias);
  const gru2Weights = tf.tensor2d(parsedWeights.gru2Weights, [64, 96]);
  const gru2RecurrentWeights = tf.tensor2d(
    parsedWeights.gru2RecurrentWeights,
    [32, 96]
  );
  const gru2Bias = tf.tensor1d(parsedWeights.gru2Bias);
  const residualWeights = tf.tensor2d(parsedWeights.residualWeights, [
    7 * 64,
    32,
  ]);
  const residualBias = tf.tensor1d(parsedWeights.residualBias);
  const dense1Weights = tf.tensor2d(parsedWeights.dense1Weights, [32, 16]);
  const dense1Bias = tf.tensor1d(parsedWeights.dense1Bias);
  const dense2Weights = tf.tensor2d(parsedWeights.dense2Weights, [16, 3]);
  const dense2Bias = tf.tensor1d(parsedWeights.dense2Bias);
  const featureMins = tf.tensor1d(parsedWeights.featureMins);
  const featureMaxs = tf.tensor1d(parsedWeights.featureMaxs);

  const timesteps = 14;
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
      isTripleBottom ? 1 : 0,
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
    .mul(tf.randomUniform([1, 32]).greater(0.2).cast("float32"))
    .div(0.8) as tf.Tensor2D;

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
  let [sellProb, holdProb, buyProb] = probabilities;

  console.log(
    `Raw Probabilities - Buy: ${buyProb.toFixed(3)}, Sell: ${sellProb.toFixed(
      3
    )}, Hold: ${holdProb.toFixed(3)}`
  );

  const sma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const stdDev20 = Math.sqrt(
    prices.slice(-20).reduce((sum, p) => sum + Math.pow(p - sma20, 2), 0) / 20
  );
  const lowerBand = sma20 - 2 * stdDev20;
  const macdHistogram = macdLine - signalLine;
  const prevMacdHistogram = prevMacdLine - signalLine;
  const { levels } = calculateFibonacciLevels(prices.slice(-30), 30);
  const fib236 = levels[1] || 0;

  const conditions: Condition[] = [
    { name: "RSI Overbought", met: !!rsi && rsi > 75, weight: 0.12 },
    {
      name: "SMA Death Cross",
      met: sma7 < sma21 && prevSma7 >= prevSma21,
      weight: 0.08,
    },
    { name: "MACD Bearish", met: macdLine < signalLine, weight: 0.12 },
    {
      name: "Above Bollinger Upper",
      met: currentPrice > upperBand,
      weight: 0.07,
    },
    {
      name: "OBV Dropping",
      met: obvValues[obvValues.length - 1] < obvValues[obvValues.length - 2],
      weight: 0.07,
    },
    {
      name: "Bearish RSI Divergence",
      met:
        currentPrice > prices[prices.length - 2] &&
        !!rsi &&
        !!prevRsi &&
        rsi < prevRsi,
      weight: 0.08,
    },
    { name: "High ATR Volatility", met: atr > atrBaseline * 2.5, weight: 0.06 },
    { name: "Z-Score High", met: zScore > 2.5, weight: 0.08 },
    { name: "Above VWAP", met: currentPrice > vwap * 1.1, weight: 0.07 },
    {
      name: "StochRSI Overbought",
      met: stochRsi > 85 && stochRsi < prevStochRsi,
      weight: 0.08,
    },
    {
      name: "Near Fibonacci 61.8%",
      met: currentPrice >= fib61_8 * 0.98 && currentPrice <= fib61_8 * 1.02,
      weight: 0.07,
    },
    {
      name: "Bearish MACD Divergence",
      met: currentPrice > prices[prices.length - 2] && macdLine < prevMacdLine,
      weight: 0.08,
    },
    {
      name: "Volume Oscillator Declining",
      met: volumeOscillator < 0 && volumeOscillator < prevVolumeOscillator,
      weight: 0.07,
    },
    { name: "Double Top Pattern", met: isDoubleTop, weight: 0.09 },
    {
      name: "Head and Shoulders Pattern",
      met: isHeadAndShoulders,
      weight: 0.09,
    },
    { name: "Triple Top Pattern", met: isTripleTop, weight: 0.09 },
    { name: "Volume Spike", met: isVolumeSpike, weight: 0.07 },
    { name: "Negative Momentum", met: momentum < 0, weight: 0.08 },
    {
      name: "Price Crossing Below VWAP",
      met: currentPrice < vwap && prices[prices.length - 2] > vwap,
      weight: 0.07,
    },
    {
      name: "Bearish Engulfing",
      met:
        prices.length >= 2 &&
        prices[prices.length - 2] < prices[prices.length - 1] &&
        currentPrice < prices[prices.length - 2] &&
        prices[prices.length - 1] - currentPrice >
          prices[prices.length - 1] - prices[prices.length - 2],
      weight: 0.07,
    },
    { name: "RSI Oversold", met: !!rsi && rsi < 25, weight: 0.128 },
    {
      name: "SMA Golden Cross",
      met: sma7 > sma21 && prevSma7 <= prevSma21,
      weight: 0.088,
    },
    { name: "MACD Bullish", met: macdLine > signalLine, weight: 0.128 },
    {
      name: "Below Bollinger Lower",
      met: currentPrice < lowerBand,
      weight: 0.078,
    },
    {
      name: "OBV Rising",
      met: obvValues[obvValues.length - 1] > obvValues[obvValues.length - 2],
      weight: 0.078,
    },
    {
      name: "Bullish RSI Divergence",
      met:
        currentPrice < prices[prices.length - 2] &&
        !!rsi &&
        !!prevRsi &&
        rsi > prevRsi,
      weight: 0.088,
    },
    { name: "Low ATR Volatility", met: atr < atrBaseline * 0.4, weight: 0.068 },
    { name: "Z-Score Low", met: zScore < -2.5, weight: 0.088 },
    { name: "Below VWAP", met: currentPrice < vwap * 0.9, weight: 0.078 },
    {
      name: "StochRSI Oversold",
      met: stochRsi < 15 && stochRsi > prevStochRsi,
      weight: 0.088,
    },
    {
      name: "Bullish MACD Divergence",
      met: currentPrice < prices[prices.length - 2] && macdLine > prevMacdLine,
      weight: 0.088,
    },
    {
      name: "Volume Oscillator Rising",
      met: volumeOscillator > 0 && volumeOscillator > prevVolumeOscillator,
      weight: 0.078,
    },
    { name: "Positive Momentum", met: momentum > 0, weight: 0.088 },
    { name: "Triple Bottom Pattern", met: isTripleBottom, weight: 0.088 },
    {
      name: "BTC Triple Bottom Pattern",
      met: btcIsTripleBottom,
      weight: 0.068,
    },
    {
      name: "Volume Breakout",
      met: isVolumeSpike && priceChangePct > 0.5,
      weight: 0.078,
    },
    {
      name: "Price Crossing Above VWAP",
      met: currentPrice > vwap && prices[prices.length - 2] < vwap,
      weight: 0.068,
    },
    {
      name: "MACD Histogram Positive Flip",
      met: macdHistogram > 0 && prevMacdHistogram <= 0,
      weight: 0.078,
    },
    {
      name: "Near Fibonacci 23.6%",
      met: currentPrice >= fib236 * 0.98 && currentPrice <= fib236 * 1.02,
      weight: 0.068,
    },
    {
      name: "Bullish Engulfing",
      met:
        prices.length >= 2 &&
        prices[prices.length - 2] > prices[prices.length - 1] &&
        currentPrice > prices[prices.length - 2] &&
        currentPrice - prices[prices.length - 1] >
          prices[prices.length - 2] - prices[prices.length - 1],
      weight: 0.068,
    },
  ];

  const metConditions = conditions
    .filter((cond) => cond.met)
    .map((cond) => cond.name);

  const isGoldenCross = sma7 > sma21 && prevSma7 <= prevSma21;
  const isDeathCross = sma7 < sma21 && prevSma7 >= prevSma21;
  const trendStrength = sma7 / sma21;

  const buyBoost = isGoldenCross
    ? Math.min(0.15 * Math.max(0, trendStrength - 1), 0.15)
    : 0;
  const sellBoost = isDeathCross
    ? Math.min(0.15 * Math.max(0, 1 - trendStrength), 0.15)
    : 0;
  buyProb += buyBoost;
  sellProb += sellBoost;

  let recommendation: Recommendation;
  const isUptrend = sma7 / sma21 > 1.05;
  if (buyProb > 0.36 || (buyProb > 0.33 && isUptrend))
    recommendation = Recommendation.Buy;
  else if (sellProb > 0.37 || (sellProb > 0.34 && !isUptrend))
    // Raised sell threshold to 0.37
    recommendation = Recommendation.Sell;
  else recommendation = Recommendation.Hold;

  console.log(
    `Adjusted Probabilities - Buy: ${buyProb.toFixed(
      3
    )}, Sell: ${sellProb.toFixed(3)}, Hold: ${holdProb.toFixed(3)}`
  );
  console.log(`Recommendation: ${recommendation}`);

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
