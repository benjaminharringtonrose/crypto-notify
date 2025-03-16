import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { Condition, Indicators, PredictSell, Recommendation } from "../types";

export const predictSell = async ({
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
}: Indicators): Promise<PredictSell> => {
  const modelRef = admin.firestore().collection("models").doc("sellPredictor");
  const modelDoc = await modelRef.get();
  const modelData = modelDoc.data();
  if (!modelData) throw new Error("Model weights not found in Firestore");

  const weights1 = tf.tensor2d(modelData.weights1, [28, 16]);
  const bias1 = tf.tensor1d(modelData.bias1);
  const weights2 = tf.tensor2d(modelData.weights2, [16, 1]);
  const bias2 = tf.scalar(modelData.bias2);
  const featureMins = tf.tensor1d(modelData.featureMins);
  const featureMaxs = tf.tensor1d(modelData.featureMaxs);

  const obvWindow = obvValues.slice(-30);
  const obvMin = Math.min(...obvWindow);
  const obvMax = Math.max(...obvWindow);
  const normalizedObv =
    obvMax !== obvMin
      ? (obvValues[obvValues.length - 1] - obvMin) / (obvMax - obvMin)
      : 0;

  const featuresRaw = tf.tensor1d([
    rsi || 0,
    prevRsi || 0,
    sma7,
    sma21,
    prevSma7,
    prevSma21,
    macdLine,
    signalLine,
    currentPrice,
    upperBand,
    normalizedObv,
    atr,
    atrBaseline,
    zScore,
    vwap,
    stochRsi,
    prevStochRsi,
    fib61_8,
    prices[prices.length - 2],
    volumeOscillator,
    prevVolumeOscillator,
    isDoubleTop ? 1 : 0,
    isHeadAndShoulders ? 1 : 0,
    prevMacdLine,
    isTripleTop ? 1 : 0,
    isVolumeSpike ? 1 : 0,
    momentum || 0,
    priceChangePct || 0,
  ]);

  const featuresNormalized = featuresRaw
    .sub(featureMins)
    .div(featureMaxs.sub(featureMins).add(1e-6));
  const features = featuresNormalized.clipByValue(0, 1);

  // Two-layer prediction
  const hidden = features.dot(weights1).add(bias1).relu();
  const logits = hidden.dot(weights2).add(bias2);
  const probabilityTensor = logits.sigmoid();
  const probability = (await probabilityTensor.data())[0];

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
  ];

  const metConditions = conditions
    .filter((cond) => cond.met)
    .map((cond) => cond.name);
  const recommendation =
    probability >= 0.5 ? Recommendation.Sell : Recommendation.Hold;

  weights1.dispose();
  bias1.dispose();
  weights2.dispose();
  bias2.dispose();
  featureMins.dispose();
  featureMaxs.dispose();
  featuresRaw.dispose();
  features.dispose();
  hidden.dispose();
  logits.dispose();
  probabilityTensor.dispose();

  return { metConditions, probability, recommendation };
};
