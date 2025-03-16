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

  const lstmWeights = tf.tensor2d(
    JSON.parse(modelData.weights).lstmWeights,
    [28, 256]
  );
  const lstmRecurrentWeights = tf.tensor2d(
    JSON.parse(modelData.weights).lstmRecurrentWeights,
    [64, 256]
  );
  const lstmBias = tf.tensor1d(JSON.parse(modelData.weights).lstmBias);
  const denseWeights = tf.tensor2d(
    JSON.parse(modelData.weights).denseWeights,
    [64, 1]
  );
  const denseBias = tf.scalar(JSON.parse(modelData.weights).denseBias);
  const featureMins = tf.tensor1d(JSON.parse(modelData.weights).featureMins);
  const featureMaxs = tf.tensor1d(JSON.parse(modelData.weights).featureMaxs);

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
  const features = featuresNormalized.clipByValue(0, 1).reshape([1, 1, 28]);

  const [Wi, Wf, Wc, Wo] = tf.split(lstmWeights, 4, 1);
  const [Ri, Rf, Rc, Ro] = tf.split(lstmRecurrentWeights, 4, 1);
  const [bi, bf, bc, bo] = tf.split(lstmBias, 4);

  const hPrev = tf.zeros([1, 64]);
  const cPrev = tf.zeros([1, 64]);

  const inputGate = tf.sigmoid(
    features.matMul(Wi).add(hPrev.matMul(Ri)).add(bi)
  );
  const forgetGate = tf.sigmoid(
    features.matMul(Wf).add(hPrev.matMul(Rf)).add(bf)
  );
  const cellGate = tf.tanh(features.matMul(Wc).add(hPrev.matMul(Rc)).add(bc));
  const outputGate = tf.sigmoid(
    features.matMul(Wo).add(hPrev.matMul(Ro)).add(bo)
  );

  const cNext = forgetGate.mul(cPrev).add(inputGate.mul(cellGate));
  const hNext = outputGate.mul(tf.tanh(cNext));

  const dropoutMask = tf.randomUniform([1, 64]).greater(0.35).cast("float32");
  const hDropped = hNext.mul(dropoutMask).div(0.65);

  const logits = hDropped.matMul(denseWeights).add(denseBias);
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
    probability >= 0.45 ? Recommendation.Sell : Recommendation.Hold;

  lstmWeights.dispose();
  lstmRecurrentWeights.dispose();
  lstmBias.dispose();
  denseWeights.dispose();
  denseBias.dispose();
  featureMins.dispose();
  featureMaxs.dispose();
  featuresRaw.dispose();
  features.dispose();
  Wi.dispose();
  Wf.dispose();
  Wc.dispose();
  Wo.dispose();
  Ri.dispose();
  Rf.dispose();
  Rc.dispose();
  Ro.dispose();
  bi.dispose();
  bf.dispose();
  bc.dispose();
  bo.dispose();
  hPrev.dispose();
  cPrev.dispose();
  inputGate.dispose();
  forgetGate.dispose();
  cellGate.dispose();
  outputGate.dispose();
  cNext.dispose();
  hNext.dispose();
  dropoutMask.dispose();
  hDropped.dispose();
  logits.dispose();
  probabilityTensor.dispose();

  return { metConditions, probability, recommendation };
};
