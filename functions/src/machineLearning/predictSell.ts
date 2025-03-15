import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import serviceAccount from "../../serviceAccountKey.json";

// Initialize Firebase Admin
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
});

export async function predictSell(indicators: {
  rsi?: number;
  prevRsi?: number;
  sma7: number;
  sma21: number;
  prevSma7: number;
  prevSma21: number;
  macdLine: number;
  signalLine: number;
  currentPrice: number;
  upperBand: number;
  obvValues: number[];
  atr: number;
  atrBaseline: number;
  zScore: number;
  vwap: number;
  stochRsi: number;
  prevStochRsi: number;
  fib61_8: number;
  prices: number[];
  volumeOscillator: number;
  prevVolumeOscillator: number;
  isDoubleTop: boolean;
  isHeadAndShoulders: boolean;
  prevMacdLine: number;
  isTripleTop: boolean;
  isVolumeSpike: boolean;
}): Promise<{
  metConditions: string[];
  probability: number;
  recommendation: "sell" | "hold";
}> {
  const {
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
  } = indicators;

  // Load trained weights from Firestore
  const modelRef = admin.firestore().collection("models").doc("sellPredictor");
  const modelDoc = await modelRef.get();
  const modelData = modelDoc.data();
  if (!modelData) throw new Error("Model weights not found in Firestore");

  const weights = tf.tensor1d(modelData.weights);
  const bias = tf.scalar(modelData.bias);
  const featureMins = tf.tensor1d(modelData.featureMins);
  const featureMaxs = tf.tensor1d(modelData.featureMaxs);

  // Features (26 total)
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
    obvValues[obvValues.length - 1],
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
  ]);

  // Normalize and clamp to [0, 1]
  const featuresNormalized = featuresRaw
    .sub(featureMins)
    .div(featureMaxs.sub(featureMins).add(1e-6));
  const features = featuresNormalized.clipByValue(0, 1); // Clamp to 0-1 range

  // Predict
  const logits = features.dot(weights).add(bias);
  const probabilityTensor = logits.sigmoid();
  const probability = (await probabilityTensor.data())[0];

  console.log("Raw Features:", await featuresRaw.array());
  console.log(
    "Normalized Features (pre-clamp):",
    await featuresNormalized.array()
  );
  console.log("Normalized Features (post-clamp):", await features.array());
  console.log("Weights:", await weights.array());
  console.log("Bias:", (await bias.data())[0]);
  console.log("Logits:", (await logits.data())[0]);
  console.log("Probability:", probability);

  // Conditions for metConditions
  const conditions: { name: string; met: boolean; weight: number }[] = [
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
  ];

  const metConditions = conditions
    .filter((cond) => cond.met)
    .map((cond) => cond.name);
  const recommendation = probability >= 0.5 ? "sell" : "hold";

  // Clean up tensors
  weights.dispose();
  bias.dispose();
  featureMins.dispose();
  featureMaxs.dispose();
  featuresRaw.dispose();
  features.dispose();
  logits.dispose();
  probabilityTensor.dispose();

  return {
    metConditions,
    probability,
    recommendation,
  };
}
