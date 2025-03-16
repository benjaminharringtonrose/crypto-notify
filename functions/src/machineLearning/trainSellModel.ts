import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { computeFeatures } from "./computeFeatures";
import { labelData } from "./labelData";
import serviceAccount from "../../../serviceAccount.json";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { Collections, Docs } from "../types";

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
});

export const trainSellModel = async () => {
  const { prices, volumes } = await getHistoricalData(
    "BTC",
    FIVE_YEARS_IN_DAYS
  );
  const X: number[][] = [];
  const y: number[] = [];

  for (let i = 34; i < prices.length; i++) {
    const features = computeFeatures(prices, volumes, i, prices[i]);
    if (features.length !== 28) {
      console.error(
        `Unexpected feature length: ${features.length} at index ${i}`
      );
      continue;
    }
    const label = labelData({
      prices,
      dayIndex: i,
      threshold: 0.01,
      horizon: 3,
    });
    X.push(features);
    y.push(label);
  }

  const sellCount = y.filter((label) => label === 1).length;
  const holdCount = y.filter((label) => label === 0).length;
  console.log(
    `Training data: ${sellCount} sell, ${holdCount} hold, total: ${y.length}`
  );

  const X_tensor = tf.tensor2d(X);
  const y_tensor = tf.tensor1d(y);

  const X_min = X_tensor.min(0);
  const X_max = X_tensor.max(0);
  const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 16, activation: "relu", inputShape: [28] })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(X_normalized, y_tensor, {
    epochs: 100,
    batchSize: 64,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch}: loss = ${logs?.loss}, accuracy = ${logs?.acc}`
        );
      },
    },
  });

  const [weights1, bias1, weights2, bias2] = model.getWeights();
  const weights1Array = await weights1.data();
  const bias1Array = await bias1.data();
  const weights2Array = await weights2.data();
  const bias2Value = (await bias2.data())[0];
  const featureMinsArray = Array.from(await X_min.data());
  const featureMaxsArray = Array.from(await X_max.data());

  const weightsJson = {
    weights1: Array.from(weights1Array),
    bias1: Array.from(bias1Array),
    weights2: Array.from(weights2Array),
    bias2: bias2Value,
    featureMins: featureMinsArray,
    featureMaxs: featureMaxsArray,
  };

  await admin
    .firestore()
    .collection(Collections.Models)
    .doc(Docs.SellPredictor)
    .set(weightsJson);

  console.log("Model weights saved to Firestore");

  X_tensor.dispose();
  y_tensor.dispose();
  X_normalized.dispose();
  weights1.dispose();
  bias1.dispose();
  weights2.dispose();
  bias2.dispose();
  X_min.dispose();
  X_max.dispose();
};

trainSellModel().catch(console.error);
