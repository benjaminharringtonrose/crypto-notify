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
    if (features.length !== 26) {
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

  console.log("X_tensor shape:", X_tensor.shape);
  console.log("y_tensor shape:", y_tensor.shape);

  const X_min = X_tensor.min(0);
  const X_max = X_tensor.max(0);
  const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 1, activation: "sigmoid", inputShape: [26] })
  );
  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(X_normalized, y_tensor, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch}: loss = ${logs?.loss}, accuracy = ${logs?.acc}`
        );
      },
    },
  });

  const [weights, bias] = model.getWeights();
  const weightsArray = await weights.data();
  const biasValue = (await bias.data())[0];
  const featureMinsArray = Array.from(await X_min.data());
  const featureMaxsArray = Array.from(await X_max.data());

  console.log("Weights shape:", weights.shape); // Should be [26, 1]
  console.log("Weights array length:", weightsArray.length); // Should be 26

  const weightsJson = {
    weights: Array.from(weightsArray),
    bias: biasValue,
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
  weights.dispose();
  bias.dispose();
  X_min.dispose();
  X_max.dispose();
};

trainSellModel().catch(console.error);
