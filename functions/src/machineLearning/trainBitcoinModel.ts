import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { computeFeatures } from "./computeFeatures";
import { labelData } from "./labelData";
import serviceAccount from "../../../serviceAccount.json";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { Collections, Docs } from "../types";
import { EarlyStopping } from "@tensorflow/tfjs-node";

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
});

export const trainBitcoinModel = async () => {
  const { prices, volumes } = await getHistoricalData(
    "BTC",
    FIVE_YEARS_IN_DAYS
  );
  const timesteps = 7;
  const X: number[][][] = [];
  const y: number[] = [];

  for (let i = 34 + timesteps - 1; i < prices.length; i++) {
    const sequence: number[][] = [];
    for (let j = i - timesteps + 1; j <= i; j++) {
      const features = computeFeatures(prices, volumes, j, prices[j]);
      if (features.length !== 28) {
        console.error(
          `Unexpected feature length: ${features.length} at index ${j}`
        );
        continue;
      }
      sequence.push(features);
    }
    const label = labelData({
      prices,
      dayIndex: i,
      threshold: 0.02,
      horizon: 3,
    });
    X.push(sequence);
    y.push(label);
  }

  console.log(`Total samples: ${X.length}`);
  console.log(`Feature sequence sample (first entry): ${JSON.stringify(X[0])}`);
  console.log(`Label sample (first 5): ${y.slice(0, 5)}`);

  const sellCount = y.filter((label) => label === 1).length;
  const holdCount = y.filter((label) => label === 0).length;
  console.log(
    `Training data: ${sellCount} sell (${((sellCount / y.length) * 100).toFixed(
      2
    )}%), ${holdCount} hold (${((holdCount / y.length) * 100).toFixed(
      2
    )}%), total: ${y.length}`
  );

  if (sellCount / y.length < 0.1 || holdCount / y.length < 0.1) {
    console.warn(
      "Data imbalance detected - one class is less than 10% of total samples"
    );
  }

  const X_tensor = tf.tensor3d(X);
  const y_tensor = tf.tensor1d(y);

  const X_min = X_tensor.min([0, 1]);
  const X_max = X_tensor.max([0, 1]);
  const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

  const model = tf.sequential();
  model.add(
    tf.layers.lstm({
      units: 64,
      inputShape: [timesteps, 28],
      returnSequences: false,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
  );
  model.add(tf.layers.dropout({ rate: 0.35 }));
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.002),
    loss: "binaryCrossentropy",
    metrics: ["accuracy", "precision"],
  });

  const earlyStoppingCallback = new EarlyStopping({
    monitor: "val_acc",
    patience: 30,
    mode: "max",
  });

  const classWeight = { 0: 1.0, 1: 5.0 };

  await model.fit(X_normalized, y_tensor, {
    epochs: 200,
    batchSize: 64,
    validationSplit: 0.2,
    callbacks: [earlyStoppingCallback],
    classWeight,
  });

  const preds = model.predict(X_normalized) as tf.Tensor;
  const predArray = Array.from(await preds.data());
  const yArray = Array.from(await y_tensor.data());
  const threshold = 0.45;
  const binaryPreds = predArray.map((p) => (p >= threshold ? 1 : 0));

  console.log(
    "Predictions vs Labels (first 5):",
    predArray.slice(0, 5).map((p) => p.toFixed(4)),
    yArray.slice(0, 5)
  );
  console.log(
    "Binary Predictions vs Labels (first 10):",
    binaryPreds.slice(0, 10),
    yArray.slice(0, 10)
  );

  const sellIndices = yArray.reduce(
    (acc: number[], label, i) => (label === 1 ? [...acc, i] : acc),
    []
  );
  console.log(
    "Predictions vs Labels (first 5 sell samples):",
    sellIndices.slice(0, 5).map((i) => predArray[i].toFixed(4)),
    sellIndices.slice(0, 5).map((i) => yArray[i])
  );

  const truePositives = binaryPreds.reduce(
    (sum: number, p, i) => sum + (p === 1 && yArray[i] === 1 ? 1 : 0),
    0 as number
  );
  const predictedPositives = binaryPreds.reduce(
    (sum: number, p) => sum + p,
    0 as number
  );
  const precision =
    predictedPositives > 0 ? truePositives / predictedPositives : 0;
  console.log(
    `Custom Precision (threshold=${threshold}): ${precision.toFixed(4)}`
  );

  const tp = binaryPreds.reduce(
    (sum: number, p, i) => sum + (p === 1 && yArray[i] === 1 ? 1 : 0),
    0 as number
  );
  const fp = binaryPreds.reduce(
    (sum: number, p, i) => sum + (p === 1 && yArray[i] === 0 ? 1 : 0),
    0 as number
  );
  const tn = binaryPreds.reduce(
    (sum: number, p, i) => sum + (p === 0 && yArray[i] === 0 ? 1 : 0),
    0 as number
  );
  const fn = binaryPreds.reduce(
    (sum: number, p, i) => sum + (p === 0 && yArray[i] === 1 ? 1 : 0),
    0 as number
  );
  console.log(`Confusion Matrix: TP=${tp}, FP=${fp}, TN=${tn}, FN=${fn}`);

  const recall = tp / (tp + fn);
  const f1 = (2 * (precision * recall)) / (precision + recall || 1);
  console.log(`F1 Score (threshold=${threshold}): ${f1.toFixed(4)}`);

  // Get weights from the LSTM layer directly
  const lstmLayerWeights = model.layers[0].getWeights();
  const [lstmKernel, lstmRecurrentKernel, lstmBias] = lstmLayerWeights;
  const denseLayerWeights = model.layers[2].getWeights();
  const [denseWeights, denseBias] = denseLayerWeights;

  // Log tensor shapes and sizes
  console.log(
    "LSTM Kernel shape:",
    lstmKernel.shape,
    "size:",
    (await lstmKernel.data()).length
  );
  console.log(
    "LSTM Recurrent Kernel shape:",
    lstmRecurrentKernel.shape,
    "size:",
    (await lstmRecurrentKernel.data()).length
  );
  console.log(
    "LSTM Bias shape:",
    lstmBias.shape,
    "size:",
    (await lstmBias.data()).length
  );
  console.log(
    "Dense Weights shape:",
    denseWeights.shape,
    "size:",
    (await denseWeights.data()).length
  );
  console.log(
    "Dense Bias shape:",
    denseBias.shape,
    "size:",
    (await denseBias.data()).length
  );

  const weightsJson = {
    weights: JSON.stringify({
      lstmWeights: Array.from(await lstmKernel.data()),
      lstmBias: Array.from(await lstmBias.data()),
      lstmRecurrentWeights: Array.from(await lstmRecurrentKernel.data()),
      denseWeights: Array.from(await denseWeights.data()),
      denseBias: (await denseBias.data())[0],
      featureMins: Array.from(await X_min.data()),
      featureMaxs: Array.from(await X_max.data()),
    }),
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
  lstmKernel.dispose();
  lstmRecurrentKernel.dispose();
  lstmBias.dispose();
  denseWeights.dispose();
  denseBias.dispose();
  X_min.dispose();
  X_max.dispose();
  preds.dispose();
};

trainBitcoinModel().catch(console.error);
