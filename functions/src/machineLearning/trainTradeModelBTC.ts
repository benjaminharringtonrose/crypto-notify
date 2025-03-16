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

export const trainTradeModelBTC = async () => {
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

  const buyCount = y.filter((label) => label === 2).length;
  const holdCount = y.filter((label) => label === 1).length;
  const sellCount = y.filter((label) => label === 0).length;
  console.log(
    `Training data: ${buyCount} buy (${((buyCount / y.length) * 100).toFixed(
      2
    )}%), ` +
      `${holdCount} hold (${((holdCount / y.length) * 100).toFixed(2)}%), ` +
      `${sellCount} sell (${((sellCount / y.length) * 100).toFixed(
        2
      )}%), total: ${y.length}`
  );

  if (
    buyCount / y.length < 0.1 ||
    holdCount / y.length < 0.1 ||
    sellCount / y.length < 0.1
  ) {
    console.warn(
      "Data imbalance detected - one class is less than 10% of total samples"
    );
  }

  const X_tensor = tf.tensor3d(X);
  const y_tensor = tf.oneHot(tf.tensor1d(y, "int32"), 3); // One-hot encode for 3 classes

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
      units: 3, // 3 classes: buy, hold, sell
      activation: "softmax", // Softmax for multi-class
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.002),
    loss: "categoricalCrossentropy", // Multi-class loss
    metrics: ["accuracy"],
  });

  const earlyStoppingCallback = new EarlyStopping({
    monitor: "val_acc",
    patience: 30,
    mode: "max",
  });

  const classWeight = { 0: 5.0, 1: 1.0, 2: 5.0 }; // Weight rare classes (buy/sell) higher

  await model.fit(X_normalized, y_tensor, {
    epochs: 200,
    batchSize: 64,
    validationSplit: 0.2,
    callbacks: [earlyStoppingCallback],
    classWeight,
  });

  const preds = model.predict(X_normalized) as tf.Tensor;
  const predArray = (await preds.array()) as number[][]; // Array of [sellProb, holdProb, buyProb]
  const yArray = Array.from(await y_tensor.argMax(-1).data()); // Ground truth labels

  const predictedLabels = predArray.map((p: number[]) =>
    p.indexOf(Math.max(...p))
  );
  console.log(
    "Predictions vs Labels (first 5):",
    predictedLabels.slice(0, 5),
    yArray.slice(0, 5)
  );

  const truePositivesBuy = predictedLabels.reduce(
    (sum, p, i) => sum + (p === 2 && yArray[i] === 2 ? 1 : 0),
    0
  );
  const truePositivesSell = predictedLabels.reduce(
    (sum, p, i) => sum + (p === 0 && yArray[i] === 0 ? 1 : 0),
    0
  );
  const predictedBuys = predictedLabels.reduce(
    (sum, p) => sum + (p === 2 ? 1 : 0),
    0
  );
  const predictedSells = predictedLabels.reduce(
    (sum, p) => sum + (p === 0 ? 1 : 0),
    0
  );
  const precisionBuy = predictedBuys > 0 ? truePositivesBuy / predictedBuys : 0;
  const precisionSell =
    predictedSells > 0 ? truePositivesSell / predictedSells : 0;
  console.log(
    `Precision Buy: ${precisionBuy.toFixed(
      4
    )}, Precision Sell: ${precisionSell.toFixed(4)}`
  );

  const lstmLayerWeights = model.layers[0].getWeights();
  const [lstmKernel, lstmRecurrentKernel, lstmBias] = lstmLayerWeights;
  const denseLayerWeights = model.layers[2].getWeights();
  const [denseWeights, denseBias] = denseLayerWeights;

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
      denseBias: Array.from(await denseBias.data()),
      featureMins: Array.from(await X_min.data()),
      featureMaxs: Array.from(await X_max.data()),
    }),
  };

  await admin
    .firestore()
    .collection(Collections.Models)
    .doc(Docs.TradePredictor)
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

trainTradeModelBTC().catch(console.error);
