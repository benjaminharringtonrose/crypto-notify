import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { computeFeatures } from "./computeFeatures";
import { labelData } from "./labelData";
import serviceAccount from "../../../serviceAccount.json";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { EarlyStopping } from "@tensorflow/tfjs-node";
import { PrecisionLogger } from "./callbacks/PrecisionLogger";
import { ExponentialDecayLR } from "./callbacks/ExponentialDecayLR";
import { ReduceLROnPlateau } from "./callbacks/ReduceLROnPlateau";

dotenv.config();

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
  storageBucket: process.env.STORAGE_BUCKET,
});

function customPrecision(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
  const truePos = yTrue.mul(yPred.round()).sum();
  const predPos = yPred.round().sum();
  return truePos.div(predPos.add(1e-6)) as tf.Scalar;
}

function customRecall(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar {
  const truePos = yTrue.mul(yPred.round()).sum();
  const actualPos = yTrue.sum();
  return truePos.div(actualPos.add(1e-6)) as tf.Scalar;
}

export const trainTradeModelADA = async () => {
  const { prices, volumes } = await getHistoricalData(
    "ADA",
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
  const y_tensor = tf.oneHot(tf.tensor1d(y, "int32"), 3);

  const X_min = X_tensor.min([0, 1]);
  const X_max = X_tensor.max([0, 1]);
  const noise = tf.randomNormal(X_tensor.shape, 0, 0.1);
  const X_noisy = X_tensor.add(noise);
  const X_normalized = X_noisy.sub(X_min).div(X_max.sub(X_min).add(1e-6));

  const model = tf.sequential();
  model.add(
    tf.layers.conv1d({
      filters: 64,
      kernelSize: 2,
      inputShape: [timesteps, 28],
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.02 }),
      padding: "same",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(
    tf.layers.conv1d({
      filters: 32,
      kernelSize: 2,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.02 }),
      padding: "same",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(
    tf.layers.conv1d({
      filters: 16,
      kernelSize: 2,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.02 }),
      padding: "same",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.02 }),
    })
  );

  function focalLoss(gamma = 2.0, alpha = 0.3) {
    return (yTrue: tf.Tensor, yPred: tf.Tensor) => {
      const ce = tf.losses.softmaxCrossEntropy(yTrue, yPred);
      const pt = yTrue.mul(yPred).sum(-1);
      const focalWeight = tf.pow(tf.sub(1, pt), gamma);
      return ce.mul(focalWeight).mul(alpha).mean();
    };
  }

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: focalLoss(),
    metrics: [tf.metrics.categoricalAccuracy, customPrecision, customRecall],
  });

  const earlyStoppingCallback = new EarlyStopping({
    monitor: "val_categoricalAccuracy",
    patience: 20,
    mode: "max",
  });

  const reduceLrCallback = new ReduceLROnPlateau({
    monitor: "val_loss",
    factor: 0.5,
    patience: 10,
    minLr: 0.0001,
    initialLr: 0.001,
  });

  const precisionLogger = new PrecisionLogger({});

  const lrDecayCallback = new ExponentialDecayLR(0.001, (500 * 1807) / 32, 0.5);

  const classWeight = { 0: 3.0, 1: 2.5, 2: 2.5 };

  await model.fit(X_normalized, y_tensor, {
    epochs: 150,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: [
      earlyStoppingCallback,
      reduceLrCallback,
      precisionLogger,
      lrDecayCallback,
    ],
    classWeight,
  });

  const preds = model.predict(X_normalized) as tf.Tensor;
  const predArray = (await preds.array()) as number[][];
  console.log("Sample predictions:", predArray.slice(0, 5));
  const yArray = Array.from(await y_tensor.argMax(-1).data());

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

  const confusionMatrix = tf.math.confusionMatrix(
    tf.tensor1d(yArray, "int32"),
    tf.tensor1d(predictedLabels, "int32"),
    3
  );
  console.log("Confusion Matrix:", await confusionMatrix.array());

  const conv1Weights = model.layers[0].getWeights();
  const conv2Weights = model.layers[3].getWeights();
  const conv3Weights = model.layers[6].getWeights();
  const denseWeights = model.layers[9].getWeights();

  const weightsJson = {
    weights: {
      conv1Weights: conv1Weights[0]
        ? Array.from(await conv1Weights[0].data())
        : [],
      conv1Bias: conv1Weights[1]
        ? Array.from(await conv1Weights[1].data())
        : [],
      conv2Weights: conv2Weights[0]
        ? Array.from(await conv2Weights[0].data())
        : [],
      conv2Bias: conv2Weights[1]
        ? Array.from(await conv2Weights[1].data())
        : [],
      conv3Weights: conv3Weights[0]
        ? Array.from(await conv3Weights[0].data())
        : [],
      conv3Bias: conv3Weights[1]
        ? Array.from(await conv3Weights[1].data())
        : [],
      denseWeights: denseWeights[0]
        ? Array.from(await denseWeights[0].data())
        : [],
      denseBias: denseWeights[1]
        ? Array.from(await denseWeights[1].data())
        : [],
      featureMins: Array.from(await X_min.data()),
      featureMaxs: Array.from(await X_max.data()),
    },
  };

  const bucket = admin.storage().bucket();
  await bucket
    .file("tradePredictorWeights.json")
    .save(JSON.stringify(weightsJson));
  console.log("Model weights saved to Cloud Storage");

  X_tensor.dispose();
  y_tensor.dispose();
  X_normalized.dispose();
  noise.dispose();
  if (conv1Weights[0]) conv1Weights[0].dispose();
  if (conv1Weights[1]) conv1Weights[1].dispose();
  if (conv2Weights[0]) conv2Weights[0].dispose();
  if (conv2Weights[1]) conv2Weights[1].dispose();
  if (conv3Weights[0]) conv3Weights[0].dispose();
  if (conv3Weights[1]) conv3Weights[1].dispose();
  if (denseWeights[0]) denseWeights[0].dispose();
  if (denseWeights[1]) denseWeights[1].dispose();
  X_min.dispose();
  X_max.dispose();
  preds.dispose();
};

trainTradeModelADA().catch(console.error);
