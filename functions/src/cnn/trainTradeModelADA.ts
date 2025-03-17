import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import * as admin from "firebase-admin";
import { computeFeatures } from "./computeFeatures";
import { labelData } from "./labelData";
import serviceAccount from "../../../serviceAccount.json";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";

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

const weightedCrossEntropy = (
  yTrue: tf.Tensor,
  yPred: tf.Tensor
): tf.Scalar => {
  const weights = tf.tensor1d([4.5, 2.5, 2.0]); // Sell, Hold, Buy
  const ce = tf.losses.softmaxCrossEntropy(yTrue, yPred);
  const weightedCe = ce.mul(yTrue.mul(weights).sum(-1)).mean() as tf.Scalar;
  weights.dispose();
  return weightedCe;
};

export const trainTradeModelADA = async () => {
  const { prices, volumes } = await getHistoricalData(
    "ADA",
    FIVE_YEARS_IN_DAYS
  );
  const timesteps = 7;
  const X: number[][][] = [];
  const y: number[] = [];

  // Precompute and standardize features globally with time warping
  const allFeatures: number[] = [];
  for (let i = 34 + timesteps - 1; i < prices.length; i++) {
    const sequence: number[][] = [];
    const warpFactor = 0.95 + Math.random() * 0.1; // Random stretch/compress
    for (
      let j = Math.floor((i - timesteps + 1) * warpFactor);
      j <= Math.floor(i * warpFactor);
      j++
    ) {
      const idx = Math.min(Math.max(j, 0), prices.length - 1); // Clamp index
      const features = computeFeatures(prices, volumes, idx, prices[idx]);
      if (features.length !== 28) {
        console.error(
          `Unexpected feature length: ${features.length} at index ${idx}`
        );
        continue;
      }
      sequence.push(features);
      allFeatures.push(...features);
    }
    while (sequence.length < timesteps)
      sequence.push(sequence[sequence.length - 1]); // Pad if needed
    while (sequence.length > timesteps) sequence.pop(); // Trim if needed
    const label = labelData({
      prices,
      dayIndex: i,
      threshold: 0.02,
      horizon: 3,
    });
    X.push(sequence);
    y.push(label);
  }

  const mean = allFeatures.reduce((a, b) => a + b, 0) / allFeatures.length;
  const std =
    Math.sqrt(
      allFeatures.reduce((a, b) => a + (b - mean) ** 2, 0) / allFeatures.length
    ) || 1;
  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < X[i].length; j++) {
      X[i][j] = X[i][j].map(
        (f) => ((f - mean) / std) * (0.95 + Math.random() * 0.1)
      );
    }
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

  const X_tensor = tf.tensor3d(X);
  const y_tensor = tf.oneHot(tf.tensor1d(y, "int32"), 3);
  const X_min = X_tensor.min([0, 1]);
  const X_max = X_tensor.max([0, 1]);
  const X_normalized = X_tensor.sub(X_min).div(X_max.sub(X_min).add(1e-6));

  const totalSamples = X.length;
  const trainSize = Math.floor(totalSamples * 0.75);
  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  tf.util.shuffle(indices);

  const trainIndices = indices.slice(0, trainSize);
  const valIndices = indices.slice(trainSize);

  const X_train = tf.gather(X_normalized, trainIndices);
  const y_train = tf.gather(y_tensor, trainIndices);
  const X_val = tf.gather(X_normalized, valIndices);
  const y_val = tf.gather(y_tensor, valIndices);

  const trainDataset = tf.data
    .zip({
      xs: tf.data.array<number[][]>(X_train.arraySync() as number[][][]),
      ys: tf.data.array<number[]>(y_train.arraySync() as number[][]),
    })
    .shuffle(trainSize)
    .batch(128)
    .prefetch(2);

  const valDataset = tf.data
    .zip({
      xs: tf.data.array<number[][]>(X_val.arraySync() as number[][][]),
      ys: tf.data.array<number[]>(y_val.arraySync() as number[][]),
    })
    .batch(128);

  const input = tf.input({ shape: [timesteps, 28] });
  const conv1 = tf.layers
    .conv1d({
      filters: 64,
      kernelSize: 2,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      padding: "same",
    })
    .apply(input) as tf.SymbolicTensor;
  const bn1 = tf.layers
    .batchNormalization({ momentum: 0.9 })
    .apply(conv1) as tf.SymbolicTensor;
  const pool1 = tf.layers
    .maxPooling1d({ poolSize: 2, strides: 2 })
    .apply(bn1) as tf.SymbolicTensor;
  const conv2 = tf.layers
    .conv1d({
      filters: 32,
      kernelSize: 2,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      padding: "same",
    })
    .apply(pool1) as tf.SymbolicTensor;
  const bn2 = tf.layers
    .batchNormalization({ momentum: 0.9 })
    .apply(conv2) as tf.SymbolicTensor;
  const gru = tf.layers
    .gru({ units: 32, returnSequences: false })
    .apply(bn2) as tf.SymbolicTensor;
  const dropout = tf.layers
    .dropout({ rate: 0.3 })
    .apply(gru) as tf.SymbolicTensor;
  const dense = tf.layers
    .dense({
      units: 3,
      activation: "softmax",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    })
    .apply(dropout) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: dense });

  console.log("Model summary:");
  model.summary();

  const initialLearningRate = 0.001;
  const optimizer = tf.train.adam(initialLearningRate);

  model.compile({
    optimizer,
    loss: weightedCrossEntropy as any,
    metrics: [tf.metrics.categoricalAccuracy, customPrecision, customRecall],
  });

  let bestValLoss = Infinity;
  let bestWeights: tf.Tensor[] | null = null;

  class BestWeightsCallback extends tf.CustomCallback {
    constructor(args: tf.CustomCallbackArgs) {
      super(args);
    }

    async onEpochEnd(epoch: number, logs?: tf.Logs) {
      if (logs && logs.val_loss < bestValLoss) {
        bestValLoss = logs.val_loss;
        bestWeights = model.getWeights();
        console.log(`New best val_loss: ${bestValLoss} at epoch ${epoch + 1}`);
      }
    }
  }

  class CosineDecayCallback extends tf.CustomCallback {
    constructor(args: tf.CustomCallbackArgs) {
      super(args);
    }

    async onEpochBegin(epoch: number) {
      const cycleLength = 75;
      const progress = (epoch % cycleLength) / cycleLength;
      const newLr =
        (initialLearningRate * (1 + Math.cos(Math.PI * progress))) / 2;
      // @ts-ignore learningRate is protected
      optimizer.learningRate = newLr;
      console.log(`Epoch ${epoch + 1}: Learning rate set to ${newLr}`);
    }
  }

  console.log("Starting model training...");
  await model.fitDataset(trainDataset, {
    epochs: 150,
    validationData: valDataset,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 30 }),
      new BestWeightsCallback({}),
      new CosineDecayCallback({}),
    ],
  });

  if (bestWeights) {
    model.setWeights(bestWeights);
    console.log("Restored best weights from training.");
  }

  console.log("Model training completed.");

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

  const conv1Weights = model.getLayer("conv1d_Conv1D1").getWeights();
  const conv2Weights = model.getLayer("conv1d_Conv1D2").getWeights();
  const gruWeights = model.getLayer("gru_GRU1").getWeights();
  const denseWeights = model.getLayer("dense_Dense1").getWeights();

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
      gruWeights: gruWeights[0] ? Array.from(await gruWeights[0].data()) : [],
      gruRecurrentWeights: gruWeights[1]
        ? Array.from(await gruWeights[1].data())
        : [],
      gruBias: gruWeights[2] ? Array.from(await gruWeights[2].data()) : [],
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
  X_min.dispose();
  X_max.dispose();
  preds.dispose();
  X_train.dispose();
  y_train.dispose();
  X_val.dispose();
  y_val.dispose();
};

trainTradeModelADA().catch(console.error);
