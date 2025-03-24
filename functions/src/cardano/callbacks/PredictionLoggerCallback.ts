import * as tf from "@tensorflow/tfjs-node";

export class PredictionLoggerCallback extends tf.CustomCallback {
  public readonly validationData: tf.Tensor;
  private model?: tf.LayersModel;
  private lossFn?: (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Scalar;
  private X_train?: tf.Tensor;
  private y_train?: tf.Tensor;

  constructor(
    validationData: tf.Tensor,
    X_train?: tf.Tensor,
    y_train?: tf.Tensor,
    lossFn?: (yTrue: tf.Tensor, yPred: tf.Tensor) => tf.Scalar
  ) {
    super({});
    this.validationData = validationData;
    this.X_train = X_train;
    this.y_train = y_train;
    this.lossFn = lossFn;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs) {
    if (!this.model)
      throw new Error("Model not set in PredictionLoggerCallback");

    // Log first 5 validation predictions (existing functionality)
    const predsFirstFive = this.model.predict(
      this.validationData.slice([0], [5])
    ) as tf.Tensor;
    const predArrayFirstFive = (await predsFirstFive.array()) as number[][];
    const predictedLabelsFirstFive = predArrayFirstFive.map((p) =>
      p[1] > 0.3 ? 1 : 0
    ); // Adjusted threshold
    const buyCountFirstFive = predictedLabelsFirstFive.filter(
      (p) => p === 1
    ).length;
    console.log(
      `Epoch ${epoch + 1} Predictions (first 5):`,
      predArrayFirstFive
    );
    console.log(
      `Epoch ${
        epoch + 1
      } Predicted Buy (first 5): ${buyCountFirstFive}, Sell: ${
        5 - buyCountFirstFive
      }`
    );

    // New logging: Class distribution across all validation data
    const predsAll = this.model.predict(this.validationData) as tf.Tensor;
    const predLabelsAll = predsAll.argMax(-1).dataSync();
    const buyCountAll = Array.from(predLabelsAll).filter((p) => p === 1).length;
    const totalValSamples = predLabelsAll.length;
    console.log(
      `Epoch ${epoch + 1} Validation Buy Count: ${buyCountAll}, Sell Count: ${
        totalValSamples - buyCountAll
      }, Buy Ratio: ${(buyCountAll / totalValSamples).toFixed(3)}`
    );

    if (logs)
      console.log(`Epoch ${epoch + 1} Val Loss: ${logs.val_loss.toFixed(4)}`);

    predsFirstFive.dispose();
    predsAll.dispose();
  }

  async onBatchEnd(batch: number, logs?: tf.Logs) {
    if (
      logs &&
      batch % 10 === 0 &&
      this.X_train &&
      this.y_train &&
      this.lossFn
    ) {
      const preds = this.model!.predict(this.X_train) as tf.Tensor;
      const loss = this.lossFn(this.y_train, preds);
      const grads = tf.variableGrads(() =>
        this.lossFn!(
          this.y_train!,
          this.model!.predict(this.X_train!) as tf.Tensor
        )
      );
      const gradList = Object.values(grads.grads) as tf.Tensor[];
      const gradNorm = tf
        .sqrt(
          gradList
            .map((g: tf.Tensor) => g.square().sum())
            .reduce((a: tf.Tensor, b: tf.Tensor) => a.add(b), tf.scalar(0))
        )
        .dataSync()[0];
      console.log(
        `Batch ${batch} Gradient Norm: ${gradNorm}, Loss: ${loss
          .dataSync()[0]
          .toFixed(4)}`
      );
      preds.dispose();
      loss.dispose();
      grads.value.dispose();
      gradList.forEach((g) => g.dispose());
    }
  }

  setParams(params: tf.CustomCallbackArgs) {
    // Required by tf.CustomCallback
  }

  setModel(model: tf.LayersModel) {
    this.model = model;
  }
}
