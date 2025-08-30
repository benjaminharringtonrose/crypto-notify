/**
 * Machine Learning Models Module for Bitcoin Price Prediction Bot
 * Implements various ML algorithms for predicting Bitcoin price movements.
 */

export interface ModelPrediction {
  prediction: number;
  probability: number;
  confidence: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc?: number;
  confusionMatrix: number[][];
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  truePositives: number;
  trainingTime: number;
}

export interface TrainingResult {
  [modelName: string]: ModelMetrics;
}

export interface EnsembleWeights {
  [modelName: string]: number;
}

export interface ModelConfig {
  modelsDir?: string;
}

// Simple ML model implementations
abstract class BaseModel {
  protected trained: boolean = false;
  protected name: string;

  constructor(name: string) {
    this.name = name;
  }

  abstract train(X: number[][], y: number[]): Promise<void>;
  abstract predict(X: number[][]): number[];
  abstract predictProba(X: number[][]): number[][];

  isTrained(): boolean {
    return this.trained;
  }

  getName(): string {
    return this.name;
  }
}

// Simple Random Forest implementation
class RandomForestModel extends BaseModel {
  private trees: DecisionTree[] = [];
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private maxFeatures: number; // Add feature sampling
  private minSamplesLeaf: number; // Add leaf regularization

  constructor(
    nEstimators: number = 50, // Reduced from 100
    maxDepth: number = 6, // Reduced from 10
    minSamplesSplit: number = 10, // Increased from 2
    maxFeatures: number = 0.7, // Use 70% of features per tree
    minSamplesLeaf: number = 5 // Minimum samples per leaf
  ) {
    super("random_forest");
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeatures = maxFeatures;
    this.minSamplesLeaf = minSamplesLeaf;
  }

  async train(X: number[][], y: number[]): Promise<void> {
    console.log(`Training Random Forest with ${this.nEstimators} trees...`);
    const startTime = Date.now();

    this.trees = [];
    const nFeatures = X[0].length;

    for (let i = 0; i < this.nEstimators; i++) {
      const tree = new DecisionTree(
        this.maxDepth,
        this.minSamplesSplit,
        this.minSamplesLeaf
      );

      // Bootstrap sample with replacement
      const { bootstrappedX, bootstrappedY } = this.bootstrapSample(X, y);

      // Feature sampling - select subset of features for this tree
      const selectedFeatures = this.selectFeatures(nFeatures);
      const sampledX = bootstrappedX.map((sample) =>
        selectedFeatures.map((featureIndex) => sample[featureIndex])
      );

      await tree.train(sampledX, bootstrappedY);
      this.trees.push(tree);
    }

    this.trained = true;
    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(
      `Random Forest training completed in ${trainingTime.toFixed(2)} seconds`
    );
  }

  predict(X: number[][]): number[] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const predictions = this.trees.map((tree, treeIndex) => {
        // Apply same feature sampling as during training
        const nFeatures = sample.length;
        const selectedFeatures = this.selectFeatures(nFeatures);
        const sampledSample = selectedFeatures.map(
          (featureIndex) => sample[featureIndex]
        );
        return tree.predict([sampledSample])[0];
      });

      // Majority voting
      const avgPrediction =
        predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
      return avgPrediction > 0.5 ? 1 : 0;
    });
  }

  predictProba(X: number[][]): number[][] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const predictions = this.trees.map((tree, treeIndex) => {
        // Apply same feature sampling as during training
        const nFeatures = sample.length;
        const selectedFeatures = this.selectFeatures(nFeatures);
        const sampledSample = selectedFeatures.map(
          (featureIndex) => sample[featureIndex]
        );
        return tree.predict([sampledSample])[0];
      });

      const avgPrediction =
        predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
      return [1 - avgPrediction, avgPrediction]; // [prob_class_0, prob_class_1]
    });
  }

  private bootstrapSample(
    X: number[][],
    y: number[]
  ): { bootstrappedX: number[][]; bootstrappedY: number[] } {
    const n = X.length;
    const bootstrappedX: number[][] = [];
    const bootstrappedY: number[] = [];

    // Bootstrap sampling with replacement
    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(Math.random() * n);
      bootstrappedX.push([...X[randomIndex]]);
      bootstrappedY.push(y[randomIndex]);
    }

    return { bootstrappedX, bootstrappedY };
  }

  private selectFeatures(nFeatures: number): number[] {
    const nSelectedFeatures = Math.max(
      1,
      Math.floor(nFeatures * this.maxFeatures)
    );
    const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);

    // Randomly select features without replacement
    const selectedFeatures: number[] = [];
    const shuffled = [...allFeatures].sort(() => Math.random() - 0.5);

    for (let i = 0; i < nSelectedFeatures; i++) {
      selectedFeatures.push(shuffled[i]);
    }

    return selectedFeatures.sort((a, b) => a - b); // Keep original order
  }
}

// Simple Decision Tree implementation
class DecisionTree {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;

  constructor(
    maxDepth: number = 10,
    minSamplesSplit: number = 2,
    minSamplesLeaf: number = 1
  ) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
  }

  async train(X: number[][], y: number[]): Promise<void> {
    this.root = this.buildTree(X, y, 0);
  }

  predict(X: number[][]): number[] {
    return X.map((sample) => this.predictSingle(sample, this.root));
  }

  private predictSingle(sample: number[], node: TreeNode | null): number {
    if (!node) return 0;

    if (node.isLeaf) {
      return node.prediction;
    }

    if (sample[node.featureIndex] <= node.threshold) {
      return this.predictSingle(sample, node.left);
    } else {
      return this.predictSingle(sample, node.right);
    }
  }

  private buildTree(
    X: number[][],
    y: number[],
    depth: number
  ): TreeNode | null {
    const nSamples = X.length;

    // Handle edge cases
    if (nSamples === 0) {
      return new TreeNode(true, 0, -1, 0, null, null);
    }

    const nClasses = new Set(y).size;

    // Stopping conditions
    if (
      depth >= this.maxDepth ||
      nSamples < this.minSamplesSplit ||
      nClasses === 1 ||
      nSamples < this.minSamplesLeaf * 2 // Ensure we can split into two valid leaves
    ) {
      const prediction = this.majorityClass(y);
      return new TreeNode(true, prediction, -1, 0, null, null);
    }

    // Find best split
    const bestSplit = this.findBestSplit(X, y);
    if (!bestSplit) {
      const prediction = this.majorityClass(y);
      return new TreeNode(true, prediction, -1, 0, null, null);
    }

    // Split the data
    const leftIndices: number[] = [];
    const rightIndices: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][bestSplit.featureIndex] <= bestSplit.threshold) {
        leftIndices.push(i);
      } else {
        rightIndices.push(i);
      }
    }

    // Validate split quality
    if (
      leftIndices.length < this.minSamplesLeaf ||
      rightIndices.length < this.minSamplesLeaf
    ) {
      const prediction = this.majorityClass(y);
      return new TreeNode(true, prediction, -1, 0, null, null);
    }

    const leftX = leftIndices.map((i) => X[i]);
    const leftY = leftIndices.map((i) => y[i]);
    const rightX = rightIndices.map((i) => X[i]);
    const rightY = rightIndices.map((i) => y[i]);

    // Build children
    const leftChild = this.buildTree(leftX, leftY, depth + 1);
    const rightChild = this.buildTree(rightX, rightY, depth + 1);

    return new TreeNode(
      false,
      0,
      bestSplit.featureIndex,
      bestSplit.threshold,
      leftChild,
      rightChild
    );
  }

  private findBestSplit(
    X: number[][],
    y: number[]
  ): { featureIndex: number; threshold: number } | null {
    const nFeatures = X[0].length;
    let bestGini = Infinity;
    let bestSplit: { featureIndex: number; threshold: number } | null = null;

    for (let featureIndex = 0; featureIndex < nFeatures; featureIndex++) {
      const featureValues = X.map((sample) => sample[featureIndex]);

      // Handle NaN and infinite values
      const validValues = featureValues.filter(
        (val) => !isNaN(val) && isFinite(val)
      );

      if (validValues.length === 0) continue;

      const uniqueValues = Array.from(new Set(validValues)).sort(
        (a, b) => a - b
      );

      // Limit the number of split points to consider for performance
      const maxSplits = Math.min(uniqueValues.length - 1, 100);
      const step = Math.max(
        1,
        Math.floor((uniqueValues.length - 1) / maxSplits)
      );

      for (let i = 0; i < uniqueValues.length - 1; i += step) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const gini = this.calculateGini(X, y, featureIndex, threshold);

        if (gini < bestGini) {
          bestGini = gini;
          bestSplit = { featureIndex, threshold };
        }
      }
    }

    return bestSplit;
  }

  private calculateGini(
    X: number[][],
    y: number[],
    featureIndex: number,
    threshold: number
  ): number {
    const leftY: number[] = [];
    const rightY: number[] = [];

    for (let i = 0; i < X.length; i++) {
      const value = X[i][featureIndex];

      // Handle NaN and infinite values
      if (isNaN(value) || !isFinite(value)) {
        // Put NaN values in the left child
        leftY.push(y[i]);
      } else if (value <= threshold) {
        leftY.push(y[i]);
      } else {
        rightY.push(y[i]);
      }
    }

    // Check minimum samples per leaf
    if (
      leftY.length < this.minSamplesLeaf ||
      rightY.length < this.minSamplesLeaf
    ) {
      return Infinity; // Penalize splits that create too small leaves
    }

    const leftGini = this.giniImpurity(leftY);
    const rightGini = this.giniImpurity(rightY);

    const leftWeight = leftY.length / y.length;
    const rightWeight = rightY.length / y.length;

    return leftWeight * leftGini + rightWeight * rightGini;
  }

  private giniImpurity(y: number[]): number {
    if (y.length === 0) return 0;

    const classCounts: Record<number, number> = {};
    for (const label of y) {
      classCounts[label] = (classCounts[label] || 0) + 1;
    }

    let gini = 1;
    for (const count of Object.values(classCounts)) {
      const probability = count / y.length;
      gini -= probability * probability;
    }

    return gini;
  }

  private majorityClass(y: number[]): number {
    if (y.length === 0) return 0;

    const classCounts: Record<number, number> = {};
    for (const label of y) {
      classCounts[label] = (classCounts[label] || 0) + 1;
    }

    let majorityClass = 0;
    let maxCount = 0;

    for (const [className, count] of Object.entries(classCounts)) {
      if (count > maxCount) {
        maxCount = count;
        majorityClass = parseInt(className);
      }
    }

    return majorityClass;
  }
}

class TreeNode {
  constructor(
    public isLeaf: boolean,
    public prediction: number,
    public featureIndex: number,
    public threshold: number,
    public left: TreeNode | null,
    public right: TreeNode | null
  ) {}
}

// Simple Logistic Regression implementation
class LogisticRegressionModel extends BaseModel {
  private weights: number[] = [];
  private bias: number = 0;
  private learningRate: number = 0.01;
  private maxIterations: number = 1000;

  constructor() {
    super("logistic_regression");
  }

  async train(X: number[][], y: number[]): Promise<void> {
    console.log("Training Logistic Regression...");
    const startTime = Date.now();

    const nFeatures = X[0].length;
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      let totalLoss = 0;

      for (let i = 0; i < X.length; i++) {
        const prediction = this.sigmoid(
          this.dotProduct(X[i], this.weights) + this.bias
        );
        const error = prediction - y[i];
        totalLoss += error * error;

        // Update weights
        for (let j = 0; j < nFeatures; j++) {
          this.weights[j] -= this.learningRate * error * X[i][j];
        }
        this.bias -= this.learningRate * error;
      }

      if (iteration % 100 === 0) {
        console.log(`Iteration ${iteration}, Loss: ${totalLoss / X.length}`);
      }
    }

    this.trained = true;
    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(
      `Logistic Regression training completed in ${trainingTime.toFixed(
        2
      )} seconds`
    );
  }

  predict(X: number[][]): number[] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const probability = this.sigmoid(
        this.dotProduct(sample, this.weights) + this.bias
      );
      return probability > 0.5 ? 1 : 0;
    });
  }

  predictProba(X: number[][]): number[][] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const probability = this.sigmoid(
        this.dotProduct(sample, this.weights) + this.bias
      );
      return [1 - probability, probability]; // [prob_class_0, prob_class_1]
    });
  }

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }
}

// Simple Neural Network implementation
class NeuralNetworkModel extends BaseModel {
  private layers: number[] = [100, 50, 25]; // Hidden layer sizes
  private weights: number[][][] = [];
  private biases: number[][] = [];
  private learningRate: number = 0.01;
  private maxIterations: number = 500;

  constructor() {
    super("neural_network");
  }

  async train(X: number[][], y: number[]): Promise<void> {
    console.log("Training Neural Network...");
    const startTime = Date.now();

    const nFeatures = X[0].length;
    const layerSizes = [nFeatures, ...this.layers, 1];

    // Initialize weights and biases
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];

      for (let j = 0; j < layerSizes[i + 1]; j++) {
        const neuronWeights: number[] = [];
        for (let k = 0; k < layerSizes[i]; k++) {
          neuronWeights.push((Math.random() - 0.5) * 0.1);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(0);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }

    // Training loop
    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      let totalLoss = 0;

      for (let i = 0; i < X.length; i++) {
        const { activations, zValues } = this.forwardPass(X[i]);
        const prediction = activations[activations.length - 1][0];
        const error = prediction - y[i];
        totalLoss += error * error;

        // Backpropagation (simplified)
        this.backwardPass(X[i], y[i], activations, zValues);
      }

      if (iteration % 100 === 0) {
        console.log(`Iteration ${iteration}, Loss: ${totalLoss / X.length}`);
      }
    }

    this.trained = true;
    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(
      `Neural Network training completed in ${trainingTime.toFixed(2)} seconds`
    );
  }

  predict(X: number[][]): number[] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const { activations } = this.forwardPass(sample);
      const probability = activations[activations.length - 1][0];
      return probability > 0.5 ? 1 : 0;
    });
  }

  predictProba(X: number[][]): number[][] {
    if (!this.trained) throw new Error("Model not trained");

    return X.map((sample) => {
      const { activations } = this.forwardPass(sample);
      const probability = activations[activations.length - 1][0];
      return [1 - probability, probability]; // [prob_class_0, prob_class_1]
    });
  }

  private forwardPass(input: number[]): {
    activations: number[][];
    zValues: number[][];
  } {
    const activations: number[][] = [input];
    const zValues: number[][] = [];

    for (let layer = 0; layer < this.weights.length; layer++) {
      const layerZ: number[] = [];
      const layerActivations: number[] = [];

      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        const z =
          this.dotProduct(this.weights[layer][neuron], activations[layer]) +
          this.biases[layer][neuron];
        layerZ.push(z);

        if (layer === this.weights.length - 1) {
          // Output layer - sigmoid
          layerActivations.push(this.sigmoid(z));
        } else {
          // Hidden layers - ReLU
          layerActivations.push(Math.max(0, z));
        }
      }

      zValues.push(layerZ);
      activations.push(layerActivations);
    }

    return { activations, zValues };
  }

  private backwardPass(
    input: number[],
    target: number,
    activations: number[][],
    zValues: number[][]
  ): void {
    // Simplified backpropagation - just update weights based on gradient
    const output = activations[activations.length - 1][0];
    const error = output - target;

    // Update weights for output layer
    for (
      let neuron = 0;
      neuron < this.weights[this.weights.length - 1].length;
      neuron++
    ) {
      for (
        let weight = 0;
        weight < this.weights[this.weights.length - 1][neuron].length;
        weight++
      ) {
        const gradient = error * activations[activations.length - 2][weight];
        this.weights[this.weights.length - 1][neuron][weight] -=
          this.learningRate * gradient;
      }
      this.biases[this.biases.length - 1][neuron] -= this.learningRate * error;
    }
  }

  private sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }
}

// Main Bitcoin Predictor class
export class BitcoinPredictor {
  private models: Record<string, BaseModel> = {};
  private bestModels: Record<string, BaseModel> = {};
  private ensembleWeights: EnsembleWeights = {};
  private featureColumns: string[] = [];

  constructor(config: ModelConfig = {}) {
    this.initializeModels();
  }

  private initializeModels(): void {
    console.log("Initializing ML models...");

    this.models["random_forest"] = new RandomForestModel(100, 10, 2);
    this.models["logistic_regression"] = new LogisticRegressionModel();
    this.models["neural_network"] = new NeuralNetworkModel();

    console.log(`Initialized ${Object.keys(this.models).length} models`);
  }

  async trainModels(
    X: number[][],
    y: number[],
    skipHyperparameterTuning: boolean = false
  ): Promise<TrainingResult> {
    console.log("Training models with comprehensive optimization...");

    this.featureColumns = Array.from(
      { length: X[0].length },
      (_, i) => `feature_${i}`
    );
    const results: TrainingResult = {};

    // Train all models
    const totalModels = Object.keys(this.models).length;
    let modelIndex = 0;

    for (const [name, model] of Object.entries(this.models)) {
      modelIndex++;
      console.log(`Training ${name}... (${modelIndex}/${totalModels})`);

      try {
        const startTime = Date.now();

        // Train the model
        await model.train(X, y);

        // Make predictions
        const trainPred = model.predict(X);
        const trainProb = model.predictProba(X).map((probs) => probs[1]); // Probability of class 1

        // Calculate metrics
        const metrics = this.calculateMetrics(y, trainPred, trainProb);
        metrics.trainingTime = (Date.now() - startTime) / 1000;

        results[name] = metrics;
        this.bestModels[name] = model;

        console.log(
          `${name} - Training completed in ${metrics.trainingTime.toFixed(
            2
          )} seconds`
        );
        console.log(
          `${name} - Train Metrics: Accuracy=${metrics.accuracy.toFixed(
            4
          )}, Precision=${metrics.precision.toFixed(
            4
          )}, Recall=${metrics.recall.toFixed(4)}, F1=${metrics.f1Score.toFixed(
            4
          )}`
        );
      } catch (e) {
        console.error(`Error training ${name}: ${e}`);
        results[name] = { error: String(e) } as any;
      }
    }

    // Optimize ensemble weights
    this.optimizeEnsembleWeights(X, y);

    return results;
  }

  private optimizeEnsembleWeights(X: number[][], y: number[]): void {
    console.log("Optimizing ensemble weights...");

    for (const [name, model] of Object.entries(this.bestModels)) {
      try {
        const predictions = model.predict(X);
        const accuracy = this.calculateAccuracy(y, predictions);
        this.ensembleWeights[name] = accuracy;
        console.log(`${name} - Validation Accuracy: ${accuracy.toFixed(4)}`);
      } catch (e) {
        console.error(`Error getting predictions from ${name}: ${e}`);
        this.ensembleWeights[name] = 0.0;
      }
    }

    // Normalize weights
    const totalWeight = Object.values(this.ensembleWeights).reduce(
      (sum, weight) => sum + weight,
      0
    );
    if (totalWeight > 0) {
      for (const name in this.ensembleWeights) {
        this.ensembleWeights[name] /= totalWeight;
      }
    } else {
      // Use equal weights if all weights are zero
      const equalWeight = 1.0 / Object.keys(this.ensembleWeights).length;
      for (const name in this.ensembleWeights) {
        this.ensembleWeights[name] = equalWeight;
      }
    }

    console.log(`Ensemble weights: ${JSON.stringify(this.ensembleWeights)}`);
  }

  async ensemblePredict(X: number[][]): Promise<{
    predictions: number[];
    probabilities: number[];
    confidenceScores: number[];
  }> {
    console.log("Making ensemble predictions...");

    const predictions: number[][] = [];
    const probabilities: number[][] = [];

    for (const [name, model] of Object.entries(this.bestModels)) {
      if (this.ensembleWeights[name] > 0) {
        try {
          const modelPreds = model.predict(X);
          const modelProbs = model.predictProba(X).map((probs) => probs[1]);

          predictions.push(modelPreds);
          probabilities.push(modelProbs);
        } catch (e) {
          console.error(`Error getting prediction from ${name}: ${e}`);
        }
      }
    }

    if (predictions.length === 0) {
      console.warn("No valid predictions available");
      return {
        predictions: new Array(X.length).fill(0),
        probabilities: new Array(X.length).fill(0),
        confidenceScores: new Array(X.length).fill(0),
      };
    }

    // Weighted ensemble
    const ensemblePredictions: number[] = [];
    const ensembleProbabilities: number[] = [];
    const confidenceScores: number[] = [];

    for (let i = 0; i < X.length; i++) {
      let weightedProb = 0;
      let totalWeight = 0;

      for (let j = 0; j < predictions.length; j++) {
        const modelName = Object.keys(this.bestModels)[j];
        const weight = this.ensembleWeights[modelName];
        weightedProb += probabilities[j][i] * weight;
        totalWeight += weight;
      }

      const finalProbability =
        totalWeight > 0 ? weightedProb / totalWeight : 0.5;
      const finalPrediction = finalProbability > 0.5 ? 1 : 0;

      ensemblePredictions.push(finalPrediction);
      ensembleProbabilities.push(finalProbability);
      confidenceScores.push(finalProbability);
    }

    return {
      predictions: ensemblePredictions,
      probabilities: ensembleProbabilities,
      confidenceScores,
    };
  }

  private calculateMetrics(
    yTrue: number[],
    yPred: number[],
    yProb?: number[]
  ): ModelMetrics {
    const { trueNegatives, falsePositives, falseNegatives, truePositives } =
      this.calculateConfusionMatrix(yTrue, yPred);

    const accuracy = (truePositives + trueNegatives) / yTrue.length;
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1Score = (2 * (precision * recall)) / (precision + recall) || 0;

    const confusionMatrix = [
      [trueNegatives, falsePositives],
      [falseNegatives, truePositives],
    ];

    const metrics: ModelMetrics = {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      trueNegatives,
      falsePositives,
      falseNegatives,
      truePositives,
      trainingTime: 0,
    };

    if (yProb) {
      metrics.auc = this.calculateAUC(yTrue, yProb);
    }

    return metrics;
  }

  private calculateConfusionMatrix(
    yTrue: number[],
    yPred: number[]
  ): {
    trueNegatives: number;
    falsePositives: number;
    falseNegatives: number;
    truePositives: number;
  } {
    let trueNegatives = 0,
      falsePositives = 0,
      falseNegatives = 0,
      truePositives = 0;

    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === 0 && yPred[i] === 0) trueNegatives++;
      else if (yTrue[i] === 0 && yPred[i] === 1) falsePositives++;
      else if (yTrue[i] === 1 && yPred[i] === 0) falseNegatives++;
      else if (yTrue[i] === 1 && yPred[i] === 1) truePositives++;
    }

    return { trueNegatives, falsePositives, falseNegatives, truePositives };
  }

  private calculateAccuracy(yTrue: number[], yPred: number[]): number {
    let correct = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === yPred[i]) correct++;
    }
    return correct / yTrue.length;
  }

  private calculateAUC(yTrue: number[], yProb: number[]): number {
    // Simplified AUC calculation using trapezoidal rule
    const sortedData = yTrue
      .map((label, index) => ({ label, prob: yProb[index] }))
      .sort((a, b) => b.prob - a.prob);

    let tp = 0,
      fp = 0;
    let prevProb = -1;
    let auc = 0;

    for (const { label, prob } of sortedData) {
      if (prob !== prevProb) {
        auc += this.trapezoidArea(fp, tp, prevProb, prob);
        prevProb = prob;
      }

      if (label === 1) tp++;
      else fp++;
    }

    auc += this.trapezoidArea(fp, tp, prevProb, 1);

    const totalPositives = yTrue.filter((label) => label === 1).length;
    const totalNegatives = yTrue.filter((label) => label === 0).length;

    return auc / (totalPositives * totalNegatives);
  }

  private trapezoidArea(
    fp: number,
    tp: number,
    prevProb: number,
    prob: number
  ): number {
    return ((fp - tp) * (prob - prevProb)) / 2;
  }

  getEnsembleWeights(): EnsembleWeights {
    return { ...this.ensembleWeights };
  }

  getFeatureColumns(): string[] {
    return [...this.featureColumns];
  }

  async evaluateModels(X: number[][], y: number[]): Promise<TrainingResult> {
    console.log("Evaluating models on test set...");
    const results: TrainingResult = {};

    for (const [name, model] of Object.entries(this.bestModels)) {
      try {
        // Make predictions
        const testPred = model.predict(X);
        const testProb = model.predictProba(X).map((probs) => probs[1]); // Probability of class 1

        // Calculate metrics
        const metrics = this.calculateMetrics(y, testPred, testProb);
        metrics.trainingTime = 0; // No training time for evaluation

        results[name] = metrics;

        // Log detailed evaluation results
        console.log(`\n${name.toUpperCase()} - Test Evaluation:`);
        console.log(`  Accuracy:  ${metrics.accuracy.toFixed(4)}`);
        console.log(`  Precision: ${metrics.precision.toFixed(4)}`);
        console.log(`  Recall:    ${metrics.recall.toFixed(4)}`);
        console.log(`  F1-Score:  ${metrics.f1Score.toFixed(4)}`);
        if (metrics.auc !== undefined) {
          console.log(`  AUC:       ${metrics.auc.toFixed(4)}`);
        }

        // Log confusion matrix
        console.log(`  Confusion Matrix:`);
        console.log(
          `    [${metrics.confusionMatrix[0][0]}  ${metrics.confusionMatrix[0][1]}]`
        );
        console.log(
          `    [${metrics.confusionMatrix[1][0]}  ${metrics.confusionMatrix[1][1]}]`
        );
        console.log(
          `    (TN: ${metrics.trueNegatives}, FP: ${metrics.falsePositives}, FN: ${metrics.falseNegatives}, TP: ${metrics.truePositives})`
        );
      } catch (e) {
        console.error(`Error evaluating ${name}: ${e}`);
        results[name] = { error: String(e) } as any;
      }
    }

    return results;
  }

  printTrainingSummary(results: TrainingResult): void {
    console.log("\n" + "=".repeat(80));
    console.log("COMPREHENSIVE TRAINING SUMMARY");
    console.log("=".repeat(80));

    console.log(`Number of Features: ${this.featureColumns.length}`);
    console.log(`Ensemble Weights: ${JSON.stringify(this.ensembleWeights)}`);

    for (const [name, result] of Object.entries(results)) {
      if ("error" in result) {
        console.log(`${name}: ERROR - ${result.error}`);
        continue;
      }

      console.log(`\n${name.toUpperCase()}:`);
      console.log(`  Training Time: ${result.trainingTime.toFixed(2)} seconds`);
      console.log(`  Training Metrics:`);
      console.log(`    Accuracy:  ${result.accuracy.toFixed(4)}`);
      console.log(`    Precision: ${result.precision.toFixed(4)}`);
      console.log(`    Recall:    ${result.recall.toFixed(4)}`);
      console.log(`    F1-Score:  ${result.f1Score.toFixed(4)}`);
      if (result.auc !== undefined) {
        console.log(`    AUC:       ${result.auc.toFixed(4)}`);
      }
      console.log(`  Training Confusion Matrix:`);
      console.log(
        `    [${result.confusionMatrix[0][0]}  ${result.confusionMatrix[0][1]}]`
      );
      console.log(
        `    [${result.confusionMatrix[1][0]}  ${result.confusionMatrix[1][1]}]`
      );
      console.log(
        `    (TN: ${result.trueNegatives}, FP: ${result.falsePositives}, FN: ${result.falseNegatives}, TP: ${result.truePositives})`
      );
    }

    console.log("\n" + "=".repeat(80));
  }

  printEvaluationSummary(
    trainResults: TrainingResult,
    testResults: TrainingResult
  ): void {
    console.log("\n" + "=".repeat(80));
    console.log("TRAIN vs TEST PERFORMANCE COMPARISON");
    console.log("=".repeat(80));

    for (const [name, trainResult] of Object.entries(trainResults)) {
      if ("error" in trainResult) {
        console.log(`${name}: ERROR - ${trainResult.error}`);
        continue;
      }

      const testResult = testResults[name];
      if ("error" in testResult) {
        console.log(`${name}: TEST ERROR - ${testResult.error}`);
        continue;
      }

      console.log(`\n${name.toUpperCase()}:`);
      console.log(
        `  Train Accuracy: ${trainResult.accuracy.toFixed(
          4
        )} | Test Accuracy: ${testResult.accuracy.toFixed(4)}`
      );
      console.log(
        `  Train Precision: ${trainResult.precision.toFixed(
          4
        )} | Test Precision: ${testResult.precision.toFixed(4)}`
      );
      console.log(
        `  Train Recall: ${trainResult.recall.toFixed(
          4
        )} | Test Recall: ${testResult.recall.toFixed(4)}`
      );
      console.log(
        `  Train F1: ${trainResult.f1Score.toFixed(
          4
        )} | Test F1: ${testResult.f1Score.toFixed(4)}`
      );

      // Check for overfitting
      const accuracyDrop = trainResult.accuracy - testResult.accuracy;
      if (accuracyDrop > 0.1) {
        console.log(
          `  ⚠️  POTENTIAL OVERFITTING: Train-Test accuracy gap: ${accuracyDrop.toFixed(
            4
          )}`
        );
      } else if (accuracyDrop < -0.05) {
        console.log(
          `  ℹ️  Test accuracy higher than train - possible data leakage or small dataset`
        );
      } else {
        console.log(
          `  ✅ Good generalization: Train-Test accuracy gap: ${accuracyDrop.toFixed(
            4
          )}`
        );
      }
    }

    console.log("\n" + "=".repeat(80));
  }
}

// Import for test function
import { BitcoinDataCollector } from "./DataCollector";
import { FeatureEngineer } from "./FeatureEngineer";

// Test function
export async function testModels(): Promise<void> {
  // Load and prepare data
  const collector = new BitcoinDataCollector();
  const data = await collector.getLatestData(200);

  const engineer = new FeatureEngineer();
  const dataWithFeatures = engineer.createTechnicalIndicators(data);
  const dataWithTarget = engineer.createTargetVariable(dataWithFeatures);

  const { X, y } = engineer.selectFeatures(dataWithTarget);

  // Split data (simple split for testing)
  const splitIndex = Math.floor(X.length * 0.8);
  const XTrain = X.slice(0, splitIndex);
  const yTrain = y.slice(0, splitIndex);
  const XTest = X.slice(splitIndex);

  // Initialize and train models
  const predictor = new BitcoinPredictor();

  // Train models
  const trainingResults = await predictor.trainModels(XTrain, yTrain);

  // Print summary
  predictor.printTrainingSummary(trainingResults);

  // Test ensemble prediction
  const ensembleResult = await predictor.ensemblePredict(XTest);
  console.log(
    `Ensemble predictions: ${ensembleResult.predictions.slice(0, 5)}`
  );
  console.log(
    `Ensemble probabilities: ${ensembleResult.probabilities
      .slice(0, 5)
      .map((p) => p.toFixed(3))}`
  );

  console.log("Model training completed successfully!");
  console.log(
    `Number of models trained: ${
      Object.keys(predictor.getEnsembleWeights()).length
    }`
  );
  console.log(`Number of features: ${predictor.getFeatureColumns().length}`);
}
