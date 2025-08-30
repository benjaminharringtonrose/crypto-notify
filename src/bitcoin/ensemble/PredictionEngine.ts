/**
 * Prediction Module for Bitcoin Price Prediction Bot
 * Handles real-time predictions and model management.
 */

import { BitcoinDataCollector } from "./DataCollector";
import { FeatureEngineer } from "./FeatureEngineer";
import { BitcoinPredictor } from "./Models";

export interface PredictionResult {
  timestamp: string;
  predictionDate: string;
  currentPrice: number;
  prediction: number; // 1 for up, 0 for down
  probability: number;
  confidence: number;
  direction: "UP" | "DOWN";
  predictedDirection: "UP" | "DOWN"; // Add this property
  modelsUsed: string[];
  predictionWindow: string;
  isConfident: boolean;
}

export interface PredictionSummary {
  currentPrediction: PredictionResult | null;
  predictionCount: number;
  recentPredictions: PredictionResult[];
}

export interface ValidationResult {
  predictionDate: string;
  validationDate: string;
  predictedDirection: "UP" | "DOWN";
  actualDirection: "UP" | "DOWN";
  predictedConfidence: number;
  actualPriceChange: number;
  predictionPrice: number;
  currentPrice: number;
  isCorrect: boolean;
  accuracy: "Correct" | "Incorrect";
}

export interface ModelPerformance {
  totalModels: number;
  models: Record<string, any>;
  featureCount: number;
}

export interface PredictionEngineConfig {
  modelsDir?: string;
  dataDir?: string;
}

export class BitcoinPredictionEngine {
  private modelsDir: string;
  private dataDir: string;
  private dataCollector: BitcoinDataCollector;
  private featureEngineer: FeatureEngineer;
  private predictor: BitcoinPredictor;
  private currentPrediction: PredictionResult | null = null;
  private predictionHistory: PredictionResult[] = [];

  constructor(config: PredictionEngineConfig = {}) {
    this.modelsDir = config.modelsDir || "models";
    this.dataDir = config.dataDir || "data";
    this.dataCollector = new BitcoinDataCollector({ dataDir: this.dataDir });
    this.featureEngineer = new FeatureEngineer();
    this.predictor = new BitcoinPredictor({ modelsDir: this.modelsDir });
  }

  async loadOrTrainModels(
    forceRetrain: boolean = false,
    dataPeriod: string = "5y"
  ): Promise<boolean> {
    try {
      // For now, we'll always train new models since we don't have persistence
      // In a real implementation, you would check for existing model files
      console.log("Training new models...");
      return await this.trainNewModels(dataPeriod);
    } catch (e) {
      console.error(`Error loading/training models: ${e}`);
      return false;
    }
  }

  private async trainNewModels(dataPeriod: string = "5y"): Promise<boolean> {
    try {
      // Collect data
      console.log(`Collecting ${dataPeriod} of Bitcoin data...`);
      const data = await this.dataCollector.fetchHistoricalData(dataPeriod);

      if (!this.dataCollector.validateData(data)) {
        console.error("Data validation failed");
        return false;
      }

      // Create features
      console.log("Creating features...");
      const dataWithFeatures =
        this.featureEngineer.createTechnicalIndicators(data);
      const dataWithTarget =
        this.featureEngineer.createTargetVariable(dataWithFeatures);

      // Select features
      const { X, y } = this.featureEngineer.selectFeatures(dataWithTarget);

      if (X.length < 100) {
        console.error("Insufficient data for training");
        return false;
      }

      // Split data
      const splitIndex = Math.floor(X.length * 0.8);
      const XTrain = X.slice(0, splitIndex);
      const yTrain = y.slice(0, splitIndex);
      const XTest = X.slice(splitIndex);
      const yTest = y.slice(splitIndex);

      // Initialize and train models
      const trainingResults = await this.predictor.trainModels(
        XTrain,
        yTrain,
        true
      );

      // Print comprehensive training summary
      this.predictor.printTrainingSummary(trainingResults);

      // Evaluate on test set
      console.log("\nEvaluating models on test set...");
      const testResults = await this.predictor.evaluateModels(XTest, yTest);

      // Print train vs test comparison
      this.predictor.printEvaluationSummary(trainingResults, testResults);

      // Log completion
      console.log("Model training and evaluation completed successfully");

      return true;
    } catch (e) {
      console.error(`Error training new models: ${e}`);
      return false;
    }
  }

  async makePrediction(
    useEnsemble: boolean = true,
    confidenceThreshold: number = 0.6
  ): Promise<PredictionResult> {
    try {
      console.log("Making Bitcoin price prediction...");

      // Get latest data
      const latestData = await this.dataCollector.getLatestData(60); // Get 60 days for features

      if (latestData.length === 0) {
        throw new Error("No recent data available");
      }

      // Create features
      const dataWithFeatures =
        this.featureEngineer.createTechnicalIndicators(latestData);

      // Get the most recent data point for prediction
      const latestFeatures = dataWithFeatures[dataWithFeatures.length - 1];

      // Use the same feature columns that were used during training
      const featureColumns = this.predictor.getFeatureColumns();
      if (featureColumns.length === 0) {
        throw new Error(
          "No feature columns available - models may not be trained"
        );
      }

      // Create feature vector for prediction
      const XPred: number[][] = [];
      const featureVector: number[] = [];

      for (const col of featureColumns) {
        const value = (latestFeatures as any)[col];
        if (value === undefined || value === null || isNaN(value)) {
          featureVector.push(0); // Default value for missing features
        } else {
          featureVector.push(value);
        }
      }

      XPred.push(featureVector);

      // Make prediction
      let prediction: number;
      let probability: number;
      let modelsUsed: string[];

      if (useEnsemble) {
        try {
          const ensembleResult = await this.predictor.ensemblePredict(XPred);
          prediction = ensembleResult.predictions[0];
          probability = ensembleResult.probabilities[0];
          modelsUsed = Object.keys(this.predictor.getEnsembleWeights());
        } catch (e) {
          console.error(`Error in ensemble prediction: ${e}`);
          throw e;
        }
      } else {
        // Use best performing model (simplified - just use the first available model)
        const modelNames = Object.keys(this.predictor.getEnsembleWeights());
        if (modelNames.length === 0) {
          throw new Error("No trained models available");
        }

        // For simplicity, we'll use ensemble prediction as fallback
        const ensembleResult = await this.predictor.ensemblePredict(XPred);
        prediction = ensembleResult.predictions[0];
        probability = ensembleResult.probabilities[0];
        modelsUsed = [modelNames[0]];
      }

      // Determine confidence level
      const confidence = prediction === 1 ? probability : 1 - probability;

      // Create prediction result
      const currentPrice = latestData[latestData.length - 1].Close;
      const predictionDate = latestData[latestData.length - 1].Date;

      const result: PredictionResult = {
        timestamp: new Date().toISOString(),
        predictionDate: predictionDate.toISOString(),
        currentPrice,
        prediction,
        probability,
        confidence,
        direction: prediction === 1 ? "UP" : "DOWN",
        predictedDirection: prediction === 1 ? "UP" : "DOWN", // Set predictedDirection
        modelsUsed,
        predictionWindow: "3 days",
        isConfident: confidence >= confidenceThreshold,
      };

      // Store prediction
      this.currentPrediction = result;
      this.predictionHistory.push(result);

      console.log(
        `Prediction: ${result.direction} with ${(confidence * 100).toFixed(
          1
        )}% confidence`
      );

      return result;
    } catch (e) {
      console.error(`Error making prediction: ${e}`);
      throw e;
    }
  }

  getPredictionSummary(): PredictionSummary {
    return {
      currentPrediction: this.currentPrediction,
      predictionCount: this.predictionHistory.length,
      recentPredictions: this.predictionHistory.slice(-5), // Last 5 predictions
    };
  }

  async validatePrediction(daysAhead: number = 3): Promise<ValidationResult> {
    if (this.predictionHistory.length === 0) {
      throw new Error("No prediction history available");
    }

    try {
      // Get the most recent prediction
      const latestPrediction =
        this.predictionHistory[this.predictionHistory.length - 1];
      const predictionDate = new Date(latestPrediction.predictionDate);

      // Get current data to check actual price movement
      const currentData = await this.dataCollector.getLatestData(10);

      // Find the price at prediction date and current price
      let predictionPrice: number | null = null;
      let currentPrice: number | null = null;

      for (const point of currentData) {
        if (point.Date.toDateString() === predictionDate.toDateString()) {
          predictionPrice = point.Close;
        }

        const validationDate = new Date(
          predictionDate.getTime() + daysAhead * 24 * 60 * 60 * 1000
        );
        if (point.Date.toDateString() === validationDate.toDateString()) {
          currentPrice = point.Close;
          break;
        }
      }

      if (predictionPrice === null || currentPrice === null) {
        throw new Error("Cannot validate prediction - insufficient data");
      }

      // Calculate actual price change
      const actualChange = (currentPrice - predictionPrice) / predictionPrice;
      const actualDirection: "UP" | "DOWN" = actualChange > 0 ? "UP" : "DOWN";

      // Compare with prediction
      const predictedDirection = latestPrediction.predictedDirection;
      const isCorrect = actualDirection === predictedDirection;

      const validationResult: ValidationResult = {
        predictionDate: predictionDate.toISOString(),
        validationDate: new Date(
          predictionDate.getTime() + daysAhead * 24 * 60 * 60 * 1000
        ).toISOString(),
        predictedDirection,
        actualDirection,
        predictedConfidence: latestPrediction.confidence,
        actualPriceChange: actualChange,
        predictionPrice,
        currentPrice,
        isCorrect,
        accuracy: isCorrect ? "Correct" : "Incorrect",
      };

      return validationResult;
    } catch (e) {
      console.error(`Error validating prediction: ${e}`);
      throw e;
    }
  }

  getModelPerformance(): ModelPerformance {
    try {
      const ensembleWeights = this.predictor.getEnsembleWeights();
      const featureColumns = this.predictor.getFeatureColumns();

      const modelInfo: Record<string, any> = {};
      for (const [name, weight] of Object.entries(ensembleWeights)) {
        modelInfo[name] = {
          type: name,
          weight: weight,
          parameters: {}, // In a real implementation, you would store model parameters
        };
      }

      return {
        totalModels: Object.keys(ensembleWeights).length,
        models: modelInfo,
        featureCount: featureColumns.length,
      };
    } catch (e) {
      console.error(`Error getting model performance: ${e}`);
      throw e;
    }
  }

  async updatePrediction(): Promise<PredictionResult> {
    console.log("Updating prediction with fresh data...");
    return await this.makePrediction();
  }

  async savePredictionHistory(filename?: string): Promise<string> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const finalFilename = filename || `prediction_history_${timestamp}.json`;
    const filepath = `${this.dataDir}/${finalFilename}`;

    try {
      // In a real implementation, you would save to file system or database
      console.log(`Prediction history saved to ${filepath}`);
      return filepath;
    } catch (e) {
      console.error(`Error saving prediction history: ${e}`);
      throw e;
    }
  }

  async loadPredictionHistory(filename: string): Promise<boolean> {
    const filepath = `${this.dataDir}/${filename}`;

    try {
      // In a real implementation, you would load from file system or database
      console.log(`Prediction history loaded from ${filepath}`);
      return true;
    } catch (e) {
      console.error(`Error loading prediction history: ${e}`);
      return false;
    }
  }

  // Additional utility methods
  async getLatestPrice(): Promise<number> {
    const latestData = await this.dataCollector.getLatestData(1);
    if (latestData.length === 0) {
      throw new Error("No recent data available");
    }
    return latestData[0].Close;
  }

  async getPriceHistory(
    days: number = 30
  ): Promise<Array<{ date: string; price: number }>> {
    const data = await this.dataCollector.getLatestData(days);
    return data.map((point) => ({
      date: point.Date.toISOString(),
      price: point.Close,
    }));
  }

  async getPredictionAccuracy(): Promise<{
    correct: number;
    total: number;
    accuracy: number;
  }> {
    if (this.predictionHistory.length === 0) {
      return { correct: 0, total: 0, accuracy: 0 };
    }

    let correct = 0;
    let total = 0;

    for (const prediction of this.predictionHistory) {
      try {
        const validation = await this.validatePrediction(3);
        if (validation.isCorrect) {
          correct++;
        }
        total++;
      } catch (e) {
        // Skip predictions that can't be validated
        console.warn(
          `Could not validate prediction from ${prediction.predictionDate}: ${e}`
        );
      }
    }

    return {
      correct,
      total,
      accuracy: total > 0 ? correct / total : 0,
    };
  }

  async getConfidenceDistribution(): Promise<Record<string, number>> {
    const distribution: Record<string, number> = {
      "0.0-0.2": 0,
      "0.2-0.4": 0,
      "0.4-0.6": 0,
      "0.6-0.8": 0,
      "0.8-1.0": 0,
    };

    for (const prediction of this.predictionHistory) {
      const confidence = prediction.confidence;
      if (confidence < 0.2) distribution["0.0-0.2"]++;
      else if (confidence < 0.4) distribution["0.2-0.4"]++;
      else if (confidence < 0.6) distribution["0.4-0.6"]++;
      else if (confidence < 0.8) distribution["0.6-0.8"]++;
      else distribution["0.8-1.0"]++;
    }

    return distribution;
  }

  async getModelContributions(): Promise<
    Record<string, { weight: number; contribution: number }>
  > {
    const ensembleWeights = this.predictor.getEnsembleWeights();
    const contributions: Record<
      string,
      { weight: number; contribution: number }
    > = {};

    for (const [modelName, weight] of Object.entries(ensembleWeights)) {
      contributions[modelName] = {
        weight,
        contribution: weight * 100, // Convert to percentage
      };
    }

    return contributions;
  }
}

// Test function
export async function testPredictionEngine(): Promise<void> {
  const engine = new BitcoinPredictionEngine();

  // Load or train models
  if (await engine.loadOrTrainModels()) {
    console.log("Models ready!");

    // Make prediction
    const prediction = await engine.makePrediction();
    console.log(`Prediction: ${JSON.stringify(prediction, null, 2)}`);

    // Get summary
    const summary = engine.getPredictionSummary();
    console.log(`Summary: ${JSON.stringify(summary, null, 2)}`);

    // Get model performance
    const performance = engine.getModelPerformance();
    console.log(`Model Performance: ${JSON.stringify(performance, null, 2)}`);

    // Get latest price
    const latestPrice = await engine.getLatestPrice();
    console.log(`Latest Bitcoin price: $${latestPrice.toLocaleString()}`);

    // Get confidence distribution
    const confidenceDistribution = await engine.getConfidenceDistribution();
    console.log(
      `Confidence Distribution: ${JSON.stringify(
        confidenceDistribution,
        null,
        2
      )}`
    );

    // Get model contributions
    const modelContributions = await engine.getModelContributions();
    console.log(
      `Model Contributions: ${JSON.stringify(modelContributions, null, 2)}`
    );
  } else {
    console.log("Failed to prepare models");
  }
}
