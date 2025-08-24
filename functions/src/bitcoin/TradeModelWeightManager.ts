import * as tf from "@tensorflow/tfjs-node";
import { FirebaseService } from "../api/FirebaseService";
import { Bucket } from "@google-cloud/storage";
import { FILE_NAMES, MODEL_CONFIG, TRAINING_CONFIG } from "../constants";
import { FeatureDetector } from "./FeatureDetector";

export class ModelWeightManager {
  private weights: any;
  private bucket: Bucket;

  constructor() {
    this.bucket = FirebaseService.getInstance().getBucket();
  }

  public async loadWeights(): Promise<void> {
    const file = this.bucket.file(FILE_NAMES.WEIGHTS);
    const [weightsData] = await file.download();
    this.weights = JSON.parse(weightsData.toString("utf8")).weights;
    this.validateWeights();
  }

  private validateWeights(): void {
    if (
      !this.weights ||
      Object.values(this.weights).some((w: any) => !w || isNaN(w[0]))
    ) {
      console.warn(
        "Invalid weights detected, falling back to random initialization."
      );
      this.weights = null;
    }
  }

  public setWeights(model: tf.LayersModel): void {
    if (!this.weights) {
      console.warn("No weights available, using random initialization");
      return;
    }

    try {
      // Set convolutional layer weights

      // Check if conv1 weights match expected shape
      // Calculate shape dynamically based on current feature count
      const kernelSize = 3; // From TradeModelFactory - proven optimal kernel size
      const filters = 48; // From TradeModelFactory - proven optimal filter count
      const featureCount = FeatureDetector.getFeatureCount();
      const expectedConv1Size = kernelSize * featureCount * filters;
      if (this.weights.conv1Weights?.length !== expectedConv1Size) {
        console.warn(
          `Conv1 weights size mismatch: expected ${expectedConv1Size}, got ${this.weights.conv1Weights?.length}`
        );
        console.warn("Using random initialization due to shape mismatch");
        return;
      }

      // Load convolutional layers
      model
        .getLayer("conv1d_input")
        .setWeights([
          tf.tensor3d(this.weights.conv1Weights, [
            kernelSize,
            featureCount,
            filters,
          ]),
          tf.tensor1d(this.weights.conv1Bias),
        ]);
      // Conv1D layer 2 - OPTIONAL (only exists in older model architectures)
      try {
        if (this.weights.conv2Weights && this.weights.conv2Weights.length > 0) {
          // Calculate conv2 shape dynamically (if it exists)
          const conv2KernelSize = 3; // Standard kernel size
          const conv2InputFilters = 48; // Output from conv1
          const conv2OutputFilters = 64; // Typical progression
          const expectedConv2Size =
            conv2KernelSize * conv2InputFilters * conv2OutputFilters;

          if (this.weights.conv2Weights.length === expectedConv2Size) {
            model
              .getLayer("conv1d_2")
              .setWeights([
                tf.tensor3d(this.weights.conv2Weights, [
                  conv2KernelSize,
                  conv2InputFilters,
                  conv2OutputFilters,
                ]),
                tf.tensor1d(this.weights.conv2Bias),
              ]);
            console.log("✅ Conv1D layer 2 weights loaded successfully");
          } else {
            console.warn(
              `Conv2 weights size mismatch: expected ${expectedConv2Size}, got ${this.weights.conv2Weights.length}`
            );
          }
        } else {
          console.log(
            "No conv2 weights found - current model uses single Conv1D layer"
          );
        }
      } catch (error) {
        console.log(
          "Conv1D layer 2 not found in current model - using simplified architecture"
        );
      }

      // Load batch normalization for conv layers
      model
        .getLayer("bn_conv1")
        .setWeights([
          tf.tensor1d(this.weights.bnConv1Gamma),
          tf.tensor1d(this.weights.bnConv1Beta),
          tf.tensor1d(this.weights.bnConv1MovingMean),
          tf.tensor1d(this.weights.bnConv1MovingVariance),
        ]);
      // Batch normalization for conv2 - OPTIONAL (only exists in older model architectures)
      try {
        if (this.weights.bnConv2Gamma && this.weights.bnConv2Gamma.length > 0) {
          model
            .getLayer("bn_conv2")
            .setWeights([
              tf.tensor1d(this.weights.bnConv2Gamma),
              tf.tensor1d(this.weights.bnConv2Beta),
              tf.tensor1d(this.weights.bnConv2MovingMean),
              tf.tensor1d(this.weights.bnConv2MovingVariance),
            ]);
          console.log("✅ Conv2 batch norm weights loaded successfully");
        } else {
          console.log(
            "No conv2 batch norm weights found - current model uses single Conv1D layer"
          );
        }
      } catch (error) {
        console.log("Conv2 batch norm layer not found in current model");
      }

      // Enhanced features layers (attention mechanism replacement) - OPTIONAL
      // These layers may not exist in old saved weights
      try {
        if (
          this.weights.enhancedFeatures1Weights &&
          this.weights.enhancedFeatures1Weights.length > 0
        ) {
          const enhanced1Shape: [number, number] = [
            MODEL_CONFIG.CONV1D_FILTERS_2,
            MODEL_CONFIG.ATTENTION_UNITS_1,
          ];
          const expectedEnhanced1Size = enhanced1Shape[0] * enhanced1Shape[1];

          if (
            this.weights.enhancedFeatures1Weights.length ===
            expectedEnhanced1Size
          ) {
            model
              .getLayer("enhanced_features1")
              .setWeights([
                tf.tensor2d(
                  this.weights.enhancedFeatures1Weights,
                  enhanced1Shape
                ),
                tf.tensor1d(this.weights.enhancedFeatures1Bias),
              ]);
            console.log("✅ Enhanced features 1 weights loaded successfully");
          } else {
            console.warn(
              `Enhanced features 1 weights size mismatch: expected ${expectedEnhanced1Size}, got ${this.weights.enhancedFeatures1Weights.length}`
            );
            console.warn("Skipping enhanced features 1 weights");
          }
        } else {
          console.log(
            "No enhanced features 1 weights found, using random initialization"
          );
        }
      } catch (error) {
        console.warn(
          "Enhanced features 1 layer not found in model, skipping weights"
        );
      }

      // Load LSTM layers
      // Note: LSTM layers get input from enhanced features layers, not directly from conv layers
      const lstm1InputSize = MODEL_CONFIG.ATTENTION_UNITS_1; // Input from enhanced_features1
      const lstm1WeightShape: [number, number] = [
        lstm1InputSize,
        MODEL_CONFIG.LSTM_UNITS_1 * 4,
      ];
      const lstm1RecurrentShape: [number, number] = [
        MODEL_CONFIG.LSTM_UNITS_1,
        MODEL_CONFIG.LSTM_UNITS_1 * 4,
      ];

      console.log(
        `LSTM1 expected shapes: weights=${lstm1WeightShape}, recurrent=${lstm1RecurrentShape}`
      );
      console.log(
        `LSTM1 actual sizes: weights=${this.weights.lstm1Weights.length}, recurrent=${this.weights.lstm1RecurrentWeights.length}`
      );

      model
        .getLayer("lstm1")
        .setWeights([
          tf.tensor2d(this.weights.lstm1Weights, lstm1WeightShape),
          tf.tensor2d(this.weights.lstm1RecurrentWeights, lstm1RecurrentShape),
          tf.tensor1d(this.weights.lstm1Bias),
        ]);

      const lstm2InputSize = MODEL_CONFIG.LSTM_UNITS_1; // Input from lstm1
      const lstm2WeightShape: [number, number] = [
        lstm2InputSize,
        MODEL_CONFIG.LSTM_UNITS_2 * 4,
      ];
      const lstm2RecurrentShape: [number, number] = [
        MODEL_CONFIG.LSTM_UNITS_2,
        MODEL_CONFIG.LSTM_UNITS_2 * 4,
      ];

      model
        .getLayer("lstm2")
        .setWeights([
          tf.tensor2d(this.weights.lstm2Weights, lstm2WeightShape),
          tf.tensor2d(this.weights.lstm2RecurrentWeights, lstm2RecurrentShape),
          tf.tensor1d(this.weights.lstm2Bias),
        ]);

      const lstm3InputSize = MODEL_CONFIG.ATTENTION_UNITS_2; // Input from enhanced_features2
      const lstm3WeightShape: [number, number] = [
        lstm3InputSize,
        MODEL_CONFIG.LSTM_UNITS_3 * 4,
      ];
      const lstm3RecurrentShape: [number, number] = [
        MODEL_CONFIG.LSTM_UNITS_3,
        MODEL_CONFIG.LSTM_UNITS_3 * 4,
      ];

      model
        .getLayer("lstm3")
        .setWeights([
          tf.tensor2d(this.weights.lstm3Weights, lstm3WeightShape),
          tf.tensor2d(this.weights.lstm3RecurrentWeights, lstm3RecurrentShape),
          tf.tensor1d(this.weights.lstm3Bias),
        ]);

      // Load batch normalization for LSTM layers - OPTIONAL (current model doesn't use LSTM batch norm)
      try {
        if (this.weights.bnLstm1Gamma && this.weights.bnLstm1Gamma.length > 0) {
          model
            .getLayer("bn_lstm1")
            .setWeights([
              tf.tensor1d(this.weights.bnLstm1Gamma),
              tf.tensor1d(this.weights.bnLstm1Beta),
              tf.tensor1d(this.weights.bnLstm1MovingMean),
              tf.tensor1d(this.weights.bnLstm1MovingVariance),
            ]);
          console.log("✅ LSTM1 batch norm weights loaded successfully");
        } else {
          console.log(
            "No LSTM1 batch norm weights found - current model doesn't use LSTM batch norm"
          );
        }
      } catch (error) {
        console.log("LSTM1 batch norm layer not found in current model");
      }

      try {
        if (this.weights.bnLstm2Gamma && this.weights.bnLstm2Gamma.length > 0) {
          model
            .getLayer("bn_lstm2")
            .setWeights([
              tf.tensor1d(this.weights.bnLstm2Gamma),
              tf.tensor1d(this.weights.bnLstm2Beta),
              tf.tensor1d(this.weights.bnLstm2MovingMean),
              tf.tensor1d(this.weights.bnLstm2MovingVariance),
            ]);
          console.log("✅ LSTM2 batch norm weights loaded successfully");
        } else {
          console.log("No LSTM2 batch norm weights found");
        }
      } catch (error) {
        console.log("LSTM2 batch norm layer not found in current model");
      }

      try {
        if (this.weights.bnLstm3Gamma && this.weights.bnLstm3Gamma.length > 0) {
          model
            .getLayer("bn_lstm3")
            .setWeights([
              tf.tensor1d(this.weights.bnLstm3Gamma),
              tf.tensor1d(this.weights.bnLstm3Beta),
              tf.tensor1d(this.weights.bnLstm3MovingMean),
              tf.tensor1d(this.weights.bnLstm3MovingVariance),
            ]);
          console.log("✅ LSTM3 batch norm weights loaded successfully");
        } else {
          console.log("No LSTM3 batch norm weights found");
        }
      } catch (error) {
        console.log("LSTM3 batch norm layer not found in current model");
      }

      // Enhanced features 2 layer (attention mechanism replacement) - OPTIONAL
      try {
        if (
          this.weights.enhancedFeatures2Weights &&
          this.weights.enhancedFeatures2Weights.length > 0
        ) {
          const expectedEnhanced2Size =
            MODEL_CONFIG.LSTM_UNITS_3 * MODEL_CONFIG.ATTENTION_UNITS_2;
          const actualSize = this.weights.enhancedFeatures2Weights.length;

          // Try to determine the correct shape based on the actual size
          let enhanced2Shape: [number, number];
          if (actualSize === expectedEnhanced2Size) {
            // Perfect match
            enhanced2Shape = [
              MODEL_CONFIG.LSTM_UNITS_3,
              MODEL_CONFIG.ATTENTION_UNITS_2,
            ];
          } else if (actualSize === 192) {
            // Legacy size - use the shape that matches our current architecture best
            enhanced2Shape = [16, 12] as [number, number];
            console.log(
              `Using legacy enhanced features 2 shape: ${enhanced2Shape[0]}x${enhanced2Shape[1]} (actual size: ${actualSize})`
            );
          } else {
            console.warn(
              `Enhanced features 2 weights size mismatch: expected ${expectedEnhanced2Size}, got ${actualSize}`
            );
            console.warn("Skipping enhanced features 2 weights");
            return;
          }

          model
            .getLayer("enhanced_features2")
            .setWeights([
              tf.tensor2d(
                this.weights.enhancedFeatures2Weights,
                enhanced2Shape
              ),
              tf.tensor1d(this.weights.enhancedFeatures2Bias),
            ]);
          console.log("✅ Enhanced features 2 weights loaded successfully");
        } else {
          console.log(
            "No enhanced features 2 weights found, using random initialization"
          );
        }
      } catch (error) {
        console.warn(
          "Enhanced features 2 layer not found in model, skipping weights"
        );
      }

      // Load batch normalization for enhanced features layers - OPTIONAL
      try {
        if (
          this.weights.bnEnhanced1Gamma &&
          this.weights.bnEnhanced1Gamma.length > 0
        ) {
          model
            .getLayer("bn_enhanced1")
            .setWeights([
              tf.tensor1d(this.weights.bnEnhanced1Gamma),
              tf.tensor1d(this.weights.bnEnhanced1Beta),
              tf.tensor1d(this.weights.bnEnhanced1MovingMean),
              tf.tensor1d(this.weights.bnEnhanced1MovingVariance),
            ]);
          console.log(
            "✅ Enhanced features 1 batch norm weights loaded successfully"
          );
        } else {
          console.log(
            "No enhanced features 1 batch norm weights found, using random initialization"
          );
        }
      } catch (error) {
        console.warn(
          "Enhanced features 1 batch norm layer not found in model, skipping weights"
        );
      }

      try {
        if (
          this.weights.bnEnhanced2Gamma &&
          this.weights.bnEnhanced2Gamma.length > 0
        ) {
          model
            .getLayer("bn_enhanced2")
            .setWeights([
              tf.tensor1d(this.weights.bnEnhanced2Gamma),
              tf.tensor1d(this.weights.bnEnhanced2Beta),
              tf.tensor1d(this.weights.bnEnhanced2MovingMean),
              tf.tensor1d(this.weights.bnEnhanced2MovingVariance),
            ]);
          console.log(
            "✅ Enhanced features 2 batch norm weights loaded successfully"
          );
        } else {
          console.log(
            "No enhanced features 2 batch norm weights found, using random initialization"
          );
        }
      } catch (error) {
        console.warn(
          "Enhanced features 2 batch norm layer not found in model, skipping weights"
        );
      }

      // Load dense layer weights
      // Calculate dense weight shapes dynamically based on current model architecture
      const lstmUnits = 64; // From TradeModelFactory - LSTM output size
      const dense1Units = 32; // From TradeModelFactory - proven optimal dense layer size

      model
        .getLayer("dense1")
        .setWeights([
          tf.tensor2d(this.weights.dense1Weights, [lstmUnits, dense1Units]),
          tf.tensor1d(this.weights.dense1Bias),
        ]);

      // Note: Current model architecture only has one dense layer (dense1)
      // dense2 is optional for backward compatibility with older saved weights
      try {
        if (
          this.weights.dense2Weights &&
          this.weights.dense2Weights.length > 0
        ) {
          const dense2Units = 16; // Typical half of dense1 if it exists
          model
            .getLayer("dense2")
            .setWeights([
              tf.tensor2d(this.weights.dense2Weights, [
                dense1Units,
                dense2Units,
              ]),
              tf.tensor1d(this.weights.dense2Bias),
            ]);
          console.log("✅ Dense layer 2 weights loaded successfully");
        } else {
          console.log(
            "No dense2 weights found - current model uses single dense layer"
          );
        }
      } catch (error) {
        console.log(
          "Dense layer 2 not found in current model - using simplified architecture"
        );
      }

      // Load batch normalization for dense layers - OPTIONAL (current model doesn't use dense batch norm)
      try {
        if (
          this.weights.bnDense1Gamma &&
          this.weights.bnDense1Gamma.length > 0
        ) {
          model
            .getLayer("bn_dense1")
            .setWeights([
              tf.tensor1d(this.weights.bnDense1Gamma),
              tf.tensor1d(this.weights.bnDense1Beta),
              tf.tensor1d(this.weights.bnDense1MovingMean),
              tf.tensor1d(this.weights.bnDense1MovingVariance),
            ]);
          console.log("✅ Dense1 batch norm weights loaded successfully");
        } else {
          console.log(
            "No dense1 batch norm weights found - current model doesn't use dense batch norm"
          );
        }
      } catch (error) {
        console.log("Dense1 batch norm layer not found in current model");
      }

      try {
        if (
          this.weights.bnDense2Gamma &&
          this.weights.bnDense2Gamma.length > 0
        ) {
          model
            .getLayer("bn_dense2")
            .setWeights([
              tf.tensor1d(this.weights.bnDense2Gamma),
              tf.tensor1d(this.weights.bnDense2Beta),
              tf.tensor1d(this.weights.bnDense2MovingMean),
              tf.tensor1d(this.weights.bnDense2MovingVariance),
            ]);
          console.log("✅ Dense2 batch norm weights loaded successfully");
        } else {
          console.log("No dense2 batch norm weights found");
        }
      } catch (error) {
        console.log("Dense2 batch norm layer not found in current model");
      }

      // Load output layer weights
      // Calculate output weight shape dynamically
      const outputClasses = TRAINING_CONFIG.OUTPUT_CLASSES; // Binary classification: Buy/Sell

      model
        .getLayer("output")
        .setWeights([
          tf.tensor2d(this.weights.outputWeights, [dense1Units, outputClasses]),
          tf.tensor1d(this.weights.outputBias),
        ]);

      console.log("✅ Weights loaded successfully");
    } catch (error) {
      console.error("Error setting weights:", error);
      console.warn("Using random initialization due to error");
    }
  }

  public getFeatureMeans(): number[] {
    return (
      this.weights?.featureMeans ||
      Array(FeatureDetector.getFeatureCount()).fill(0)
    );
  }

  public getFeatureStds(): number[] {
    return (
      this.weights?.featureStds ||
      Array(FeatureDetector.getFeatureCount()).fill(1)
    );
  }
}
