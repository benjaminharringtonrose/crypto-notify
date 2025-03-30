import dotenv from "dotenv";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { TradeModelTrainer } from "../cardano/TradeModelTrainer";
import { onRequest } from "firebase-functions/https";

dotenv.config();

const runTraining = async () => {
  console.log("Training started...");
  const trainer = new TradeModelTrainer();
  try {
    await trainer.train();
    console.log("Training completed successfully.");
  } catch (error) {
    console.error("Training failed:", error);
    throw error;
  }
};

export const runModelTrainingADA = onSchedule(
  {
    schedule: `*/30 * * * *`,
    memory: "4GiB",
    timeoutSeconds: 540,
  },
  runTraining
);

export const triggerTrainingNow = onRequest(
  { memory: "4GiB", timeoutSeconds: 540 },
  async (req, res) => {
    await runTraining();
    res.send("Training triggered successfully.");
  }
);
