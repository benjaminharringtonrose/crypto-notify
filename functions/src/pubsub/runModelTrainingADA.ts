import { onSchedule } from "firebase-functions/v2/scheduler";
import { TradeModelTrainer } from "../cardano/TradeModelTrainer";
import { onRequest } from "firebase-functions/https";

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
    schedule: "0 0,12 * * *", // run once at midnight and once at noon
    memory: "4GiB",
    timeoutSeconds: 540,
    timeZone: "America/New_York",
  },
  runTraining
);

export const triggerTrainingNow = onRequest(
  {
    memory: "4GiB",
    timeoutSeconds: 540,
  },
  async (req, res) => {
    try {
      const startTime = performance.now();
      await runTraining();
      const endTime = performance.now();
      const executionTime = (endTime - startTime) / 1000;
      res.send(
        `Training triggered successfully. Execution time: ${executionTime} milliseconds`
      );
    } catch (error) {
      res.status(504);
      console.log("Error training: ", JSON.stringify(error));
    }
  }
);
