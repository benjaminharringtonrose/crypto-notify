import { onSchedule } from "firebase-functions/v2/scheduler";
import { TradeModelTrainer } from "../bitcoin/TradeModelTrainer";
import { onRequest } from "firebase-functions/https";

const runTraining = async () => {
  console.log("Bitcoin model training started...");
  const trainer = new TradeModelTrainer(42); // Use fixed seed for consistency
  try {
    await trainer.train();
    console.log("Bitcoin model training completed successfully.");
  } catch (error) {
    console.error("Bitcoin model training failed:", error);
    throw error;
  }
};

export const runModelTrainingBTC = onSchedule(
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
        `Bitcoin model training triggered successfully. Execution time: ${executionTime} milliseconds`
      );
    } catch (error) {
      res.status(504);
      console.log("Error training Bitcoin model: ", JSON.stringify(error));
    }
  }
);
