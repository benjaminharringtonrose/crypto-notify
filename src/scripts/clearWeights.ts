import { FirebaseService } from "../api/FirebaseService";
import { FILE_NAMES } from "../constants";

async function clearWeights() {
  try {
    console.log("🗑️  Clearing saved model weights...");

    const bucket = FirebaseService.getInstance().getBucket();
    const file = bucket.file(FILE_NAMES.WEIGHTS);

    // Check if the file exists
    const [exists] = await file.exists();

    if (exists) {
      await file.delete();
      console.log("✅ Successfully deleted saved weights");
      console.log(
        "🔄 Next training run will create new weights with current architecture"
      );
    } else {
      console.log("ℹ️  No saved weights found to delete");
    }
  } catch (error) {
    console.error("❌ Error clearing weights:", error);
  }
}

clearWeights();
