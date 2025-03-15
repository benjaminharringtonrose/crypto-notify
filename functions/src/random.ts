import { onSchedule } from "firebase-functions/v2/scheduler";
import { ANALYSIS_SCHEDULE } from "./constants";

export const random = onSchedule(ANALYSIS_SCHEDULE, async () => {});
