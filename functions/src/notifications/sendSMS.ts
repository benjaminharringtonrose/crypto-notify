import axios from "axios";
import { TEXTBELT_API_URL } from "../constants";
import { sendSmsErrorMessage } from "../utils";

export const sendSMS = async (message: string) => {
  try {
    await axios.post(`${TEXTBELT_API_URL}/text`, {
      phone: process.env.PHONE_NUMBER,
      message,
      key: process.env.TEXTBELT_API_KEY,
      replyWebhookUrl: process.env.WEBHOOK_URL,
    });
  } catch (error) {
    console.log(sendSmsErrorMessage(error));
    throw error;
  }
};
