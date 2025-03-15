import axios from "axios";
import { TEXTBELT_API_URL } from "../constants";
import { sendSmsErrorMessage, textMessage } from "../utils";

export const sendSmsNotification = async (
  currentPrice: number,
  threshold: number
) => {
  try {
    await axios.post(`${TEXTBELT_API_URL}/text`, {
      phone: process.env.PHONE_NUMBER,
      message: textMessage(threshold, currentPrice),
      key: process.env.TEXTBELT_API_KEY,
    });
  } catch (error) {
    console.log(sendSmsErrorMessage(error));
    throw error;
  }
};
