import axios from "axios";
import { COINGECKO_API_URL, CryptoIds, Currencies } from "../constants";
import { fetchCardanoPriceErrorMessage } from "../utils";

export const getCurrentPrice = async ({
  id,
  currency,
}: {
  id: CryptoIds;
  currency: Currencies;
}): Promise<number> => {
  try {
    const params = {
      ids: id,
      vs_currencies: currency,
    };

    const response = await axios.get(`${COINGECKO_API_URL}/simple/price`, {
      params,
    });

    return response.data.cardano.usd;
  } catch (error) {
    console.log(fetchCardanoPriceErrorMessage(error));
    throw error;
  }
};
