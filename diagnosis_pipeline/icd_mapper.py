import requests
from datetime import datetime, timedelta
import logging
from typing import Union, List, Dict

logger = logging.getLogger(__name__)

class ICD10Mapper:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expiry = None
        self.cache = {}
        self.headers = {
            "Authorization": None,
            "Accept": "application/json",
            "Accept-Language": "en",
            "API-Version": "v2"
        }

    def _refresh_token(self):
        """Refresh OAuth token if expired"""
        if self.token and datetime.now() < self.expiry:
            return

        try:
            response = requests.post(
                "https://icdaccessmanagement.who.int/connect/token",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                    "scope": "icdapi_access"
                },
                timeout=5
            )
            response.raise_for_status()
            token_data = response.json()
            self.token = token_data["access_token"]
            self.expiry = datetime.now() + timedelta(seconds=token_data["expires_in"] - 300)
            self.headers["Authorization"] = f"Bearer {self.token}"
        except Exception as e:
            raise ConnectionError(f"Failed to refresh token: {str(e)}")

    def get_codes(self, diseases: Union[str, List[str]]) -> Dict[str, str]:
        """Get ICD codes for one or multiple diseases"""
        if isinstance(diseases, str):
            diseases = [diseases]

        results = {}
        uncached = []

        for disease in diseases:
            key = disease.lower()
            if key in self.cache:
                results[disease] = self.cache[key]
            else:
                uncached.append(disease)

        if not uncached:
            return results

        self._refresh_token()

        for disease in uncached:
            try:
                response = requests.get(
                    "https://id.who.int/icd/release/11/2022-02/mms/search",
                    headers=self.headers,
                    params={
                        "q": disease,
                        "flatResults": "true",
                        "useFlexisearch": "true"
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if "destinationEntities" in data and data["destinationEntities"]:
                        code = data["destinationEntities"][0].get("theCode", "Unknown")
                    else:
                        code = "Not_Found"
                else:
                    code = f"API_Error_{response.status_code}"
            except Exception as e:
                code = f"Lookup_Error: {str(e)}"

            self.cache[disease.lower()] = code
            results[disease] = code

        return results
