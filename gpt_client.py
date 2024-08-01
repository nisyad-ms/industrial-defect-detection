import dataclasses

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import tenacity


@dataclasses.dataclass
class GPTClient:
    # your base endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    api_base: str = None
    deployment_name: str = None
    api_version: str = "2024-02-15-preview"
    api_key: str = None

    def __post_init__(self):
        assert self.api_base is not None, "API base is required"
        assert self.deployment_name is not None, "Deployment name is required"

        self._client = self._get_client()

    @property
    def client(self):
        return self._client

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=2, min=30, max=128),
        retry=tenacity.retry_if_exception_type(Exception))
    def get_response(self, base64_image: str, prompt: str, max_tokens: int = 500, mime_type="image/png", raise_exception: bool = True) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant that returns output in JSON format."},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64," + base64_image,
                            }
                        }
                    ]}
                ],
                max_tokens=max_tokens)
        except Exception as e:
            if raise_exception:
                raise e
            response = None

        return response

    def _get_client(self):
        if self.api_key is not None:
            return AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                base_url=f"{self.api_base}openai/deployments/{self.deployment_name}/extensions")
        else:
            token_provider = self.get_token_provider()
            return AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                azure_ad_token_provider=token_provider)

    def extract_result(self, response):
        if response is None:
            return None
        return response.choices[0]['message']['content']

    @staticmethod
    def get_token_provider():
        return get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
