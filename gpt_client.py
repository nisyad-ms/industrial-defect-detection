import dataclasses
from enum import Enum

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import tenacity

SYSTEM_PROMPT = "You are an intelligent assistant that returns output in JSON format."


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclasses.dataclass
class GPTClient:
    api_base: str = None  # https://YOUR_RESOURCE_NAME.openai.azure.com/
    deployment_name: str = None
    api_version: str = "2024-02-15-preview"
    api_key: str = None

    def __post_init__(self):
        assert self.api_base is not None, "API base is required"
        assert self.deployment_name is not None, "Deployment name is required"

        self._client = self.get_client()
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def client(self):
        return self._client

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=2, min=30, max=128),
        retry=tenacity.retry_if_exception_type(Exception))
    def get_response(self, max_tokens: int = 500, raise_exception: bool = True) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self._messages,
                max_tokens=max_tokens)
        except Exception as e:
            if raise_exception:
                raise e
            response = None

        return response

    def get_client(self):
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

    def update_messages(self, content: list[dict]):
        message = {"role": "user", "content": []}
        for item in content:
            if item["type"] == ContentType.TEXT:
                message["content"].append({"type": "text", "text": item["text"]})
            elif item["type"] == ContentType.IMAGE:
                message["content"].append({"type": "image_url", "image_url": {"url": item["url"]}})
        self._messages.append(message)

    # def create_single_message(self, content: list[dict]):
    #     message = {"role": "user", "content": []}
    #     for item in content:
    #         if item["type"] == ContentType.TEXT:
    #             message["content"].append({"type": "text", "text": item["text"]})
    #         elif item["type"] == ContentType.IMAGE:
    #             message["content"].append({"type": "image_url", "image_url": {"url": item["url"]}})
    #     return message

    def extract_output(self, response):
        if response is None:
            return None
        return response.choices[0].message.content

    @staticmethod
    def get_token_provider():
        return get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
