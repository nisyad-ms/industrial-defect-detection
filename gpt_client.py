import dataclasses

from openai import AzureOpenAI


@dataclasses.dataclass
class GPTClient:
    # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    api_base: str = None
    api_key: str = None
    deployment_name: str = None
    api_version: str = None

    def __post_init__(self):
        assert self.api_key is not None, "API key is required"
        assert self.api_base is not None, "API base is required"
        assert self.deployment_name is not None, "Deployment name is required"
        assert self.api_version is not None, "API version is required"

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}openai/deployments/{self.deployment_name}/extensions",
        )

    def get_response(self, base64_image: str, prompt: str, max_tokens: int = 500, raise_exception: bool = True) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64_image,
                            }
                        }
                    ]}
                ],
                max_tokens=max_tokens)
        except:
            if raise_exception:
                raise
            response = None

        return response

    def extract_result(self, response):
        if response is None:
            return None
        return response.choices[0]['message']['content']
