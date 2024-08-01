from enum import Enum
import instructor
from pydantic import BaseModel
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from openai import AzureOpenAI
import base64


# Define an Enum class for defect types
class DefectType(str, Enum):
    NO_DEFECT = "no_defect"
    CRACK = "crack"
    BROKEN = "broken"
    SCRATCH = "scratch"

# Define your Pydantic model


class IndustrialDefect(BaseModel):
    defect_type: DefectType
    reason: str = ""


def get_messages_body(prompt, base64_image, mime_type):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                {
                        "type": "image_url",
                        "image_url": {
                                "url": f"data:{mime_type};base64," + f"{base64_image}",
                        }
                    }
            ]
        }
    ]


def convert_to_base64(image_path: str):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("ascii")
    return base64_image, mime_type


def get_prompt():
    return """You are an expert visual inspector for a manufacturing company that makes glass bottles. You will be shown a top-view image of a glass bottle and your task is to identify if it is defective or not. Think step-by-step - first identify if there is a defect or not. Second, if there is a defect, identify the type of defect. **IF** present, the defect can only be of the following types: 1.broken 2.contamination. Third, explain your reasoning for the defect if present. Finally, identify where the defect is located in the image and provide the relative coordinates (between 0-1) of the bounding box enclosing the defect in the format [x_top, y_top, x_bottom, y_bottom]. Please return your response **strictly** as a valid JSON object as defined by the IndustrialDefect Pydantic model.

    Note: If  there is no defect or if you are unsure, please return defect_type as 'no_defect' and leave the other fields empty.
    """


def main():

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    client = AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint="https://customvision-dev-aoai.openai.azure.com/",
        azure_ad_token_provider=token_provider)

    # Patch the OpenAI client
    client = instructor.patch(client)

    prompt = get_prompt()
    sample_img_path = "datasets/raw/mvtec-ad/bottle/test/broken_large/000.png"
    base64_image, mime_type = convert_to_base64(sample_img_path)

    # Extract structured data from natural language
    defect = client.chat.completions.create(
        model="gpt4o-001",
        response_model=IndustrialDefect,
        messages=get_messages_body(prompt, base64_image, mime_type),
    )

    print(defect)


if __name__ == '__main__':
    main()
    
