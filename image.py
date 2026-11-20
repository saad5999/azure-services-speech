import base64
from openai import OpenAI

endpoint = "openai/v1"
deployment_name = "gpt-5.4-mini"
api_key = ""

# Helper function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your local image
image_path = "/content/mahesh babu.jpg"
base64_image = encode_image(image_path)

client = OpenAI(
     base_url=endpoint,
     api_key=api_key
 )

response = client.responses.create(
     model=deployment_name,
     input=[{
         "role": "user",
         "content": [
             {"type": "input_text", "text": "what's in this image?"},
             {
                 "type": "input_image",
                 "image_url":"https://www.imdb.com/name/nm1121870/"
             },
         ],
     }],
)

print(f"answer: {response.output_text}")