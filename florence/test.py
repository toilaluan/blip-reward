import requests
from PIL import Image
from transformers import AutoProcessor
from florence.modeling_florence2 import Florence2ForConditionalGeneration

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)
model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base")
model.to("cuda")

prompt = "<CAPTION>"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
print(inputs)
print(inputs.keys())
raise
outputs = model(**inputs, output_hidden_states=True)

print(outputs)
