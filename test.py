from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


image = Image.open("test/image/cat.png")
texts = ["a cat with green background", "a cat with black background"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)  # 어떤 설명이 가장 잘 맞는지 확률로 출력
