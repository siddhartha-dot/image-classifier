import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import gradio as gr

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

def predict(inp):
    inp = preprocess(inp).unsqueeze(0)

    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)

    confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=10),
).launch(share=True)