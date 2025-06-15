import gradio as gr
import os
import torch
from pathlib import Path

from src.config import (
    MODEL_PATH,
    CLASS_NAMES,
    EXAMPLES_DIR,
    APP_TITLE,
    APP_DESCRIPTION,
    APP_ARTICLE,
)
from src.modeling.models import create_effnetb2_model
from src.modeling.predict import predict

# Load model and transforms
model, transforms = create_effnetb2_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

# Destination path
examples_folder = Path(EXAMPLES_DIR)

# Create examples list from "examples/" directory
examples_list = [[f"{EXAMPLES_DIR}/{f}"] for f in os.listdir(examples_folder)]

# Create demo interface with `Interface`
demo = gr.Interface(
    fn=lambda img: predict(img, model, transforms, CLASS_NAMES),
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=examples_list,
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    article=APP_ARTICLE,
)

# Create demo interface with `Blocks`
# with gr.Blocks(
#     css=".gradio-container { max-width: 900px !important; margin: auto; }"
# ) as demo:
#     gr.Markdown(f"# {APP_TITLE}")
#     gr.Markdown(f"### {APP_DESCRIPTION}")

#     with gr.Row():
#         with gr.Column(scale=1):
#             image_input = gr.Image(type="pil", label="Upload Food Image")
#             predict_btn = gr.Button("🔍 Predict")
#             gr.Examples(
#                 examples=examples_list,
#                 inputs=image_input,
#                 label="Try an example",
#                 examples_per_page=6,
#             )
#         with gr.Column(scale=1):
#             label_output = gr.Label(label="Top Predictions", num_top_classes=3)
#             time_output = gr.Number(label="Prediction Time (s)")

#     predict_btn.click(
#         fn=lambda img: predict(img, model, transforms, CLASS_NAMES),
#         inputs=image_input,
#         outputs=[label_output, time_output],
#     )

#     gr.Markdown(APP_ARTICLE)
