import torch
from torchvision.transforms import Compose
from typing import Tuple, Dict
from timeit import default_timer as timer
from PIL import Image


def predict(
    img: Image.Image,
    model: torch.nn.Module,
    transform: Compose,
    class_names: list,
    device: str = "cpu",
) -> Tuple[Dict, float]:
    """Transforms and performs prediction on img, returns (class_probs_dict, time_taken)."""
    start_time = timer()

    # Transform and batch the image
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)

    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time
