import torch
import json
from pathlib import Path
from datetime import datetime
from utils.path_utils import resolve_root

# Add project root to sys.path
resolve_root(3)

from src.config import CLASS_NAMES
from src.features import create_dataloaders
from src.modeling.models import create_effnetb2_model
from src.modeling.train import train
from src.modeling.utils import save_model

# Set hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Paths
DATA_DIR = Path("data/pizza_steak_sushi_20_percent")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = Path("models")
MODEL_NAME = f"effnetb2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
RESULTS_PATH = Path("reports/effnetb2_results.json")

# Create model
model, transform = create_effnetb2_model(num_classes=len(CLASS_NAMES))

# Load data
train_dataloader, test_dataloader, _ = create_dataloaders(
    train_dir=str(TRAIN_DIR),
    test_dir=str(TEST_DIR),
    transform=transform,
    batch_size=BATCH_SIZE,
)

# Setup optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    device=DEVICE,
)

# Save model
save_model(model=model, target_dir=MODEL_DIR, model_name=MODEL_NAME)

# ---- Save training results ----
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Training complete. Model saved to: {MODEL_DIR / MODEL_NAME}")
print(f"Results saved to: {RESULTS_PATH}")
