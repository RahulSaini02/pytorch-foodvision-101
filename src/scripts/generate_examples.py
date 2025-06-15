import random
import shutil
from pathlib import Path

# -------- Setup -------- #
NUM_EXAMPLES = 3
EXAMPLES_DIR = Path("examples")
TEST_DIR = Path("data/pizza_steak_sushi/test")


# -------- Get test images -------- #
test_data_paths = list(TEST_DIR.glob("*/*.jpg"))
example_paths = random.sample(test_data_paths, k=NUM_EXAMPLES)

# --------  Delete if exists, then recreate -------- #
if EXAMPLES_DIR.exists():
    print(f"[INFO] Removing existing folder: {EXAMPLES_DIR}")
    shutil.rmtree(EXAMPLES_DIR)

# -------- Create examples folder & copy images -------- #
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

for example_path in example_paths:
    destination = EXAMPLES_DIR / example_path.name
    print(f"[INFO] Copying {example_path} to {destination}")
    shutil.copy2(src=example_path, dst=destination)
