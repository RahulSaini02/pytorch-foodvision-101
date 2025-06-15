# Food Vision 101

A PyTorch-based image classification project using **EfficientNetB2** and **Vision Transformer (ViT)** to classify food images into `pizza`, `steak`, or `sushi`. The final model is deployed using **Gradio** and hosted on **Hugging Face Spaces**.

---

## 🚀 Project Structure

```
FOODVISION-101/
│
├── data/                           # Raw and extracted datasets
│   ├── pizza_steak_sushi/          # Preprocessed food image dataset
|   └── food-101.zip                # Original ZIP archive
│
├── models/                         # Trained PyTorch model weights (.pth)
│   ├── effnetb2.pth
│   ├── effnetb2_food101.pth
│   ├── tinyvgg.pth
│   └── vit.pth
│
├── examples/                       # Example images used by Gradio app
│
├── notebooks/                      # Jupyter notebooks for EDA, training, testing
│   ├── 1.0-rs-exploratory-data-analysis.ipynb
│   ├── 2.0-rs-train-eval-models.ipynb
│   └── 3.0-rs-interface-test.ipynb
│   └── 4.0-rs-foodvision-101.ipynb
│
├── reports/                        # Visualizations and evaluation reports
│   └── figures/
│       └── foodvision-mini-inference-speed-vs-performance.jpg
│
├── src/                            # Source code (modular structure)
│   ├── config.py                    # Global config variables (paths, class names)
│   ├── dataset.py                  # Data download/unzipping helpers
│   ├── features.py                 # Transforms and DataLoader creation
│   ├── plots.py                    # Visualization utilities
│   │
│   ├── modeling/                   # Model logic
│   │   ├── models.py               # Model creation: EffNetB2, ViT, etc.
│   │   ├── train.py                # Training loop
│   │   ├── predict.py              # Inference helpers
│   │   └── utils.py                # Save/load model, metrics, etc.
│   │
│   ├── services/                   # External integrations (e.g. Gradio app)
│   │   └── demo.py                 # Gradio Blocks interface
│   │
│   └── scripts/                    # CLI scripts to automate training/testing
│       ├── generate_examples.py    # Generate and copy example images
│       ├── train_effnetb2.py       # Train EfficientNetB2 model
│       └── train_vit.py            # Train ViT model
│
├── .gitignore
├── app.py                          # Launch Gradio app (entry point)
├── requirements.txt                # Python package dependencies
└── README.md

```
---

## 🧠 Key Features

- 🧠 Transfer learning with **EfficientNetB2** and **ViT**
- 🧹 Clean modular code (`src/` structure)
- 📈 Training via script (`src/scripts/train_effnetb2.py`, `train_vit.py`)
- 🖼️ Real-time image predictions with **Gradio**
- ☁️ Deployment-ready for Hugging Face Spaces

---

## 📦 Installation

```bash
git clone git@github.com:RahulSaini02/pytorch-foodvision-101.git
cd foodvision-101
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run Locally

```bash
python src/scripts/generative_examples.py

python src/scripts/train_effnetb2.py
# or
python src/scripts/train_vit.py
```

## Launch Gradio App
```bash
python app.py
```
