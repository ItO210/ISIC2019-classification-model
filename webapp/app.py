from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Constants
# ---------------------------
IMG_SIZE = 300

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASSES = ['MEL', 'NV', 'AK', 'BCC', 'BKL', 'VASC', 'DF', 'SCC']
NUM_CLASSES = len(CLASSES)

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# ---------------------------
# Model
# ---------------------------
def build_model(num_classes):
    model = efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = build_model(NUM_CLASSES)

    state_dict = torch.load(path, map_location=device)

    # Handle different save formats
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../cnn/efficientnet_b3_300_best.pth"
)

model = load_model(MODEL_PATH)

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------
# Grad-CAM
# ---------------------------
def generate_gradcam(model, input_tensor, original_image, target_class=None):
    input_tensor = input_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True

    # Better layer choice for EfficientNet
    target_layer = model.features[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(
        original_image.resize((IMG_SIZE, IMG_SIZE))
    ) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.75
    )

    return visualization


def encode_image_to_base64(image_array):
    image_pil = Image.fromarray(image_array)
    buffer = BytesIO()
    image_pil.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", result={"error": "No file uploaded"})

        if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
            return render_template("index.html", result={"error": "Invalid file type"})

        try:
            # Load image
            img_pil = Image.open(file).convert("RGB")

            # Transform
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # ---------------------------
            # Prediction
            # ---------------------------
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)

                top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            top_indices = top_indices.squeeze().tolist()
            top_probs = top_probs.squeeze().tolist()

            predictions = [
                (CLASSES[idx], float(prob))
                for idx, prob in zip(top_indices, top_probs)
            ]

            # ---------------------------
            # Grad-CAM (use top-1 class)
            # ---------------------------
            cam_image = generate_gradcam(
                model,
                img_tensor,
                img_pil,
                target_class=top_indices[0]
            )

            cam_base64 = encode_image_to_base64(cam_image)

            result = {
                "predictions": predictions,
                "cam": cam_base64
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)