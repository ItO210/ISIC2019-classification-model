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
import threading
from urllib.request import urlopen, Request as URLRequest

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

_model = None
_model_lock = threading.Lock()


def get_model():
    """Carga el modelo una sola vez; en CPU puede tardar 30–90 s la primera vez."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            print("Cargando pesos del modelo (EfficientNet-B3, ~47 MB)...", flush=True)
            print("En CPU esto puede tardar un minuto; no cierres la terminal.", flush=True)
            _model = load_model(MODEL_PATH)
            print("Modelo listo.", flush=True)
        return _model

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
        file      = request.files.get("image")
        image_url = request.form.get("image_url", "").strip()

        has_file = file and file.filename != ""
        has_url  = bool(image_url)

        if not has_file and not has_url:
            return render_template("index.html", result={"error": "No se recibió ninguna imagen."})

        try:
            if has_file:
                if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
                    return render_template("index.html", result={"error": "Tipo de archivo no válido."})
                img_pil = Image.open(file).convert("RGB")
            else:
                req_obj = URLRequest(image_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req_obj, timeout=10) as resp:
                    img_pil = Image.open(BytesIO(resp.read())).convert("RGB")

            # Transform
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            model = get_model()

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
    # 5000 suele estar ocupado en macOS (Receptor AirPlay)
    port = int(os.environ.get("PORT", 5001))
    # debug=True + reloader por defecto carga el modelo DOS veces y suele “congelar” la terminal.
    print(
        f"Iniciando en http://127.0.0.1:{port} — el modelo carga en segundo plano; "
        "la primera petición puede esperar hasta que termine.",
        flush=True,
    )
    threading.Thread(target=get_model, daemon=True).start()
    app.run(debug=True, use_reloader=False, threaded=True, port=port)