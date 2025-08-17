import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# =========================
# Class names & device
# =========================
CLASS_NAMES = [
    "Amrapali Mango",
    "Banana Mango",
    "Chaunsa Mango",
    "Fazli Mango",
    "Haribhanga Mango",
    "Himsagar Mango"
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Custom CNN definition
# =========================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# Model loaders
# =========================
def load_custom_cnn():
    model = CustomCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("custom_cnn_model.pth", map_location=DEVICE))
    model.eval()
    return model

def load_resnet50():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_resnet50.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_densenet121():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_densenet121.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_efficientnet_b0():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_efficientnet_b0.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_vgg16():
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_vgg16.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# =========================
# Target layer selector
# =========================
def get_target_layer(model):
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4[-1]
    elif hasattr(model, "features"):  # VGG, DenseNet, EfficientNet, Custom CNN
        conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            # Fallback: try to find any Conv2d layer in the model
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    return module
            raise ValueError("No convolutional layers found in the model.")
        return conv_layers[-1]
    else:
        raise ValueError("Target layer not found")

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# =========================
# Streamlit UI
# =========================
st.title("Image dataset of Bangladeshi mango leaf")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ("Custom CNN", "ResNet-50", "DenseNet121", "EfficientNet-B0", "VGG16")
)
uploaded_file = st.sidebar.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# =========================
# Load selected model
# =========================
if model_choice == "Custom CNN":
    model = load_custom_cnn()
elif model_choice == "ResNet-50":
    model = load_resnet50()
elif model_choice == "DenseNet121":
    model = load_densenet121()
elif model_choice == "EfficientNet-B0":
    model = load_efficientnet_b0()
elif model_choice == "VGG16":
    model = load_vgg16()

# =========================
# Prediction + XAI
# =========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    tensor_img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top 3 predictions
        top3_idxs = np.argsort(probs)[-3:][::-1]  # Indices of top 3 predictions
        top3_classes = [CLASS_NAMES[i] for i in top3_idxs]
        top3_probs = [probs[i] * 100 for i in top3_idxs]
        
        pred_idx = top3_idxs[0]  # Top prediction
        pred_class = top3_classes[0]
        pred_prob = top3_probs[0]

    st.subheader("Predictions")
    for i in range(3):
        st.write(f"{top3_classes[i]}: {top3_probs[i]:.2f}%")

    # Prepare CAM methods
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    target_layers = [get_target_layer(model)]

    # Create CAM instances
    gradcam = GradCAM(model=model, target_layers=target_layers)
    gradcam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
    eigencam = EigenCAM(model=model, target_layers=target_layers)
    ablationcam = AblationCAM(model=model, target_layers=target_layers)

    # Generate CAM visualizations - Make sure variable names match what you use later
    grayscale_gradcam = gradcam(input_tensor=tensor_img)[0]  # Changed from grayscale_gc
    grayscale_gradcam_pp = gradcam_pp(input_tensor=tensor_img)[0]  # Changed from grayscale_gcpp
    grayscale_eigencam = eigencam(input_tensor=tensor_img)[0]  # Changed from grayscale_eig
    grayscale_ablationcam = ablationcam(input_tensor=tensor_img)[0]  # Changed from grayscale_ab

    # Make sure these variable names match what you use in cam_images list
    vis_gradcam = show_cam_on_image(rgb_img, grayscale_gradcam, use_rgb=True)
    vis_gradcam_pp = show_cam_on_image(rgb_img, grayscale_gradcam_pp, use_rgb=True)
    vis_eigencam = show_cam_on_image(rgb_img, grayscale_eigencam, use_rgb=True)
    vis_ablationcam = show_cam_on_image(rgb_img, grayscale_ablationcam, use_rgb=True)

    # Display CAM visualizations in a grid - Updated variable names to match
    st.subheader("CAM Visualizations")
    cols = st.columns(4)
    cam_images = [vis_gradcam, vis_gradcam_pp, vis_eigencam, vis_ablationcam]  # Fixed variable names
    cam_titles = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM"]
    
    for col, title, img_cam in zip(cols, cam_titles, cam_images):
        col.image(img_cam, caption=f"{title}\n(Predicted: {pred_class})", use_container_width=True)

    # LIME visualization
    st.subheader("LIME Explanation")
    
    def batch_predict(images):
        batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
        logits = model(batch.to(DEVICE))
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img), batch_predict, top_labels=1, hide_color=0, num_samples=100
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    lime_vis = mark_boundaries(temp / 255.0, mask)

    # Display original and LIME side by side
    col1, col2 = st.columns(2)
    col1.image(np.array(img.resize((224, 224))), caption=f"Original Image\n(Predicted: {pred_class})", use_container_width=True)
    col2.image(lime_vis, caption=f"LIME\n(Predicted: {pred_class})", use_container_width=True)
