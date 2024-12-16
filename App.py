import streamlit as st
import torch
from torchvision import models, transforms, datasets
from torch import nn
from PIL import Image, ImageOps
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
import gdown  # Pour télécharger les fichiers depuis Google Drive

# Configuration
num_classes = 10  # Nombre de classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels des classes
class_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Transformation personnalisée pour l'égalisation d'histogramme
class HistogramEqualization:
    def __call__(self, img):
        return ImageOps.equalize(img)

# Pipeline de transformation des images
transform = transforms.Compose([
    HistogramEqualization(),  # Égalisation d'histogramme
    transforms.Resize((224, 224)),  # Redimensionnement
    transforms.RandomHorizontalFlip(p=0.5),  # Retour horizontal
    transforms.RandomRotation(10),  # Rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Jitter de couleur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
])

# Chemins des fichiers modèles sur Google Drive
MODEL_LINKS = {
    "ResNet50": "https://drive.google.com/uc?id=1t1eIFQCA7FU9i8OLwzkHalrN6aapffae",  # Remplacez avec l'ID exact
    "ConvNeXt": "https://drive.google.com/uc?id=1t1eIFQCA7FU9i8OLwzkHalrN6aapffae"  # Remplacez avec l'ID exact
}

# Fonction pour télécharger et charger un modèle
def download_and_load_model(model_name):
    # Déterminer le chemin local pour sauvegarder le modèle
    model_path = f"{model_name}.pth"
    
    # Télécharger le fichier s'il n'existe pas localement
    if not os.path.exists(model_path):
        st.info(f"Téléchargement du modèle {model_name} depuis Google Drive...")
        gdown.download(MODEL_LINKS[model_name], model_path, quiet=False)
        st.success(f"Modèle {model_name} téléchargé avec succès.")
    
    # Charger le modèle
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    elif model_name == "ConvNeXt":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes, bias=True)
    else:
        st.error("Modèle inconnu sélectionné.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Prédire l'image
def predict_image(image, model_name):
    model = download_and_load_model(model_name)  # Télécharger et charger le modèle
    if model is None:
        return None
    image_tensor = transform(image).unsqueeze(0).to(device)  # Transformer l'image et l'envoyer au GPU/CPU
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)  # Récupérer la classe prédite
    label = class_labels[predicted_class.item()]  # Associer à son label
    return label

# Interface utilisateur avec Streamlit
st.title("Dashboard de classification d'images avec ResNet50 et ConvNeXt")

# Étape 1 : Téléchargement d'une image pour la prédiction
st.subheader("1. Prédiction d'image")
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

# Étape 2 : Sélectionner un modèle
st.subheader("2. Sélectionnez un modèle")
model_name = st.selectbox("Choisissez un modèle", ["ResNet50", "ConvNeXt"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Prédiction sur l'image téléchargée
    if st.button("Faire une prédiction"):
        predicted_label = predict_image(image, model_name)
        if predicted_label:
            st.write(f"Classe prédite : **{predicted_label}**")
