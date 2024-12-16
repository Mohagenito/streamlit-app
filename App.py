import streamlit as st
import torch
from torchvision import models, transforms, datasets
from torch import nn
from PIL import Image, ImageOps
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np

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

# Charger les modèles
def load_model(model_name):
    model_path = ""
    if model_name == "ResNet50":
        model_path = "ResNet50_optimized.pth"
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
    elif model_name == "ConvNeXt":
        model_path = "ConvNeXt_transfer_learning_quantized.pth"
        model = models.convnext_base(weights="DEFAULT")
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes, bias=True)
    
    # Vérifier si le fichier existe
    if not os.path.exists(model_path):
        st.error(f"Fichier de modèle introuvable : {model_path}")
        st.stop()
    
    # Charger le modèle
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Prédire l'image
def predict_image(image, model_name):
    model = load_model(model_name)  # Charger le modèle choisi
    image_tensor = transform(image).unsqueeze(0).to(device)  # Transformer l'image et l'envoyer au GPU/CPU
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)  # Récupérer la classe prédite
    label = class_labels[predicted_class.item()]  # Associer à son label
    return label

# Analyse exploratoire
def display_exploratory_analysis(dataset):
    st.subheader("Analyse exploratoire des données")
    if len(dataset) > 0:
        # Convertir les images Tensor en PIL.Image pour les afficher
        images = [to_pil_image(dataset[i][0]) for i in range(min(len(dataset), 5))]
        st.image(images, width=100)
        st.write("Exemple d'images provenant de notre dataset.")
    else:
        st.write("Aucune image n'a été trouvée dans le dossier sélectionné.")

# Interface utilisateur avec Streamlit
st.title("Dashboard de classification d'images avec ResNet50 et ConvNeXt")

# Étape 1 : Chargement des données
st.subheader("1. Chargement des données")
data_path = st.text_input("Entrez le chemin du dossier contenant vos données :", "")
if data_path and os.path.isdir(data_path):
    dataset = datasets.ImageFolder(root=data_path, transform=transforms.ToTensor())
    st.success(f"Données chargées depuis : {data_path}")
else:
    dataset = None
    st.warning("Veuillez entrer un chemin valide vers un dossier contenant des images.")

# Étape 2 : Analyse exploratoire (si les données sont disponibles)
if dataset:
    display_exploratory_analysis(dataset)

# Étape 3 : Sélectionner un modèle
st.subheader("2. Sélectionnez un modèle")
model_name = st.selectbox("Choisissez un modèle", ["ResNet50", "ConvNeXt"])

# Étape 4 : Téléchargement d'une image pour la prédiction
st.subheader("3. Prédiction d'image")
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Affichage de l'image transformée
    transformed_image = transform(image)
    st.image(transformed_image.permute(1, 2, 0).numpy(), caption="Image transformée", use_column_width=True, clamp=True)

    # Prédiction sur l'image téléchargée
    if st.button("Faire une prédiction"):
        predicted_label = predict_image(image, model_name)
        st.write(f"Classe prédite : **{predicted_label}**")

        # Afficher les graphiques de comparaison des pertes et précisions
        st.subheader("Comparaison des pertes et précisions")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Exemple de graphes de pertes
        epochs = range(1, 11)  # 10 époques
        axs[0].plot(epochs, np.random.rand(10), label="Train Loss")
        axs[0].plot(epochs, np.random.rand(10), label="Validation Loss")
        axs[0].set_title('Comparaison des pertes')
        axs[0].set_xlabel('Époques')
        axs[0].set_ylabel('Perte')
        axs[0].legend()

        # Exemple de graphes de précisions
        axs[1].plot(epochs, np.random.rand(10) * 100, label="Train Accuracy")
        axs[1].plot(epochs, np.random.rand(10) * 100, label="Validation Accuracy")
        axs[1].set_title('Comparaison des précisions')
        axs[1].set_xlabel('Époques')
        axs[1].set_ylabel('Précision (%)')
        axs[1].legend()

        st.pyplot(fig)
