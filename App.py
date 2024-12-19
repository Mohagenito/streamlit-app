import streamlit as st
from PIL import Image, ImageOps
import os
import gdown  # Pour télécharger les fichiers depuis Google Drive

# Configuration
NUM_CLASSES = 10  # Nombre de classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels des classes
CLASS_LABELS = {
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
TRANSFORM = transforms.Compose([
    HistogramEqualization(),  # Égalisation d'histogramme
    transforms.Resize((224, 224)),  # Redimensionnement
    transforms.RandomHorizontalFlip(p=0.5),  # Retour horizontal
    transforms.RandomRotation(10),  # Rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Jitter de couleur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
])

# Liens des modèles Google Drive (ajustez ces liens pour obtenir le fichier directement)
MODEL_LINKS = {
    "ResNet50": "https://drive.google.com/uc?id=1WTO_F6CG6NLd_BxnTCZZDv1oPkX2I00f",  # Remplacez avec l'ID exact
    "ConvNeXt": "https://drive.google.com/uc?id=14di4RyyKzeBuRUlB_N1n6EkDvo-AVvqn"  # Remplacez avec l'ID exact
}

# Fonction pour télécharger et charger un modèle
def download_and_load_model(model_name):
    model_path = f"{model_name}.pth"  # Déterminer le chemin local pour sauvegarder le modèle
    
    # Télécharger le modèle s'il n'existe pas
    if not os.path.exists(model_path):
        st.info(f"Téléchargement du modèle {model_name} depuis Google Drive...")
        try:
            gdown.download(MODEL_LINKS[model_name], model_path, quiet=False)
            st.success(f"Modèle {model_name} téléchargé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
            return None
    
    # Charger le modèle selon le type
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES, bias=True)
    elif model_name == "ConvNeXt":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=NUM_CLASSES, bias=True)
    else:
        st.error("Modèle inconnu sélectionné.")
        return None

    # Charger les poids du modèle et le mettre sur le bon appareil (CPU ou GPU)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()  # Passage en mode évaluation
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None
    return model

# Prédire l'image
def predict_image(image, model_name):
    model = download_and_load_model(model_name)  # Télécharger et charger le modèle
    if model is None:
        return None
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)  # Transformer l'image et l'envoyer au GPU/CPU
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)  # Récupérer la classe prédite
    label = CLASS_LABELS[predicted_class.item()]  # Associer à son label
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
