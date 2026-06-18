import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# ============================================
# KONFIGURASI
# ============================================
MODEL_PATH = "ikan_model_deploy.pt"
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = [
    "Busuk Tidak Segar (5-6 Hari)",
    "Sangat Segar (1-2 Hari)",
    "Segar (3-4 Hari)"
]

# Emoji dan warna per kelas
CLASS_INFO = {
    "Busuk Tidak Segar (5-6 Hari)": { "color": "#e74c3c", "status": "TIDAK LAYAK KONSUMSI"},
    "Sangat Segar (1-2 Hari)": { "color": "#2ecc71", "status": "SANGAT LAYAK KONSUMSI"},
    "Segar (3-4 Hari)": { "color": "#f39c12", "status": "LAYAK KONSUMSI"},
}

# ============================================
# FUNGSI LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load model PyTorch dari file .pth (state_dict)."""
    device = torch.device('cpu')

    # Bangun ulang arsitektur model (sama persis dengan saat training)
    base_model = models.mobilenet_v2(weights=None)
    in_features = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    return base_model, device


def preprocess_image(image: Image.Image):
    """Preprocess gambar sesuai format input MobileNetV2."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # Tambah dimensi batch


def predict(model, image_tensor, device):
    """Prediksi kelas dan probabilitas."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    all_probs = {CLASS_NAMES[i]: probabilities[0][i].item() * 100 for i in range(NUM_CLASSES)}

    return predicted_class, confidence_pct, all_probs


# ============================================
# TAMPILAN STREAMLIT
# ============================================
st.set_page_config(
    page_title="Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .prob-label {
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Klasifikasi Kesegaran Ikan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unggah foto mata ikan untuk mendeteksi tingkat kesegarannya<br>Menggunakan MobileNetV2 (PyTorch)</div>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write(
        "Aplikasi ini mengklasifikasikan tingkat kesegaran ikan "
        "berdasarkan **citra mata ikan** menggunakan model Deep Learning "
        "MobileNetV2 yang telah di-fine-tuning."
    )
    st.divider()
    st.subheader(" Kelas Klasifikasi")
    st.markdown(" **Sangat Segar** (1-2 Hari)")
    st.markdown(" **Segar** (3-4 Hari)")
    st.markdown(" **Busuk / Tidak Segar** (5-6 Hari)")
    st.divider()
    st.subheader(" Tips Foto")
    st.write("- Ambil foto bagian **mata ikan** secara dekat.")
    st.write("- Pastikan pencahayaan cukup terang.")
    st.write("- Hindari foto yang buram atau gelap.")

# Upload gambar
uploaded_file = st.file_uploader(
    " Pilih gambar mata ikan...",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload foto mata ikan dalam format JPG, JPEG, PNG, atau BMP."
)

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    # Load model dan prediksi
    with st.spinner("🔍 Menganalisis gambar..."):
        model, device = load_model()
        image_tensor = preprocess_image(image)
        predicted_class, confidence, all_probs = predict(model, image_tensor, device)

    info = CLASS_INFO[predicted_class]

    with col2:
        # Kotak hasil prediksi
        st.markdown(
            f"""
            <div class="result-box" style="background-color: {info['color']}22; border: 2px solid {info['color']};">
                <div style="font-size: 3rem;">{info['emoji']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {info['color']};">{predicted_class}</div>
                <div style="font-size: 1rem; margin-top: 0.5rem;">Confidence: <b>{confidence:.1f}%</b></div>
                <div style="font-size: 0.9rem; margin-top: 0.3rem; color: #555;">{info['status']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Probabilitas semua kelas
    st.subheader(" Probabilitas Setiap Kelas")
    for cls_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        cls_info = CLASS_INFO[cls_name]
        st.markdown(f"**{cls_info['emoji']} {cls_name}**")
        st.progress(prob / 100)
        st.caption(f"{prob:.2f}%")

else:
    # Placeholder jika belum ada gambar
    st.info("👆 Silakan unggah gambar mata ikan di atas untuk memulai klasifikasi.")
