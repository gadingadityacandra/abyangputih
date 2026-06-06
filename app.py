import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Klasifikasi Bawang",
    page_icon="🧅",
    layout="wide"
)

# =====================================
# NAMA KELAS DATASET
# =====================================

class_names = ['bawang_merah', 'bawang_putih']

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    # Membuat arsitektur MobileNetV2
    model = models.mobilenet_v2(weights=None)

    # Mengubah output menjadi 2 kelas
    model.classifier[1] = nn.Linear(
        model.last_channel,
        2
    )

    # Memuat model hasil training
    model.load_state_dict(
        torch.load(
            "model_bawang_mobilenetv2.pth",
            map_location=torch.device("cpu")
        )
    )

    # Mode evaluasi
    model.eval()

    return model

model = load_model()

# =====================================
# PREPROCESSING GAMBAR
# =====================================

transform = transforms.Compose([

    # Menyesuaikan ukuran gambar
    transforms.Resize((224, 224)),

    # Mengubah gambar menjadi tensor
    transforms.ToTensor(),

    # Normalisasi ImageNet
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================
# FUNGSI PREDIKSI
# =====================================

def prediksi_gambar(image):

    # Preprocessing
    img_tensor = transform(image).unsqueeze(0)

    # Prediksi
    with torch.no_grad():

        output = model(img_tensor)

        prob = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(prob, 1)

    # Ambil label
    label = class_names[predicted.item()]

    # Confidence (%)
    conf = confidence.item() * 100

    return label, conf, prob

# =====================================
# TAMPILAN UTAMA
# =====================================

st.title("🧅 Klasifikasi Bawang Merah dan Bawang Putih")

st.write(
    "Pilih metode input gambar yang ingin digunakan."
)

# =====================================
# PILIH METODE INPUT
# =====================================

opsi = st.radio(
    "Metode Input",
    [
        "📁 Upload Banyak Gambar",
        "📷 Kamera"
    ]
)

# =====================================
# OPSI UPLOAD BANYAK GAMBAR
# =====================================

if opsi == "📁 Upload Banyak Gambar":

    uploaded_files = st.file_uploader(
        "Upload gambar bawang",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} gambar berhasil dipilih"
        )

        for file in uploaded_files:

            # Membuka gambar
            image = Image.open(file).convert("RGB")

            # Prediksi
            label, conf, prob = prediksi_gambar(image)

            # Menampilkan nama file
            st.subheader(f"📂 {file.name}")

            # Menampilkan gambar
            st.image(
                image,
                width=300
            )

            # Menampilkan hasil dengan Threshold (Batas Keyakinan)
            THRESHOLD = 75.0
            
            if conf < THRESHOLD:
                st.error(f"❌ Tidak dapat mendeteksi. Model ragu ({conf:.2f}%). Pastikan ini gambar Bawang Merah/Putih.")
            elif label == "bawang_merah":
                st.success(f"🧅 Bawang Merah ({conf:.2f}%)")
            else:
                st.success(f"🧄 Bawang Putih ({conf:.2f}%)")

            # Menampilkan confidence tiap kelas
            st.write("Tingkat Kepercayaan")

            st.write({
                "Bawang Merah":
                round(float(prob[0][0]) * 100, 2),

                "Bawang Putih":
                round(float(prob[0][1]) * 100, 2)
            })

            st.markdown("---")

# =====================================
# OPSI KAMERA
# =====================================

else:

    foto = st.camera_input(
        "Ambil foto bawang"
    )

    if foto is not None:

        # Membuka foto kamera
        image = Image.open(foto).convert("RGB")

        # Prediksi
        label, conf, prob = prediksi_gambar(image)

        # Menampilkan foto
        st.image(
            image,
            caption="Foto Kamera",
            width=350
        )

        # Menampilkan hasil dengan Threshold (Batas Keyakinan)
        THRESHOLD = 75.0
        
        if conf < THRESHOLD:
            st.error(f"❌ Tidak dapat mendeteksi. Model ragu ({conf:.2f}%). Pastikan ini gambar Bawang Merah/Putih.")
        elif label == "bawang_merah":
            st.success(f"🧅 Bawang Merah ({conf:.2f}%)")
        else:
            st.success(f"🧄 Bawang Putih ({conf:.2f}%)")

        # Menampilkan confidence
        st.write("Tingkat Kepercayaan")

        st.write({
            "Bawang Merah":
            round(float(prob[0][0]) * 100, 2),

            "Bawang Putih":
            round(float(prob[0][1]) * 100, 2)
        })