import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(HERE, "logo.png")


import streamlit as st
from PIL import Image
import tempfile

# Import your own utilities
from predict_cassava_disease import predict_image, load_model, transform, class_names  # adjust this to your actual module


# Load model
@st.cache_resource
def _load_model(path='cassava_model.pth'):
    # load model saved earlier 
    return load_model()

# App config
st.set_page_config(page_title="Image Classifier", page_icon="logo.png", layout="centered")
cols = st.columns([2, 3, 2])
with cols[1]:
    st.image("logo.png", width=150)
st.markdown(
    """
    <style>
    body {
        background-color: #1e293b;
        color: white;
    }

    .stApp {
        background-color: #1e293b;
    }

    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption {
        color: white;
    }

    .css-18e3th9 {  /* main content container */
        background-color: #1e293b;
    }

    .css-1d391kg {  /* sidebar */
        background-color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 PyTorch Image Classifier")
st.markdown("Upload an image and let the model classify it into one of the categories.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
model = _load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image_path = tmp_file.name
        image.save(image_path)

    with st.spinner("Classifying..."):
        prediction = predict_image(image_path, model, transform, class_names)
        try:
            prediction = prediction.split('_')
            prediction.remove('')
            prediction.remove('Cassava')
        except:
            if not isinstance(predict_image, list):
                prediction = [prediction]
    st.success(f"**Prediction:** {' '.join(prediction)}")
    os.remove(image_path)
