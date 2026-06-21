import io
from typing import Tuple, Union, Optional

import streamlit as st
from PIL import Image
import requests
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision import models
from sentence_transformers import SentenceTransformer

# --- Embedded feature extractor (merged for single-folder deploy) ---
transforms_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_models(
    image_model_name: str = 'resnet18', text_model_name: str = 'all-MiniLM-L6-v2'
) -> Tuple[torch.nn.Module, SentenceTransformer]:
    """Load and cache image and text models for the Streamlit app."""
    if image_model_name == 'resnet18':
        image_model = models.resnet18(pretrained=True)
    else:
        image_model = models.__dict__.get(image_model_name, models.resnet18)(pretrained=True)
    image_model.eval()

    text_model = SentenceTransformer(text_model_name)
    return image_model, text_model


def get_text_embedding(text_query: str, text_model: Optional[SentenceTransformer] = None) -> np.ndarray:
    if text_model is None:
        _, text_model = load_models()
    return np.array(text_model.encode(text_query))


def get_image_features(
    image_input: Union[str, Image.Image], image_model: Optional[torch.nn.Module] = None
) -> Tuple[np.ndarray, Image.Image]:
    if isinstance(image_input, str):
        response = requests.get(image_input)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        image = image_input.convert('RGB')

    if image_model is None:
        image_model, _ = load_models()

    img_tensor = transforms_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        original_fc = image_model.fc
        image_model.fc = torch.nn.Identity()
        features = image_model(img_tensor)
        image_model.fc = original_fc

    return features.squeeze().cpu().numpy(), image

# --- End embedded extractor ---

# Configure page
st.set_page_config(page_title="Multimodal Feature Extractor", layout="wide")
st.title("🖼️ Multimodal Feature Extractor")
st.markdown("Extract and compare image and text features using AI models — no coding required")

# Load models (cached)
with st.spinner("Loading models (this may take a minute the first time)..."):
    image_model, text_model = load_models()

st.markdown(
    """
    **How to use (3 simple steps):**
    1. Upload an image or paste an image URL.
    2. Enter a short text question describing the image.
    3. Click **Extract Features** and view/download results.
    """
)

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

# Image input section
with col1:
    st.subheader("📸 Image Input")
    image_input_type = st.radio("Choose image input method:", ["Upload", "URL"], key="image_input")
    
    image = None
    image_features = None
    
    if image_input_type == "Upload":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
    else:
        image_url = st.text_input(
            "Enter image URL:",
            value="https://www.nasa.gov/sites/default/files/thumbnails/image/main_image_deep_field_smacs0723-5mb.jpg",
        )
        if image_url:
            try:
                image_features, image = get_image_features(image_url, image_model)
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error("Could not load image from the URL. Try a different URL or upload an image.")

# Text input section
with col2:
    st.subheader("💬 Text Input")
    text_query = st.text_area(
        "Enter a text query:",
        value="What is in this picture? Is it a cat or a dog?",
        height=100
    )
    
    text_embedding = None
    if text_query:
        try:
            text_embedding = get_text_embedding(text_query, text_model)
        except Exception:
            st.error("Error computing text embedding.")

# Process button
if st.button("🚀 Extract Features", use_container_width=True):
    if image is None and image_input_type == "Upload":
        st.warning("Please upload an image first")
    else:
        # Extract image features if not already done
        if image is not None and image_features is None:
            try:
                with st.spinner("Extracting image features..."):
                    image_features, _ = get_image_features(image, image_model)
            except Exception:
                st.error("Error extracting image features. Make sure the image is valid.")
        
        # Display results
        if image_features is not None:
            st.success("✅ Features extracted successfully!")
            
            # Create tabs for results
            tab1, tab2, tab3 = st.tabs(["Image Features", "Text Embedding", "Comparison"])
            
            with tab1:
                st.header("Image Features")
                st.write("Vector shape:", image_features.shape)
                st.write("First 10 values:")
                st.write(image_features[:10])
                st.bar_chart(image_features[:50])
                # Download button
                buf = io.BytesIO()
                np.save(buf, image_features)
                buf.seek(0)
                st.download_button("Download image features (.npy)", buf, file_name="image_features.npy")
            
            with tab2:
                if text_embedding is not None:
                    st.header("Text Embedding")
                    st.write("Vector shape:", text_embedding.shape)
                    st.write("First 10 values:")
                    st.write(text_embedding[:10])
                    st.bar_chart(text_embedding[:50])
                    buf2 = io.BytesIO()
                    np.save(buf2, text_embedding)
                    buf2.seek(0)
                    st.download_button("Download text embedding (.npy)", buf2, file_name="text_embedding.npy")
            
            with tab3:
                if text_embedding is not None:
                    # Calculate similarity
                    from sklearn.metrics.pairwise import cosine_similarity

                    img_feat_norm = image_features / (np.linalg.norm(image_features) + 1e-8)
                    text_emb_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)

                    similarity = cosine_similarity(
                        img_feat_norm.reshape(1, -1), text_emb_norm.reshape(1, -1)
                    )[0][0]

                    st.header("Image ↔ Text Similarity")
                    st.metric("Cosine Similarity", f"{similarity:.4f}")

                    if similarity > 0.7:
                        st.success("High similarity — the text closely matches the image.")
                    elif similarity > 0.4:
                        st.info("Moderate similarity — some shared content detected.")
                    else:
                        st.warning("Low similarity — the text and image seem unrelated.")

                    # Friendly explanation for non-technical users
                    if similarity > 0.7:
                        st.write("Interpretation: Your text description matches the image well.")
                    elif similarity > 0.4:
                        st.write("Interpretation: There's some overlap between text and image.")
                    else:
                        st.write("Interpretation: The text does not describe this image closely.")

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **Multimodal Feature Extractor** uses:
    - **ResNet18** for image features
    - **Sentence Transformers** (all-MiniLM-L6-v2) for text embeddings
    
    Features extracted:
    - Image: 512-dimensional vector
    - Text: 384-dimensional vector
    """)
    
    st.markdown("---")
    st.subheader("📚 Example Queries")
    st.markdown("""
    - "What is in this picture?"
    - "Is this a cat or a dog?"
    - "Describe the landscape"
    - "What objects are visible?"
    """)
