# Streamlit Multimodal Feature Extractor

Run locally:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r streamlit_feature_extractor/requirements.txt
streamlit run streamlit_feature_extractor/app.py
```

Deploy on Streamlit Cloud:

- In the Streamlit Cloud UI, point the app entry file to `streamlit_feature_extractor/app.py`.
- Ensure `streamlit_feature_extractor/requirements.txt` is used (Cloud will install it).

Notes:
- Models (PyTorch & sentence-transformers) are listed in the requirements; install may take time and needs sufficient RAM.
- If you prefer smaller footprint, change the models in `multimodal_feature_extractor/feature_extractor.py`.
# 🖼️ Multimodal Feature Extractor

A Streamlit web application that extracts and compares image and text features using pre-trained AI models.

## Features

- **Image Feature Extraction**: Uses ResNet18 to extract 512-dimensional feature vectors from images
- **Text Embedding**: Uses Sentence Transformers (all-MiniLM-L6-v2) to generate 384-dimensional embeddings from text
- **Similarity Analysis**: Calculates cosine similarity between image and text features
- **Multiple Input Methods**: Upload images or provide URLs
- **Interactive Visualization**: View feature distributions and comparisons

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Usage

1. **Upload or provide an image** using either the upload form or image URL
2. **Enter a text query** describing the image or asking about it
3. **Click "Extract Features"** to process both inputs
4. **View the results** in tabs showing:
   - Image features
   - Text embeddings
   - Similarity comparison

## Models Used

- **ResNet18**: Pre-trained ImageNet model for image feature extraction
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Efficient text embedding model

## Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your repository
4. Set main file to `streamlit_feature_extractor/app.py`
5. Deploy!

### Deploy on Other Platforms

- **Heroku**: Use Procfile with `streamlit run app.py --server.port=$PORT`
- **Docker**: Create a Dockerfile with Python 3.9+ and required packages
- **AWS/GCP/Azure**: Use container deployment services

## Requirements

- Python 3.8+
- 2GB+ RAM for model loading
- Internet connection for model downloads on first run
