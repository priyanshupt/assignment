import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload a brain MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded MRI.", use_column_width=True)

    if st.button('Segment'):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)
        prediction = np.array(response.json()["prediction"])
        st.image(prediction[0], caption="Segmented Metastasis.", use_column_width=True)
