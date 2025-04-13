import streamlit as st
from PIL import Image
import numpy as np
import moviepy.editor as mpy
import tempfile
import os

st.set_page_config(page_title="Gerador de Vídeo com Imagem", layout="centered")
st.title("Gerador de Vídeo com Imagem (MoviePy)")

uploaded_file = st.file_uploader("Envie uma imagem (JPG, PNG...)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("Gerar vídeo"):
        with st.spinner("Gerando vídeo..."):
            np_img = np.array(image)
            clip = mpy.ImageClip(np_img).set_duration(10)
            clip = clip.set_fps(24)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                video_path = tmp.name
                clip.write_videofile(video_path, codec="libx264", audio=False)

            st.success("Vídeo gerado com sucesso!")
            st.video(video_path)

            with open(video_path, "rb") as f:
                st.download_button("Baixar vídeo", f, "video.mp4", mime="video/mp4")
