import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

st.set_page_config(page_title="Gerador de Vídeo com Imagem", layout="centered")

st.title("Gerador de Vídeo com Imagem (OpenCV)")
st.write("Faça upload de uma imagem e gere um vídeo de 10 segundos!")

uploaded_file = st.file_uploader("Envie uma imagem (JPG, PNG...)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("Gerar vídeo"):
        with st.spinner("Gerando vídeo..."):
            np_img = np.array(image)
            height, width, _ = np_img.shape
            fps = 24
            duration = 10
            total_frames = fps * duration

            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = os.path.join(tmpdir, "output.mp4")

                # Cria o vídeo com OpenCV
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for _ in range(total_frames):
                    frame = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    video.write(frame)

                video.release()

                st.success("Vídeo gerado com sucesso!")
                st.video(video_path)

                with open(video_path, "rb") as f:
                    st.download_button("Baixar vídeo", f, "video.mp4", mime="video/mp4")
