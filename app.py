import streamlit as st
from moviepy.editor import *
from PIL import Image
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Gerador de Vídeo do Boneco", layout="centered")

st.title("Gerador de Vídeo com Imagem")
st.write("Faça o upload de uma imagem e gere um vídeo de 10 segundos com ela!")

# Upload da imagem
uploaded_file = st.file_uploader("Envie uma imagem (JPG, PNG...)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("Gerar vídeo"):
        with st.spinner("Gerando vídeo..."):
            image_array = np.array(image)

            # Diretório temporário para evitar problemas de permissão no Streamlit Cloud
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "video_output.mp4")

            # Criar e salvar o vídeo
            clip = ImageClip(image_array).set_duration(10)
            clip.write_videofile(output_path, fps=24, logger=None)

            st.success("Vídeo gerado com sucesso!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("Baixar vídeo", f, file_name="video_com_imagem.mp4", mime="video/mp4")
else:
    st.info("Por favor, envie uma imagem para continuar.")
