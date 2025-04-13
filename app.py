import streamlit as st
from PIL import Image, ImageSequence
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Gerador de GIF com Imagem", layout="centered")
st.title("Gerador de GIF com Imagem")

uploaded_file = st.file_uploader("Envie uma imagem (JPG, PNG...)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGBA")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("Gerar GIF"):
        with st.spinner("Gerando GIF..."):
            frames = []

            # Criar leve animação: zoom in/out alternando tamanho
            for i in range(20):
                scale = 1 + 0.01 * (i if i < 10 else 20 - i)
                new_size = (int(image.width * scale), int(image.height * scale))
                frame = image.resize(new_size, resample=Image.LANCZOS)
                # Centralizar
                background = Image.new("RGBA", image.size, (255, 255, 255, 0))
                pos = ((image.width - frame.width) // 2, (image.height - frame.height) // 2)
                background.paste(frame, pos, frame)
                frames.append(background)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
                frames[0].save(
                    tmp.name,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0,
                    transparency=0,
                    disposal=2
                )
                gif_path = tmp.name

            st.success("GIF gerado com sucesso!")
            st.image(gif_path)
            with open(gif_path, "rb") as f:
                st.download_button("Baixar GIF", f, file_name="animacao.gif", mime="image/gif")
