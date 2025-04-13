from moviepy.editor import *
from PIL import Image
import numpy as np

image = Image.open("boneco.png")
image_array = np.array(image)

clip = ImageClip(image_array).set_duration(10)
clip.write_videofile("boneco_na_caixa.mp4", fps=24)
