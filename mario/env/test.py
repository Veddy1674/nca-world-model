from PIL import Image
import numpy as np

COLOR_MAP = "mario/env/color_map.npz" # output

c = np.load(COLOR_MAP)

# i = 30
# print(c["colors"][i].tolist())
# Image.fromarray(np.array(c["sprites"][i])).show()

# create a image with all the sprites and show
img = Image.new("RGB", (16 * 36, 16))
for i in range(36):
    img.paste(Image.fromarray(np.array(c["sprites"][i])), (16 * i, 0))
img.show()
