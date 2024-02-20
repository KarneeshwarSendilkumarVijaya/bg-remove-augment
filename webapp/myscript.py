import os
import uuid
import engine

from PIL import Image
import requests
from io import BytesIO

UPLOAD_FOLDER = "images-input"
OUTPUT_FOLDER = "images-output"
url = "https://c-static.copart.com/v1/AUTH_svc.pdoc00001/LPP467/20e086540347477ba05f132462258b9d_ful.jpg"

new_name = str(uuid.uuid4()).split("-")[0]
ext = url.split(".")[-1]
file_name = f"{UPLOAD_FOLDER}/{new_name}.{ext}"

response = requests.get(url)
input_image = Image.open(BytesIO(response.content))
img_pil = engine.remove_bg_mult(input_image)

output_image = os.path.join(OUTPUT_FOLDER, f'{new_name}.png')
img_pil.save(output_image)
