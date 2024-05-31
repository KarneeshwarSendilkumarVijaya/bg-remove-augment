import os
import uuid
import engine
import pandas as pd
import requests

from PIL import Image
from io import BytesIO

UPLOAD_FOLDER = "images-input"
OUTPUT_FOLDER = "images-output"
input_type = '.jpg'
output_type = '.png'

lot_image_file = "../blu_car_lot_image.csv"

input_path = "../webapp/images-input/"


def do_processing(row):
    lot = row['lot_nbr']
    dict = {
        'fp': row['fp'],
        'fd': row['fd'],
        'rd': row['rd'],
        'rp': row['rp'],
        'ff': row['ff'],
        'rr': row['rr']
    }

    for item in dict.keys():
        side = item
        url = dict[item]

        file_name = f'{lot}_{side}'

        ext = url.split(".")[-1]
        complete_file_name = f"{OUTPUT_FOLDER}/{file_name}_1_original.{ext}"
        try:
            response = requests.get(url)
        except Exception as e:
            print(f'Error in {lot} - {e}')
            return

        input_image = Image.open(BytesIO(response.content))
        if ext != 'jpg':
            input_image = input_image.convert("RGB")
            complete_file_name = f"{OUTPUT_FOLDER}/{file_name}.jpg"

        input_image.save(complete_file_name)

        bg_blur_50 = engine.blur_bg(input_image, threshold=50)
        bg_blur_100 = engine.blur_bg(input_image, threshold=100)
        bg_removed = engine.remove_bg(input_image)

        bg_blur_50.save(f'{OUTPUT_FOLDER}/{file_name}_3_low_blur{input_type}')
        bg_blur_100.save(f'{OUTPUT_FOLDER}/{file_name}_4_high_blur{input_type}')
        bg_removed.save(f'{OUTPUT_FOLDER}/{file_name}_2_bg{output_type}')


for file in os.listdir(input_path):
    file_path = os.path.join(input_path, file)
    if os.path.isfile(file_path) and file.endswith(input_type):
        try:
            img = Image.open(os.path.join(input_path, file))
        except Exception as e:
            print(e)
            exit(0)

        name = file.split(".")[0]
        ext = input_path.split(".")[-1]

        # response = requests.get(url)
        # input_image = Image.open(BytesIO(response.content))
        bg_blur_50 = engine.blur_bg(img, threshold=50)
        bg_blur_100 = engine.blur_bg(img, threshold=100)
        bg_removed = engine.remove_bg(img)

        output_image = os.path.join(OUTPUT_FOLDER, f'{name}_bg{output_type}')
        bg_removed.save(output_image)

        output_image = os.path.join(OUTPUT_FOLDER, f'{name}_low_blur{input_type}')
        bg_blur_50.save(output_image)

        output_image = os.path.join(OUTPUT_FOLDER, f'{name}_high_blur{input_type}')
        bg_blur_100.save(output_image)


# sample = pd.read_csv(lot_image_file)
# sample = sample.head(30)
#
# sample.apply(lambda row: do_processing(row.to_dict()), axis=1)
