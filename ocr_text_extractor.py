from google.cloud import vision
from google.cloud.vision import types
from google.protobuf.json_format import MessageToDict
import pandas as pd
import numpy as np
import sys
import kyc_config as cfg

client = vision.ImageAnnotatorClient.from_service_account_file(
cfg.gcv_api_key_path
)

def get_text_response_from_path(path):

    output = None

    try:
        if path.startswith('http') or path.startswith('gs:'):
            image = types.Image()
            image.source.image_uri = path
        else:
            with open(path, 'rb') as image_file:
                content = image_file.read()
            image = types.Image(content=content)

    except ValueError:
        output = "Cannot Read Input File"
        return output

    text_response = client.text_detection(image=image)
    text_response = MessageToDict(text_response)
    return text_response

def process_ocr(img_path):
    text_response = get_text_response_from_path(img_path)

    #save the output file
    img_name = img_path.split('/')[-1].split('.')[0]
    json_name = cfg.json_loc+'ocr_'+img_name+'.npy'
    np.save(json_name, text_response)

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        # input: image path
        img_path = sys.argv[1]
        print('OCR processing '+img_path)
        process_ocr(img_path)
    else:
        print('argument is missing: image path')
