try:
    import unzip_requirements
except ImportError:
    pass

"""import torch
import torchvision
import torchvision.transforms as transforms"""
from PIL import Image

import boto3
import os
import io
import json
import base64
import numpy as np

from requests_toolbelt.multipart import decoder
from wrapper import align_face, face_swap
import traceback

# lambda console log
print('Imports done... - Logger')


# define env variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4-phase2'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'shape_predictor_5_face_landmarks.dat'

# lambda console log
print('Downloaded model - Logger')


s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        # lambda console log
        print('Creating bytesteam - Logger')
        bytesteam = io.BytesIO(obj['Body'].read())

        print('Loading Model - Logger')
        model = torch.jit.load(bytesteam)
        print('Model loaded- Logger')

except Exception as e:
    # lambda console log
    print(repr(e) + ' - Logger')
    raise(e)


def get_response_image(image_bytes):
    byte_arr = io.BytesIO()
    #convert PIL img to byte array
    image_bytes.save(byte_arr, format = 'JPEG')
    #encode as base64
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img


def align_img(event, context):
    try:
        content_type_header = event['headers']['content-type']
        # lambda console log
        print(event['body'][:10] + ' - Logger')
        body = base64.b64decode(event['body'])
        print('Body loader - Logger')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        image_bytes = picture.content

        print("Aligning image now..")

        image = Image.open(io.BytesIO(image_bytes))
        print("Image shape: ", image.size)
        
        np_image = np.array(image)
        align_im = align_face(np_image)
        
        print("Function align_image called..")

        pil_image = Image.fromarray(align_im)

        print("Converted np_image back to image.")

        res_im = get_response_image(pil_image)

        print("Got back response base64 image.")

        message ="Aligned and stored response image."

        return {
            "statusCode": 200,
            "headers" : {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({'Status': 'Success', 'message': message, 'ImageBytes': res_im})
        }

    except Exception as e:
        print(repr(e), '- Logger')
        return {
            "statusCode": 500,
            "headers" : {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({'error': repr(e)})        
        }
        
        
def swap_face(event, context):
    try:
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event["body"])

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[1]
        
        image_bytes=picture.content
        image_bytes1=picture1.content
        
        print("Starting Face swap...")
        image = Image.open(io.BytesIO(image_bytes))
        image1 = Image.open(io.BytesIO(image_bytes1))
        print("Image Shape:", image.size)
        
        np_image,np_image1,  = np.array(image),np.array(image1)
        swapped_image = face_swap(np_image,np_image1)
        
        pil_image = Image.fromarray(swapped_image)
        res_image = get_response_image(pil_image)
        
        img = get_response_image(image)
        img1 = get_response_image(image1)
        
        message = 'Face swap makes faces funny'
        response = {'Status': 'Success', 'message': message, 'img': img, 'img1': img1, 'swapped_image':res_image}
        # img_bytes = image.tobytes()
        # bytes_encoded = base64.b64encode(img_bytes)
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            # "body": response, "isBase64Encoded":"true"
            "body" : json.dumps(response)
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'multipart/form-data',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
