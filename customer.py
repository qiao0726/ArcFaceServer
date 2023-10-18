import argparse
import threading
import io
import time
import urllib
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from flask import Flask, request

from hr_service import hr_product, hr_cutimage, recongnition, hr_web
from service import product, get_bytes, customer11, cutimage, web

app = Flask(__name__)
models = {}


# @app.route('/img/<model>', methods=['POST'])
# def predict(model):
#     if request.method != 'POST':
#         return
#
#     if request.files.get('image'):
#         # Method 2
#         im_file = request.files['image']
#         im_bytes = im_file.read()
#         im = Image.open(io.BytesIO(im_bytes))
#
#         if model in models:
#             results = models[model](im, size=640)  # reduce size=320 for faster inference
#             return results.pandas().xywh[0].to_json(orient='records')


# @app.route('/message/<model>', methods=['POST', 'GET'])
# def boolean(model):
#     code, message = None, None
#     path = request.args.get('url')
#     res = urllib.request.urlopen(path)
#     img = np.asarray(bytearray(res.read()), dtype="uint8")
#     img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#     success, encoded_image = cv2.imencode(".jpg", img)
#     # 将数组转为bytes
#     byte_data = encoded_image.tobytes()
#     im = Image.open(io.BytesIO(byte_data))
#     if model in models:
#         box_and_point = models[model](im, size=640)  # reduce size=320 for faster inference
#     box_and_point = box_and_point.pandas().xywh[0].to_json(orient='records')
#     df = pd.read_json(box_and_point)
#     dict = df.to_dict(orient='index')
#     list = []
#     if box_and_point != {} and len(dict) >= 1:
#         code = 200
#         message = '检测到人脸'
#         for i in dict.keys():
#             res = {"faceRect": {'x': None, 'y': None, 'w': None, 'h': None}, 'url': path}
#             res['faceRect']['x'] = int(dict[i]['xcenter'])
#             res['faceRect']['y'] = int(dict[i]['ycenter'])
#             res['faceRect']['w'] = int(dict[i]['width'])
#             res['faceRect']['h'] = int(dict[i]['height'])
#             list.append(res)
#     else:
#         code = 500
#         message = '未检测到人脸'
#     return {
#         "code": code,
#         "message": message,
#         "data": list
#     }


@app.route('/start', methods=['POST'])
def start_test():
    list1 = []
    production = threading.Thread(target=product)
    list1.append(production)
    consume = threading.Thread(target=get_bytes)
    list1.append(consume)
    consume11 = threading.Thread(target=customer11)
    list1.append(consume11)
    post_image = threading.Thread(target=cutimage)
    list1.append(post_image)
    open = threading.Thread(target=web)
    list1.append(open)
    for task in list1:
        task.start()
    for task in list1:
        task.join()


@app.route('/hr', methods=['POST'])
def hr():
    list2 = []
    hr_start = threading.Thread(target=hr_product)
    list2.append(hr_start)
    recon = threading.Thread(target=recongnition)
    list2.append(recon)
    get_cut = threading.Thread(target=hr_cutimage)
    list2.append(get_cut)
    send = threading.Thread(target=hr_web)
    list2.append(send)
    for i in list2:
        i.start()
    for i in list2:
        i.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['mask'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    # opt = parser.parse_args()
    #
    # for m in opt.model:
    #     models[m] = torch.hub.load('D:/pycharmcode/yolov5-flask', m, source='local', force_reload=True,
    #                                skip_validation=True)
    opt = parser.parse_args()
    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat