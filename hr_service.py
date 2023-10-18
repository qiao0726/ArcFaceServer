# -*- coding: utf-8 -*-

import argparse
import copy
import io
import json
import os
import queue
import threading
import time
import urllib
import cv2
import jpype
import numpy as np
import logging
import pandas as pd
import torch
import websocket
from PIL import Image
from minio import Minio
from datetime import datetime

models = {}

logger = logging.getLogger('customer-product')
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler(filename='hrtest.log', encoding='utf-8', mode='w')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

inputUrl = 'rtsp://admin:123uestc@192.168.1.32:554/h264'
# 存放图片地址
PATH = 'C:/Users/10095/Desktop/img'
# 裁剪图片本地存放地址
save_path = 'C:/Users/10095/Desktop/cut'
# 虹软jar包
jar_path = 'E:\\javacode\\arcface\\arcfacetest.jar'
# 默认JVM路径
jvmPath = jpype.getDefaultJVMPath()
jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % jar_path)

client = Minio(
    endpoint='192.168.1.133:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)
jz_image = queue.Queue(maxsize=2)
face_image = queue.Queue(maxsize=2)
get_result = queue.Queue(maxsize=2)
image_num = queue.Queue(maxsize=2)
put_url = queue.Queue(maxsize=2)


def hr_product():
    num = 0
    gap = 5
    cap = cv2.VideoCapture(inputUrl)
    if not cap.isOpened():
        print('Cap is Not Open')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            num += 1
            if num % gap == 0 and not jz_image.full():
                now_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                jz_name = PATH + '/' + str(num) + '.jpg'
                image_num.put(num)
                img_name = 'jz' + '/' + datetime.now().strftime('%Y-%m-%d') + '-' + now_time + '-' + str(num) + '.jpg'
                cv2.imwrite(jz_name, frame) # Write the decoded frame to a path
                logger.info("Get JZ-image:" + jz_name)
                client.fput_object('xhtest', img_name, jz_name)
                # 解帧获得
                jz_image.put(img_name)
                logger.info("Queue Get:" + img_name)


def recongnition():
    while True:
        now_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        num = image_num.get()
        image_num.task_done()
        jz_name = PATH + '/' + str(num) + '.jpg'
        javaClass = jpype.JClass("FaceTestMd")
        result = javaClass.detectFace(jz_name)
        logger.info("HR GET:" + str(result))
        result1 = str(result)
        result1 = pd.read_json(result1)
        result1 = result1.to_dict(orient='index')
        for i in result1.keys():
            x1 = result1[i]['rect']['left']
            x2 = result1[i]['rect']['right']
            y1 = result1[i]['rect']['top']
            y2 = result1[i]['rect']['bottom']
            image = cv2.imread(jz_name)
            img_roi = image[y1:y2, x1:x2]
            cv2.imwrite(save_path + '/' + str(num) + '-' + str(i) + '.jpg', img_roi)
            cut_name = str(num) + '-' + str(i) + '-' + datetime.now().strftime('%Y-%m-%d') + '-' + now_time + '.jpg'
            client.fput_object("xhtest", cut_name, save_path + '/' + str(num) + '-' + str(i) + '.jpg')
            logger.info("Minio GET:" + cut_name)
            if not face_image.full():
                face_image.put(cut_name)
        logger.info("frame: %d , box_num: %d" % (num, len(result1.keys())))


def hr_cutimage():
    while True:
        dict1 = {'cphoto': None, 'time': None, 'jzphoto': None}
        dict2 = {'data': None, 'code': '200', 'message': 'okpro'}
        cut_image = []
        now_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        times = str(now_time)
        name = face_image.get()
        face_image.task_done()
        jz_names = jz_image.get()
        jz_image.task_done()
        url1 = client.presigned_get_object("xhtest", name)
        # 截取人脸url
        put_url.put(url1)
        url2 = client.presigned_get_object("xhtest", jz_names)
        dict1['cphoto'] = url1
        dict1['jzphoto'] = url2
        dict1['time'] = times
        cut_image.append(dict1)
        cut_image = copy.deepcopy(cut_image)
        dict2['data'] = cut_image
        logger.info("Get-Cut-Mes:" + str(dict2))
        # get_result.put(dict2)


def hr_web():
    while True:
        def on_open(wsapp):
            print("on_open")
            while True:
                mes = get_result.get()
                get_result.task_done()
                logger.info("WebSocket Get:" + str(mes))
                time.sleep(0.5)
                wsapp.send(json.dumps(mes))
                logger.info("SEND: " + str(json.dumps(mes)))

        wsapp = websocket.WebSocketApp("ws://192.168.3.137:8889/face/webSocketServer",
                                       on_open=on_open
                                       # on_message=on_message
                                       )
        wsapp.run_forever()
