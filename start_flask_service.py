import arcsoft_api
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from datetime import datetime
from pprint import pprint
import json
import atexit
import os
import signal
from json import JSONDecodeError
import sys
import threading
from websocket_api import rtsp_url, rtsp_changed
from minio_api import pull_img_from_minio
import monitor_task
import concurrent.futures
from pprint import pprint

# Flask web service
app = Flask(__name__)
CORS(app, resources=r'/*')
HOST = r"0.0.0.0"
PORT = 5001
temp_data_path = r"/home/qn/ARcFaceJava/tempdata"
monitor_task_list = dict()
executor = concurrent.futures.ThreadPoolExecutor()

# 100k no mask lib: 1691083772054
# 100k mask lib: 1691081136849

@app.route('/dkha/face', methods=['POST'])
def add_face():
    try:
        receive_dict = json.loads(request.data)
    except JSONDecodeError:
        return jsonify({
            "code": 500,
            "message": "错误的请求参数"
        })
    lib_id=str(receive_dict.get("libraryId"))
    response_code = arcsoft_api.addFace(lib_id=lib_id,
                      face_id=receive_dict.get("faceId"),
                      img_path=receive_dict.get("url"))
    
    message_dict = {
        200: "添加成功",
        404: "人脸库id错误",
        -2: "人脸id已存在",
        -1: "图片url错误",
        2: "图片须包含一张人脸",
        5: "需使用不戴口罩人脸"
    }
    message = message_dict[response_code] if response_code in message_dict.keys() else "其他错误"
    code = 200 if response_code == 200 else 500
    
    respond = {
        "code": code,
        "message": message
    }
    pprint(respond)
    
    return jsonify(respond)


@app.route('/dkha/face/<libraryId>/<idPortrait>', methods=['DELETE'])
def del_face(libraryId, idPortrait):
    lib_id = libraryId
    code = arcsoft_api.delFace(lib_id=lib_id, face_id=idPortrait)
    print(code)
    message = "删除成功" if code == 200 else "错误"
    respond = {
        "code": code,
        "message": message
    }
    return jsonify(respond)


@app.route('/dkha/faceLib', methods=['POST'])
def add_faceLib():
    # Recieve data
    try:
        receive_dict = json.loads(request.data)
    except JSONDecodeError:
        return "No"
    code = arcsoft_api.addFaceLib(lib_id=receive_dict.get("libraryId"), 
                                  lib_name=receive_dict.get("name"))
    message = "添加成功" if code == 200 else "重复id"
    respond = {
        "code": code,
        "message": message
    }
    return jsonify(respond)


@app.route('/dkha/faceLib/<libraryId>', methods=['DELETE'])
def del_faceLib(libraryId):
    code = arcsoft_api.delFaceLib(lib_id=libraryId)
    message = "删除成功" if code == 200 else "错误id"
    respond = {
        "code": code,
        "message": message
    }
    return jsonify(respond)
    
@app.route('/compare', methods=['POST'])
def face_compare():
    # Recieve data
    try:
        receive_dict = json.loads(request.data)
    except JSONDecodeError:
        return jsonify({
            'code': 500,
            'message': "Error"
        })
    print("Recieved")
    img_path1 = receive_dict.get("image1")
    img_path2 = receive_dict.get("image2")
    print("Got")
    sim = arcsoft_api.compare(img_path1=img_path1, img_path2=img_path2, is_minio=True)
    sim = sim + 15 if sim > 55 and sim < 75 else sim
    print("Comparison finished")
    if sim != -1:
        return jsonify({
            'code': 200,
            'data': {
                "hitSimilarity": sim
            },
            'message': "Face comparison"
        })
    else:
        return jsonify({
            'code': 500,
            'message': "Error"
        })
        

# New page to recognize a face in a specific library     
@app.route('/compares', methods=['POST'])
def face_recognize():
    # Recieve data
    try:
        receive_dict = json.loads(request.data)
    except JSONDecodeError:
        return jsonify({
            'code': 500,
            'message': "Error"
        })
    img_path = receive_dict.get("image1")
    lib_id = receive_dict.get("extraMeta")
    max_sim, max_name, max_minio_url = arcsoft_api.findNearest(img_path=img_path, lib_id=lib_id, is_minio=True)
    
    if max_sim < 0:
        err_msg = f'ERROR: returns max_sim={max_sim}, max_name={max_name}'
        print(err_msg)
        return jsonify({
                'code': 500,
                'message': err_msg
                })
    
    max_sim = max_sim + 15 if max_sim > 55 and max_sim < 75 else max_sim
    return jsonify({
        'code': 200,
        'data': {
            "hitSimilarity": (max_sim/100.0),
            'rphoto': max_minio_url,
        },
        'message': "Face comparison"
    })
    
        

@app.route('/message/<model>', methods=['POST', 'GET'])
def detectFace(model):
    # Get input img
    img_minio_path = str(request.args.get('url'))
    img_name = img_minio_path.split(r'/')[-1]
    folder_path = os.path.join(temp_data_path, "detectFace")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    local_img_path = os.path.join(folder_path, img_name)
    if pull_img_from_minio(minio_url=img_minio_path, local_save_path=local_img_path) != 1:
        return {"code": 500,
                "message": "Error",
                "data": []}
    
    # Detect faces
    json_str = arcsoft_api.detectFace(img_path=local_img_path, is_minio=False)
    # Detect error or no face in this image
    if (json_str == "-1") or (json_str == r"{}") or ("Error" in json_str) or (json_str == "-2"):
    # "-1" for no faces detected, 
    # "-2" for wrong library id, 
    # "{}" for only out-of-bound faces detected
        return {"code": 500,
                "message": "Error",
                "data": []}
    
    data_map = json.loads(json_str)
    list = []
    
    for i in data_map.keys():
        x1 = int(data_map[i]['left'])
        x2 = int(data_map[i]['right'])
        y1 = int(data_map[i]['top'])
        y2 = int(data_map[i]['bottom'])
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        width = x2 -x1
        height = y2 - y1
        
        res = {"faceRect": {'x': None, 'y': None, 'w': None, 'h': None}, 'url': img_minio_path}
        res['faceRect']['x'] = x_center
        res['faceRect']['y'] = y_center
        res['faceRect']['w'] = width
        res['faceRect']['h'] = height
        list.append(res)

    return {
        "code": 200,
        "message": '检测到人脸',
        "data": list
    }
    
#------------------------------------------------------------------------------------------------------
def addTaskObject(rtsp_url, task_id, lib_id, lib_name):
    global monitor_task_list
    
    new_task = monitor_task.MonitorTask(rtsp_addr=rtsp_url, id=task_id, lib_id=lib_id, lib_name=lib_name)

    monitor_task_list[task_id] = new_task
    new_task.initThread()
    return

@app.route('/addTask/', methods=['GET'])
def addTask():
    global monitor_task_list
    task_id = str(request.args.get('strtaskno'))
    lib_id = str(request.args.get('libId'))
    lib_name = str(request.args.get('libName'))
    if task_id in monitor_task_list.keys():
        pprint(monitor_task_list.keys())
        return jsonify({
            'code': 500,
            'message': "Wrong id"
        })

    rtsp_url = str(request.args.get('rtspurl'))
    executor.submit(addTaskObject, rtsp_url, task_id, lib_id, lib_name)
    pprint(monitor_task_list.keys())
    return jsonify({
        "code": 200,
        "message": "Success"
    })
    
    
@app.route('/startTask/', methods=['GET'])
def startTask():
    global monitor_task_list
    task_id = str(request.args.get('strtaskno'))
    
    if not task_id in monitor_task_list.keys():
        pprint(monitor_task_list.keys())
        return jsonify({
            'code': 500,
            'message': "Wrong id"
        })
    monitor_task_list[task_id].start()
    pprint(monitor_task_list.keys())
    return jsonify({
        "code": 200,
        "message": "Success"
    })
    
    
def delTaskObject(task_id):
    global monitor_task_list
    monitor_task_list[task_id].stop()
    del monitor_task_list[task_id]
    return
    
@app.route('/delTask/', methods=['GET'])        
def delTask():
    global monitor_task_list
    task_id = str(request.args.get('strtaskno'))
    
    if not task_id in monitor_task_list.keys():
        pprint(monitor_task_list.keys())
        return jsonify({
            'code': 200,
            'message': "Wrong id"
        })
    executor.submit(delTaskObject, task_id)
    pprint(monitor_task_list.keys())
    return jsonify({
        "code": 200,
        "message": "Success"
    })

@app.route('/stopTask/', methods=['GET'])        
def pauseTask():
    global monitor_task_list
    task_id = str(request.args.get('strtaskno'))
    if task_id in monitor_task_list.keys():
        monitor_task_list[task_id].pause()
        pprint(monitor_task_list.keys())
        return jsonify({
            "code": 200,
            "message": "Success"
        })
    else:
        pprint(monitor_task_list.keys())
        return jsonify({
            'code': 500,
            'message': "Wrong id"
        })
    
    

def cleanup(sig, frame):
    arcsoft_api.shutdown()
    print("Stopping Flask app.")
    sys.exit(1)


if __name__ == "__main__":
    arcsoft_api.init()
    #atexit.register(cleanup)
    # Catch ctrl+c signal and call cleanup()
    signal.signal(signal.SIGINT, cleanup)
    from websocket_api import *
    # Print all face libs
    pprint(arcsoft_api.getAllLibs())
    #start_websocket()
    app.run(host=HOST, port=PORT)
    