import websocket
import os
from queue import Queue
import minio
import cv2
from datetime import datetime
from arcsoft_api import findNearest, detectAndRecognizeFace
import json
import time
import copy
import logging
import ffmpeg
import numpy as np
from multiprocessing import Manager


logger = logging.getLogger('customer-product')
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler(filename='hrtest.log', encoding='utf-8', mode='w')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger2 = logging.getLogger('decode-send')
logger2.setLevel(level=logging.DEBUG)
handler2 = logging.FileHandler(filename='decode_send.log', encoding='utf-8', mode='w')
handler2.setLevel(logging.DEBUG)
formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler2.setFormatter(formatter2)
logger2.addHandler(handler2)



websocket_url = r"ws://192.168.1.196:8889/face/webSocketServer"
temp_data_path = r"/home/qn/ARcFaceJava/tempdata"
now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

manager = Manager()
#rtsp_url = manager.Value(str, r'rtsp://192.168.1.138:8557/h264')
# rtsp://admin:123uestc@192.168.1.32:554/h264/ch1/main/av_stream
rtsp_url = manager.Value(str, r'NULL')
rtsp_changed = manager.Value(int, 0)

QUEUE_SIZE = 5
GAP = 10 # Decode one frame every GAP frames

THRESHOLD = 0 # Face with similarity lower than this will not be put into the result queue

queue_dict = {
    #"decoded_frames": Queue(maxsize=QUEUE_SIZE), # Frame decoding thread decodes frames from rtsp stream, save the minio img name to this queue
    "decoded_frameImg_num": Queue(maxsize=QUEUE_SIZE), # Store num of each decoded frame
    "detected_faces": Queue(maxsize=QUEUE_SIZE), # Store minio img name of each detected face
    "result": Queue(maxsize=QUEUE_SIZE)
}

search_result = dict() # Dict for search result for each face, key: minio_name for each face, value: dict{"sim": sim, "name": name}


# Minio client
MINIO_CONFIG = {
    'endpoint': '192.168.1.133:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin',
    'secure': False
}
client = minio.Minio(**MINIO_CONFIG)


def decodeFrames():
    """ Get the RTSP stream from openCV's VideoCapture, save the decoded frame image to local file,
    upload it to minio, then put the local file path to the decoded_frames queue
    """
    global queue_dict
    num = 0 # number of frames
    
    #-------------------------RTSP Stream------------------
    args = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "flags": "low_delay"
    }
    # probe = ffmpeg.probe(rtsp_url.value)
    # cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
    width = None           # 获取视频流的宽度
    height = None         # 获取视频流的高度
    process1 = None
    #------------------------------------------------------
    
    while True:
        #----------------Change RTSP Stream------------------
        if rtsp_changed.value != 0 and rtsp_url.value != r'NULL':
            # Empty all queues
            # for key in queue_dict.keys():
            #     q = queue_dict[key]
            #     while not q.empty():
            #         q.get()
            #         q.task_done()
            
            # Build new stream
            if process1 is not None:
                process1.kill()
            new_rtsp_url = rtsp_url.value
            probe = ffmpeg.probe(new_rtsp_url)
            cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
            width = cap_info['width']           # 获取视频流的宽度
            height = cap_info['height']         # 获取视频流的高度
            process1 = (
                ffmpeg
                .input(new_rtsp_url, **args)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .overwrite_output()
                .run_async(pipe_stdout=True)
            )
            rtsp_changed.value = 0
        #-----------------------------------------------------
        # RTSP url has not been initialized
        if rtsp_url.value == r'NULL' or process1 == None:
            continue
        
        in_bytes = process1.stdout.read(width * height * 3)     # Read one frame
        num += 1
        if not in_bytes:
            continue
        
        if (num % GAP != 0) or (queue_dict['decoded_frameImg_num'].full()):
            continue
        # To ndarray
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        #in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
        frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # To BGR
        
        # Save this frame to local file and upload to minio
        saveFrame(frame=frame, frame_num=num)
        
        logger2.info(f"Frame{num} decoded.")
        # Put the frame number to queue
        queue_dict['decoded_frameImg_num'].put(num)
        logger.info(f"queue_dict['decoded_frameImg_num'].put(num={num})")
        logAllQueue(title="decodeFrames")
                

def detectAndSaveFaces():
    """Consume decoded frame imgs in queue_dict['decoded_frameImg_num'],
    produce minio image name to queue_dict['detected_faces']
    """
    global queue_dict
    while True:
        num = queue_dict['decoded_frameImg_num'].get()
        queue_dict['decoded_frameImg_num'].task_done()
        
        # RTSP has changed, ignore this frame
        if rtsp_changed.value != 0:
            continue
        
        logger.info(f"detectAndSaveFaces: queue_dict['decoded_frameImg_num'].get: {str(num)}")
        logAllQueue(title="detectAndSaveFaces1")
        
        result = detectAndSaveFacesForOneFrame(frame_num=num)
        logger.info(result)
        logAllQueue(title="detectAndSaveFaces2")
        
                

def produceResult():
    """Produce return result for websocket, put it into queue_dict['result']
    """
    global queue_dict
    while True:
        minio_face_name = queue_dict['detected_faces'].get()
        logger.info("produce_result: queue_dict['detected_faces'].get:" + minio_face_name)
        queue_dict['detected_faces'].task_done()
        
        # RTSP has changed, ignore this face
        if rtsp_changed.value != 0:
            continue
        
        result = produce_one_result(minio_face_name)
        logger.info(result)
        logAllQueue(title="produceResult")
        
# def run_websocket():
#     global queue_dict
#     while True:
#         def on_open(wsapp):
#             print("on_open")
#             while True:
#                 mes = queue_dict['result'].get()
#                 queue_dict['result'].task_done()
#                 logger.info(f"queue_dict['result'].get: {str(json.dumps(mes))}")
#                 #time.sleep(0.5)
#                 wsapp.send(json.dumps(mes))
#                 logger.info("SEND: " + str(json.dumps(mes)))

#         wsapp = websocket.WebSocketApp("ws://192.168.1.196:8889/face/webSocketServer",
#                                        on_open=on_open
#                                        # on_message=on_message
#                                        )
#         wsapp.run_forever()

def run_websocket():
    """ Get result from queue_dict['result']
    """
    global queue_dict
    ws = websocket.WebSocket()
    ws.connect(websocket_url)
    logger.info("WebSocket connection success")
    while True:
        try:
            while True:
                mes = queue_dict["result"].get()
                # RTSP has changed, ignore this result
                if rtsp_changed.value != 0:
                    continue
                if mes is None:
                    break  # 结束线程
                try:
                    ws.send(json.dumps(mes))
                    logger.info("SEND: " + str(json.dumps(mes)))
                    
                    face_name = mes['data'][0]['cphoto']
                    if '?' in face_name:
                        face_name = face_name.split('?')[0]
                    face_name = face_name.split('_')[-2] + '_' + face_name.split('_')[-1]
                    logger2.info(f"SEND: {face_name}")
                except websocket.WebSocketException as e:
                    logger.error(f"Send error: {str(json.dumps(mes))}")
        except websocket.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Other error: {e}")
        finally:
            if not ws.connected:
                ws = websocket.WebSocket()
                ws.connect(websocket_url)
                logger.info(f"Reconnected")

#----------------------------------------------UTILS FUNCTIONS--------------------------------------------------------
def logAllQueue(title=None):
        qsize2 = queue_dict["detected_faces"].qsize()
        qsize3 = queue_dict["decoded_frameImg_num"].qsize()
        qsize4 = queue_dict["result"].qsize()
        if title is not None:
            log_info = title + f": Current queue size: detected_faces:{qsize2}, decoded_frameImg_num:{qsize3}, result:{qsize4}" 
        else:
            log_info = f"Current queue size: detected_faces:{qsize2}, decoded_frameImg_num:{qsize3}, result:{qsize4}"
        logger.info(log_info)
        

def saveFrame(frame, frame_num):
    """Save one frame to local file, then upload it to minio

    Args:
        frame (_type_): _description_
        frame_num (int): Frame number which is about to be saved
    """
    # Path of this image frame on the local system
    local_img_path = os.path.join(temp_data_path, now_time)
    if not os.path.exists(local_img_path):
        os.mkdir(local_img_path)
    local_img_path = os.path.join(local_img_path, "decodeFrames")
    if not os.path.exists(local_img_path):
        os.mkdir(local_img_path)
    local_img_path = os.path.join(local_img_path, str(frame_num) + '.bmp')
    # Write the decoded frame to local file
    cv2.imwrite(local_img_path, frame) 
    
    # Upload to minio
    minio_img_name = "decodeFrames_" + now_time + "_" + str(frame_num) + '.bmp'
    client.fput_object(bucket_name='qntest', 
                        object_name=minio_img_name, 
                        file_path=local_img_path,
                        content_type='application/picture')
    
def detectAndSaveFacesForOneFrame(frame_num):
    """ Take a frame number, with this frame number we can locate a local frame image file, 
    then detect and recognize every face in this frame, upload every face to minio seperatly,
    and put the detection and recognition result for each face into the search_result dict,
    use minio image name as key.

    Args:
        frame_num (int): frame number which is already decoded and saved in a local file.

    Returns:
        _type_: _description_
    """
    # Entire decoded frame image
    frame_img_local_path = os.path.join(temp_data_path, now_time, "decodeFrames", str(frame_num) + '.bmp')
    # Detect with ArcSoft api
    # Fixed lib id
    json_str = detectAndRecognizeFace(img_path=frame_img_local_path, lib_id="1691145066922", is_minio=False)
    
    # Detect error or no face in this frame
    if (json_str == "-1") or (json_str == r"{}") or ("Error" in json_str) or (json_str == "-2"):
    # "-1" for no faces detected, 
    # "-2" for wrong library id, 
    # "{}" for only out-of-bound faces detected
        return f"ERROR: detectAndSaveFacesForOneFrame({frame_num}), returns {json_str}"
        
    data_map = json.loads(json_str)
    img = cv2.imread(frame_img_local_path)
    
    global queue_dict
    global search_result
    for i in data_map.keys():
        x1 = int(data_map[i]['left'])
        x2 = int(data_map[i]['right'])
        y1 = int(data_map[i]['top'])
        y2 = int(data_map[i]['bottom'])
        is_masked = int(data_map[i]['mask'])
        max_sim = float(data_map[i]["sim"])
        max_sim = round(max_sim * 100, 2)
        # If sim is lower than THRESHOLD, ignore this face
        if max_sim < THRESHOLD:
            continue
        
        max_name = str(data_map[i]["name"])
        #max_name = "33010519841102311X_4"
        img_crop = img[y1:y2, x1:x2]
        
        # Save this face to local file
        this_face_local_path = os.path.join(temp_data_path, now_time, "detectFaces")
        if not os.path.exists(this_face_local_path):
            os.mkdir(this_face_local_path)
        this_face_local_path = os.path.join(this_face_local_path, str(frame_num) + '_' + str(i) + '.bmp')
        cv2.imwrite(this_face_local_path, img_crop)
        
        # Upload to minio
        minio_img_name = "detectFaces_" + now_time + "_" + str(frame_num) + '_' + str(i) + '.bmp'
        client.fput_object("qntest", minio_img_name, this_face_local_path)
        if not queue_dict['detected_faces'].full():
            # Put this face image name to queue
            queue_dict['detected_faces'].put(minio_img_name)
            # Save search result for this face
            search_result[minio_img_name] = {
                "sim": max_sim,
                "name": max_name
            }
        
        
    result = f"Detect and save {len(data_map.keys())} faces from frame {frame_num}"
    return result


def produce_one_result(minio_face_name):
    """Take the minio image name for a face, generate the result json string for this face

    Args:
        minio_face_name (str): Face image name on minio

    Returns:
        _type_: _description_
    """
    global queue_dict
    num = minio_face_name.split('_')[2]
    decoded_frame = "decodeFrames_" + now_time + "_" + str(num) + '.bmp'
    
    # Generate monio url
    face_url = client.presigned_get_object("qntest", minio_face_name)
    frame_url = client.presigned_get_object("qntest", decoded_frame)
    
    now = str(datetime.now().strftime('%H:%M:%S.%f')[:-3])
    
    dict_data = {'cphoto': None, # minio_url for detected face image
                 'jzphoto': None, # minio_url for decoded frame image
                 'time': None}
    dict_result = {'data': None, 'code': '200', 'message': 'okplus'}
    cut_image = []
    dict_data['cphoto'] = face_url
    dict_data['jzphoto'] = frame_url
    dict_data['time'] = now
    
    # max_sim, max_name = findNearest(img_path=face_url, lib_id="10k", minio=True)
    max_name = search_result[minio_face_name]["name"]
    max_sim = search_result[minio_face_name]["sim"]
    dict_data['xsd'] = max_sim
    dict_data["name"] = max_name
    dict_data['libName'] = '1w人像库_new_5_5'
    cut_image.append(dict_data)
    cut_image = copy.deepcopy(cut_image)
    
    dict_result['data'] = cut_image
    if isinstance(max_sim, float) and max_sim >= 0:
        queue_dict['result'].put(dict_result)
        return f"produce_result: queue_dict['result'].put: {str(dict_result)}"
    else:
        return f"No result saved for {minio_face_name}"