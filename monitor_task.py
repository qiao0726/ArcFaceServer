import threading
import time
from datetime import datetime
from queue import Queue
import minio
import os
import logging
import cv2
from arcsoft_api import detectAndRecognizeFace
import json
import copy
import ffmpeg
import websocket
import numpy as np
from shutil import rmtree, move


# Minio client
MINIO_CONFIG = {
    'endpoint': '192.168.1.133:9000',
    'access_key': 'minioadmin',
    'secret_key': 'minioadmin',
    'secure': False
}
client = minio.Minio(**MINIO_CONFIG)
QUEUE_SIZE = 10
GAP = 10 # Decode one frame every GAP frames
THRESHOLD = 0 # Face with similarity lower than this will not be put into the result queue
websocket_url = r"ws://192.168.1.196:8889/face/webSocketServer"

class MonitorTask:
    def __init__(self, rtsp_addr, id, lib_id, lib_name,
                 temp_data_path = r"/home/qn/ARcFaceJava/tempdata") -> None:
        self.rtsp_addr = rtsp_addr
        self.id = id
        self.lib_id = lib_id
        self.lib_name = lib_name
        self.thread_list = (threading.Thread(target=self.decodeFrames), # decode_thread
                            threading.Thread(target=self.detectAndSaveFaces), # detect_thread
                            threading.Thread(target=self.produceResult), # result_thread
                            threading.Thread(target=self.sendResult)) # send_thread
        
        self.queue_dict = {
            #"decoded_frames": Queue(maxsize=QUEUE_SIZE), # Frame decoding thread decodes frames from rtsp stream, save the minio img name to this queue
            "decoded_frameImg_num": Queue(maxsize=QUEUE_SIZE), # Store num of each decoded frame
            "detected_faces": Queue(maxsize=QUEUE_SIZE), # Store minio img name of each detected face
            "result": Queue(maxsize=QUEUE_SIZE)
        }
        self.search_result = dict() # Dict for search result for each face, key: minio_name for each face, value: dict{"sim": sim, "name": name}
        
        self.temp_data_path = temp_data_path # Folder to save decoded frame images and detected face images
        self.now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        
        self.is_stopped = 0 # Set to 1 when this task is stopped
        self.is_paused = 0 # Set to 1 when this task is paused
        
        # Logger
        self.logger = logging.getLogger('decode-send')
        self.logger.setLevel(level=logging.DEBUG)
        file_name = f"{self.id}.log"
        handler = logging.FileHandler(filename=file_name, encoding='utf-8', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def __del__(self):
        # Delete temp data
        local_img_path = os.path.join(self.temp_data_path, self.now_time)
        if os.path.exists(local_img_path):
            rmtree(local_img_path)
        print(f"Task {self.id} has been deleted")
        return
            
    def initThread(self):
        # Start task
        for t in self.thread_list:
            t.start()
        return


    def start(self):
        self.is_stopped = 0
        self.is_paused = 0
        print(f"Task {self.id} has been started")
        return

    def stop(self):
        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
        self.is_stopped = 1
        for t in self.thread_list:
            t.join()
        print(f"Task {self.id} has been stopped")
        return
        
    def pause(self):
        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
        self.is_paused = 1
        print(f"Task {self.id} has been paused")
        return
        
    def decodeFrames(self):
        """ Get the RTSP stream from openCV's VideoCapture, save the decoded frame image to local file,
        upload it to minio, then put the local file path to the decoded_frames queue
        """
        num = 0 # number of frames
        #-------------------------RTSP Stream------------------
        args = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay"
        }
        # Build new stream
        try:
            probe = ffmpeg.probe(self.rtsp_addr)
        except ffmpeg._run.Error:
            print('error')
            return
        cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
        width = cap_info['width']           # 获取视频流的宽度
        height = cap_info['height']         # 获取视频流的高度
        process1 = (
            ffmpeg
            .input(self.rtsp_addr, **args)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )
        #------------------------------------------------------
        
        # Start decoding
        while True:
            # if self.is_paused != 0:
            #     # Empty all queues
            #     for key in self.queue_dict.keys():
            #         q = self.queue_dict[key]
            #         while not q.empty():
            #             q.get()
            #             q.task_done()
            #     continue
            in_bytes = process1.stdout.read(width * height * 3)     # Read one frame
            num += 1
            if not in_bytes:
                continue
            
            if (num % GAP != 0) or (self.queue_dict['decoded_frameImg_num'].full()):
                continue
            # To ndarray
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            #in_frame = cv2.resize(in_frame, (1280, 720))   # 改变图片尺寸
            frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # To BGR
            
            # Save this frame to local file and upload to minio
            self.saveFrame(frame=frame, frame_num=num)
            
            self.logger.info(f"Frame{num} decoded.")
            # Put the frame number to queue
            self.queue_dict['decoded_frameImg_num'].put(num)
            #logger.info(f"queue_dict['decoded_frameImg_num'].put(num={num})")
            self.logAllQueue(title="decodeFrames")
            
            
        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
    
    
    def detectAndSaveFaces(self):
        """Consume decoded frame imgs in queue_dict['decoded_frameImg_num'],
        produce minio image name to queue_dict['detected_faces']
        """
        while True:
            # if self.is_paused != 0:
            #     # Empty all queues
            #     for key in self.queue_dict.keys():
            #         q = self.queue_dict[key]
            #         while not q.empty():
            #             q.get()
            #             q.task_done()
            #     continue
            num = self.queue_dict['decoded_frameImg_num'].get()
            self.queue_dict['decoded_frameImg_num'].task_done()
            
            self.logger.info(f"detectAndSaveFaces: queue_dict['decoded_frameImg_num'].get: {str(num)}")
            self.logAllQueue(title="detectAndSaveFaces1")
            
            result = self.detectAndSaveFacesForOneFrame(frame_num=num)
            self.logger.info(result)
            self.logAllQueue(title="detectAndSaveFaces2")

        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
    
    
    def produceResult(self):
        """Produce return result for websocket, put it into queue_dict['result']
        """
        while True:
            # if self.is_paused != 0:
            #     # Empty all queues
            #     for key in self.queue_dict.keys():
            #         q = self.queue_dict[key]
            #         while not q.empty():
            #             q.get()
            #             q.task_done()
            #     continue
            minio_face_name = self.queue_dict['detected_faces'].get()
            self.logger.info("produce_result: queue_dict['detected_faces'].get:" + minio_face_name)
            self.queue_dict['detected_faces'].task_done()
            
            result = self.produce_one_result(minio_face_name)
            self.logger.info(result)
            self.logAllQueue(title="produceResult")

        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
            
    def sendResult(self):
        """ Get result from queue_dict['result']
        """
        ws = websocket.WebSocket()
        ws.connect(websocket_url)
        self.logger.info("WebSocket connection success")
        while True:
            try:
                while True:
                    # if self.is_paused != 0:
                    #     # Empty all queues
                    #     for key in self.queue_dict.keys():
                    #         q = self.queue_dict[key]
                    #         while not q.empty():
                    #             q.get()
                    #             q.task_done()
                    #     continue
                    
                    mes = self.queue_dict["result"].get()
                    self.queue_dict["result"].task_done()
                    self.logger.info(f"GET FROM RESULT QUEUE: {mes}")
                    
                    
                    if self.is_paused == 0:
                        ws.send(json.dumps(mes))
                        self.logger.info(f"SEND: {mes}")
            except websocket.WebSocketException as e:
                pass
            except Exception as e:
                pass
            finally:
                if not ws.connected:
                    ws = websocket.WebSocket()
                    ws.connect(websocket_url)
                    self.logger.info(f"Reconnected")

        # Empty all queues
        # for key in self.queue_dict.keys():
        #     q = self.queue_dict[key]
        #     while not q.empty():
        #         q.get()
        #         q.task_done()
            
            
#----------------------------------------------UTILS FUNCTIONS--------------------------------------------------------
    def logAllQueue(self, title=None):
            qsize2 = self.queue_dict["detected_faces"].qsize()
            qsize3 = self.queue_dict["decoded_frameImg_num"].qsize()
            qsize4 = self.queue_dict["result"].qsize()
            if title is not None:
                log_info = title + f": Current queue size: detected_faces:{qsize2}, decoded_frameImg_num:{qsize3}, result:{qsize4}" 
            else:
                log_info = f"Current queue size: detected_faces:{qsize2}, decoded_frameImg_num:{qsize3}, result:{qsize4}"
            self.logger.info(log_info)
            

    def saveFrame(self, frame, frame_num):
        """Save one frame to local file, then upload it to minio

        Args:
            frame (_type_): _description_
            frame_num (int): Frame number which is about to be saved
        """
        # Path of this image frame on the local system
        local_img_path = os.path.join(self.temp_data_path, self.now_time)
        if not os.path.exists(local_img_path):
            os.mkdir(local_img_path)
        local_img_path = os.path.join(local_img_path, "decodeFrames")
        if not os.path.exists(local_img_path):
            os.mkdir(local_img_path)
        local_img_path = os.path.join(local_img_path, str(frame_num) + '.bmp')
        # Write the decoded frame to local file
        cv2.imwrite(local_img_path, frame) 
        
        # Upload to minio
        minio_img_name = "decodeFrames_" + self.now_time + "_" + str(frame_num) + '.bmp'
        client.fput_object(bucket_name='qntest', 
                            object_name=minio_img_name, 
                            file_path=local_img_path,
                            content_type='application/picture')
        
    def detectAndSaveFacesForOneFrame(self, frame_num):
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
        frame_img_local_path = os.path.join(self.temp_data_path, self.now_time, "decodeFrames", str(frame_num) + '.bmp')
        # Detect with ArcSoft api
        json_str = detectAndRecognizeFace(img_path=frame_img_local_path, lib_id=self.lib_id, is_minio=False)
        
        # Detect error or no face in this frame
        if (json_str == "-1") or (json_str == r"{}") or ("Error" in json_str) or (json_str == "-2"):
        # "-1" for no faces detected, 
        # "-2" for wrong library id, 
        # "{}" for only out-of-bound faces detected
            return f"ERROR: detectAndSaveFacesForOneFrame({frame_num}), returns {json_str}"
            
        data_map = json.loads(json_str)
        img = cv2.imread(frame_img_local_path)

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
            this_face_local_path = os.path.join(self.temp_data_path, self.now_time, "detectFaces")
            if not os.path.exists(this_face_local_path):
                os.mkdir(this_face_local_path)
            this_face_local_path = os.path.join(this_face_local_path, str(frame_num) + '_' + str(i) + '.bmp')
            cv2.imwrite(this_face_local_path, img_crop)
            
            # Upload to minio
            minio_img_name = "detectFaces_" + self.now_time + "_" + str(frame_num) + '_' + str(i) + '.bmp'
            client.fput_object("qntest", minio_img_name, this_face_local_path)
            if not self.queue_dict['detected_faces'].full():
                # Put this face image name to queue
                self.queue_dict['detected_faces'].put(minio_img_name)
                # Save search result for this face
                self.search_result[minio_img_name] = {
                    "sim": max_sim,
                    "name": max_name
                }
            
            
        result = f"Detect and save {len(data_map.keys())} faces from frame {frame_num}"
        return result


    def produce_one_result(self, minio_face_name:str):
        """Take the minio image name for a face, generate the result json string for this face

        Args:
            minio_face_name (str): Face image name on minio

        Returns:
            _type_: _description_
        """
        num = minio_face_name.split('_')[2]
        decoded_frame = "decodeFrames_" + self.now_time + "_" + str(num) + '.bmp'
        
        # Generate monio url
        face_url = client.presigned_get_object("qntest", minio_face_name)
        frame_url = client.presigned_get_object("qntest", decoded_frame)
        
        now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        
        dict_data = {'cphoto': None, # minio_url for detected face image
                    'jzphoto': None, # minio_url for decoded frame image
                    'time': None}
        dict_result = {'data': None, 'code': '200', 'message': 'okplus'}
        cut_image = []
        dict_data['cphoto'] = face_url
        dict_data['jzphoto'] = frame_url
        dict_data['time'] = now
        
        # To which rtsp source
        dict_data['rtspUrl'] = self.rtsp_addr
        dict_data['cameraId'] = self.id
        
        # max_sim, max_name = findNearest(img_path=face_url, lib_id="10k", minio=True)
        max_name = self.search_result[minio_face_name]["name"]
        max_sim = self.search_result[minio_face_name]["sim"]
        dict_data['xsd'] = max_sim
        dict_data["name"] = max_name
        dict_data['libName'] = self.lib_name
        cut_image.append(dict_data)
        cut_image = copy.deepcopy(cut_image)
        
        dict_result['data'] = cut_image
        if isinstance(max_sim, float) and max_sim >= 0:
            self.queue_dict['result'].put(dict_result)
            return f"produce_result: queue_dict['result'].put: {str(dict_result)}"
        else:
            return f"No result saved for {minio_face_name}"
    