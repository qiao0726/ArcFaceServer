import jpype
import os
import glob
import numpy as np
from shutil import rmtree, move
from minio_api import pull_img_from_minio
from json_utils import save_dict_to_json, load_dict_from_json

# Path of the java class file(use current directory here)
java_class_path = os.path.abspath(os.path.dirname(__file__))
java_class_name = r"ArcSoftFaceEngine"

# Folder which includes all JAR files
jar_directory = r"/home/qn/ARcFaceJava/libs"

# Get all JAR file path
jar_files = glob.glob(os.path.join(jar_directory, "*.jar"))

# Concat
classpath = f"{java_class_path}:{':'.join(jar_files)}"

# Path to store all face lib files
database_path = r"/home/qn/ARcFaceJava/database"

dict_libId2engine = dict()
dict_libId2libName = dict()
FaceEngine = None
detect_engine = None

all_libs_file_path = r"/home/qn/ARcFaceJava/database/all_libs.npy"
temp_data_path = r"/home/qn/ARcFaceJava/tempdata"
faceId2minioUrl_file_path = r"/home/qn/ARcFaceJava/database/faceId2minioUrl.json"

dict_faceId2minioUrl = dict()

def init():
    # Start the JVM with the class path
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea",
                   f"-Djava.class.path={classpath}")

    global FaceEngine
    # Load the Java class
    FaceEngine = jpype.JClass(java_class_name)
    
    # Init detect engine
    global detect_engine
    detect_engine = FaceEngine("detect_only", "detect_only", 0)
    
    # Init dict_libId2libName
    global dict_libId2libName
    if not os.path.exists(all_libs_file_path):
        dict_libId2libName = dict()
        np.save(all_libs_file_path, dict_libId2libName)
    else:
        dict_libId2libName = np.load(all_libs_file_path, allow_pickle=True).item()
    
    # Init dict_libId2engine
    global dict_libId2engine
    for lib_id in dict_libId2libName.keys():
        lib_name = dict_libId2libName[lib_id]
        dict_libId2engine[lib_id] = FaceEngine(lib_id, lib_name, 1)
    
    # Init dict_faceId2minioUrl
    global dict_faceId2minioUrl
    dict_faceId2minioUrl = load_dict_from_json(faceId2minioUrl_file_path)
    
    return


def shutdown():
    global dict_libId2libName
    global dict_libId2engine
    # Save dict_libId2libName
    np.save(all_libs_file_path, dict_libId2libName)
    
    # Save all engines' features
    for lib_id in dict_libId2libName.keys():
        this_engine_path = os.path.join(database_path, dict_libId2libName[lib_id])
        dict_libId2engine[lib_id].SaveFaceFeat(this_engine_path)
        print(f'Save face features of lib_id: {lib_id}, lib_name: {dict_libId2libName[lib_id]} successfully')
    
    # Save dict_faceId2minioUrl
    save_dict_to_json(dict_faceId2minioUrl, faceId2minioUrl_file_path)
    print('Save dict_faceId2minioUrl successfully')
    
    # Shutdown the JVM
    jpype.shutdownJVM()
    return

def addFaceLib(lib_id:str, lib_name:str):
    global dict_libId2libName
    global dict_libId2engine
    if lib_id in dict_libId2libName.keys():
        return 500
    if lib_name in dict_libId2libName.values():
        return 500
    
    # Create lib folder
    lib_path = os.path.join(database_path, lib_name)
    os.mkdir(lib_path)
    
    dict_libId2libName[lib_id] = lib_name
    
    # Create face engine for this lib
    engine = FaceEngine(lib_id, lib_name, 0)
    dict_libId2engine[lib_id] = engine
    return 200

def delFaceLib(lib_id:str):
    global dict_libId2libName
    global dict_libId2engine
    # Lib not exists
    if not lib_id in dict_libId2libName.keys():
        return 500
    
    # Delete lib folder
    lib_name = dict_libId2libName[lib_id]
    lib_path = os.path.join(database_path, lib_name)
    if os.path.exists(lib_path):
        rmtree(lib_path)
    
    # Delete this lib from dict
    del dict_libId2libName[lib_id]
    
    # Del face engine for this lib
    del dict_libId2engine[lib_id]
    return 200

def addFace(lib_id, face_id, img_path):
    global dict_libId2engine
    if not lib_id in dict_libId2engine.keys():
        print(f'lib_id: {lib_id} not exists')
        return 404 
    minio_path = img_path
    # If img_path is not on the local system
    if "http" in img_path:
        local_path = os.path.join(temp_data_path, "minio")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        if '?' in img_path:
            img_path = img_path.split('?')[0]
        img_name = img_path.split('/')[-1]
        local_path = os.path.join(local_path, img_name)
        pull_img_from_minio(minio_url=minio_path, local_save_path=local_path)
        img_path = local_path
    
    code = dict_libId2engine[lib_id].registerFace(img_path, face_id)
    code = int(str(code))
    if code == 200:
        # Add faceId to dict_faceId2minioUrl
        dict_faceId2minioUrl[face_id] = minio_path

    return code

def delFace(lib_id, face_id):
    global dict_libId2engine
    if not lib_id in dict_libId2engine.keys():
        return 500
    code = dict_libId2engine[lib_id].delFace(face_id)
    if code != 200:
        return 500
    else:
        del dict_faceId2minioUrl[face_id]
        return 200

def findNearest(img_path, lib_id, is_minio = False):
    """Find the nearest face in a library

    Args:
        img_path (str): _description_
        lib_id (str): _description_

    Returns:
        float, str: The similarity and name of the nearest face,
        return  (negative, negative) if anything wrong(like wrong lib_id or empty library)
    """
    if is_minio: # img_path is a minio url, not a local path
        minio_path = img_path
        local_path = os.path.join(temp_data_path, "minio")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        if '?' in img_path:
            img_path = img_path.split('?')[0]
        img_name = img_path.split('/')[-1]
        local_path = os.path.join(local_path, img_name)
        pull_img_from_minio(minio_url=minio_path, local_save_path=local_path)
        img_path = local_path
    global dict_libId2engine
    if not lib_id in dict_libId2engine.keys():
        return 0, 0
    nearest_info = str(dict_libId2engine[lib_id].findNearestFace(img_path))
    # If anything wrong
    if not '$' in nearest_info:
        if "detectFaces" in nearest_info:
            return -1, -1
        elif "contain" in nearest_info:
            return -2, -2
        elif "extractFaceFeature" in nearest_info:
            return -3, -3
        elif "searchFaceFeature" in nearest_info:
            return -4, -4
        return -5, -5
    max_sim = float(nearest_info.split('$')[0]) * 100
    max_sim = round(max_sim, 2)
    #max_id = int(nearest_info.split('$')[1])
    max_tag = nearest_info.split('$')[2]
    max_minio_url = dict_faceId2minioUrl[max_tag]
    return max_sim, max_tag, max_minio_url


def compare(img_path1:str, img_path2:str, is_minio=False):
    if is_minio: # img_path is a minio url, not a local path
        local_path = os.path.join(temp_data_path, "minio")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
            
        if '?' in img_path1:
            raw_img_path1 = img_path1.split('?')[0]
        else:
            raw_img_path1 = img_path1
        if '?' in img_path2:
            raw_img_path2 = img_path2.split('?')[0]
        else:
            raw_img_path2 = img_path2
            
        img_name1 = raw_img_path1.split('/')[-1]
        img_name2 = raw_img_path2.split('/')[-1]
        local_path1 = os.path.join(local_path, img_name1)
        local_path2 = os.path.join(local_path, img_name2)
        code1 = pull_img_from_minio(minio_url=img_path1, local_save_path=local_path1)
        code2 = pull_img_from_minio(minio_url=img_path2, local_save_path=local_path2)
        if code1 + code2 < 2:
            return -1
        img_path1 = local_path1
        img_path2 = local_path2
    
    sim = str(detect_engine.Compare(img_path1, img_path2))
    if "Error" in sim:
        return -1
    else:
        return float(sim)
    

def detectAndRecognizeFace(img_path:str, lib_id:str, is_minio=False):
    if is_minio: # img_path is a minio url, not a local path
        minio_path = img_path
        local_path = os.path.join(temp_data_path, "minio")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        if '?' in img_path:
            img_path = img_path.split('?')[0]
        img_name = img_path.split('/')[-1]
        local_path = os.path.join(local_path, img_name)
        pull_img_from_minio(minio_url=minio_path, local_save_path=local_path)
        img_path = local_path
    global dict_libId2engine
    if not lib_id in dict_libId2engine.keys():
        return "-2"
    # Detect with ArcSoft api
    json_str = str(dict_libId2engine[lib_id].detectAndRecognizeFace(img_path))
    return json_str


def detectFace(img_path:str, is_minio=False):
    if is_minio: # img_path is a minio url, not a local path
        local_path = os.path.join(temp_data_path, "minio")
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        if '?' in img_path:
            raw_img_path = img_path.split('?')[0]
        img_name = raw_img_path.split('/')[-1]
        local_path = os.path.join(local_path, img_name)
        pull_img_from_minio(minio_url=img_path, local_save_path=local_path)
        img_path = local_path
    # Detect with ArcSoft api
    json_str = str(detect_engine.detectFace(img_path, 0))
    return json_str


def getFaceNum(lib_id:str):
    """Return the registered face number of this library

    Args:
        lib_id (str): library id

    Returns:
        int: -1 if lib_id is incorrect
    """
    global dict_libId2engine
    if not lib_id in dict_libId2engine.keys():
        return -1
    else:
        return dict_libId2engine[lib_id].getFaceNum()
    
def getAllLibs() -> list:
    global dict_libId2libName
    global dict_libId2engine
    result = list()
    for lib_id in dict_libId2libName.keys():
        face_num = dict_libId2engine[lib_id].getFaceNum()
        result.append((lib_id, dict_libId2libName[lib_id], face_num))
    
    return result
