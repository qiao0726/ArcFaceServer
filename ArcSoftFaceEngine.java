import com.arcsoft.face.*;
import com.arcsoft.face.enums.*;
import com.arcsoft.face.toolkit.ImageFactory;
import com.arcsoft.face.toolkit.ImageInfo;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileOutputStream;
import javax.imageio.ImageIO;
// import javax.naming.directory.SearchResult;
import com.arcsoft.face.SearchResult;


import java.awt.image.BufferedImage;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.arcsoft.face.toolkit.ImageInfoEx;


public class ArcSoftFaceEngine {
    //激活码，从官网获取
    private static final String appId = "CUDRiMDFwPVqHjycTsHSr5DsRj2PJGmWQJBkNzJhqCme";
    private static final String sdkKey = "98ZUeBqv8VVpbWd2WKFb6x6zVghBtdpzNWqsYswKBgo3";
    private static final String activeKey = "082K-11DH-H2BL-VLDX";

    private static final String engine_path = "/home/qn/ARcFaceJava/LINUX64";
    private static final String database_path = "/home/qn/ARcFaceJava/database";

    //Library id
    private String lib_id;
    private String lib_name;

    private FaceEngine faceEngine;
    private int registered_face_num = 0;

    private Map<String, Integer> faceTag2faceId = new HashMap<>();
    private Map<String, byte[]> faceTag2Feat = new HashMap<>();


    // Static method to load faceTag2Feat
    public static Map<String, byte[]> loadByteArraysWithKeysFromFile(String filePath) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (HashMap<String, byte[]>) ois.readObject();
        }
    }
    // Static method to save faceTag2Feat
    public static void saveByteArraysWithKeysToFile(Map<String, byte[]> byteArrayMap, String filePath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(byteArrayMap);
        }
    }

    // Static method to save faceTag2faceId
    public static void saveIntegerMapToFile(Map<String, Integer> integerMap, String filePath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(integerMap);
        }
    }
    // Static method to load faceTag2faceId
    public static Map<String, Integer> loadIntegerMapFromFile(String filePath) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return (HashMap<String, Integer>) ois.readObject();
        }
    }

    // Static method to transform Map to json string
    public static String mapToJsonString(Map<String, Map<String, String>> jsonData) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            String resultJsonString = objectMapper.writeValueAsString(jsonData);
            return resultJsonString;
        } catch (IOException e) {
            System.err.println("处理JSON数据时发生错误: " + e.getMessage());
            return null;
        }
    }

    public ArcSoftFaceEngine(String lib_id, String lib_name, int need_load) {
        /*
         * If need_load == 1, this engine needs to load faceTag2faceId and faceTag2Feat file,
         * else, this is a new built engine
         */
        this.lib_id = lib_id;
        this.lib_name = lib_name;

        this.faceEngine = new FaceEngine(engine_path);
        //激活引擎
        int errorCode = this.faceEngine.activeOnline(appId, sdkKey, activeKey);

        ActiveDeviceInfo activeDeviceInfo = new ActiveDeviceInfo();
        //采集设备信息（可离线）
        errorCode = this.faceEngine.getActiveDeviceInfo(activeDeviceInfo);
        ActiveFileInfo activeFileInfo = new ActiveFileInfo();
        errorCode = this.faceEngine.getActiveFileInfo(activeFileInfo);

        //引擎配置
        EngineConfiguration engineConfiguration = new EngineConfiguration();
        engineConfiguration.setDetectMode(DetectMode.ASF_DETECT_MODE_IMAGE);
        engineConfiguration.setDetectFaceOrientPriority(DetectOrient.ASF_OP_ALL_OUT);
        engineConfiguration.setDetectFaceMaxNum(10);
        //功能配置
        FunctionConfiguration functionConfiguration = new FunctionConfiguration();
        functionConfiguration.setSupportAge(true);
        functionConfiguration.setSupportFaceDetect(true);
        functionConfiguration.setSupportFaceRecognition(true);
        functionConfiguration.setSupportGender(true);
        functionConfiguration.setSupportLiveness(true);
        functionConfiguration.setSupportIRLiveness(true);
        functionConfiguration.setSupportImageQuality(true);
        functionConfiguration.setSupportMaskDetect(true);
        functionConfiguration.setSupportUpdateFaceData(true);
        engineConfiguration.setFunctionConfiguration(functionConfiguration);

        //初始化引擎
        errorCode = this.faceEngine.init(engineConfiguration);

        //Load faceTag2faceId and faceTag2Feat
        if(need_load == 1) {
            try {
                this.faceTag2faceId = loadIntegerMapFromFile(database_path + '/' + this.lib_name + "/faceTag2faceId.dat");
                this.faceTag2Feat = loadByteArraysWithKeysFromFile(database_path + '/' + this.lib_name + "/faceTag2Feat.dat");
                

                // For each faceId
                this.faceTag2faceId.forEach((face_tag, face_id) -> {
                    //注册人脸信息
                    FaceFeatureInfo faceFeatureInfo = new FaceFeatureInfo();
                    faceFeatureInfo.setSearchId(face_id);
                    faceFeatureInfo.setFaceTag(face_tag);

                    byte[] featureData = this.faceTag2Feat.get(face_tag);

                    faceFeatureInfo.setFeatureData(featureData);
                    this.faceEngine.registerFaceFeature(faceFeatureInfo);
                });

                this.registered_face_num = this.getFaceNum();
            } catch(IOException|ClassNotFoundException e) {
                System.err.println("Load file error");
            }
            
        }

    }

    public String getId() {
        return this.lib_id;
    }

    public String getName() {
        return this.lib_name;
    }

    public int getFaceNum() {
        //获取注册人脸个数
        FaceSearchCount faceSearchCount = new FaceSearchCount();
        int errorCode = this.faceEngine.getFaceCount(faceSearchCount);
        return faceSearchCount.getCount();
    }

    public String detectFace(String img_path, int ignore_out_of_bounds) {
        File imgFile = new File(img_path);
        ImageInfo imageInfo = ImageFactory.getRGBData(imgFile);
        List<FaceInfo> faceInfoList = new ArrayList<FaceInfo>();
        int errorCode = this.faceEngine.detectFaces(imageInfo, faceInfoList);
        if(errorCode != 0) {
            return "Detect Error code: " + errorCode;
        }

        if(faceInfoList.size() == 0) {
            return "-1";
        }

        //人脸属性检测
        FunctionConfiguration configuration = new FunctionConfiguration();
        configuration.setSupportAge(false);
        configuration.setSupportGender(false);
        configuration.setSupportLiveness(false);
        configuration.setSupportMaskDetect(true);
        errorCode = this.faceEngine.process(imageInfo, faceInfoList, configuration);
        if(errorCode != 0) {
            return "Process Error code: " + errorCode;
        }

        //口罩检测
        List<MaskInfo> maskInfoList = new ArrayList<MaskInfo>();
        errorCode = this.faceEngine.getMask(maskInfoList);
        if(errorCode != 0) {
            return "Mask detection error: " + errorCode;
        }

        Map<String, Map<String, String>> jsonData = new HashMap<>();

        for(int i=0; i < faceInfoList.size(); i++) {
            int isWithinBoundary = faceInfoList.get(i).getIsWithinBoundary();
            if(isWithinBoundary == 0 && ignore_out_of_bounds != 0) {
                continue;
            }
            int left = faceInfoList.get(i).getRect().left;
            int right = faceInfoList.get(i).getRect().right;
            int top = faceInfoList.get(i).getRect().top;
            int bottom = faceInfoList.get(i).getRect().bottom;
            int mask = maskInfoList.get(i).getMask(); //1戴口罩，0不戴口罩

            // If out of bounds
            if(left < 0 || right < 0 || top < 0 || bottom < 0) {
                continue;
            }
            //
            Map<String, String> thisFaceData = new HashMap<>();
            thisFaceData.put("left", left + "");
            thisFaceData.put("right", right + "");
            thisFaceData.put("top", top + "");
            thisFaceData.put("bottom", bottom + "");
            thisFaceData.put("mask", mask + "");
            jsonData.put(i + "", thisFaceData);
        }

        return mapToJsonString(jsonData);
    }

    public String registerFace(String img_path, String face_id) {
        // If this face_id already exists
        if(this.faceTag2faceId.containsKey(face_id)) {
            return "-2";
        }

        ImageInfo imageInfo;
        try {
            imageInfo = ImageFactory.getRGBData(new File(img_path));
        }catch(Exception e) {
            return "-1";
        }
        List<FaceInfo> faceInfoList = new ArrayList<FaceInfo>();
        int errorCode = this.faceEngine.detectFaces(imageInfo, faceInfoList);
        if(errorCode != 0) {
            return "1";
        }
        // More than 1 face(or no faces)
        if(faceInfoList.size() != 1) {
            return "2";
        }
        //人脸属性检测
        FunctionConfiguration configuration = new FunctionConfiguration();
        configuration.setSupportAge(false);
        configuration.setSupportGender(false);
        configuration.setSupportLiveness(false);
        configuration.setSupportMaskDetect(true);
        errorCode = this.faceEngine.process(imageInfo, faceInfoList, configuration);
        if(errorCode != 0) {
            return "3";
        }
        //口罩检测
        List<MaskInfo> maskInfoList = new ArrayList<MaskInfo>();
        errorCode = this.faceEngine.getMask(maskInfoList);
        if(errorCode != 0) {
            return "4";
        }
        // int mask = maskInfoList.get(0).getMask(); //1戴口罩，0不戴口罩
        // if(mask == 1) {
        //     return "5";
        // }

        //特征提取
        FaceFeature faceFeature = new FaceFeature();
        errorCode = this.faceEngine.extractFaceFeature(imageInfo, faceInfoList.get(0), ExtractType.REGISTER, 0, faceFeature);
        if(errorCode != 0) {
            return "6";
        }

        //注册人脸信息1
        FaceFeatureInfo faceFeatureInfo = new FaceFeatureInfo();
        faceFeatureInfo.setSearchId(this.registered_face_num);
        faceFeatureInfo.setFaceTag(face_id);

        byte[] featureData = faceFeature.getFeatureData();


        faceFeatureInfo.setFeatureData(featureData);
        errorCode = this.faceEngine.registerFaceFeature(faceFeatureInfo);
        System.out.println("errorCode: " + errorCode);
        if(errorCode != 0) {
            return "7";
        }

        //Add (tag, id) pair to map
        this.faceTag2faceId.put(face_id, this.registered_face_num);
        // Add (tag, feat) pair to map
        this.faceTag2Feat.put(face_id, featureData);

        this.registered_face_num++;

        return "200";
    }

    public int delFace(String face_id) {
        if(!this.faceTag2faceId.containsKey(face_id)) {
            return 500;
        }
        Integer removedValue = this.faceTag2faceId.remove(face_id);
        this.faceTag2Feat.remove(face_id);
        //移除人脸信息
        int errorCode = this.faceEngine.removeFaceFeature(removedValue);
        return 200;
        // Set<String> keys = this.faceTag2faceId.keySet();
        // for (String key : keys) {
        //     Integer removedValue = this.faceTag2faceId.remove(key);
        //     this.faceTag2Feat.remove(key);
        //     //移除人脸信息
        //     int errorCode = this.faceEngine.removeFaceFeature(removedValue);
        //     System.out.println(key);
        //     break;
        // }
        // return 0;
    }



    public String findNearestFace(String img_path) {
        ImageInfo imageInfo = ImageFactory.getRGBData(new File(img_path));
        List<FaceInfo> faceInfoList = new ArrayList<FaceInfo>();
        int errorCode = this.faceEngine.detectFaces(imageInfo, faceInfoList);
        if(errorCode != 0) {
            return "detectFaces Error";
        }
        // More than 1 face(or no faces)
        if(faceInfoList.size() == 0) {
            return "Error: The image should contain 1 face.";
        }

        //特征提取
        FaceFeature faceFeature = new FaceFeature();
        errorCode = this.faceEngine.extractFaceFeature(imageInfo, faceInfoList.get(0), ExtractType.REGISTER, 0, faceFeature);
        if(errorCode != 0) {
            return "extractFaceFeature Error";
        }
        
        //搜索最相似人脸
        SearchResult searchResult = new SearchResult();
        errorCode = this.faceEngine.searchFaceFeature(faceFeature, CompareModel.LIFE_PHOTO, searchResult);
        if(errorCode != 0) {
            return "searchFaceFeature Error";
        }

        String result_str = searchResult.getMaxSimilar() + "$" + searchResult.getFaceFeatureInfo().getSearchId();
        result_str = result_str + "$" + searchResult.getFaceFeatureInfo().getFaceTag();
        //return searchResult.getFaceFeatureInfo().getSearchId();
        return result_str;
    }


    //Compare two faces
    public String Compare(String img_path1, String img_path2) {
        //Face detection 1
        ImageInfo imageInfo1 = ImageFactory.getRGBData(new File(img_path1));
        List<FaceInfo> faceInfoList1 = new ArrayList<FaceInfo>();
        int errorCode = this.faceEngine.detectFaces(imageInfo1, faceInfoList1);
        if(errorCode != 0) {
            return "detectFaces Error";
        }
        // More than 1 face(or no faces)
        if(faceInfoList1.size() != 1) {
            return "Error: The image should contain 1 face.";
        }

        //Extract face feature1
        FaceFeature faceFeature1 = new FaceFeature();
        errorCode = this.faceEngine.extractFaceFeature(imageInfo1, faceInfoList1.get(0), ExtractType.REGISTER, 0, faceFeature1);
        if(errorCode != 0) {
            return "extractFaceFeature Error";
        }


        //Face detection 2
        ImageInfo imageInfo2 = ImageFactory.getRGBData(new File(img_path2));
        List<FaceInfo> faceInfoList2 = new ArrayList<FaceInfo>();
        errorCode = this.faceEngine.detectFaces(imageInfo2, faceInfoList2);
        if(errorCode != 0) {
            return "detectFaces Error";
        }
        // More than 1 face(or no faces)
        if(faceInfoList2.size() != 1) {
            return "Error: The image should contain 1 face.";
        }

        //Extract face feature2
        FaceFeature faceFeature2 = new FaceFeature();
        errorCode = this.faceEngine.extractFaceFeature(imageInfo2, faceInfoList2.get(0), ExtractType.REGISTER, 0, faceFeature2);
        if(errorCode != 0) {
            return "extractFaceFeature Error";
        }


        //特征比对
        FaceFeature targetFaceFeature = new FaceFeature();
        targetFaceFeature.setFeatureData(faceFeature1.getFeatureData());
        FaceFeature sourceFaceFeature = new FaceFeature();
        sourceFaceFeature.setFeatureData(faceFeature2.getFeatureData());
        FaceSimilar faceSimilar = new FaceSimilar();

        errorCode = faceEngine.compareFaceFeature(targetFaceFeature, sourceFaceFeature, faceSimilar);
        if(errorCode != 0) {
            return "compareFaceFeature Error";
        }
        return faceSimilar.getScore() + "";
    }



    public String detectAndRecognizeFace(String img_path) {
        File imgFile = new File(img_path);
        if(!imgFile.exists()) {
            return "Image file not exists";
        }
        ImageInfo imageInfo = ImageFactory.getRGBData(imgFile);
        List<FaceInfo> faceInfoList = new ArrayList<FaceInfo>();
        int errorCode = this.faceEngine.detectFaces(imageInfo, faceInfoList);
        if(errorCode != 0) {
            return "Detect Error code: " + errorCode;
        }

        // No faces detected
        if(faceInfoList.size() == 0) {
            return "-1";
        }

        //人脸属性检测
        FunctionConfiguration configuration = new FunctionConfiguration();
        configuration.setSupportAge(false);
        configuration.setSupportGender(false);
        configuration.setSupportLiveness(false);
        configuration.setSupportMaskDetect(true);
        errorCode = this.faceEngine.process(imageInfo, faceInfoList, configuration);
        if(errorCode != 0) {
            return "Process Error code: " + errorCode;
        }

        //口罩检测
        List<MaskInfo> maskInfoList = new ArrayList<MaskInfo>();
        errorCode = this.faceEngine.getMask(maskInfoList);
        if(errorCode != 0) {
            return "Mask detection error: " + errorCode;
        }

        Map<String, Map<String, String>> jsonData = new HashMap<>();

        for(int i=0; i < faceInfoList.size(); i++) {
            int isWithinBoundary = faceInfoList.get(i).getIsWithinBoundary();
            if(isWithinBoundary == 0) {
                continue;
            }

            // Extract face feature
            FaceFeature faceFeature = new FaceFeature();
            errorCode = this.faceEngine.extractFaceFeature(imageInfo, faceInfoList.get(i), ExtractType.REGISTER, 0, faceFeature);
            if(errorCode != 0) {
                return "extractFaceFeature Error";
            }
            
            // Search nearest face
            SearchResult searchResult = new SearchResult();
            errorCode = this.faceEngine.searchFaceFeature(faceFeature, CompareModel.LIFE_PHOTO, searchResult);
            if(errorCode != 0) {
                return "searchFaceFeature Error";
            }
            
            String max_sim = searchResult.getMaxSimilar() + "";
            String search_id = searchResult.getFaceFeatureInfo().getFaceTag() + "";

            int left = faceInfoList.get(i).getRect().left;
            int right = faceInfoList.get(i).getRect().right;
            int top = faceInfoList.get(i).getRect().top;
            int bottom = faceInfoList.get(i).getRect().bottom;
            int mask = maskInfoList.get(i).getMask(); //1戴口罩，0不戴口罩

            // If out of bounds
            if(left < 0 || right < 0 || top < 0 || bottom < 0) {
                continue;
            }

            Map<String, String> thisFaceData = new HashMap<>();
            thisFaceData.put("left", left + "");
            thisFaceData.put("right", right + "");
            thisFaceData.put("top", top + "");
            thisFaceData.put("bottom", bottom + "");
            thisFaceData.put("mask", mask + "");
            thisFaceData.put("sim", max_sim + "");
            thisFaceData.put("name", search_id + "");
            jsonData.put(i + "", thisFaceData);
        }

        return mapToJsonString(jsonData);
    }





    // When the http service shutdown, save faceTag2faceId and faceTag2feat
    public void SaveFaceFeat(String save_path) {
        try {
            saveByteArraysWithKeysToFile(this.faceTag2Feat, save_path + "/faceTag2Feat.dat");
            saveIntegerMapToFile(this.faceTag2faceId, save_path + "/faceTag2faceId.dat");
        }catch(IOException e) {
            System.err.println("Save file error");
        }
        
    }
}
