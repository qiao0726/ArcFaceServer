import com.arcsoft.face.*;
import com.arcsoft.face.enums.*;
import com.arcsoft.face.toolkit.ImageFactory;
import com.arcsoft.face.toolkit.ImageInfo;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.arcsoft.face.toolkit.ImageInfoEx;



public class FaceRecognition {
    public static void main(String[] args) {
        //激活码，从官网获取
        String appId = "GcymLjqt3RbQQqFg1KYNa96DbF4qzmwzoouDX9VKrAt6";
        String sdkKey = "CxD9mmpS4rPAFRZhxTSgK5LrFqfqqTF9H4h1yuAMjUJW";
        String activeKey = "82K1-11HP-X12G-9DHJ";


        //人脸识别引擎库存放路径
        FaceEngine faceEngine = new FaceEngine("/home/qn/ARcFaceJava/LINUX64");
        //激活引擎
        int errorCode = faceEngine.activeOnline(appId, sdkKey, activeKey);
        //System.out.println("引擎激活errorCode:" + errorCode);

        ActiveDeviceInfo activeDeviceInfo = new ActiveDeviceInfo();
        //采集设备信息（可离线）
        errorCode = faceEngine.getActiveDeviceInfo(activeDeviceInfo);
        //System.out.println("采集设备信息errorCode:" + errorCode);
        //System.out.println("设备信息:" + activeDeviceInfo.getDeviceInfo());

        ActiveFileInfo activeFileInfo = new ActiveFileInfo();
        errorCode = faceEngine.getActiveFileInfo(activeFileInfo);
        //System.out.println("获取激活文件errorCode:" + errorCode);
        //System.out.println("激活文件信息:" + activeFileInfo.toString());


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
        errorCode = faceEngine.init(engineConfiguration);
        //System.out.println("初始化引擎errorCode:" + errorCode);
        VersionInfo version = faceEngine.getVersion();
        //System.out.println(version);

        //人脸检测
        String path = "/home/qn/ARcFaceJava/testImg/hsq.jpg";
        ImageInfo imageInfo = ImageFactory.getRGBData(new File(path));
        List<FaceInfo> faceInfoList = new ArrayList<FaceInfo>();
        errorCode = faceEngine.detectFaces(imageInfo, faceInfoList);
        //System.out.println("人脸检测errorCode:" + errorCode);
        System.out.println("检测到人脸数:" + faceInfoList.size());

        //特征提取
        FaceFeature faceFeature = new FaceFeature();
        errorCode = faceEngine.extractFaceFeature(imageInfo, faceInfoList.get(0), ExtractType.REGISTER, 0, faceFeature);
        //System.out.println("特征提取errorCode:" + errorCode);

        //获取注册人脸个数
        FaceSearchCount faceSearchCount = new FaceSearchCount();
        errorCode = faceEngine.getFaceCount(faceSearchCount);
        System.out.println("注册人脸个数:" + faceSearchCount.getCount());

        //注册人脸信息1
        FaceFeatureInfo faceFeatureInfo = new FaceFeatureInfo();
        faceFeatureInfo.setSearchId(1);
        faceFeatureInfo.setFaceTag("hsq");
        faceFeatureInfo.setFeatureData(faceFeature.getFeatureData());
        errorCode = faceEngine.registerFaceFeature(faceFeatureInfo);


        errorCode = faceEngine.getFaceCount(faceSearchCount);
        System.out.println("注册人脸个数:" + faceSearchCount.getCount());
    }
}
