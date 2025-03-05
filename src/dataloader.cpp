#include "dataloader.h"
int imageload(cv::Mat& img, const std::string& filename) {
    // 使用 OpenCV 的 imread 函数加载图像
    img = cv::imread(filename);

    // 检查图像是否成功加载
    if (img.empty()) {
        std::cerr << "Error: Could not read image file " << filename << std::endl;
        return -1; // 返回错误码
    }else{
        std:cout << "Image loaded successfully" << std::endl;
    }

    return 0; // 成功
}

std::string detectionsToJson(const std::vector<Detection>& detections){
    nlohmann::json detectionsJson;
    for (const auto& detection : detections) {
        nlohmann::json detectionJson;
        detectionJson["class"] = detection.class_id;
        detectionJson["score"] = detection.conf;
        detectionJson["bbox"]["x"] = detection.bbox.x;
        detectionJson["bbox"]["y"] = detection.bbox.y;
        detectionJson["bbox"]["width"] = detection.bbox.width;
        detectionJson["bbox"]["height"] = detection.bbox.height;
        detectionsJson.push_back(detectionJson);
    }
    return detectionsJson.dump();
}
