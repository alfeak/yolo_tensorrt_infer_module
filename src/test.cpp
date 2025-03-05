#include <cstdio>
#include "dataloader.h"
#include "engine.h"

//全局engine
Engine engine;
std::string engine_filename = "yolo11n.trt";
std::string image_filename = "bus.jpg";
int main() {
    cv::Mat img;
    int img_loade = imageload(img, image_filename);
    if (img_loade != 0) {
        engine.gLogger.log(ILogger::Severity::kERROR, "Error: Could not load image");
        return -1;
    }else{
        engine.gLogger.log(ILogger::Severity::kINFO, "image load successfully");
    }
    if (!engine.loadEngine(engine_filename)) {
        engine.gLogger.log(ILogger::Severity::kERROR, ("Error: Could not load engine " + engine_filename).c_str());
        return -1;
    } else{
        engine.gLogger.log(ILogger::Severity::kINFO, "engine load successfully");
    }
    engine.preprocess(img);
    

    return 0;
}