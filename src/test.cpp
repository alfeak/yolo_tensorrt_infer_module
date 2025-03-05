#include <cstdio>
#include "dataloader.h"
#include "engine.h"

//全局engine
Engine engine;
std::string engine_filename = "yolo11n.trt";
std::string image_filename = "bus.jpg";
int main() {
    cv::Mat img;
    
    if (!engine.loadEngine(engine_filename)) {
        std::cerr << "Error: Could not load engine " << engine_filename << std::endl;
        return -1;
    } else{
        printf("engine load successfully\n");
    }

    return 0;
}