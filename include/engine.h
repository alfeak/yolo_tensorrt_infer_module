#ifndef ENGINE_H
#define ENGINE_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cuda_runtime.h>
#include "dataloader.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "postprocess.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger {
    public:
    void log(Severity severity, const char* msg) noexcept override {
        // Get current time
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm = *std::localtime(&now_time);

        const char* severity_str = "";
        const char* color = "";
        
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                color = "\033[1;31m"; severity_str = "INTERNAL_ERROR"; break; // Bold Red
            case nvinfer1::ILogger::Severity::kERROR:
                color = "\033[31m"; severity_str = "ERROR"; break; // Red
            case nvinfer1::ILogger::Severity::kWARNING:
                color = "\033[33m"; severity_str = "WARNING"; break; // Yellow
            case nvinfer1::ILogger::Severity::kINFO:
                color = "\033[36m"; severity_str = "I"; break; // Cyan
            case nvinfer1::ILogger::Severity::kVERBOSE:
                color = "\033[37m"; severity_str = "DEBUG"; break; // White/Gray
        }
        std::cout << color 
                  << "[" << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << severity_str << "] "
                  << msg
                  << "\033[0m" << std::endl; // Reset color
    }
    };

class Engine {
    public:
        Engine();
        ~Engine();

        Logger gLogger;
        bool loadEngine(const std::string& enginePath);
        void preprocess(Mat& image);
        void infer();
        void postprocess(vector<Detection>& output);
        void postprocess_cpu(vector<Detection>& output);
        void draw(Mat& image, const vector<Detection>& output);
        void inference(Mat& image, vector<Detection>& output,const bool use_gpu);
    
    private:
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
        cudaStream_t stream;
        float* gpu_buffers[3]; //!< The vector of device buffers needed for engine execution.
        float* cpu_output_buffer; //!< Pointer to the output buffer on the host.
        int input_w; //!< Width of the input image.
        int input_h; //!< Height of the input image.
        const int MAX_IMAGE_SIZE = 4096 * 4096; //!< Maximum allowed input image size.

    public:
        char const * Inptensorname;
        char const * Outtensorname;
        int max_detections = 100;
        int num_classes = 80; //!< Number of object classes that can be detected.
        int num_detections; //!< Number of detections output by the model.
        int detection_attribute_size; //!< Size of each detection attribute.
        float conf_threshold = 0.6f; //!< Confidence threshold for filtering detections.
        float nms_threshold = 0.6f; //!< Non-Maximum Suppression (NMS) threshold for filtering overlapping boxes.
    };

#endif // ENGINELOAD_H