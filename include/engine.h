#ifndef ENGINE_H
#define ENGINE_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include "dataloader.h"
#include "cuda_utils.h"
#include "preprocess.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {  // Only log warnings and errors
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };

class Engine {
    public:
        Engine();
        ~Engine();
    
        bool loadEngine(const std::string& enginePath);
        // bool infer(const std::vector<float>& inputData, std::vector<float>& outputData);
        void preprocess(Mat& image);
        void infer();
        void postprocess(float* cpu_output_buffer);
        void draw(Mat& image, const vector<Detection>& output);

    private:
        Logger gLogger;
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
        cudaStream_t stream;

        char const * Inptensorname;
        char const * Outtensorname;
        float* gpu_buffers[2]; //!< The vector of device buffers needed for engine execution.
        // float* cpu_output_buffer; //!< Pointer to the output buffer on the host.
        int input_w; //!< Width of the input image.
        int input_h; //!< Height of the input image.
        int num_classes = 80; //!< Number of object classes that can be detected.
        const int MAX_IMAGE_SIZE = 4096 * 4096; //!< Maximum allowed input image size.
        
        public:
        int num_detections; //!< Number of detections output by the model.
        int detection_attribute_size; //!< Size of each detection attribute.
        float conf_threshold = 0.3f; //!< Confidence threshold for filtering detections.
        float nms_threshold = 0.4f; //!< Non-Maximum Suppression (NMS) threshold for filtering overlapping boxes.
    };

    
    
void nms(float* cpu_output_buffer, vector<Detection>& output,int num_detections, int detection_attribute_size, float nms_threshold , float conf_threshold);
#endif // ENGINELOAD_H