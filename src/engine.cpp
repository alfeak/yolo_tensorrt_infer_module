#include "engine.h"

Engine::Engine() : runtime(nullptr), engine(nullptr), context(nullptr) {
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        this->gLogger.log(Severity::kERROR, "Failed to create TensorRT runtime");
    }        
}

Engine::~Engine() {
    // Synchronize and destroy the CUDA stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Free allocated GPU buffers
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    // // Free CPU output buffer
    // delete[] cpu_output_buffer;
    // Destroy CUDA preprocessing resources
    cuda_preprocess_destroy();
    // Delete TensorRT context, engine, and runtime

    if (context) delete context;
    if (&gLogger) delete &gLogger;
    if (engine) delete engine;
    if (runtime) delete runtime;
}
bool Engine::loadEngine(const std::string& enginePath){
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        this->gLogger.log(Severity::kERROR, "Failed to open engine file");
        return false;
    }

    // Read the engine file into a buffer
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
    if (!engine) {
        this->gLogger.log(Severity::kERROR, "Failed to deserialize CUDA engine");
        delete engine;
        delete runtime;
        return false;
    }else{
        this->gLogger.log(Severity::kINFO, "CUDA engine deserialized successfully.");
    }
    // Create an execution context for the engine
    context = engine->createExecutionContext();

    // Retrieve input dimensions from the engine
    Inptensorname = engine->getIOTensorName(0);
    Outtensorname = engine->getIOTensorName(1);
    input_h = engine->getTensorShape(Inptensorname).d[2];
    input_w = engine->getTensorShape(Inptensorname).d[3];
    // Retrieve detection attributes and number of detections
    detection_attribute_size = engine->getTensorShape(Outtensorname).d[1];
    num_detections = engine->getTensorShape(Outtensorname).d[2];
    // Calculate the number of classes based on detection attributes
    num_classes = detection_attribute_size - 4;
    this->gLogger.log(Severity::kINFO, ("Input dimensions: " + std::to_string(input_w) + "x" + std::to_string(input_h)).c_str());
    this->gLogger.log(Severity::kINFO, ("Detection attributes: " + std::to_string(detection_attribute_size)).c_str());
    this->gLogger.log(Severity::kINFO, ("Number of detections: " + std::to_string(num_detections)).c_str());
    this->gLogger.log(Severity::kINFO, ("Number of classes: " + std::to_string(num_classes)).c_str());
    
    // // Allocate CPU memory for output buffer
    // cpu_output_buffer = new float[detection_attribute_size * num_detections];
    // Allocate GPU memory for input buffer (assuming 3 channels: RGB)
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Allocate GPU memory for output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));
    // Initialize CUDA preprocessing with maximum image size
    cuda_preprocess_init(MAX_IMAGE_SIZE);
    // Create a CUDA stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&stream));
    cv::Mat warmup_image = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    for (int i = 0; i < 10; i++) {
        this->preprocess(warmup_image); // Preprocess the warmup image
        this->infer(); // Run inference to warm up the model
    }
    // printf("model warmup 10 times\n");
    this->gLogger.log(Severity::kINFO, "model warmup 10 times");
    return true;
}
// Preprocess the input image and transfer it to the GPU buffer
void Engine::preprocess(Mat& image) {
    // Perform CUDA-based preprocessing
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    // Synchronize the CUDA stream to ensure preprocessing is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
void Engine::infer() {
    // Set the input tensor address
    context->setTensorAddress(Inptensorname, gpu_buffers[0]);
    // Set the output tensor address
    context->setTensorAddress(Outtensorname, gpu_buffers[1]);
    this->context->enqueueV3(this->stream);
}

void Engine::postprocess(float* cpu_output_buffer)
{
     // Asynchronously copy output from GPU to CPU
     CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], this->num_detections * this->detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
     // Synchronize the CUDA stream to ensure copy is complete
     CUDA_CHECK(cudaStreamSynchronize(stream));
}
void nms(float* cpu_output_buffer, vector<Detection>& output,int num_detections, int detection_attribute_size, float nms_threshold , float conf_threshold){
     vector<Rect> boxes;          // Bounding boxes
     vector<int> class_ids;       // Class IDs
     vector<float> confidences;   // Confidence scores
 
     // Create a matrix view of the detection output
     const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
 
     // Iterate over each detection
     for (int i = 0; i < det_output.cols; ++i) {
         // Extract class scores for the current detection
         const Mat classes_scores = det_output.col(i).rowRange(4, detection_attribute_size);
         Point class_id_point;
         double score;
         // Find the class with the maximum score
         minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
 
         // Check if the confidence score exceeds the threshold
         if (score > conf_threshold) {
             // Extract bounding box coordinates
             const float cx = det_output.at<float>(0, i);
             const float cy = det_output.at<float>(1, i);
             const float ow = det_output.at<float>(2, i);
             const float oh = det_output.at<float>(3, i);
             Rect box;
             // Calculate top-left corner of the bounding box
             box.x = static_cast<int>((cx - 0.5 * ow));
             box.y = static_cast<int>((cy - 0.5 * oh));
             // Set width and height of the bounding box
             box.width = static_cast<int>(ow);
             box.height = static_cast<int>(oh);
 
             // Store the bounding box, class ID, and confidence
             boxes.push_back(box);
             class_ids.push_back(class_id_point.y);
             confidences.push_back(score);
         }
     }
 
     vector<int> nms_result; // Indices after Non-Maximum Suppression (NMS)
     // Apply NMS to remove overlapping boxes
     dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
 
     // Iterate over NMS results and populate the output detections
     for (int i = 0; i < nms_result.size(); i++)
     {
         Detection result;
         int idx = nms_result[i];
         result.class_id = class_ids[idx];
         result.conf = confidences[idx];
         result.bbox = boxes[idx];
         output.push_back(result);
     }
}
void Engine::draw(Mat& image, const vector<Detection>& output)
{
    // Calculate the scaling ratios between input and original image dimensions
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // Iterate over each detection
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        // Assign a color based on the class ID
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        // Adjust bounding box coordinates based on aspect ratio
        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        // Draw the bounding box on the image
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Prepare the label text with class name and confidence
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        // Calculate the size of the text for background rectangle
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        // Define the background rectangle for the text
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        // Draw the background rectangle
        rectangle(image, text_rect, color, FILLED);
        // Put the text label on the image
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}