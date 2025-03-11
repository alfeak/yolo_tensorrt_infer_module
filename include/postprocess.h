#pragma once

#include "NvInfer.h"
#include "dataloader.h"
#include <cuda_runtime.h>
#include "cuda_utils.h"

void cuda_decode(float* src, float* dst,vector<Rect>& boxes, vector<int>& class_ids, vector<float>& confidences,
    int detection_attribute_size, int num_detections, float conf_threshold, float nms_threshold, cudaStream_t stream);
