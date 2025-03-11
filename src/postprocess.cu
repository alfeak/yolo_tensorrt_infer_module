#include "postprocess.h"


__global__ void decode_kernel(float* dst, float* src, int num_detections, int detection_attribute_size, float conf_threshold) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_detections) return;

    // 按列优先访问数据
    float* ptr = src + idx; // 每列的第 idx 个检测结果
    float* ptr_conf = src + num_detections * 4 + idx; // 置信度部分

    float* ptr_dst = dst + idx * 7;

    // 提取边界框坐标
    const float cx = ptr[0 * num_detections];
    const float cy = ptr[1 * num_detections];
    const float ow = ptr[2 * num_detections];
    const float oh = ptr[3 * num_detections];

    // 查找最大置信度及其类别
    float max_conf = 0.0f;
    int cls_id = -1;
    for (int i = 0; i < detection_attribute_size - 4; ++i) {
        float conf = ptr_conf[i * num_detections];
        if (conf > max_conf) {
            max_conf = conf;
            cls_id = i; // 索引从 0 开始
        }
    }

    // 如果最大置信度大于阈值，则存储结果
    if (max_conf >= conf_threshold) {
        ptr_dst[0] = cx;
        ptr_dst[1] = cy;
        ptr_dst[2] = ow;
        ptr_dst[3] = oh;
        ptr_dst[4] = max_conf;
        ptr_dst[5] = cls_id;
        ptr_dst[6] = 1; // 标记为有效检测
    }
}

static __device__ float box_iou(float ax, float ay, float aw, float ah, float bx, float by,
    float bw, float bh) {
float aleft = ax - aw / 2.0f;
float atop = ay - ah / 2.0f;
float aright = ax + aw / 2.0f;
float abottom = ay + ah / 2.0f;
float bleft = bx - bw / 2.0f;
float btop = by - bh / 2.0f;
float bright = bx + bw / 2.0f;
float bbottom = by + bh / 2.0f;
float cleft = max(aleft, bleft);
float ctop = max(atop, btop);
float cright = min(aright, bright);
float cbottom = min(abottom, bbottom);
float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
if (c_area == 0.0f)
return 0.0f;

float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= max_objects)
        return;

    float* pcurrent = bboxes  + position * 7;
    for (int i = 0; i < max_objects; ++i) {
        float* pitem = bboxes + i * 7;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou =
                    box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > threshold) {
                atomicExch(&pcurrent[6], 0);
                return;
            }
        }
    }
}

void cuda_decode(float* src,float* dst, vector<Rect>& boxes, vector<int>& class_ids, vector<float>& confidences,
    int detection_attribute_size,int num_detections, float conf_threshold, float nms_threshold,cudaStream_t stream){

    int block = 256;
    int grid = ceil(num_detections / (float)block);
    float *dst_cpu = (new float[num_detections * 7]);

    CUDA_CHECK(cudaMemset(dst, 0, num_detections * 7 * sizeof(float)));
    
    decode_kernel<<<grid, block, 0, stream>>>(dst,src,num_detections,detection_attribute_size,conf_threshold);
    nms_kernel<<<grid, block, 0, stream>>>(dst, num_detections, nms_threshold);
    CUDA_CHECK(cudaMemcpyAsync(dst_cpu, dst, num_detections * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < num_detections; ++i) {
        float* ptr = dst_cpu + i * 7;
        if (ptr[6] == 1) {
            Rect box;
            const float cx = ptr[0];
            const float cy = ptr[1];
            const float ow = ptr[2];
            const float oh = ptr[3];
            const float score = ptr[4];
            const int class_id = static_cast<int>(ptr[5]);
            // Calculate top-left corner of the bounding box
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            // Set width and height of the bounding box
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            // Store the bounding box, class ID, and confidence
            boxes.push_back(box);
            class_ids.push_back(class_id);
            confidences.push_back(score);
        }
    }
        delete[] dst_cpu;
    }