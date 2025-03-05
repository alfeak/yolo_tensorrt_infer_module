cmake_minimum_required(VERSION 3.10)
project(yolo_infer LANGUAGES  CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cuda setup for jetson
find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")
message("-- CUDA INC PATH: ${CUDA_INCLUDE_DIRS}")
message("-- CUDA LIB PATH: ${CUDA_LIBRARIES}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
set(CMAKE_CUDA_ARCHITECTURES 53 62 72 87)

# openCV setup
find_package(OpenCV REQUIRED)
message("-- OpenCV version: ${OpenCV_VERSION}")

# tensorRT setup
set(TensorRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARY /usr/lib/aarch64-linux-gnu)
set(TensorRT_LIBRARIES nvinfer nvinfer_plugin)

include_directories(${OpenCV_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# add headers
include_directories(include)
file(GLOB SOURCES "src/*.cpp" "src/*.cu" )

# add executable
add_executable(yolo_infer ${SOURCES})

# link libraries
target_link_libraries(yolo_infer ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})