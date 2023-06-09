cmake_minimum_required(VERSION 3.0)

project(yolov8_seg)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


include_directories(${PROJECT_SOURCE_DIR})

# opencv 
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt 8
include_directories(/home/ruoxi/3rdparty/TensorRT-8.0.1.6/include)
link_directories(/home/ruoxi/3rdparty/TensorRT-8.0.1.6/lib)

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 


add_executable(onnx2trt ${PROJECT_SOURCE_DIR}/onnx2trt.cpp)
target_link_libraries(onnx2trt nvinfer)
target_link_libraries(onnx2trt nvonnxparser)
target_link_libraries(onnx2trt cudart)
target_link_libraries(onnx2trt ${OpenCV_LIBS})

add_executable(trt_infer_seg ${PROJECT_SOURCE_DIR}/trt_infer_seg.cpp)
target_link_libraries(trt_infer_seg nvinfer)
target_link_libraries(trt_infer_seg nvonnxparser)
target_link_libraries(trt_infer_seg cudart)
target_link_libraries(trt_infer_seg nvinfer_plugin)
target_link_libraries(trt_infer_seg ${OpenCV_LIBS})

add_executable(trt_infer_det ${PROJECT_SOURCE_DIR}/trt_infer_det.cpp)
target_link_libraries(trt_infer_det nvinfer)
target_link_libraries(trt_infer_det nvonnxparser)
target_link_libraries(trt_infer_det cudart)
target_link_libraries(trt_infer_det nvinfer_plugin)
target_link_libraries(trt_infer_det ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

