#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>

using namespace nvinfer1;
using namespace std;

// network and the input/output
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int CLASSES = 80;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES + 4);  // detect  


static const float CONF_THRESHOLD = 0.2;
static const float NMS_THRESHOLD = 0.5;

const char* INPUT_BLOB_NAME = "images";
// const char* INPUT_BLOB_NAME = "video";
const char* OUTPUT_BLOB_NAME = "output0"; //detect;

static Logger gLogger;                    // log

struct OutputDet {
    int id;            // 类别id
    float confidence;  // 置信度 
    cv::Rect box;      // 矩形框
};

void DrawPred_Det(Mat& img, std::vector<OutputDet> result) {
    // 生成随机颜色
    std::vector<Scalar> color;
    srand(time(0));

    // 为不同类别生成不同的对应颜色
    for (int i = 0; i < CLASSES; i++) {
        int b = rand() % 256;   
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }

    for (int i = 0; i < result.size(); i++ ) {
        int left,top;           // BoungingBox的左边和上边
        int color_num = i;      
        left = result[i].box.x;
        top  = result[i].box.y;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);

        char label[100];        // 标签

        const std::vector<std::string> coco80 = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        // sprintf(label, "%d:%.2f", result[i].id, result[i].confidence);  // 打印标签和置信度
        sprintf(label, "%s:%.2f", coco80[result[i].id].c_str(), result[i].confidence);

        int baseline;           // 起始位置
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        top = max(top, labelSize.height);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
        // putText(img, coco80.at(int(*label)), Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
        std::cout << "idx: " << label << std::endl;
    }
}

void doInfer_Det(IExecutionContext& context, float* input, float* output, int batchSize) 
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // know the name of the input and output for binding the buffers
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    
    // 常用API函数
    // cudaMallocHost : 分配CPU内存
    // cudaMalloc :     分配GPU内存
    // cudaMemcpy ：    显式同步阻塞（同步）传输（GPU ——— CPU）
    // cudaMemcpyAsync ： 显式非同步阻塞（异步）传输


    // create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        argv[1] = "../models/yolov8n.engine";
        argv[2] = "../images/bus.jpg";
    }

    // -----------------------------load model--------------------------------
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr};
    size_t size{ 0 };

    std::ifstream file(argv[1], std::ios::binary);
    if (file.good()) {
        std::cout << "load engine suceess" << std::endl;
        file.seekg(0, file.end);            // 指向文件的最后地址
        size = file.tellg();                // 把文件长度告诉给size

        std::cout << "\nfile: " << argv[1] << std::endl;
        std::cout << "size is:  " << size  << std::endl;

        file.seekg(0, file.beg);            // 指向文件的最后地址
        trtModelStream = new char [size];   // 开辟一个char 长度是文件的长度
        assert(trtModelStream);
        file.read(trtModelStream, size);    // 将文件内容传给trtModelStream
        file.close();   
    }
    else{
        std::cout << "load engine failed" << std::endl;
        return 1;
    }

    // -----------------------------read video--------------------------------- 
    Mat src = imread(argv[2], 1);
    if (src.empty()) {std::cout << "imag load failed" << std::endl; return 1;}
    int img_width = src.cols;
    int img_height = src.rows;
    std::cout << "宽高" << img_width << " " << img_height << std::endl;

    cv::VideoCapture capture;

    // capture.read(src);

    // Subtract mean from image

    static float data[3 * INPUT_H * INPUT_W];
    Mat pre_img;

    // std::cout << "debug0" << std::endl;

    std::vector<int> padsize;           // resize
    pre_img = preprocess_img(src, INPUT_H, INPUT_W, padsize);
    int new_h = padsize[0],  new_w = padsize[1],  padh = padsize[2],  padw = padsize[3];
    float ratio_h = (float)src.rows / new_h;
    float ratio_w = (float)src.cols / new_w;

    // [1, 3, INPUT_H, INPUT_W]
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row)
    {
        uchar* uc_pixel = pre_img.data + row * pre_img.step;
        // pre_img.step = width * 3  每一行有width个3通道的值
        for (int col = 0; col < INPUT_W; ++col)
        {
            data[i] = (float)uc_pixel[2] /255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        } 
    }

    // std::cout << "debug1" << std::endl;

    // runtime
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // inference
    static float prob[OUTPUT_SIZE];

    // 计算10次的推理速度
    // for (int i = 0; i < 10; i++)
    // {
    //     auto start = std::chrono::system_clock::now();
    //     doInfer_Det(*context, data, prob, 1);
    //     auto end =std::chrono::system_clock::now();
    //     std::cout << "10次推理: " << std::chrono::duration_cast <std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // }

    auto start = std::chrono::system_clock::now();
    doInfer_Det(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << "推理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // display result
    std::vector<int> classIds;                  // id数组
    std::vector<float> confidences;            // id对应的置信度数组
    std::vector<cv::Rect> boxes;                // 每个id对应的锚框

    // back process
    int net_length = CLASSES + 4;
    cv::Mat out =  cv::Mat(net_length, Num_box, CV_32F, prob);

    start = std::chrono::system_clock::now();
    
    // std::cout << "debug2" << std::endl;

    // 输出：1*net_length*Num_box;     
    // 所以每个box的属性是每隔Num_box取一个值，共net_length个值
    for (int i = 0; i < Num_box; i++)
    {   
        // std::cout << "debug-----0" << std::endl;
        // cv::Mat scores = out(Rect(i, 0, 1, CLASSES)).clone();
        cv::Mat scores = out(Rect(i, 4, 1, CLASSES)).clone();
        // std::cout << "debug-----1" << std::endl;
        Point classIdPoint;
        double max_class_score;
        // std::cout << "debug-----2" << std::endl;

        minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
        max_class_score = (float)max_class_score;
        
        if (max_class_score >= CONF_THRESHOLD) {
            float x = (out.at<float>(0, i) - padw) * ratio_w;  
            float y = (out.at<float>(1, i) - padh) * ratio_h;
            float w =  out.at<float>(2, i) * ratio_w;
            float h =  out.at<float>(3, i) * ratio_h;
            int left = MAX((x - 0.5 * w), 0);
            int  top = MAX((y - 0.5 * h), 0);
            int  width = (int)w;
            int height = (int)h;
            if (width <= 0 || height <= 0) {
                continue;
            }
            classIds.push_back(classIdPoint.y);
            confidences.push_back(max_class_score);
            boxes.push_back(Rect(left, top, width, height));
        }
    }

    // std::cout << "debug3" << std::endl;

    // NMS 执行非最大抑制以消除具有较低置信度的冗余重叠框
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<OutputDet> output;

    // std::cout << "debug4" << std::endl;

    Rect holeImgRect(0, 0, src.cols, src.rows);
    for (int i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        OutputDet result;
        result.id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx]& holeImgRect;
        output.push_back(result);
    }

    end = std::chrono::system_clock::now();
    std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    DrawPred_Det(src, output);
    cv::imshow("output.jpg", src);
    char c = cv::waitKey(0);

    // Destory the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    system("pause");

    return 0;
}

