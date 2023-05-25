#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "include/loguru.hpp"
#include "include/detector.h"
using namespace cv;
using namespace std;
/* main */

int main(int argc, char *argv[])
{
    // 默认参数
//    string model_path = argv[1];
//    string img_path = argv[2];
    string model_path = "D:/work/AI/python/yolov5-7.0/fls700Ds.onnx";
    string img_path = "D:/FFOutput/image_12269.jpg";
    loguru::init(argc, argv);
    Config config = {0.25f, 0.45f, model_path, "D:/work/AI/python/yolov5-7.0/datasets/fls700D/predefined_classes.txt", Size(640, 640),false,true};
    LOG_F(INFO,"Start main process");
    Detector detector(config);
    LOG_F(INFO,"Load model done ..");
    Mat img = imread(img_path, IMREAD_COLOR);

    LOG_F(INFO,"Read image from %s", img_path.c_str());
    Detection detection = detector.detect(img);
    LOG_F(INFO,"Detect process finished");
    Colors cl = Colors();
    detector.postProcess(img, detection,cl);

    cv::Mat dst = img.rowRange(160,480);
    cv::resize(dst,dst,cv::Size(1000,500));
    imshow("rst", dst);
    waitKey(0);
    LOG_F(INFO,"Post process done save image to assets/output.jpg");
    imwrite("D:/FFOutput/output.jpg", img);
    cout << "detect Image And Save to assets/output.jpg" << endl;
    return 0;
}
