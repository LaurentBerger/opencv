#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h          | | Print help message. }"
    "{ @alias           | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo              | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i          | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ initial_width    | 0 | Preprocess input image by initial resizing to a specific width.}"
    "{ initial_height   | 0 | Preprocess input image by initial resizing to a specific height.}"
    "{ std              | 0.0 0.0 0.0 | Preprocess input image by dividing on a standard deviation.}"
    "{ crop             | false | Preprocess input image by center cropping.}"
    "{ framework f      | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ needSoftmax      | false | Use Softmax to post-process the output of the net.}"
    "{ classes          | | Optional path to a text file with names of classes. }"
    "{ backend          | 0 | Choose one of computation backends: "
                            "0: automatically (by default), "
                            "1: Halide language (http://halide-lang.org/), "
                            "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "3: OpenCV implementation, "
                            "4: VKCOM, "
                            "5: CUDA, "
                            "6: WebNN }"
    "{ target           | 0 | Choose one of target computation devices: "
                            "0: CPU target (by default), "
                            "1: OpenCL, "
                            "2: OpenCL fp16 (half-float precision), "
                            "3: VPU, "
                            "4: Vulkan, "
                            "6: CUDA, "
                            "7: CUDA fp16 (half-float preprocess) }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    {
        Net netNihil;
        std::vector< String > inputBlobNames = {"inpa", "inpb" };
        std::vector<int> sza = { 1, 2 };
        LayerParams params;
        // netNihil.addLayer(inputBlobNames[0], "__NetInputLayer__", params);
        params.blobs = { Mat(sza, CV_32FC1) };
        netNihil.addLayer(inputBlobNames[0], "Const", params);
        params.blobs = { Mat(sza, CV_32FC1) };
        netNihil.addLayer(inputBlobNames[1], "Const", params);
        netNihil.setInputsNames(inputBlobNames);
        netNihil.setInputShape(inputBlobNames[0], MatShape(sza));
        netNihil.setInputShape(inputBlobNames[1], MatShape(sza));
        params.blobs = { Mat(sza, CV_32FC1), Mat(sza, CV_32FC1) };

        netNihil.addLayer("Concat1", "Concat", params);
        int idInpa = netNihil.getLayerId(inputBlobNames[0]);
        int idInpb = netNihil.getLayerId(inputBlobNames[1]);
        int idAdd1 = netNihil.getLayerId("Concat1");
        netNihil.connect(inputBlobNames[0], "Concat1");
        netNihil.connect(idInpb, 0, idAdd1, 1);
        netNihil.setInput(Mat(sza, CV_32FC1), inputBlobNames[0]);
        netNihil.setInput(Mat(sza, CV_32FC1), inputBlobNames[1]);
        netNihil.forward();
        std::string texte = netNihil.dump();
    }
    {
        Net netNihil;
        std::vector< String > inputBlobNames = { "inpx", "inpa", "inpb" };
        std::vector<int> szx = { 3, 2 };
        std::vector<int> sza = { 1, 2 };
        LayerParams params;
        params.blobs = { Mat(szx, CV_32FC1) };
        // netNihil.addLayer(inputBlobNames[0], "__NetInputLayer__", params);
        netNihil.addLayer(inputBlobNames[0], "Const", params);
        params.blobs = { Mat(sza, CV_32FC1) };
        netNihil.addLayer(inputBlobNames[1], "Const", params);
        params.blobs = { Mat(sza, CV_32FC1) };
        netNihil.addLayer(inputBlobNames[2], "Const", params);
        netNihil.setInputsNames(inputBlobNames);
        netNihil.setInputShape(inputBlobNames[0], MatShape(szx));
        netNihil.setInputShape(inputBlobNames[1], MatShape(sza));
        netNihil.setInputShape(inputBlobNames[2], MatShape(sza));
        params.blobs = { Mat(sza, CV_32FC1), Mat(sza, CV_32FC1) };

        params.set<int>("axis", 0);
        netNihil.addLayer("Concat1", "Concat", params);
        int idInpx = netNihil.getLayerId(inputBlobNames[0]);
        int idInpa = netNihil.getLayerId(inputBlobNames[1]);
        int idInpb = netNihil.getLayerId(inputBlobNames[2]);
        int idConcat1 = netNihil.getLayerId("Concat1");
        netNihil.connect(inputBlobNames[0], "Concat1");
        netNihil.connect(idInpa,0, idConcat1, 1);
        netNihil.setInput(Mat(szx, CV_32FC1), inputBlobNames[0]);
        netNihil.setInput(Mat(sza, CV_32FC1), inputBlobNames[1]);
        netNihil.setInput(Mat(sza, CV_32FC1), inputBlobNames[2]);
        netNihil.forward();
        std::string texte=netNihil.dump();
    }
    {
        Net netAdd = readNet("c:/users/laurent/desktop/onnx_test/test_add.onnx");
        std::vector<int> szx = { 1, 4, 2 ,2 };
        Mat inpx(szx, CV_32FC1, Scalar::all(1));
        netAdd.setInput(inpx, "inpx");
        //        Mat c = netAdd.forward();
    }
    {
        Net netPad = readNet("c:/users/laurent/desktop/onnx_test/test_pad.onnx");
        Mat inpx = (Mat_<float>(3, 2) << 10, 11, 12, 13, 14, 15);
        Mat inpa = (Mat_<float>(2, 1) << 1, 2);
        Mat inpb = (Mat_<float>(2, 1) << 3, 1);
        netPad.setInput(inpx, "inpx");
        netPad.setInput(inpa, "inpa");
        netPad.setInput(inpb, "inpb");
        Mat c = netPad.forward();
    }
    {
        // Net netAdd = readNet("C:/Users/laurent/Downloads/saved_model.pb");
    }



    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int rszWidth = parser.get<int>("initial_width");
    int rszHeight = parser.get<int>("initial_height");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    Scalar std = parser.get<Scalar>("std");
    bool swapRB = parser.get<bool>("rgb");
    bool crop = parser.get<bool>("crop");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findFile(parser.get<String>("model"));
    String config = findFile(parser.get<String>("config"));
    String framework = parser.get<String>("framework");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");
    bool needSoftmax = parser.get<bool>("needSoftmax");
    std::cout<<"mean: "<<mean<<std::endl;
    std::cout<<"std: "<<std<<std::endl;

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);
    //! [Open a video file or an image file or a camera stream]

    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        if (rszWidth != 0 && rszHeight != 0)
        {
            resize(frame, frame, Size(rszWidth, rszHeight));
        }

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);

        // Check std values.
        if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
        {
            // Divide blob by std.
            divide(blob, std, blob);
        }
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
        // double t_sum = 0.0;
        // double t;
        int classId;
        double confidence;
        cv::TickMeter timeRecorder;
        timeRecorder.reset();
        Mat prob = net.forward();
        double t1;
        timeRecorder.start();
        prob = net.forward();
        timeRecorder.stop();
        t1 = timeRecorder.getTimeMilli();

        timeRecorder.reset();
        for(int i = 0; i < 200; i++) {
            //! [Make forward pass]
            timeRecorder.start();
            prob = net.forward();
            timeRecorder.stop();

            //! [Get a class with a highest score]
            Point classIdPoint;
            minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            classId = classIdPoint.x;
            //! [Get a class with a highest score]

            // Put efficiency information.
            // std::vector<double> layersTimes;
            // double freq = getTickFrequency() / 1000;
            // t = net.getPerfProfile(layersTimes) / freq;
            // t_sum += t;
        }
        if (needSoftmax == true)
        {
            float maxProb = 0.0;
            float sum = 0.0;
            Mat softmaxProb;

            maxProb = *std::max_element(prob.begin<float>(), prob.end<float>());
            cv::exp(prob-maxProb, softmaxProb);
            sum = (float)cv::sum(softmaxProb)[0];
            softmaxProb /= sum;
            Point classIdPoint;
            minMaxLoc(softmaxProb.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            classId = classIdPoint.x;
        }
        std::string label = format("Inference time of 1 round: %.2f ms", t1);
        std::string label2 = format("Average time of 200 rounds: %.2f ms", timeRecorder.getTimeMilli()/200);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        putText(frame, label2, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        // Print predicted class.
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
        putText(frame, label, Point(0, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}
