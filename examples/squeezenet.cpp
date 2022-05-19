
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <cstdio>
#include <vector>
#include <benchmark.h>
#include <thread>

int detect_run(std::vector<cv::Mat>& bgrs)
{
    ncnn::Net squeezenet;

    squeezenet.opt.use_vulkan_compute = false;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    squeezenet.load_param("../../examples/vgg16-12.param");
    squeezenet.load_model("../../examples/vgg16-12.bin");

//    squeezenet.load_param("../../examples/squeezenet_v1.1.param");
//    squeezenet.load_model("../../examples/squeezenet_v1.1.bin");

//    squeezenet.load_param("../../examples/unet.param"); // 13.74 GMacs
//    squeezenet.load_model("../../examples/unet.bin");

    double total_time = 0.0;
    bool first = true;
    for (int c=0; c<bgrs.size() - 1; c++){
        auto bgr = bgrs[c];
//        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 256, 256);
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
//        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 28, 28);
//        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 32, 32);

//        const float mean_vals[3] = {104.f, 117.f, 123.f};
//        const float mean_vals[3] = {1.f, 1.f, 1.f};
//        const float std_vals[3] = {1/255.f, 1/255.f, 1/255.f};
//        in.substract_mean_normalize(mean_vals, 0);
//        in.substract_mean_normalize(mean_vals, std_vals);
        ncnn::Extractor ex = squeezenet.create_extractor();
        ex.input("data", in);
//        ex.input("input.1", in);


        ncnn::Mat out;
        double start = ncnn::get_current_time();
//        ex.extract("prob", out);
        ex.extract("vgg0_dense2_fwd", out);
//        ex.extract("onnx::Sigmoid_185", out);


        double end = ncnn::get_current_time();
//        fprintf(stderr, "%f\n", end - start);
//        if (c!=0){
            if (first){
                total_time = end - start;
                first = false;
            } else{
                total_time = total_time * c / (c+1) + (end - start)/(c+1);
            }
//        }

//        fprintf(stderr, "\n");
         fprintf(stderr, " %f -> %f\n",  end - start, total_time);
         ex.clear();
         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
//    fprintf(stderr, " %.2f\n", total_time);

//    sleep for 3s
    squeezenet.clear();

    return 0;
}

int main(int argc, char** argv)
{
//    std::string temp;
//    temp.c_str()

    std::vector<std::string> video_dataset = {
//    "../../images/moving_change_video.mp4",
//            "../../images/tiff_ct.mp4",
//        "../../images/all_number.mp4",
//        "../../images/all_black_video.mp4",
        "../../images/all_imagenet_100_images.mp4",
//        "../../images/all_cifar_1000_images.mp4",
//        "../../images/cts/BreastMRI_ct.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/WindmillFarm/WindmillFarm_static_cam_19.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/FallingTrees/FallingTrees_static_cam_1.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Fountain/Fountain_static_cam_10.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/WavingFlags/WavingFlags_static_cam_2.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Ocean/Ocean_static_cam_2.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/BuildingCollapse/BuildingCollapse_static_cam_3.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Escalator/Escalator_static_cam_30.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Waterfall/Waterfall_static_cam_9.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Highway/Highway_static_cam_23.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Fireworks/Fireworks_static_cam_5.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Railway/Railway_static_cam_30.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Marathon/Marathon_static_cam_27.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/LightningStorm/LightningStorm_static_cam_14.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/ForestFire/ForestFire_static_cam_28.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/SkyClouds/SkyClouds_static_cam_2.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Street/Street_static_cam_26.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Snowing/Snowing_static_cam_24.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Beach/Beach_static_cam_12.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/Elevator/Elevator_static_cam_23.mp4",
//        "/Users/kr/Downloads/YUP++/camera_stationary/RushingRiver/RushingRiver_static_cam_6.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/FallingTrees/FallingTrees_moving_cam_10.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/WindmillFarm/WindmillFarm_moving_cam_10.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Fountain/Fountain_moving_cam_21.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/WavingFlags/WavingFlags_moving_cam_1.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Ocean/Ocean_moving_cam_13.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/BuildingCollapse/BuildingCollapse_moving_cam_3.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Escalator/Escalator_moving_cam_6.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Waterfall/Waterfall_moving_cam_10.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Highway/Highway_moving_cam_11.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Fireworks/Fireworks_moving_cam_18.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Railway/Railway_moving_cam_23.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Marathon/Marathon_moving_cam_4.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/LightningStorm/LightningStorm_moving_cam_6.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/ForestFire/ForestFire_moving_cam_17.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/SkyClouds/SkyClouds_moving_cam_19.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Street/Street_moving_cam_16.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Snowing/Snowing_moving_cam_22.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Beach/Beach_moving_cam_4.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/Elevator/Elevator_moving_cam_23.mp4",
//        "/Users/kr/Downloads/YUP++/camera_moving/RushingRiver/RushingRiver_moving_cam_10.mp4"
    };


    for (const auto& video_name: video_dataset){
        fprintf(stderr, "%s\n", video_name.c_str());
        std::vector<cv::Mat> ms;
        cv::VideoCapture capture;
//        cv::Mat frame;
    //    capture.open("../../images/all_black_video.mp4");
    //    capture.open("../../images/all_number.mp4");

    //    capture.open("../../images/cts/BreastMRI_ct.mp4");
    //    capture.open("../../images/all_imagenet_mini.mp4");
    //    capture.open("../../images/cts/Head_ct.mp4");
    //    capture.open("/Users/kr/Downloads/YUP++/camera_moving/Fireworks/Fireworks_moving_cam_27.mp4");
    //    capture.open("/Users/kr/Downloads/YUP++/camera_stationary/Elevator/Elevator_static_cam_1.mp4");
        capture.open(video_name);
        if(!capture.isOpened())
        {
            printf("can not open ...\n");
            return -1;

        }

        int c = 0;
        while (1)
        {
            ms.emplace_back(cv::Mat());
            capture.read(ms[c]);
            if (ms[c].empty())
                break;
//            cv::imshow("w", ms[c]);
//            cv::waitKey(0); // waits to display frame
            c += 1;
        }

//        ms.erase(ms.begin());
//        ms.erase(ms.begin());

        detect_run(ms);
        capture.release();
        ms.clear();


    }
    return 0;
}