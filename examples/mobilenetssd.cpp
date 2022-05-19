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

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <benchmark.h>
#include <thread>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;

    mobilenet.opt.use_vulkan_compute = true;

    // model is converted from https://github.com/chuanqi305/MobileNet-SSD
    // and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    mobilenet.load_param("mobilenet_ssd_voc_ncnn.param");
    mobilenet.load_model("mobilenet_ssd_voc_ncnn.bin");

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static int detect_run(std::vector<cv::Mat> bgrs)
{
    ncnn::Net mobilenet;

    mobilenet.opt.use_vulkan_compute = true;

    // model is converted from https://github.com/chuanqi305/MobileNet-SSD
    // and can be downloaded from https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    mobilenet.load_param("../../examples/mobilenet_ssd_voc_ncnn.param");
    mobilenet.load_model("../../examples/mobilenet_ssd_voc_ncnn.bin");

    const int target_size = 300;
    double infer_time = 0.0;
    double start, end;

    for (int c =0; c*10 <= bgrs.size()-1; c++){
        //        double infer_time = 0;

        // scale to target detect size
        auto bgr = bgrs[c*10];

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

//        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
//        const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
//        in.substract_mean_normalize(mean_vals, norm_vals);
        const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
        //        const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f};
        in.substract_mean_normalize(mean_vals, 0);

        ncnn::Extractor ex = mobilenet.create_extractor();

        ex.input("data", in);

        ncnn::Mat out;

        start = ncnn::get_current_time();
        ex.extract("detection_out", out);
        end = ncnn::get_current_time();

        if (c!=0){
            infer_time += end - start;
            fprintf(stderr, " %.2f\n", infer_time / (c));
        }
        //    sleep for 3s
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

//    fprintf(stderr, "%.2f\n", infer_time / (bgrs.size()-2));


    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main2(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_mobilenet(m, objects);

    draw_objects(m, objects);


    return 0;
}
int main(int argc, char** argv)
{
    //    std::string temp;
    //    temp.c_str()

    std::vector<std::string> video_dataset = {
        "/Users/kr/Downloads/Marathon_moving_cam_1.mp4",
//        "/Users/kr/Downloads/Highway_static_cam_8.mp4",
//        "../../images/all_black_video.mp4"
//        "../../images/moving_change_video.mp4"
//                "../../images/all_coco_1000_images.mp4",
//        "../../images/all_biaoshi.mp4",
        //        "../../images/all_imagenet_mini.mp4",
        //        "../../images/cts/BreastMRI_ct.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/FallingTrees/FallingTrees_moving_cam_10.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/WindmillFarm/WindmillFarm_moving_cam_10.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Fountain/Fountain_moving_cam_21.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/WavingFlags/WavingFlags_moving_cam_1.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Ocean/Ocean_moving_cam_13.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/BuildingCollapse/BuildingCollapse_moving_cam_3.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/Escalator/Escalator_moving_cam_6.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Waterfall/Waterfall_moving_cam_10.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/Highway/Highway_moving_cam_11.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/Fireworks/Fireworks_moving_cam_18.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Railway/Railway_moving_cam_23.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Marathon/Marathon_moving_cam_4.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/LightningStorm/LightningStorm_moving_cam_6.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/ForestFire/ForestFire_moving_cam_17.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/SkyClouds/SkyClouds_moving_cam_19.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Street/Street_moving_cam_16.mp4",
//                "/Users/kr/Downloads/YUP++/camera_moving/Snowing/Snowing_moving_cam_22.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/Beach/Beach_moving_cam_4.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/Elevator/Elevator_moving_cam_23.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_moving/RushingRiver/RushingRiver_moving_cam_10.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_stationary/FallingTrees/FallingTrees_static_cam_1.mp4",
        //        "/Users/kr/Downloads/YUP++/camera_stationary/WindmillFarm/WindmillFarm_static_cam_19.mp4",
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
        //        "/Users/kr/Downloads/YUP++/camera_stationary/RushingRiver/RushingRiver_static_cam_6.mp4"
    };


    for (const auto& video_name: video_dataset){
        fprintf(stderr, "%s \n", video_name.c_str());
        std::vector<cv::Mat> ms;
        cv::VideoCapture capture;
        cv::Mat frame;
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
//                    cv::imshow("w", ms[c]);
//                    cv::waitKey(0); // waits to display frame
            c += 1;
        }

        ms.erase(ms.begin());
        ms.erase(ms.begin());
        detect_run(ms);
        capture.release();
        ms.clear();


    }
    return 0;
}
