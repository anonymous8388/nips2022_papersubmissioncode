// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <math.h>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <benchmark.h>
#include <thread>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static int detect_fasterrcnn(const cv::Mat& bgr)
{
    ncnn::Net fasterrcnn;

    fasterrcnn.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/rbgirshick/py-faster-rcnn
    // py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt
    // https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0
    // ZF_faster_rcnn_final.caffemodel
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    fasterrcnn.load_param("../../examples/ZF_faster_rcnn_final.param");
    fasterrcnn.load_model("../../examples/ZF_faster_rcnn_final.bin");

    double start = ncnn::get_current_time();
    double end = ncnn::get_current_time();
    double infer_time = 0;

    // hyper parameters taken from
    // py-faster-rcnn/lib/fast_rcnn/config.py
    // py-faster-rcnn/lib/fast_rcnn/test.py
    const int target_size = 600; // __C.TEST.SCALES

    const int max_per_image = 100;
    const float confidence_thresh = 0.05f;

    const float nms_threshold = 0.3f; // __C.TEST.NMS

    // scale to target detect size
    int w = bgr.cols;
    int h = bgr.rows;
    float scale = 1.f;
    if (w < h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, w, h);

    const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Mat im_info(3);
    im_info[0] = h;
    im_info[1] = w;
    im_info[2] = scale;

    // step1, extract feature and all rois
    ncnn::Extractor ex1 = fasterrcnn.create_extractor();

    ex1.input("data", in);
    ex1.input("im_info", im_info);

    ncnn::Mat conv5_relu5; // feature
    ncnn::Mat rois;        // all rois
    start = ncnn::get_current_time();
    ex1.extract("conv5_relu5", conv5_relu5);
    ex1.extract("rois", rois);
    end = ncnn::get_current_time();
    infer_time += end - start;

    // step2, extract bbox and score for each roi
    std::vector<std::vector<Object> > class_candidates;
    for (int i = 0; i < rois.c; i++)
    {
        ncnn::Extractor ex2 = fasterrcnn.create_extractor();

        ncnn::Mat roi = rois.channel(i); // get single roi
        ex2.input("conv5_relu5", conv5_relu5);
        ex2.input("rois", roi);

        ncnn::Mat bbox_pred;
        ncnn::Mat cls_prob;
        start = ncnn::get_current_time();
        ex2.extract("bbox_pred", bbox_pred);
        ex2.extract("cls_prob", cls_prob);
        end = ncnn::get_current_time();
        infer_time += end - start;

    }

    fprintf(stderr, "End-to-end time: %.4f\n", infer_time);

    return 0;
}


static int detect_run1(std::vector<cv::Mat> bgrs)
{
    ncnn::Net fasterrcnn;

    fasterrcnn.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/rbgirshick/py-faster-rcnn
    // py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt
    // https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0
    // ZF_faster_rcnn_final.caffemodel
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    fasterrcnn.load_param("../../examples/ZF_faster_rcnn_final.param");
    fasterrcnn.load_model("../../examples/ZF_faster_rcnn_final.bin");


    // hyper parameters taken from
    // py-faster-rcnn/lib/fast_rcnn/config.py
    // py-faster-rcnn/lib/fast_rcnn/test.py
    const int target_size = 600; // __C.TEST.SCALES
    double infer_time = 0.0;

    for (int c =0; c <bgrs.size()-1; c++){
        double start = ncnn::get_current_time();
        double end = ncnn::get_current_time();
//        double infer_time = 0;

        // scale to target detect size
        auto bgr = bgrs[c];

        int w = bgr.cols;
        int h = bgr.rows;
        float scale = 1.f;
        if (w < h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, w, h);

//        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
//        const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f};
        in.substract_mean_normalize(mean_vals, 0);

        ncnn::Mat im_info(3);
        im_info[0] = h;
        im_info[1] = w;
        im_info[2] = scale;

        // step1, extract feature and all rois
        ncnn::Extractor ex1 = fasterrcnn.create_extractor();

        ex1.input("data", in);
        ex1.input("im_info", im_info);

        ncnn::Mat conv5_relu5; // feature
        ncnn::Mat rois;        // all rois
        start = ncnn::get_current_time();
        ex1.extract("conv5_relu5", conv5_relu5);
        ex1.extract("rois", rois);
        end = ncnn::get_current_time();
        infer_time += end - start;
//
//        // step2, extract bbox and score for each roi
        std::vector<std::vector<Object> > class_candidates;
        for (int i = 0; i < rois.c; i++)
        {
            ncnn::Extractor ex2 = fasterrcnn.create_extractor();

            ncnn::Mat roi = rois.channel(i); // get single roi
            ex2.input("conv5_relu5", conv5_relu5);
            ex2.input("rois", roi);

            ncnn::Mat bbox_pred;
            ncnn::Mat cls_prob;
            start = ncnn::get_current_time();
            ex2.extract("bbox_pred", bbox_pred);
            ex2.extract("cls_prob", cls_prob);
            end = ncnn::get_current_time();
            infer_time += end - start;

        }
//        fprintf(stderr, "\n");
        fprintf(stderr, "\t%.2f\n", infer_time / (c+1));
    }
    fprintf(stderr, "%.2f\n", infer_time / (bgrs.size()-1));
    //    sleep for 3s
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    return 0;
}

static int detect_run2(std::vector<cv::Mat> bgrs)
{
    ncnn::Net fasterrcnn;

    fasterrcnn.opt.use_vulkan_compute = true;

    // original pretrained model from https://github.com/rbgirshick/py-faster-rcnn
    // py-faster-rcnn/models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt
    // https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0
    // ZF_faster_rcnn_final.caffemodel
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    fasterrcnn.load_param("../../examples/ZF_faster_rcnn_final.param");
    fasterrcnn.load_model("../../examples/ZF_faster_rcnn_final.bin");


    // hyper parameters taken from
    // py-faster-rcnn/lib/fast_rcnn/config.py
    // py-faster-rcnn/lib/fast_rcnn/test.py
    const int target_size = 600; // __C.TEST.SCALES
    double infer_time = 0.0;

    for (int c =0; c <bgrs.size(); c++){
        double start = ncnn::get_current_time();
        double end = ncnn::get_current_time();
        //        double infer_time = 0;

        // scale to target detect size
        auto bgr = bgrs[c];

        int w = bgr.cols;
        int h = bgr.rows;
        float scale = 1.f;
        if (w < h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, w, h);

//        const float mean_vals[3] = {0.f, 0.f, 0.f};
//        in.substract_mean_normalize(mean_vals, 0);

        const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f};
        in.substract_mean_normalize(mean_vals, 0);

        ncnn::Mat im_info(3);
        im_info[0] = h;
        im_info[1] = w;
        im_info[2] = scale;

        // step1, extract feature and all rois
        ncnn::Extractor ex1 = fasterrcnn.create_extractor();

        ex1.input("data", in);
        ex1.input("im_info", im_info);

        ncnn::Mat conv5_relu5; // feature
        ncnn::Mat rois;        // all rois
        start = ncnn::get_current_time();
        ex1.extract("conv5_relu5", conv5_relu5);
        ex1.extract("rois", rois);
        end = ncnn::get_current_time();
        infer_time += end - start;
        //
        //        // step2, extract bbox and score for each roi
        std::vector<std::vector<Object> > class_candidates;
        for (int i = 0; i < rois.c; i++)
        {
            ncnn::Extractor ex2 = fasterrcnn.create_extractor();

            ncnn::Mat roi = rois.channel(i); // get single roi
            ex2.input("conv5_relu5", conv5_relu5);
            ex2.input("rois", roi);

            ncnn::Mat bbox_pred;
            ncnn::Mat cls_prob;
            start = ncnn::get_current_time();
            ex2.extract("bbox_pred", bbox_pred);
            ex2.extract("cls_prob", cls_prob);
            end = ncnn::get_current_time();
            infer_time += end - start;

        }
        //        fprintf(stderr, "\n");
        fprintf(stderr, "\t%.2f\n", infer_time / (c+1));
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
//    fprintf(stderr, "%.2f\n", infer_time / (bgrs.size()-1));
    //    sleep for 3s

    return 0;
}

int main1(int argc, char** argv)
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

//    std::vector<Object> objects;
//    detect_fasterrcnn(m, objects);

//    draw_objects(m, objects);

    return 0;
}


int main2(int argc, char** argv)
{
    std::vector<cv::Mat> ms;
    cv::VideoCapture capture;
    cv::Mat frame;
//    capture.open("../../images/all_black_video.mp4");
//    capture.open("../../images/all_number.mp4");
//        capture.open("../../images/all_imagenet_mini.mp4");
//        capture.open("../../images/cts/BreastMRI_ct.mp4");
        capture.open("/Users/kr/Downloads/YUP++/camera_moving/WindmillFarm/WindmillFarm_moving_cam_10.mp4");
    if(!capture.isOpened()){
        printf("can not open ...\n");
        return -1;

    }

    int c = 0;
    while (1){
        ms.emplace_back(cv::Mat());
        capture.read(ms[c]);
        if (ms[c].empty())
            break;
        //        cv::imshow("w", ms[c]);
        //        cv::waitKey(0); // waits to display frame
        c += 1;
    }

    detect_run1(ms);
    capture.release();
    ms.clear();

    return 0;
}


int main(int argc, char** argv)
{
    //    std::string temp;
    //    temp.c_str()

    std::vector<std::string> video_dataset = {
//        "../../images/moving_change_video.mp4"
//        "../../images/all_black_video.mp4",
        "../../images/all_coco_1000_images.mp4",
//        "../../images/all_biaoshi.mp4",
//        "../../images/all_imagenet_mini.mp4",
//        "../../images/cts/BreastMRI_ct.mp4",





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
            //        cv::imshow("w", ms[c]);
            //        cv::waitKey(0); // waits to display frame
            c += 1;
        }

//        fprintf(stderr, "run1\n");
//        detect_run1(ms);
//        ms.erase(ms.begin());
//        ms.erase(ms.begin());

//        fprintf(stderr, "run2\n");
        detect_run2(ms);
        capture.release();
        ms.clear();


    }
    return 0;
}