#include <iostream>
#include <opencv2/opencv.hpp>
#include "gtest/gtest.h"
#include "cvMatUtils.h"
#include "cv_testHelper.h"

TEST(CVTests, TransposeTest) {
    constexpr int channel = 3;
    constexpr int rows = 2;
    constexpr int cols = 2;
    float data[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    cv::Mat im(rows, cols, CV_32FC(channel), data);
    // input
    // [1, 2, 3, 4, 5, 6;
    //  7, 8, 9, 10, 11, 12]

    cv::Mat im_transposed(channel, rows, CV_32FC(cols));
    transpose_DWH_to_WHD<float, 1, rows, cols, channel>((float (*)[rows][cols][channel]) im.data,
                                                        (float (*)[channel][rows][cols]) im_transposed.data);
    float data_gt[12]{1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
    cv::Mat im_gt(channel, rows, CV_32FC(cols), data_gt);
    // expected output
    // [1, 4, 7, 10;
    //  2, 5, 8, 11;
    //  3, 6, 9, 12]
    EXPECT_TRUE(cvTestAllClose(im_transposed, im_gt));
}

TEST(CVTests, TransposeTest2) {
    constexpr int channel = 4;
    constexpr int rows = 2;
    constexpr int cols = 3;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    std::vector<cv::Mat> splits;
    cv::split(im, splits);
    splits[0].setTo(1);
    cv::merge(splits, im);
    // input
    // [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0;
    //  1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    cv::Mat im_transposed(rows, channel, CV_32FC(cols));
    transpose_DWH_to_WDH<float, 1, rows, cols, channel>((float (*)[rows][cols][channel]) im.data,
                                                        (float (*)[rows][channel][cols]) im_transposed.data);

    float data_gt[rows * cols * channel]{1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    cv::Mat im_gt(channel, rows, CV_32FC(cols), data_gt);
    // expected output
    // [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    EXPECT_TRUE(cvTestAllClose(im_transposed, im_gt));
}

TEST(CVTests, Gradient3DTest) {
    constexpr int channel = 3;
    constexpr int rows = 5;
    constexpr int cols = 5;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im.col(2).setTo(1);
    // input
    // [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    cv::Mat grads;
    derivate3D<rows, cols, channel>(im, grads);

    // expected output
    // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cv::Mat grads_gt = cv::Mat::zeros(rows, cols, CV_32FC(3 * channel));
    grads_gt.col(1).setTo(0.5);
    grads_gt.col(3).setTo(-0.5);
    std::vector<cv::Mat> splits;
    cv::split(grads_gt, splits);
    splits[3].setTo(0);
    splits[4].setTo(0);
    splits[5].setTo(0);;
    splits[6].setTo(0);
    splits[7].setTo(0);
    splits[8].setTo(0);
    cv::merge(splits, grads_gt);
    EXPECT_TRUE(cvTestAllClose(grads, grads_gt));
}

TEST(CVTests, Gradient3DTest0) {
    constexpr int channel = 3;
    constexpr int rows = 5;
    constexpr int cols = 5;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im.row(2).setTo(1);
    // input
    // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cv::Mat grads;
    derivate3D<rows, cols, channel>(im, grads);
    // expected output
    // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, -0.5, -0.5, -0.5, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cv::Mat grads_gt = cv::Mat::zeros(rows, cols, CV_32FC(3 * channel));
    grads_gt.row(1).setTo(0.5);
    grads_gt.row(3).setTo(-0.5);
    std::vector<cv::Mat> splits;
    cv::split(grads_gt, splits);
    splits[0].setTo(0);
    splits[1].setTo(0);
    splits[2].setTo(0);
    splits[6].setTo(0);
    splits[7].setTo(0);
    splits[8].setTo(0);
    cv::merge(splits, grads_gt);
    EXPECT_TRUE(cvTestAllClose(grads, grads_gt));
}

TEST(CVTests, Gradient3DTest1) {
    constexpr int channel = 3;
    constexpr int rows = 3;
    constexpr int cols = 3;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im.col(1).setTo(1);
    // input
    // [0, 0, 0, 1, 1, 1, 0, 0, 0;
    //  0, 0, 0, 1, 1, 1, 0, 0, 0;
    //  0, 0, 0, 1, 1, 1, 0, 0, 0]

    cv::Mat grads;
    derivate3D<rows, cols, channel>(im, grads);

    // expected output
    // [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0;
    //  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0;
    //  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0]

    cv::Mat grads_gt = cv::Mat::zeros(rows, cols, CV_32FC(3 * channel));
    grads_gt.col(0).setTo(1);
    grads_gt.col(2).setTo(-1);
    std::vector<cv::Mat> splits;
    cv::split(grads_gt, splits);
    splits[3].setTo(0);
    splits[4].setTo(0);
    splits[5].setTo(0);
    splits[6].setTo(0);
    splits[7].setTo(0);
    splits[8].setTo(0);
    cv::merge(splits, grads_gt);
    EXPECT_TRUE(cvTestAllClose(grads, grads_gt));
}

TEST(CVTests, ClipTest) {
    constexpr int channel = 3;
    constexpr int rows = 2;
    constexpr int cols = 2;
    float data[12]{-2, -3, -4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10, 11, 12};
    cv::Mat im(rows, cols, CV_32FC(channel), data);

    // input
    // [-2, -3, -4, 0.4, 0.5, 0.6;
    //  0.7, 0.8, 0.9, 10, 11, 12]

    cv::Mat clipped;
    clip<float>(im, clipped, 0.f, 1.f);

    // expected output
    // [0, 0, 0, 0.4, 0.5, 0.6;
    //  0.7, 0.8, 0.9, 1, 1, 1]
    float data_gt[rows * cols * channel]{0, 0, 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1};
    cv::Mat clip_gt(rows, cols, CV_32FC(channel), data_gt);
    EXPECT_TRUE(cvTestAllClose(clipped, clip_gt));
}

TEST(CVTests, NormTest) {
    constexpr int channel = 2;
    constexpr int rows = 2;
    constexpr int cols = 2;
    float data[8]{3, 4, 3, 4, 3, 4, 3, 4};
    cv::Mat im(rows, cols, CV_32FC(channel), data);

    // input
    // [3, 4, 3, 4;
    //  3, 4, 3, 4]

    cv::Mat norm;
    normL2<channel>(im, norm);

    // expected output
    // [5, 5;
    //  5, 5]

    float data_gt[rows * cols * channel]{5, 5, 5, 5};
    cv::Mat norm_gt(rows, cols, CV_32FC(1), data_gt);
    EXPECT_TRUE(cvTestAllClose(norm, norm_gt));
}

TEST(CVTests, ReduceChannelTest) {
    constexpr int channel = 6;
    constexpr int rows = 2;
    constexpr int cols = 2;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im.at<cv::Vec<float, channel>>(0, 0) = cv::Vec<float, channel>{0, 1, 1, 0, 1, 1};
    im.at<cv::Vec<float, channel>>(1, 1) = cv::Vec<float, channel>{0, 1, 1, 0, 1, 1};

    // input
    // [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0;
    //  0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]

    cv::Mat reduced;
    reduceChannels(im, reduced, cv::REDUCE_SUM);

    // expected output
    // [4, 0;
    //  0, 4]
    float data_gt[rows * cols * channel]{4, 0, 0, 4};
    cv::Mat reduce_gt(rows, cols, CV_32FC(1), data_gt);
    EXPECT_TRUE(cvTestAllClose(reduced, reduce_gt));
}
