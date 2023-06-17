#include <iostream>
#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include "primalDualStereo.h"
#include "cv_testHelper.h"

TEST(PrimalDualStereoTests, ProjectionTest1) {
    constexpr int channel = 2;
    constexpr int rows = 2;
    constexpr int cols = 3;
    float data[12]{3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4};
    cv::Mat im(rows, cols, CV_32FC(channel), data);
    // input
    // [3, 4, 3, 4, 3, 4;
    //  3, 4, 3, 4, 3, 4]

    cv::Mat norm, result;
    normL2<channel>(im, norm);
    cv::Mat scale = 2 * cv::Mat::ones(rows, cols, CV_32FC(1));
    projection(im, norm, scale, result);

    float data_gt[rows * cols * channel]{1.2, 1.6, 1.2, 1.6, 1.2, 1.6, \
                                     1.2, 1.6, 1.2, 1.6, 1.2, 1.6};
    cv::Mat im_gt(channel, rows, CV_32FC(cols), data_gt);
    // expected output
    // [1.2, 1.6, 1.2, 1.6, 1.2, 1.6;
    //  1.2, 1.6, 1.2, 1.6, 1.2, 1.6]
    EXPECT_TRUE(cvTestAllClose(result, im_gt));
}

TEST(PrimalDualStereoTests, ProjectionSingleTest) {
    constexpr int channel = 1;
    constexpr int rows = 2;
    constexpr int cols = 3;
    float data[6]{3, 4, 3, 4, 3, 4};
    cv::Mat im(rows, cols, CV_32FC(channel), data);
    // input
    // [3, 4, 3;
    //  4, 3, 4]

    cv::Mat norm, result;
    norm = cv::abs(im);
    cv::Mat scale = 3.5 * cv::Mat::ones(rows, cols, CV_32FC(1));
    projectionSingle(im, norm, scale, result);

    float data_gt[rows * cols * channel]{3, 3.5, 3, \
                                     3.5, 3, 3.5};
    cv::Mat im_gt(channel, rows, CV_32FC(cols), data_gt);
    // expected output
    // [3, 3.5, 3;
    //  3,5, 3, 3.5]
    EXPECT_TRUE(cvTestAllClose(result, im_gt));

}

TEST(PrimalDualStereoTestsTests, WarpTest) {
    constexpr int channel = 1;
    constexpr int rows = 5;
    constexpr int cols = 6;
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im.col(3) = 1;
    // input
    // [0, 0, 0, 1, 0, 0;
    //  0, 0, 0, 1, 0, 0;
    //  0, 0, 0, 1, 0, 0;
    //  0, 0, 0, 1, 0, 0;
    //  0, 0, 0, 1, 0, 0]

    cv::Mat disparity = 1 * cv::Mat::ones(rows, cols, CV_32FC(channel));
    cv::Mat warped;
    warp<rows, cols, channel>(im, disparity, warped);

    cv::Mat im_gt = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    im_gt.col(2).setTo(1);
    // expected output
    // [0, 0, 1, 0, 0, 0;
    //  0, 0, 1, 0, 0, 0;
    //  0, 0, 1, 0, 0, 0;
    //  0, 0, 1, 0, 0, 0;
    //  0, 0, 1, 0, 0, 0]
    EXPECT_TRUE(cvTestAllClose(warped, im_gt));
}