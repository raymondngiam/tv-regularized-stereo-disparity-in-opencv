#ifndef CV_TESTHELPER_H
#define CV_TESTHELPER_H

#include <opencv2/opencv.hpp>

inline size_t getCvMatSize(const cv::Mat &mat) {
    return mat.rows * mat.cols * mat.elemSize();
}

inline bool cvTestAllExact(const cv::Mat &a, const cv::Mat &b) {
    auto sizeA = getCvMatSize(a);
    auto sizeB = getCvMatSize(b);
    assert(sizeA == sizeB && "Mat size should be equal.");
    return 0 == std::memcmp(a.data, b.data, sizeA);
}

inline bool cvTestAllClose(const cv::Mat &a, const cv::Mat &b, float epsilon = 1e-6) {
    auto sizeA = getCvMatSize(a);
    auto sizeB = getCvMatSize(b);
    assert(sizeA == sizeB && "Mat size should be equal.");
    cv::Mat aFloat, bFloat;
    a.convertTo(aFloat, CV_32F);
    b.convertTo(bFloat, CV_32F);
    aFloat = aFloat.reshape(1, 1);
    bFloat = bFloat.reshape(1, 1);
    cv::Mat all_diff = cv::abs(aFloat - bFloat);
    cv::Mat thres;
    cv::threshold(all_diff, thres, epsilon, 1.0, cv::ThresholdTypes::THRESH_BINARY);
    cv::Mat nonZero;
    cv::findNonZero(thres, nonZero);
    bool isEqual = nonZero.rows == 0;
    if (!isEqual) {
        std::cout << nonZero.rows << " elements are with difference > " << epsilon << std::endl;
    }
    return isEqual;
}


#endif //CV_TESTHELPER_H
