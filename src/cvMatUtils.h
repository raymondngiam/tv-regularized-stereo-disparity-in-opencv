#ifndef CVMATUTILS_H
#define CVMATUTILS_H

#include <execution>

#define OMP_THREAD 32

template<typename T, int dim0, int dim1, int dim2, int dim3>
inline void transpose_DWH_to_WDH(const T src[dim0][dim1][dim2][dim3], T dst[dim0][dim1][dim3][dim2]) {
#pragma omp parallel for num_threads(OMP_THREAD)
    for (int i0 = 0; i0 < dim0; i0++) {
        for (int i1 = 0; i1 < dim1; i1++) {
            for (int i2 = 0; i2 < dim2; i2++) {
#pragma omp simd
                for (int i3 = 0; i3 < dim3; i3++) {
                    dst[i0][i1][i3][i2] = src[i0][i1][i2][i3];
                }
            }
        }
    }
}

template<int rows, int cols, int channels>
inline void derivate3D(const cv::Mat &im, cv::Mat &grads_stack) {
    assert(rows > 2 && cols > 2 && channels > 2 && "Each dimension must have >= 3 elements");
    cv::Mat gradX, gradY;
    float d_kernelX[3]{-0.5, 0, 0.5};
    cv::Mat kernelX(1, 3, CV_32F, d_kernelX);
    cv::filter2D(im, gradX, CV_32F, kernelX, cv::Point2i(-1, 0));
    gradX.col(0) = im.col(1) - im.col(0);
    gradX.col(cols - 1) = im.col(cols - 1) - im.col(cols - 2);

    float d_kernelY[3]{-0.5, 0, 0.5};
    cv::Mat kernelY(3, 1, CV_32F, d_kernelY);
    cv::filter2D(im, gradY, CV_32F, kernelY, cv::Point2i(0, -1));
    gradY.row(0) = im.row(1) - im.row(0);
    gradY.row(rows - 1) = im.row(rows - 1) - im.row(rows - 2);

    cv::Mat tmp(rows, channels, CV_32FC(cols));
    cv::Mat tmp2;
    cv::Mat gradZ(rows, cols, CV_32FC(channels));
    transpose_DWH_to_WDH<float, 1, rows, cols, channels>((float (*)[rows][cols][channels]) im.data,
                                                         (float (*)[rows][channels][cols]) tmp.data);
    cv::filter2D(tmp, tmp2, CV_32F, kernelX, cv::Point2i(-1, 0));
    tmp2.col(0) = tmp.col(1) - tmp.col(0);
    tmp2.col(channels - 1) = tmp.col(channels - 1) - tmp.col(channels - 2);
    transpose_DWH_to_WDH<float, 1, rows, channels, cols>((float (*)[rows][channels][cols]) tmp2.data,
                                                         (float (*)[rows][cols][channels]) gradZ.data);

    std::vector<cv::Mat> grads;
    grads.emplace_back(gradX);
    grads.emplace_back(gradY);
    grads.emplace_back(gradZ);
    cv::merge(grads, grads_stack);
}

template<typename T>
inline void clip(const cv::Mat &im, cv::Mat &clipped, T min, T max) {
    im.copyTo(clipped);
    cv::Mat maskHigh, maskLow;
    cv::threshold(clipped, maskHigh, max, 1.0, cv::ThresholdTypes::THRESH_BINARY);
    cv::threshold(clipped, maskLow, min, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);
    maskHigh.convertTo(maskHigh, CV_8U);
    maskLow.convertTo(maskLow, CV_8U);
    clipped = clipped.setTo(max, maskHigh);
    clipped = clipped.setTo(min, maskLow);
}

template<int channels>
inline void normL2(const cv::Mat &im, cv::Mat &norm) {
    cv::Mat sq[channels];
    for (int i = 0; i < channels; i++) {
        cv::extractChannel(im, sq[i], i);
        cv::pow(sq[i], 2, sq[i]);
    }

    norm = cv::Mat::zeros(im.rows, im.cols, CV_32FC1);
    for (int i = 0; i < channels; i++) {
        norm += sq[i];
    }
    cv::sqrt(norm, norm);
}

inline void reduceChannels(const cv::Mat &src, cv::Mat &result, int _rtype) {
    cv::Mat reshaped;
    reshaped = src.reshape(1, src.cols * src.rows);
    cv::reduce(reshaped, result, 1, _rtype);
    result = result.reshape(1, src.rows);
}

#endif // CVMATUTILS_H