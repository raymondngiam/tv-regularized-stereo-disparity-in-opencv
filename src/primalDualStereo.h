#ifndef PRIMALDUALSTEREO_H
#define PRIMALDUALSTEREO_H

#include "cvMatUtils.h"

constexpr int rows = 288;
constexpr int cols = 384;
constexpr int channels = 1;
constexpr int disparity_range = 64;
constexpr float min_disparity = -32;
constexpr float tau_p = 0.57735f;
constexpr float tau_phi = 0.57735f;

template<int rows, int cols, int channel>
inline void warp(const cv::Mat &im, const cv::Mat &disparity, cv::Mat &warped) {
    std::vector<float> ind_per_row;
    cv::Mat ind = cv::Mat::zeros(rows, cols, CV_32FC(channel));
    for (int i = 0; i < cols; i++) {
        ind_per_row.emplace_back(i);
        ind.col(i) = i;
    }
    warped = cv::Mat::zeros(rows, cols, CV_32FC(channel));

    cv::Mat mappedIndex = ind + disparity;

    cv::Mat mappedIndexClipped;
    clip<float>(mappedIndex, mappedIndexClipped, 0.f, cols - 1);

    for (int y = 0; y < rows; y++) {
        std::for_each(std::execution::par,
                      ind_per_row.begin(),
                      ind_per_row.end(),
                      [&y, &im, &warped, &mappedIndexClipped](auto x) {
                          auto newX = (int) mappedIndexClipped.at<float>(y, x);
                          warped.at<float>(y, x) = im.at<float>(y, newX);
                      });
    }
}

inline void projectionSingle(const cv::Mat &im, const cv::Mat &norm, const cv::Mat &scale, cv::Mat &projected) {
    cv::Mat mask;
    mask = norm > scale;
    mask.convertTo(mask, CV_8U);

    im.copyTo(projected);

    cv::Mat_<cv::Point2i> id;
    cv::findNonZero(mask, id);
    std::for_each(std::execution::par,
                  id.begin(),
                  id.end(),
                  [&projected, &scale](cv::Point2i item) {
                      projected.at<float>(item.y, item.x) = scale.at<float>(item.y, item.x);
                  });
}

inline void projection(const cv::Mat &im, const cv::Mat &norm, const cv::Mat &scale, cv::Mat &projected) {
    cv::Mat mask;
    cv::Mat normCopy;
    norm.copyTo(normCopy);
    cv::divide(normCopy, scale, normCopy);
    cv::threshold(normCopy, mask, 1, 1.0, cv::ThresholdTypes::THRESH_BINARY_INV);
    mask.convertTo(mask, CV_8U);

    cv::Mat norms_stack, result;
    normCopy.setTo(1.f, mask);

    std::vector<cv::Mat> norms;
    for (int i = 0; i < im.channels(); i++) {
        norms.emplace_back(normCopy);
    }
    cv::merge(norms, norms_stack);
    cv::divide(im, norms_stack, projected);
}

inline void compute_rho(const cv::Mat &imL, const cv::Mat &imR, cv::Mat &rhos, std::vector<float> gammas) {
    std::vector<cv::Mat> v;
    v.reserve(disparity_range);
    for (int i = 0; i < disparity_range; i++) {
        cv::Mat disparity = gammas[i] * cv::Mat::ones(imL.rows, imL.cols, CV_32F);
        cv::Mat warped;
        warp<rows, cols, channels>(imR, disparity, warped);
        cv::Mat absdiff;
        cv::absdiff(imL, warped, absdiff);
        v.emplace_back(absdiff);
    }
    cv::merge(v, rhos);
}

inline void compute_disparity(const cv::Mat &phi, cv::Mat &disparity, float min_disparity) {
    cv::Mat phiThres;
    cv::threshold(phi, phiThres, 0.5, 1.0, cv::THRESH_BINARY);
    cv::Mat phiSum;
    reduceChannels(phiThres, phiSum, cv::REDUCE_SUM);
    disparity = phiSum + min_disparity;
}

template<int rows, int cols, int channels>
inline void divergence3D(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &p3, cv::Mat &div) {
    cv::Mat gradX, gradY;
    float d_kernelX[3]{-0.5, 0, 0.5};
    cv::Mat kernelX(1, 3, CV_32F, d_kernelX);
    cv::filter2D(p1, gradX, CV_32F, kernelX, cv::Point2i(-1, 0));
    gradX.col(0) = p1.col(1) - p1.col(0);
    gradX.col(cols - 1) = p1.col(cols - 1) - p1.col(cols - 2);

    float d_kernelY[3]{-0.5, 0, 0.5};
    cv::Mat kernelY(3, 1, CV_32F, d_kernelY);
    cv::filter2D(p2, gradY, CV_32F, kernelY, cv::Point2i(0, -1));
    gradY.row(0) = p2.row(1) - p2.row(0);
    gradY.row(rows - 1) = p2.row(rows - 1) - p2.row(rows - 2);

    cv::Mat tmp(rows, channels, CV_32FC(cols));
    cv::Mat tmp2;
    cv::Mat gradZ(rows, cols, CV_32FC(channels));
    transpose_DWH_to_WDH<float, 1, rows, cols, channels>((float (*)[rows][cols][channels]) p3.data,
                                                         (float (*)[rows][channels][cols]) tmp.data);
    cv::filter2D(tmp, tmp2, CV_32F, kernelX, cv::Point2i(-1, 0));
    tmp2.col(0) = tmp.col(1) - tmp.col(0);
    tmp2.col(channels - 1) = tmp.col(channels - 1) - tmp.col(channels - 2);
    transpose_DWH_to_WDH<float, 1, rows, channels, cols>((float (*)[rows][channels][cols]) tmp2.data,
                                                         (float (*)[rows][cols][channels]) gradZ.data);
    div = gradX + gradY + gradZ;
}

inline void update_phi(const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &p3, cv::Mat &phi, float step_size) {
    cv::Mat div;
    divergence3D<rows, cols, disparity_range>(p1, p2, p3, div);
    phi = phi + step_size * div;
}

inline void update_p(const cv::Mat &phi, cv::Mat &p1, cv::Mat &p2, cv::Mat &p3, float step_size) {
    cv::Mat dPhi;
    derivate3D<rows, cols, disparity_range>(phi, dPhi);

    std::vector<cv::Mat> dPhis;
    cv::split(dPhi, dPhis);
    std::vector<cv::Mat> tmp1(dPhis.begin(), dPhis.begin() + disparity_range);
    std::vector<cv::Mat> tmp2(dPhis.begin() + disparity_range, dPhis.begin() + 2 * disparity_range);
    std::vector<cv::Mat> tmp3(dPhis.begin() + 2 * disparity_range, dPhis.end());

    cv::Mat matTmp1, matTmp2, matTmp3;
    cv::merge(tmp1, matTmp1);
    cv::merge(tmp2, matTmp2);
    cv::merge(tmp3, matTmp3);
    p1 = p1 + step_size * matTmp1;
    p2 = p2 + step_size * matTmp2;
    p3 = p3 + step_size * matTmp3;
}

inline void proj_phi(cv::Mat &phi) {
    clip<float>(phi, phi, 0.f, 1.0f);
    std::vector<cv::Mat> splits;
    cv::split(phi, splits);
    splits[0].setTo(1);
    splits[splits.size() - 1].setTo(0);
    cv::merge(splits, phi);
}

inline void proj_p(cv::Mat &p1,
                   cv::Mat &p2,
                   cv::Mat &p3,
                   const cv::Mat &rho,
                   float lambda) {
    std::vector<cv::Mat> splits;
    splits.emplace_back(p1.reshape(1, p1.rows));
    splits.emplace_back(p2.reshape(1, p2.rows));
    cv::Mat firstTwoPs;
    cv::merge(splits, firstTwoPs);

    cv::Mat norm1, result1;
    normL2<2>(firstTwoPs, norm1);
    cv::Mat scale1 = lambda * cv::Mat::ones(rows, cols * disparity_range, CV_32FC(1));
    projection(firstTwoPs, norm1, scale1, result1);

    cv::Mat norm2, result2;
    p3 = p3.reshape(1, p3.rows);
    norm2 = cv::abs(p3);
    projectionSingle(p3, norm2.reshape(1, norm2.rows), rho.reshape(1, rho.rows), result2);

    std::vector<cv::Mat> results;
    cv::split(result1, results);

    results[0].copyTo(p1);
    results[1].copyTo(p2);
    result2.copyTo(p3);

    p1 = p1.reshape(rho.channels(), p1.rows);
    p2 = p2.reshape(rho.channels(), p2.rows);
    p3 = p3.reshape(rho.channels(), p3.rows);
}

inline void compute_error(const cv::Mat &phi, const cv::Mat &p1, const cv::Mat &p2, const cv::Mat &p3, cv::Mat &error) {
    cv::Mat dPhi;
    derivate3D<rows, cols, disparity_range>(phi, dPhi);

    std::vector<cv::Mat> dPhis;
    cv::split(dPhi, dPhis);
    std::vector<cv::Mat> tmp1(dPhis.begin(), dPhis.begin() + disparity_range);
    std::vector<cv::Mat> tmp2(dPhis.begin() + disparity_range, dPhis.begin() + 2 * disparity_range);
    std::vector<cv::Mat> tmp3(dPhis.begin() + 2 * disparity_range, dPhis.end());

    cv::Mat matTmp1, matTmp2, matTmp3;
    cv::merge(tmp1, matTmp1);
    cv::merge(tmp2, matTmp2);
    cv::merge(tmp3, matTmp3);

    cv::Mat sumStage, sum;
    sumStage = p1.mul(matTmp1) + p2.mul(matTmp2) + p3.mul(matTmp3);
    cv::reduce(sumStage.reshape(1, 1), sum, 1, cv::REDUCE_SUM);
    sum.copyTo(error);
}

template<typename T>
inline void saveBinary(std::string filepath, T *data, size_t size) {
    FILE *file;
    file = fopen(filepath.c_str(), "wb");
    fwrite(data, sizeof(T), size, file);
    fclose(file);
}

#endif // PRIMALDUALSTEREO_H