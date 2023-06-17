#include <iostream>
#include <opencv2/opencv.hpp>
#include "cvMatUtils.h"
#include "primalDualStereo.h"
#include <chrono>
#include <fstream>
#include "../thirdparty/matplotlib-cpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;

#define EPOCH 1000

int main() {
    std::cout << "Loading data..." << std::endl;
    cv::Mat imL, imR;
    imL = cv::imread("../../data/tsukuba_l.png", cv::IMREAD_GRAYSCALE);
    imR = cv::imread("../../data/tsukuba_r.png", cv::IMREAD_GRAYSCALE);
    imL.convertTo(imL, CV_32F);
    imR.convertTo(imR, CV_32F);
    std::cout << "Data loaded." << std::endl;

    // initialize phi
    cv::Mat phi = cv::Mat::zeros(imL.rows, imL.cols, CV_32FC(disparity_range));

    // initialize p
    cv::Mat p1 = cv::Mat::zeros(imL.rows, imL.cols, CV_32FC(disparity_range));
    cv::Mat p2 = cv::Mat::zeros(imL.rows, imL.cols, CV_32FC(disparity_range));
    cv::Mat p3 = cv::Mat::zeros(imL.rows, imL.cols, CV_32FC(disparity_range));

    // initialize gammas
    std::vector<float> gammas;
    gammas.reserve(disparity_range);
    for (int i = 0; i < disparity_range; i++) {
        gammas.emplace_back(min_disparity + i);
    }

    // initialize rhos
    cv::Mat rhos;
    compute_rho(imL, imR, rhos, gammas);

    // smoothness regularization
    float Lambda = 2.0f;

    std::ofstream log("../../data/log.csv", std::ofstream::out);
    log << "primalError,dualError\n";
    std::cout << "Start primal dual iteration for " << EPOCH << " epochs" << std::endl;
    std::cout << "Executing..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < EPOCH; i++) {
        cv::Mat primalError, dualError;
        update_phi(p1, p2, p3, phi, tau_phi);
        proj_phi(phi);
        compute_error(phi, p1, p2, p3, primalError);
        update_p(phi, p1, p2, p3, tau_p);
        proj_p(p1, p2, p3, rhos, Lambda);
        compute_error(phi, p1, p2, p3, dualError);
        log << primalError.at<float>(0) << "," << dualError.at<float>(0) << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = 1e-6 * (end - start).count();
    std::cout << "Execution time =" << elapsed_ms << " milliseconds\n";

    cv::Mat disparity;
    compute_disparity(phi, disparity, min_disparity);

    saveBinary<float>("../../data/disparity_288_384_1.bin", (float *) disparity.data, rows * cols * channels);

    std::map<std::string, std::string> keywords;
    keywords["cmap"]="gray";
    plt::imshow(disparity, keywords);
    plt::save("../../data/disparity_matplotlib.png");

    return 0;
}