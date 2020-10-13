#include "detector.h"

Figure::Figure() { null = true; }

Figure::Figure(std::vector<cv::Point>& landmarks,
    std::vector<float>& headPose) {
    setLandmarks(landmarks);
    setHeadPose(headPose);
    null = false;
}

Figure::~Figure() {}

void Figure::setLandmarks(std::vector<cv::Point>& l) {
    landmarks.clear();
    std::vector<cv::Point>(landmarks).swap(landmarks);
    landmarks.assign(l.begin(), l.end());
}

std::vector<cv::Point> Figure::getLandmarks() { return landmarks; }

void Figure::setHeadPose(std::vector<float>& h) {
    headPose.clear();
    std::vector<float>(headPose).swap(headPose);
    headPose.assign(h.begin(), h.end());
}

std::vector<float> Figure::getHeadPose() { return headPose; }

bool Figure::is_null() { return null; }
