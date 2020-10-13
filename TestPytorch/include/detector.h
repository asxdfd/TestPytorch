#pragma once

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "torch/script.h"
#include "torch/nn.h"
#include "torch/autograd.h"

class Figure
{
public:
	Figure();
	Figure(std::vector<cv::Point>&, std::vector<float>&);
	~Figure();
	void setLandmarks(std::vector<cv::Point>&);
	std::vector<cv::Point> getLandmarks();
	void setHeadPose(std::vector<float>&);
	std::vector<float> getHeadPose();
	bool is_null();

private:
	std::vector<cv::Point> landmarks;
	std::vector<float> headPose;
	bool null;
};

class FaceDetector
{
public:
	FaceDetector(const std::string&);
	~FaceDetector();
	std::vector<std::vector<float>> predict(std::string, std::string outpath = "");
	std::vector<std::vector<float>> predict(cv::Mat&, std::string outpath = "");

private:
	torch::jit::script::Module module;
	at::Tensor default_boxes;
	at::Tensor detect(cv::Mat&);
	std::tuple<at::Tensor, at::Tensor, at::Tensor> decode(at::Tensor&, at::Tensor&);
	at::Tensor nms(at::Tensor&, at::Tensor&, float threshold = 0.5);
};

//class KeypointsDetector
//{
//public:
//	KeypointsDetector(std::string&);
//	~KeypointsDetector();
//	std::vector<Figure> predict(std::string, std::vector<std::vector<float>>&);
//	std::vector<Figure> predict(cv::Mat&, std::vector<std::vector<float>>&);
//
//private:
//	int min_face;
//	int point_num;
//	float base_extend_range[2];
//	cv::Size size;
//	torch::jit::script::Module module;
//	Figure onShotRun(cv::Mat&, std::vector<float>);
//	Figure simple_predict(cv::Mat&);
//};
