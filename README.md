# TestPytorch
使用libtorch完成人脸识别和关键点识别

## 环境和依赖
项目运行环境
- Windows 10
- Visual Studio 2019
- CMake

项目依赖
- [libtorch 1.6.0 CPU](https://pytorch.org/get-started/locally/)
- OpenCV4.3.0-contrib

## 简介
本项目用到的代码和模型来自以下两个项目（Python）
- [人脸识别](https://github.com/lxg2015/faceboxes)
- [人脸对齐](https://github.com/610265158/face_landmark_pytorch)

流程简介：
1. 将模型转换为C++可用的TorchScript模型。[参考链接](https://pytorch.org/tutorials/advanced/cpp_export.html#step-1-converting-your-pytorch-model-to-torch-script)
2. 加载模型，参考官网[《在C++中加载TorchScript模型》](https://pytorch.org/tutorials/advanced/cpp_export.html#step-3-loading-your-script-module-in-c)。
3. 用OpenCV加载图片，预处理并转换为at::Tensor。[参考链接](http://discuss.seekloud.org:50080/d/572-human-mattingpytorchcpython)
4. 输出处理，解析模型输出内容。（decode函数和nms函数）
    - `decode`：用于解析模型输出内容
    - `nms`：非极大值抑制
    - 对应[Python代码](https://github.com/lxg2015/faceboxes/blob/master/encoderl.py)中的decode和nms函数
5. 结果输出到图片。

C++代码暂未整理，功能完成后将对代码进行适当的封装以便于使用。

## TODOS
- [x] 使用Pytorch完成人脸识别
- [ ] 使用Pytorch完成关键点识别
- [ ] 代码整理
- [ ] 将代码迁移到UE4中