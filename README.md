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

### 流程简介
1. 将模型转换为C++可用的TorchScript模型。[参考链接](https://pytorch.org/tutorials/advanced/cpp_export.html#step-1-converting-your-pytorch-model-to-torch-script)
2. 加载人脸识别模型，参考官网[《在C++中加载TorchScript模型》](https://pytorch.org/tutorials/advanced/cpp_export.html#step-3-loading-your-script-module-in-c)。
3. 用OpenCV加载图片，预处理并转换为at::Tensor。[参考链接](http://discuss.seekloud.org:50080/d/572-human-mattingpytorchcpython)
4. 输出处理，解析模型输出内容。（decode函数和nms函数）
    - `decode`：用于解析模型输出内容
    - `nms`：非极大值抑制
    - 对应[Python代码](https://github.com/lxg2015/faceboxes/blob/master/encoderl.py)中的decode和nms函数
5. （未完成）加载关键点识别模型。
6. （未完成）输入原始图片和人脸框。
7. （未完成）输出处理，解析模型输出内容。
8. 结果输出到图片（可略）。

#### 使用方法
1. 引用头文件`#include "detector.h"`
2. 创建FaceDetector的实例
  ```
  FaceDetector face(model_path);
  ```
3. 人脸识别
  ```
  face.predict(input, output);
  
  /**
  * 识别图片中的人脸
  * @param path - 输入图片的路径
  * @param outpath - 输出图片的路径
  *                - default ""
  *
  * @return 人脸框的vector
  **/
  std::vector<std::vector<float>> predict(std::string path, std::string outpath);
  
  /**
  * 识别图片中的人脸
  * @param img - 输入的图片Mat
  * @param outpath - 输出图片的路径
  *                - default ""
  *
  * @return 人脸框的vector
  **/
  std::vector<std::vector<float>> predict(std::string path, std::string outpath); 
  ```
4. 关键点识别

#### 实体类
- Figure：存放图片中的关键点和头部姿态信息
- FaceDetector：人脸检测器
- （未完成）KeypointsDetector：关键点识别器

## TODOS
- [x] 使用Pytorch完成人脸识别
- [ ] 使用Pytorch完成关键点识别
- [x] 代码整理
- [ ] 将代码迁移到UE4中