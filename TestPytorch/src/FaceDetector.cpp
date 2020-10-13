#include "detector.h"

FaceDetector::FaceDetector(const std::string& path)
{
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        //module = torch::jit::load("D:\\PycharmProjects\\tf2torch\\keypoints.pt");
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    module.to(torch::kCPU);
    assert(module != nullptr);

    float scale = 1024;
    float steps[] = { 32 / scale, 64 / scale, 128 / scale };
    float sizes[] = { 32 / scale, 256 / scale, 512 / scale };
    std::vector<std::vector<int>> aspect_ratios = { {1, 2, 4}, {1}, {1} };
    int feature_map_sizes[] = { 32, 16, 8 };

    std::vector<std::vector<int>> density = { {-3, -1, 1, 3},{-1, 1},{0} };

    int num_layers = sizeof(feature_map_sizes) / sizeof(int);
    std::vector<at::Tensor> boxes;
    for (int i = 0; i < num_layers; i++)
    {
        int fmsize = feature_map_sizes[i];
        for (int h = 0; h < fmsize; h++)
            for (int w = 0; w < fmsize; w++)
            {
                float cx = (w + 0.5) * steps[i];
                float cy = (h + 0.5) * steps[i];

                float s = sizes[i];
                for (int j = 0; j < aspect_ratios[i].size(); j++)
                {
                    int ar = aspect_ratios[i][j];
                    if (i == 0)
                        for (int m = 0; m < density[j].size(); m++)
                            for (int n = 0; n < density[j].size(); n++)
                            {
                                int dx = density[j][m];
                                int dy = density[j][n];
                                std::vector<float> box;
                                box.push_back(cx + dx / 8.0 * s * ar);
                                box.push_back(cy + dy / 8.0 * s * ar);
                                box.push_back(s * ar);
                                box.push_back(s * ar);
                                boxes.push_back(torch::tensor(box, torch::kFloat));
                            }
                    else
                    {
                        std::vector<float> box;
                        box.push_back(cx);
                        box.push_back(cy);
                        box.push_back(s * ar);
                        box.push_back(s * ar);
                        boxes.push_back(torch::tensor(box, torch::kFloat));
                    }
                }
            }
    }
    default_boxes = torch::stack(boxes);
}

FaceDetector::~FaceDetector()
{

}

std::vector<std::vector<float>> FaceDetector::predict(std::string path, std::string outpath)
{
    cv::Mat img = cv::imread(path);
    return predict(img, outpath);
}

std::vector<std::vector<float>> FaceDetector::predict(cv::Mat& img, std::string outpath)
{
    at::Tensor boxes = detect(img);
    cv::Mat img_faceboxes = img.clone();
    std::vector<std::vector<float>> boxes_vec;
    int h = img.size().height;
    int w = img.size().width;
    at::Tensor scaler = torch::tensor({ w, h, w, h });
    for (int i = 0; i < boxes.size(0); i++)
    {
        at::Tensor point = boxes[i] * scaler;
        int x1 = point[0].item().toInt();
        int x2 = point[2].item().toInt();
        int y1 = point[1].item().toInt();
        int y2 = point[3].item().toInt();
        std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << w << " " << h << " " << std::endl;
        cv::rectangle(img_faceboxes, cv::Point(x1, y1 + 4), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
        point[1].add_(4);
        std::vector<float> box_vec(point.data_ptr<float>(), point.data_ptr<float>() + point.numel());
        boxes_vec.push_back(box_vec);
    }
    if (outpath != "")
        cv::imwrite(outpath, img_faceboxes);
    std::cout << "ok\n";

    return boxes_vec;
}

at::Tensor FaceDetector::detect(cv::Mat& img)
{
    cv::Mat img1;
    cv::resize(img.clone(), img1, cv::Size(1024, 1024));

    cv::Mat img2;
    img1.convertTo(img2, CV_32F, 1.0 / 255); // ÏñËØÖµ¹éÒ»»¯

    at::Tensor tensor = torch::from_blob(img2.data, { 1, img2.rows, img2.cols, 3 }, torch::kFloat32);
    //tensor = torch::from_blob();
    tensor = tensor.permute({ 0, 3, 1, 2 });
    tensor = tensor.toType(torch::kFloat).to(torch::kCPU);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    torch::jit::IValue output = module.forward(inputs);
    at::Tensor output1 = output.toTuple()->elements()[0].toTensor().data().squeeze(0);
    at::Tensor output2 = torch::nn::functional::softmax(output.toTuple()->elements()[1].toTensor().squeeze(0), 1).data();

    at::Tensor boxes, labels, probs;
    std::tie(boxes, labels, probs) = decode(output1, output2);
    std::cout << boxes << std::endl;

    return boxes;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> FaceDetector::decode(at::Tensor& loc, at::Tensor& conf)
{
    float variances[] = { 0.1, 0.2 };

    at::Tensor cxcy = loc.slice(1, 0, 2) * variances[0] * default_boxes.slice(1, 2) + default_boxes.slice(1, 0, 2);
    at::Tensor wh = torch::exp(loc.slice(1, 2) * variances[1]) * default_boxes.slice(1, 2);
    at::Tensor box = torch::cat({ cxcy - wh / 2, cxcy + wh / 2 }, 1);

    conf.select(1, 0) = 0.4;
    at::Tensor max_conf;
    at::Tensor labels;
    std::tie(max_conf, labels) = conf.max(1);

    if (labels.to(torch::kLong).sum().item().toLong() == 0)
    {
        at::Tensor sconf;
        at::Tensor slabel;
        std::tie(sconf, slabel) = conf.max(0);
        max_conf[slabel.slice(0, 0, 5)] = sconf.slice(0, 0, 5);
        labels[slabel.slice(0, 0, 5)] = 1;
    }

    at::Tensor ids = labels.nonzero().squeeze(1);

    at::Tensor keep = nms(box.index({ ids }), max_conf.index({ ids }));

    return std::make_tuple(box.index({ ids }).index({ keep }), labels.index({ ids }).index({ keep }), max_conf.index({ ids }).index({ keep }));
}

at::Tensor FaceDetector::nms(at::Tensor& bboxes, at::Tensor& scores, float threshold)
{
    // bboxes(tensor)[N, 4]
    // scores(tensor)[N, ]
    at::Tensor x1 = bboxes.select(1, 0);
    at::Tensor y1 = bboxes.select(1, 1);
    at::Tensor x2 = bboxes.select(1, 2);
    at::Tensor y2 = bboxes.select(1, 3);
    at::Tensor areas = (x2 - x1) * (y2 - y1);

    at::Tensor order;
    std::tie(std::ignore, order) = scores.sort(0, true);
    std::vector<at::Tensor> keep;
    while (order.numel() > 0)
    {
        at::Tensor i = order[0];
        keep.push_back(i);

        if (order.numel() == 1)
            break;
        at::Tensor xx1 = x1.index({ order.slice(0, 1) }).clamp(x1[i].item());
        at::Tensor yy1 = y1.index({ order.slice(0, 1) }).clamp(y1[i].item());
        at::Tensor xx2 = x2.index({ order.slice(0, 1) }).clamp(c10::nullopt, x2[i].item());
        at::Tensor yy2 = y2.index({ order.slice(0, 1) }).clamp(c10::nullopt, y2[i].item());

        at::Tensor w = (xx2 - xx1).clamp(0);
        at::Tensor h = (yy2 - yy1).clamp(0);
        at::Tensor inter = w * h;

        at::Tensor ovr = inter / (areas[i] + areas.index({ order.slice(0, 1) }) - inter);
        at::Tensor ids = (ovr <= threshold).nonzero().squeeze();
        if (ids.numel() == 0)
            break;
        order = order.index({ ids + 1 });
    }

    return torch::stack(keep).to(torch::kLong);
}
