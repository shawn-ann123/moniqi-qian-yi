// 离线 NN 探针：复刻 NNDetector 的预处理/后处理，对模拟器帧直接跑 model-0624.onnx
// usage: probe_nn <image.ppm|png> [--model PATH] [--conf 0.65] [--kconf 0.5] [--out PREFIX]
// 输出：类别最大置信度、阈值统计、best anchors 与关键点（映射回原图），并存图便于目检
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
constexpr int kKeypointCount = 5;
constexpr int kKeypointDimension = 3;

struct LetterboxResult
{
    cv::Mat image;
    float scale = 1.0F;
    int padding_x = 0;
    int padding_y = 0;
};

LetterboxResult preprocess_letterbox(const cv::Mat &source, int input_width, int input_height)
{
    LetterboxResult result;
    result.scale = std::min(static_cast<float>(input_width) / static_cast<float>(source.cols),
                            static_cast<float>(input_height) / static_cast<float>(source.rows));
    const int resized_width = static_cast<int>(std::round(source.cols * result.scale));
    const int resized_height = static_cast<int>(std::round(source.rows * result.scale));
    result.padding_x = (input_width - resized_width) / 2;
    result.padding_y = (input_height - resized_height) / 2;
    const int right_padding = input_width - resized_width - result.padding_x;
    const int bottom_padding = input_height - resized_height - result.padding_y;

    cv::Mat resized;
    if (resized_width != source.cols || resized_height != source.rows)
        cv::resize(source, resized, cv::Size(resized_width, resized_height));
    else
        resized = source;

    if (result.padding_x > 0 || result.padding_y > 0 || right_padding > 0 || bottom_padding > 0)
        cv::copyMakeBorder(resized, result.image, result.padding_y, bottom_padding,
                           result.padding_x, right_padding, cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));
    else
        result.image = resized;
    return result;
}
} // namespace

int main(int argc, char **argv)
{
    std::string image_path;
    std::string model_path =
        "/home/shawn/桌面/usaa/RP-26Rune/src/app_plugin/detector/config/openvino/model-0624.onnx";
    std::string out_prefix = "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/nn";
    float conf_threshold = 0.65F;
    float kconf_threshold = 0.5F;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--conf" && i + 1 < argc)
            conf_threshold = std::atof(argv[++i]);
        else if (arg == "--kconf" && i + 1 < argc)
            kconf_threshold = std::atof(argv[++i]);
        else if (arg == "--out" && i + 1 < argc)
            out_prefix = argv[++i];
        else if (image_path.empty())
            image_path = arg;
    }
    if (image_path.empty())
    {
        std::fprintf(stderr,
                     "usage: probe_nn <image.ppm|png> [--model PATH] [--conf C] [--kconf K] [--out PREFIX]\n");
        return 2;
    }

    cv::Mat source = cv::imread(image_path, cv::IMREAD_COLOR);
    if (source.empty())
    {
        std::fprintf(stderr, "cannot read image: %s\n", image_path.c_str());
        return 1;
    }
    std::printf("image=%s size=%dx%d\n", image_path.c_str(), source.cols, source.rows);

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    const ov::Shape input_shape = model->input().get_shape();
    if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3)
    {
        std::fprintf(stderr, "unexpected model input shape\n");
        return 1;
    }
    const int input_height = static_cast<int>(input_shape[2]);
    const int input_width = static_cast<int>(input_shape[3]);
    std::printf("model=%s input=%dx%d\n", model_path.c_str(), input_width, input_height);

    ov::preprocess::PrePostProcessor preprocessor(model);
    preprocessor.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    preprocessor.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0F);
    preprocessor.input().model().set_layout("NCHW");
    model = preprocessor.build();

    ov::CompiledModel compiled =
        core.compile_model(model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    ov::InferRequest request = compiled.create_infer_request();

    LetterboxResult letterbox = preprocess_letterbox(source, input_width, input_height);
    ov::Tensor input_tensor(
        ov::element::u8,
        {1, static_cast<std::size_t>(input_height), static_cast<std::size_t>(input_width), 3});
    std::memcpy(input_tensor.data<std::uint8_t>(), letterbox.image.data,
                letterbox.image.total() * letterbox.image.elemSize());
    request.set_input_tensor(input_tensor);
    request.infer();

    const ov::Tensor output_tensor = request.get_output_tensor();
    const ov::Shape output_shape = output_tensor.get_shape();
    const float *output = output_tensor.data<float>();
    bool output_is_nca = true;
    int channels = 0;
    int anchors = 0;
    if (output_shape.size() == 3 && output_shape[0] == 1)
    {
        if (output_shape[1] <= 256 && output_shape[2] >= 100)
        {
            output_is_nca = true;
            channels = static_cast<int>(output_shape[1]);
            anchors = static_cast<int>(output_shape[2]);
        }
        else
        {
            output_is_nca = false;
            anchors = static_cast<int>(output_shape[1]);
            channels = static_cast<int>(output_shape[2]);
        }
    }
    std::printf("output=[%llu,%llu,%llu] layout=%s channels=%d anchors=%d\n",
                static_cast<unsigned long long>(output_shape[0]),
                static_cast<unsigned long long>(output_shape[1]),
                static_cast<unsigned long long>(output_shape[2]),
                output_is_nca ? "NCA" : "NAC", channels, anchors);
    if (channels != 3 + kKeypointCount * kKeypointDimension)
    {
        std::fprintf(stderr, "unexpected channel count %d\n", channels);
        return 1;
    }
    const int class_count = 3;

    const auto value_at = [&](int channel, int anchor)
    {
        return output_is_nca ? output[channel * anchors + anchor]
                             : output[anchor * channels + channel];
    };

    // 0) 逐通道统计（判定通道语义）
    for (int channel = 0; channel < channels; ++channel)
    {
        float min_value = 1e9F;
        float max_value = -1e9F;
        double sum = 0.0;
        for (int anchor = 0; anchor < anchors; ++anchor)
        {
            const float value = value_at(channel, anchor);
            if (value < min_value)
                min_value = value;
            if (value > max_value)
                max_value = value;
            sum += value;
        }
        std::printf("channel %2d: min=%.4f max=%.4f mean=%.4f\n", channel, min_value, max_value,
                    sum / anchors);
    }

    // 1) 类别置信度统计
    for (int cls = 0; cls < class_count; ++cls)
    {
        float best = -1.0F;
        int best_anchor = -1;
        std::array<int, 6> counts{}; // >0.1 >0.25 >0.4 >0.5 >0.65 >0.8
        const float levels[6] = {0.1F, 0.25F, 0.4F, 0.5F, 0.65F, 0.8F};
        for (int anchor = 0; anchor < anchors; ++anchor)
        {
            const float score = value_at(cls, anchor);
            if (score > best)
            {
                best = score;
                best_anchor = anchor;
            }
            for (int level = 0; level < 6; ++level)
                if (score > levels[level])
                    ++counts[level];
        }
        std::printf("class %d: max=%.4f @anchor %d | counts >0.1:%d >0.25:%d >0.4:%d >0.5:%d >0.65:%d >0.8:%d\n",
                    cls, best, best_anchor, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);
    }

    // 2) top anchors（任意类别）
    struct AnchorScore
    {
        float score;
        int anchor;
        int cls;
    };
    std::vector<AnchorScore> top;
    top.reserve(anchors);
    for (int anchor = 0; anchor < anchors; ++anchor)
    {
        float best = -1.0F;
        int best_cls = -1;
        for (int cls = 0; cls < class_count; ++cls)
        {
            const float score = value_at(cls, anchor);
            if (score > best)
            {
                best = score;
                best_cls = cls;
            }
        }
        top.push_back({best, anchor, best_cls});
    }
    std::partial_sort(top.begin(), top.begin() + 10, top.end(),
                      [](const AnchorScore &a, const AnchorScore &b) { return a.score > b.score; });
    std::printf("top-10 anchors (score, class, kpt-pixel-area):\n");
    for (int i = 0; i < 10; ++i)
    {
        const AnchorScore &entry = top[i];
        std::printf("  #%d score=%.4f cls=%d anchor=%d | ", i, entry.score, entry.cls, entry.anchor);
        for (int kpt = 0; kpt < kKeypointCount; ++kpt)
        {
            const int base = class_count + kpt * kKeypointDimension;
            const float x = (value_at(base, entry.anchor) - static_cast<float>(letterbox.padding_x)) /
                            letterbox.scale;
            const float y = (value_at(base + 1, entry.anchor) - static_cast<float>(letterbox.padding_y)) /
                            letterbox.scale;
            const float kc = value_at(base + 2, entry.anchor);
            std::printf("(%.0f,%.0f,%.2f) ", x, y, kc);
        }
        std::printf("\n");
    }

    // 3) 按 NNDetector 规则产出检测（conf 阈值 + kpt 有效数 >= 5）
    int detections = 0;
    cv::Mat overlay = source.clone();
    for (const AnchorScore &entry : top)
    {
        if (entry.score < conf_threshold)
            continue;
        std::array<cv::Point2f, kKeypointCount> keypoints{};
        std::array<float, kKeypointCount> keypoint_confidences{};
        int valid = 0;
        bool bad = false;
        for (int kpt = 0; kpt < kKeypointCount; ++kpt)
        {
            const int base = class_count + kpt * kKeypointDimension;
            const float x = (value_at(base, entry.anchor) - static_cast<float>(letterbox.padding_x)) /
                            letterbox.scale;
            const float y = (value_at(base + 1, entry.anchor) - static_cast<float>(letterbox.padding_y)) /
                            letterbox.scale;
            const float kc = value_at(base + 2, entry.anchor);
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F)
            {
                bad = true;
                break;
            }
            keypoints[kpt] = {std::clamp(x, 0.0F, static_cast<float>(source.cols - 1)),
                              std::clamp(y, 0.0F, static_cast<float>(source.rows - 1))};
            keypoint_confidences[kpt] = kc;
            if (kc >= kconf_threshold)
                ++valid;
        }
        if (bad || valid < kKeypointCount)
            continue;
        ++detections;
        std::printf("DETECTION cls=%d conf=%.3f anchor=%d kpts:", entry.cls, entry.score, entry.anchor);
        for (int kpt = 0; kpt < kKeypointCount; ++kpt)
            std::printf(" (%.1f,%.1f,%.2f)", keypoints[kpt].x, keypoints[kpt].y,
                        keypoint_confidences[kpt]);
        std::printf("\n");

        const cv::Scalar colors[5] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                                      cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255),
                                      cv::Scalar(0, 0, 255)};
        for (int kpt = 0; kpt < kKeypointCount; ++kpt)
        {
            cv::circle(overlay, keypoints[kpt], kpt == 2 ? 8 : 5, colors[kpt], cv::FILLED, cv::LINE_AA);
            cv::circle(overlay, keypoints[kpt], kpt == 2 ? 8 : 5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }
    std::printf("total detections (conf=%.2f, kconf=%.2f, min_valid=5): %d\n", conf_threshold,
                kconf_threshold, detections);

    cv::imwrite(out_prefix + "_input.png", letterbox.image);
    cv::imwrite(out_prefix + "_overlay.png", overlay);
    cv::imwrite(out_prefix + "_src.png", source);
    std::printf("saved %s_{input,overlay,src}.png\n", out_prefix.c_str());
    return 0;
}
