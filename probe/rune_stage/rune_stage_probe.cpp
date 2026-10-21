// 打符算法分阶段探针（诊断用，不改动算法代码）：
// 直接调用 SZU power_rune 的公开类（RuneObservationRefiner / PowerRunePlane /
// PhaseMotionEstimator / RuneDecisionModule），逐阶段打印内部状态。
// 联调背景：桥接(app)侧 NN 有稳定检出，但 power_rune 一直不锁定 —— 本工具用于
// 定位是哪一阶段把观测丢掉的（轮廓约束 / 描述符 / 平面投影 / 相位滤波 / 决策）。
//
// 运行前提：
//   1) 模拟器运行中，且 app 未运行（模拟器图像流只服务单客户端）；
//   2) 二进制必须位于 RP-26Rune/output/ 下（power_rune 库内 J_POWER_RUNE 用
//      /proc/self/exe 相对路径找 config/power_rune.json）。
// usage: rune_stage_probe [--frames N] [--no-aim] [--fire] [--conf C] [--kconf K]
//                         [--model PATH] [--save-every N] [--outdir DIR]

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>
#include <glog/logging.h>

#include "power_rune_interface.hpp"
#include "RuneObservationRefiner.hpp"
#include "PowerRunePlane.hpp"
#include "PhaseMotionEstimator.hpp"
#include "RuneDecisionModule.hpp"
#include "common/power_rune_global.hpp"
#include "common/power_rune_function.hpp"
#include "transform_tools/transform_tools.h"
#include "time/time.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace daedalus::sim::sdk::v1;

// ---------- 模拟器 -> 算法 桥接常量（与 sim_bridge/SimCameraManager.cpp 一致） ----------
using V3 = std::array<double, 3>;
using M3 = std::array<double, 9>; // row-major

M3 mat_mul(const M3 &a, const M3 &b)
{
    M3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k)
                sum += a[i * 3 + k] * b[k * 3 + j];
            r[i * 3 + j] = sum;
        }
    return r;
}

M3 mat_transpose(const M3 &a)
{
    M3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i * 3 + j] = a[j * 3 + i];
    return r;
}

M3 quat_wxyz_to_rotation(const float q[4])
{
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
}

constexpr M3 kVehicleFromEcs{0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};
constexpr M3 kStableToCarRdf{0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
constexpr M3 kStableToCarRdfTranspose{0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
constexpr M3 kGimbalFromCameraTranspose{0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};

double to_degree(double rad) { return rad * 57.29577951308232; }

std::string default_ipc_directory()
{
    if (const char *configured = std::getenv("TALOS_IPC_DIR");
        configured != nullptr && configured[0] != '\0')
        return configured;

    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    const std::string base = xdg != nullptr ? xdg : "/tmp";
    return base + "/daedalus-learning-" + std::to_string(::getuid());
}

cv::Mat to_bgr_image(const TcpImageFrame &frame)
{
    using daedalus::sim::sdk::v1::tcp_image::PixelFormat;
    const auto &header = frame.header;
    if (header.width == 0 || header.height == 0)
        return {};
    const int channels = header.format == PixelFormat::Rgba32 ? 4 : 3;
    const std::size_t expected = static_cast<std::size_t>(header.width) *
                                 static_cast<std::size_t>(header.height) *
                                 static_cast<std::size_t>(channels);
    if (frame.payload.size() < expected)
        return {};
    const cv::Mat view(static_cast<int>(header.height), static_cast<int>(header.width),
                       channels == 4 ? CV_8UC4 : CV_8UC3,
                       const_cast<std::uint8_t *>(frame.payload.data()));
    cv::Mat bgr;
    cv::cvtColor(view, bgr, channels == 4 ? cv::COLOR_RGBA2BGR : cv::COLOR_RGB2BGR);
    return bgr;
}

// ---------- NN（与 app_plugin/detector NNDetector 一致的预处理/后处理） ----------
constexpr int kKeypointCount = 5;
constexpr int kKeypointDimension = 3;
constexpr int kClassCount = 3;

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

struct Detection
{
    int class_id = 0;
    float confidence = 0.0F;
    float quality = 0.0F;
    std::array<cv::Point2f, kKeypointCount> keypoints{};
    std::array<float, kKeypointCount> keypoint_confidences{};
};

class RuneDetector
{
public:
    RuneDetector(const std::string &model_path, float conf, float kconf, float nms_distance)
        : m_conf(conf), m_kconf(kconf), m_nms_distance(nms_distance)
    {
        std::shared_ptr<ov::Model> model = m_core.read_model(model_path);
        const ov::Shape input_shape = model->input().get_shape();
        m_input_height = static_cast<int>(input_shape[2]);
        m_input_width = static_cast<int>(input_shape[3]);

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

        m_compiled = m_core.compile_model(
            model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        m_request = m_compiled.create_infer_request();
        LOG(INFO) << "[rune_stage] NN ready; model=" << model_path;
    }

    std::vector<Detection> infer(const cv::Mat &source)
    {
        LetterboxResult letterbox = preprocess_letterbox(source, m_input_width, m_input_height);
        ov::Tensor input_tensor(
            ov::element::u8,
            {1, static_cast<std::size_t>(m_input_height), static_cast<std::size_t>(m_input_width), 3});
        std::memcpy(input_tensor.data<std::uint8_t>(), letterbox.image.data,
                    letterbox.image.total() * letterbox.image.elemSize());
        m_request.set_input_tensor(input_tensor);
        m_request.infer();

        const ov::Tensor output_tensor = m_request.get_output_tensor();
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
        const auto value_at = [&](int channel, int anchor)
        {
            return output_is_nca ? output[channel * anchors + anchor]
                                 : output[anchor * channels + channel];
        };

        std::vector<Detection> detections;
        for (int anchor = 0; anchor < anchors; ++anchor)
        {
            int best_class = -1;
            float best_confidence = 0.0F;
            for (int class_index = 0; class_index < kClassCount; ++class_index)
            {
                const float score = value_at(class_index, anchor);
                if (std::isfinite(score) && score > best_confidence)
                {
                    best_confidence = score;
                    best_class = class_index;
                }
            }
            if (best_class < 0 || best_confidence < m_conf)
                continue;

            Detection detection;
            detection.class_id = best_class;
            detection.confidence = best_confidence;
            int valid_keypoints = 0;
            float keypoint_confidence_sum = 0.0F;
            cv::Point2f valid_center{0.0F, 0.0F};
            bool invalid_keypoint = false;
            for (int keypoint = 0; keypoint < kKeypointCount; ++keypoint)
            {
                const int base = kClassCount + keypoint * kKeypointDimension;
                float x = (value_at(base, anchor) - static_cast<float>(letterbox.padding_x)) /
                          letterbox.scale;
                float y = (value_at(base + 1, anchor) - static_cast<float>(letterbox.padding_y)) /
                          letterbox.scale;
                const float keypoint_confidence = value_at(base + 2, anchor);
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(keypoint_confidence) ||
                    x < 0.0F || y < 0.0F)
                {
                    invalid_keypoint = true;
                    break;
                }
                x = std::clamp(x, 0.0F, static_cast<float>(source.cols - 1));
                y = std::clamp(y, 0.0F, static_cast<float>(source.rows - 1));
                detection.keypoints[keypoint] = {x, y};
                detection.keypoint_confidences[keypoint] = keypoint_confidence;
                if (keypoint_confidence >= m_kconf)
                {
                    ++valid_keypoints;
                    keypoint_confidence_sum += keypoint_confidence;
                    valid_center += detection.keypoints[keypoint];
                }
            }
            if (invalid_keypoint || valid_keypoints < kKeypointCount)
                continue;
            const float mean_keypoint_confidence =
                keypoint_confidence_sum / static_cast<float>(valid_keypoints);
            detection.quality = detection.confidence * mean_keypoint_confidence;
            detections.push_back(std::move(detection));
        }

        // center-distance NMS（NNDetector 同款）
        std::sort(detections.begin(), detections.end(),
                  [](const Detection &a, const Detection &b) { return a.quality > b.quality; });
        std::vector<Detection> kept;
        const float threshold_squared = m_nms_distance * m_nms_distance;
        for (std::size_t i = 0; i < detections.size(); ++i)
        {
            bool suppressed = false;
            cv::Point2f center_i{0, 0};
            for (int k = 0; k < kKeypointCount; ++k)
                center_i += detections[i].keypoints[k];
            center_i *= (1.0F / kKeypointCount);
            for (const Detection &k : kept)
            {
                cv::Point2f center_k{0, 0};
                for (int kk = 0; kk < kKeypointCount; ++kk)
                    center_k += k.keypoints[kk];
                center_k *= (1.0F / kKeypointCount);
                const cv::Point2f d = center_i - center_k;
                if (d.dot(d) < threshold_squared)
                {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed)
                kept.push_back(detections[i]);
        }
        return kept;
    }

private:
    ov::Core m_core;
    ov::CompiledModel m_compiled;
    ov::InferRequest m_request;
    int m_input_width = 0;
    int m_input_height = 0;
    float m_conf = 0.65F;
    float m_kconf = 0.5F;
    float m_nms_distance = 30.0F;
};

// ---------- TF 树（复刻 RuneTrackerManager::initialize_tf_tree / update_tf_tree） ----------
using transform_tools::Angle;
using transform_tools::Pitch;
using transform_tools::Roll;
using transform_tools::RotationAxis;
using transform_tools::Yaw;

struct SimTf
{
    transform_tools::TFTree tree;
    // 与 tracker_manager/config/track.json 保持一致：t_off = M_hc·A·o_g（相机 RDF 约定）
    double x_offset_m = -0.25631;
    double y_offset_m = -0.00183;
    double z_offset_m = 0.09543;
    double gimbal_true_y_degree = 0.0;
    double gimbal_true_x_degree = 0.0;

    void init()
    {
        tree.add_TF(transform_tools::TF(), car_frame,
                    ecs_world_frame);
        Eigen::Matrix3d vehicle_from_ecs;
        vehicle_from_ecs << 0.0, 1.0, 0.0,
                            0.0, 0.0, -1.0,
                            -1.0, 0.0, 0.0;
        tree[ecs_world_frame].set_rotation(vehicle_from_ecs);

        tree.add_TF(transform_tools::TF(), ecs_world_frame,
                    gimbal_frame);
        tree.add_TF(
            transform_tools::TF(Eigen::Vector3d(x_offset_m, y_offset_m, z_offset_m)),
            gimbal_frame, camera_frame);

        Eigen::Matrix3d gimbal_from_camera;
        gimbal_from_camera << 0.0, 0.0, -1.0,
                              1.0, 0.0, 0.0,
                              0.0, -1.0, 0.0;
        tree[camera_frame].set_rotation(gimbal_from_camera);
        tree.add_TF(transform_tools::TF(), car_frame,
                    unbiased_camera_frame);
    }

    void update(double ecs_yaw_deg, double ecs_pitch_deg, double ecs_roll_deg)
    {
        const Eigen::Matrix3d ecs_from_gimbal =
            Yaw(Angle::from_degree(ecs_yaw_deg), RotationAxis::Z).to_rotation_mat() *
            Pitch(Angle::from_degree(ecs_pitch_deg), RotationAxis::Y).to_rotation_mat() *
            Roll(Angle::from_degree(ecs_roll_deg), RotationAxis::X).to_rotation_mat();
        const Eigen::Matrix3d gimbal_bias =
            (Eigen::AngleAxisd(Angle::from_degree(gimbal_true_y_degree).to_rad(),
                               Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(Angle::from_degree(gimbal_true_x_degree).to_rad(),
                               Eigen::Vector3d::UnitX()))
                .matrix();
        tree[gimbal_frame].set_rotation(ecs_from_gimbal * gimbal_bias);

        const transform_tools::TF car_from_camera =
            tree.calculate_absolute_TF(camera_frame);
        tree[unbiased_camera_frame].setTF(
            car_from_camera.get_offset(), car_from_camera.calculate_yaw());
    }
};

// ---------- convert2rune_observation 复刻（PowerRuneProcessor 同款） ----------
RuneObservation build_rune_observation(const cv::Mat &ori_mat,
                                       const timetool::Timestamp &timestamp,
                                       const transform_tools::TFTree &tf_tree,
                                       bool is_big_rune,
                                       int cd_my_color,
                                       const std::vector<Detection> &detections,
                                       double extend_ratio)
{
    RuneObservation rune_observation;
    rune_observation.is_big_rune = is_big_rune;
    rune_observation.ori_img = ori_mat;
    rune_observation.tf_tree = tf_tree;
    rune_observation.timestamp = PRF::timestamp_to_foxglove_time(timestamp);
    rune_observation.rune_infos.reserve(detections.size());
    for (const auto &detection : detections)
    {
        std::vector<cv::Point2i> points;
        points.reserve(5);
        points.push_back(detection.keypoints[0]); // top
        points.push_back(detection.keypoints[1]); // left
        points.push_back(detection.keypoints[2]); // point_R
        points.push_back(detection.keypoints[3]); // right
        points.push_back(detection.keypoints[4]); // bottom

        cv::RotatedRect rotated_rect = cv::minAreaRect(points);
        rotated_rect.size = rotated_rect.size * static_cast<float>(extend_ratio);

        cv::Rect view_rect = rotated_rect.boundingRect();
        view_rect &= cv::Rect(0, 0, ori_mat.cols, ori_mat.rows);
        if (view_rect.empty())
        {
            LOG(ERROR) << "[rune_stage] bad view_rect";
            continue;
        }

        RuneInfo rune_info;
        rune_info.top = detection.keypoints[0];
        rune_info.left = detection.keypoints[1];
        rune_info.bottom = detection.keypoints[4];
        rune_info.right = detection.keypoints[3];
        rune_info.point_R = detection.keypoints[2];
        rune_info.class_id = detection.class_id;
        rune_info.rotated_rect = rotated_rect;
        rune_info.view_rect = view_rect;
        rune_info.view = ori_mat(view_rect).clone();
        rune_info.color = cd_my_color == 0 ? RuneInfo::RED : RuneInfo::BLUE;
        rune_observation.rune_infos.emplace_back(std::move(rune_info));
    }
    return rune_observation;
}

// 复刻 extract_contours（用于把"提取"和"约束"两个环节的问题分开）
std::vector<std::vector<cv::Point>> raw_extract_contours(const RuneInfo &rune_info,
                                                         double threshold_override = -1.0,
                                                         cv::Mat *mask_out = nullptr)
{
    const cv::Mat &view = rune_info.view;
    std::vector<cv::Mat> bgr;
    cv::split(view, bgr);
    const bool is_red = rune_info.color == RuneInfo::RED;
    cv::Mat minus_mat = is_red ? bgr[2] - bgr[0] : bgr[0] - bgr[2];
    double threshold = threshold_override;
    if (threshold < 0.0)
        threshold = is_red ? 50.0 : 62.0; // 与 power_rune.json 的 red/blue_minus_*_2bin_threshold 一致
    cv::GaussianBlur(minus_mat, minus_mat, cv::Size(5, 5), 0);
    cv::Mat mask;
    cv::threshold(minus_mat, mask, threshold, 255, cv::THRESH_BINARY);
    if (mask_out != nullptr)
        mask_out->create(view.rows, view.cols, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (auto &contour : contours)
    {
        for (auto &pt : contour)
        {
            pt.x += rune_info.view_rect.x;
            pt.y += rune_info.view_rect.y;
        }
    }
    if (mask_out != nullptr)
        *mask_out = mask.clone();
    return contours;
}

const char *state_name(::RuneState state)
{
    switch (state)
    {
    case ::RuneState::BigInactive:
        return "BigInactive";
    case ::RuneState::BigActivated:
        return "BigActivated";
    case ::RuneState::SmallInactive:
        return "SmallInactive";
    case ::RuneState::SmallActivated:
        return "SmallActivated";
    }
    return "?";
}

// 复刻 constrain_contours 的三个 filter + RectangularDescriptor，
// 用于逐轮廓打印"为什么被判为/没被判为灯臂"（防止只看到最终 usable 位）
bool line_pass_through(const cv::Point2f &A, const cv::Point2f &B,
                       const std::vector<cv::Point> &contour, int samples)
{
    for (int i = 0; i <= samples; ++i)
    {
        const float t = static_cast<float>(i) / samples;
        const cv::Point2f P = A + t * (B - A);
        if (cv::pointPolygonTest(contour, P, false) > 0)
            return true;
    }
    return false;
}

void diagnose_contours(const RuneInfo &info, const std::vector<std::vector<cv::Point>> &raw)
{
    const cv::Point2f center = (info.top + info.left + info.bottom + info.right) / 4.f;
    const double nn_ellipse_area =
        0.25f * cv::norm(info.top - info.bottom) * cv::norm(info.left - info.right) * CV_PI;
    const double armor_area_tol = 0.35;
    const int line_samples = 15;

    for (std::size_t c = 0; c < raw.size(); ++c)
    {
        const auto &contour = raw[c];
        if (contour.size() <= 5)
            continue;

        const double area = cv::contourArea(contour);
        const double rel_err = std::fabs(area - nn_ellipse_area) / nn_ellipse_area;
        const bool is_armor =
            cv::pointPolygonTest(contour, center, false) > 0 && rel_err <= armor_area_tol;

        const bool center_out = cv::pointPolygonTest(contour, center, false) < 0;
        const bool R_out = cv::pointPolygonTest(contour, info.point_R, false) < 0;
        const bool cr_line = line_pass_through(center, info.point_R, contour, line_samples);
        const bool lr_line = line_pass_through(info.left, info.right, contour, line_samples);
        const bool is_light = center_out && R_out && cr_line && !lr_line;

        const bool is_R =
            cv::pointPolygonTest(contour, info.point_R, false) > 0 &&
            cv::pointPolygonTest(contour, info.top, false) < 0 &&
            cv::pointPolygonTest(contour, info.left, false) < 0 &&
            cv::pointPolygonTest(contour, info.bottom, false) < 0 &&
            cv::pointPolygonTest(contour, info.right, false) < 0 &&
            cv::pointPolygonTest(contour, center, false) < 0;

        std::string tag = is_armor ? "ARMOR" : (is_light ? "LIGHT" : (is_R ? "R" : "-"));

        // 灯臂候选：打印矩形描述符的两个判据
        double solidity = 0.0, aspect = 0.0;
        std::string extra;
        if (is_light)
        {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            const double hull_area = cv::contourArea(hull);
            solidity = hull_area > 0 ? area / hull_area : 0.0;
            const cv::RotatedRect rr = cv::minAreaRect(contour);
            const double wmax = std::max(static_cast<double>(rr.size.width), static_cast<double>(rr.size.height));
            const double wmin = std::max(1e-3, std::min(static_cast<double>(rr.size.width), static_cast<double>(rr.size.height)));
            aspect = wmax / wmin;
            const double asp_rel = std::fabs(5.0 - aspect) / 5.0;
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          " solidity=%.3f(%s<=0.66) aspect=%.2f(rel=%.3f,%s>=0.42)",
                          solidity, solidity <= 0.66 ? "FAIL" : "pass", aspect, asp_rel,
                          asp_rel >= 0.42 ? "FAIL" : "pass");
            extra = buf;
        }
        else if (is_armor)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), " rel_err=%.3f(<=%.2f)", rel_err, armor_area_tol);
            extra = buf;
        }

        std::printf("      diag[%zu] %s area=%.0f ellipse=%.0f%s\n", c, tag.c_str(), area,
                    nn_ellipse_area, extra.c_str());
    }
}

} // namespace

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    int max_frames = 0; // 0 = 无限
    bool aim_enabled = true;
    bool fire_enabled = false;
    float conf = 0.65F;
    float kconf = 0.5F;
    int save_every = 15;
    const std::filesystem::path local_rp =
        std::filesystem::canonical("/proc/self/exe").parent_path().parent_path();
    std::string model_path =
        (local_rp / "src/app_plugin/detector/config/openvino/model-1012.onnx").string();
    std::string out_dir = "/tmp/rune_stage";
    double extend_ratio = 1.4;
    double yaw_scale = -1.0;
    double yaw_offset_deg = 0.0;
    double pitch_scale = 1.0;
    double pitch_offset_deg = 65.0;
    int my_color = 0; // 0=RED(默认) 1=BLUE
    std::optional<double> hold_yaw;
    std::optional<double> hold_pitch;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc)
            max_frames = std::atoi(argv[++i]);
        else if (arg == "--no-aim")
            aim_enabled = false;
        else if (arg == "--fire")
            fire_enabled = true;
        else if (arg == "--conf" && i + 1 < argc)
            conf = std::atof(argv[++i]);
        else if (arg == "--kconf" && i + 1 < argc)
            kconf = std::atof(argv[++i]);
        else if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--save-every" && i + 1 < argc)
            save_every = std::atoi(argv[++i]);
        else if (arg == "--outdir" && i + 1 < argc)
            out_dir = argv[++i];
        else if (arg == "--extend" && i + 1 < argc)
            extend_ratio = std::atof(argv[++i]);
        else if (arg == "--color" && i + 1 < argc)
            my_color = std::string(argv[++i]) == "blue" ? 1 : 0;
        else if (arg == "--hold" && i + 1 < argc)
        {
            hold_yaw = std::atof(argv[++i]);
            if (i + 1 < argc)
                hold_pitch = std::atof(argv[++i]);
        }
        else if (arg == "--map" && i + 4 < argc)
        {
            // --map YAW_SCALE YAW_OFFSET PITCH_SCALE PITCH_OFFSET
            yaw_scale = std::atof(argv[++i]);
            yaw_offset_deg = std::atof(argv[++i]);
            pitch_scale = std::atof(argv[++i]);
            pitch_offset_deg = std::atof(argv[++i]);
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    std::filesystem::create_directories(out_dir + "/stage1", ec);
    std::filesystem::create_directories(out_dir + "/raw", ec);
    std::filesystem::create_directories(out_dir + "/bin", ec);

    const std::string ipc_dir = default_ipc_directory();
    TalosMetadataMapping mapping;
    if (const auto status = mapping.open(ipc_dir + "/talos_ipc_meta"); !status.ok())
    {
        LOG(ERROR) << "metadata open failed: " << status.message;
        return 1;
    }

    TcpImageClient client;
    if (const auto status = client.connect(); !status.ok())
    {
        LOG(ERROR) << "sim connect failed: " << status.message;
        return 1;
    }
    LOG(INFO) << "[rune_stage] connected to simulator; ipc=" << ipc_dir;

    UdpGimbalClient gimbal;

    RuneDetector detector(model_path, conf, kconf, 30.0F);

    SimTf sim_tf;
    sim_tf.init();

    // 算法各阶段对象（进程内单套，等价于 PowerRuneProcessor 内部的成员）
    RuneObservationRefiner refiner;
    PowerRunePlane plane;
    PhaseMotionEstimator estimator;
    RuneDecisionModule decision;

    std::uint64_t last_sequence = 0;
    int frames_done = 0;
    int acquisitions = 0; // is_find_buff 为真的帧数
    int fire_advices = 0;

    LOG(INFO) << "[rune_stage] entering loop; aim=" << (aim_enabled ? 1 : 0)
              << " fire=" << (fire_enabled ? 1 : 0) << " conf=" << conf
              << " kconf=" << kconf << " extend=" << extend_ratio;

    while (max_frames == 0 || frames_done < max_frames)
    {
        auto frame = client.waitForLatest(last_sequence, std::chrono::milliseconds(1000));
        if (!frame.ok())
        {
            LOG_EVERY_N(WARNING, 120) << "[rune_stage] nextFrame failed: " << frame.status.message;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        const auto &sdk_frame = *frame.value;
        const std::uint64_t sequence = sdk_frame.header.source_sequence;
        last_sequence = sequence;

        auto reader = mapping.reader();
        if (!reader.ok())
        {
            LOG_EVERY_N(WARNING, 120) << "[rune_stage] reader unavailable: " << reader.status.message;
            continue;
        }
        auto frame_snapshot = reader.value->readGroundTruthForFrame(sequence);
        if (!frame_snapshot.ok())
        {
            LOG_EVERY_N(WARNING, 120) << "[rune_stage] no exposure state; seq=" << sequence;
            continue;
        }
        const auto &exposure = frame_snapshot.value->exposure_state;

        // ECS 映射（与 sim_bridge/SimCameraManager 完全一致）
        // Q = V^T · A · R_ch^T · R_cam · A^T · M_hc^T：相机为 FLU 四元数，
        // 算法 camera_frame 为 RDF（右下前），R_cam 后补 A^T 得到 RDF->车系旋转。
        const M3 r_chassis = quat_wxyz_to_rotation(exposure.chassis_quaternion_world_wxyz);
        const M3 r_camera = quat_wxyz_to_rotation(exposure.camera_quaternion_world_wxyz);
        const M3 q = mat_mul(mat_mul(mat_mul(mat_mul(mat_mul(mat_transpose(kVehicleFromEcs),
                                                              kStableToCarRdf),
                                                      mat_transpose(r_chassis)),
                                              r_camera),
                                      kStableToCarRdfTranspose),
                             kGimbalFromCameraTranspose);
        const double ecs_yaw = to_degree(std::atan2(q[3], q[0]));
        const double ecs_pitch = to_degree(std::atan2(-q[6], std::sqrt(q[7] * q[7] + q[8] * q[8])));
        const double ecs_roll = to_degree(std::atan2(q[7], q[8]));

        // 模拟器命令空间云台反馈（yaw_deg 与 UdpGimbalCommand 同零点；pitch 90=水平瞄准）
        double cmd_yaw = 0.0;
        double cmd_pitch = 90.0;
        if (const auto gs = reader.value->readRuntimeState(); gs.ok())
        {
            cmd_yaw = to_degree(gs.value->gimbal_yaw_rad);
            cmd_pitch = 90.0 + to_degree(gs.value->gimbal_pitch_rad);
        }

        cv::Mat image = to_bgr_image(sdk_frame);
        if (image.empty())
            continue;

        sim_tf.update(ecs_yaw, ecs_pitch, ecs_roll);
        const timetool::Timestamp timestamp = timetool::now();

        // 定角保持模式（标定云台命令空间用）
        if (hold_yaw.has_value())
        {
            UdpGimbalCommand command;
            command.yaw_deg = static_cast<float>(*hold_yaw);
            command.pitch_deg = static_cast<float>(hold_pitch.value_or(90.0));
            command.fire_advice = fire_enabled;
            gimbal.send(command);
        }

        // ---------- 阶段 0：NN 检测 ----------
        const std::vector<Detection> detections = detector.infer(image);

        std::printf("\n=== frame %d seq=%llu t=%lld ecs(yaw=%.2f pitch=%.2f roll=%.2f)"
                    " cmd(yaw=%.2f pitch=%.2f) dets=%zu ===\n",
                    frames_done, static_cast<unsigned long long>(sequence),
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                        timestamp.time_since_epoch())
                                               .count()),
                    ecs_yaw, ecs_pitch, ecs_roll, cmd_yaw, cmd_pitch, detections.size());
        std::printf("  [pose] ch=(%.3f,%.3f,%.3f) cq=(%.4f,%.4f,%.4f,%.4f)"
                    " cam=(%.3f,%.3f,%.3f) mq=(%.4f,%.4f,%.4f,%.4f)\n",
                    exposure.chassis_position_world[0], exposure.chassis_position_world[1],
                    exposure.chassis_position_world[2], exposure.chassis_quaternion_world_wxyz[0],
                    exposure.chassis_quaternion_world_wxyz[1], exposure.chassis_quaternion_world_wxyz[2],
                    exposure.chassis_quaternion_world_wxyz[3], exposure.camera_position_world[0],
                    exposure.camera_position_world[1], exposure.camera_position_world[2],
                    exposure.camera_quaternion_world_wxyz[0], exposure.camera_quaternion_world_wxyz[1],
                    exposure.camera_quaternion_world_wxyz[2], exposure.camera_quaternion_world_wxyz[3]);
        if (const auto gt = reader.value->readGroundTruthForFrame(sequence); gt.ok())
        {
            const auto &batch = gt.value->ground_truth;
            std::printf("  [truth] runes=%u", batch.rune_count);
            for (std::uint32_t ri = 0; ri < batch.rune_count && ri < 4; ++ri)
            {
                const auto &r = batch.runes[ri];
                std::printf(" | r%u team=%u c=(%.3f,%.3f,%.3f) ang=%.2f act=%u%u%u%u%u",
                            ri, r.team, r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2],
                            r.current_angle, r.target_activations[0], r.target_activations[1],
                            r.target_activations[2], r.target_activations[3], r.target_activations[4]);
            }
            std::printf("\n");
        }
        for (std::size_t i = 0; i < detections.size(); ++i)
        {
            const auto &det = detections[i];
            std::printf("  [det %zu] cls=%d conf=%.3f R=(%.0f,%.0f) top=(%.0f,%.0f) left=(%.0f,%.0f)"
                        " right=(%.0f,%.0f) bottom=(%.0f,%.0f)\n",
                        i, det.class_id, det.confidence, det.keypoints[2].x, det.keypoints[2].y,
                        det.keypoints[0].x, det.keypoints[0].y, det.keypoints[1].x,
                        det.keypoints[1].y, det.keypoints[3].x, det.keypoints[3].y,
                        det.keypoints[4].x, det.keypoints[4].y);
        }

        RuneObservation observation = build_rune_observation(
            image, timestamp, sim_tf.tree, true /*is_big_rune*/, my_color, detections, extend_ratio);

        // ---------- 阶段 1：refiner（轮廓提取 + 约束 + 描述符） ----------
        RefinedRuneObservation refined = refiner.refine(observation);
        for (std::size_t i = 0; i < refined.rune_blade_2D.size(); ++i)
        {
            const SingleRuneBlade2D &blade = refined.rune_blade_2D[i];
            const ConstrainedContours &cc = blade.constrained_contours;
            const auto raw = raw_extract_contours(blade.rune_info, -1.0);
            std::printf("  [blade %zu] %s view_rect=(%d,%d,%dx%d) cls=%d | raw_contours=%zu\n",
                        i, state_name(blade.rune_state), blade.rune_info.view_rect.x,
                        blade.rune_info.view_rect.y, blade.rune_info.view_rect.width,
                        blade.rune_info.view_rect.height, blade.rune_info.class_id, raw.size());
            for (std::size_t c = 0; c < raw.size(); ++c)
            {
                const double area = cv::contourArea(raw[c]);
                const cv::Rect bbox = cv::boundingRect(raw[c]);
                std::printf("      raw[%zu] pts=%zu area=%.0f bbox=(%d,%d,%dx%d)\n", c,
                            raw[c].size(), area, bbox.x, bbox.y, bbox.width, bbox.height);
            }
            std::printf("      -> armor:%s(usable=%d) light_arm:%s(usable=%d) R:%s(usable=%d)\n",
                        cc.armor_module_opt.has_value() ? "yes" : "NO",
                        static_cast<int>(blade.is_armor_module_usable),
                        cc.light_arm_opt.has_value() ? "yes" : "NO",
                        static_cast<int>(blade.is_light_arm_usable),
                        cc.center_R_opt.has_value() ? "yes" : "NO",
                        static_cast<int>(blade.is_center_R_usable));
            diagnose_contours(blade.rune_info, raw);
        }

        // ---------- 阶段 2：符平面（投影/切比雪夫） ----------
        plane.update_power_rune_plane(refined);
        InactiveTargets inactive = plane.get_inactive_targets();
        inactive.is_big_rune = true; // 与 PowerRuneProcessor 一致（大符模式）
        std::printf("  [plane] is_valid=%d pieces=%zu", static_cast<int>(inactive.is_vaild),
                    inactive.rune_pieces.size());
        if (inactive.is_vaild)
        {
            for (const auto &piece : inactive.rune_pieces)
                std::printf(" | c=(%.2f,%.2f,%.2f) a=(%.2f,%.2f,%.2f)", piece.rune_center.x(),
                            piece.rune_center.y(), piece.rune_center.z(), piece.armor_center.x(),
                            piece.armor_center.y(), piece.armor_center.z());
        }
        std::printf("\n");

        // ---------- 阶段 3：相位/运动估计 ----------
        if (inactive.is_vaild)
        {
            estimator.estimate_phase_motion(inactive);
            std::optional<std::vector<RuneTarget>> targets = estimator.try_get_rune_targets();
            if (!targets.has_value())
            {
                std::printf("  [phase] no filtered targets yet\n");
            }
            else
            {
                std::printf("  [phase] targets=%zu\n", targets->size());
                for (const auto &target : *targets)
                {
                    std::printf("      target phase=%.3f armor=(%.2f,%.2f,%.2f) center=(%.2f,%.2f,%.2f)"
                                " n=%d\n",
                                target.phase, target.armor_module_center.x(),
                                target.armor_module_center.y(), target.armor_module_center.z(),
                                target.rune_center.x(), target.rune_center.y(),
                                target.rune_center.z(), target.inactivate_target_num);
                }

                // ---------- 阶段 4：决策（弹道） ----------
                decision.update_decision_module(std::move(targets.value()));
            }
        }

        const power_rune::RuneSendData send = decision.get_send_data(true);
        if (send.is_find_buff)
            ++acquisitions;
        std::printf("  [decision] find_buff=%d enable_fire=%d mode=%d yaw=%.2fdeg pitch=%.2fdeg\n",
                    static_cast<int>(send.is_find_buff), static_cast<int>(send.is_enable_fire),
                    static_cast<int>(send.mode),
                    send.is_find_buff ? to_degree(-static_cast<double>(send.yaw)) : 0.0,
                    send.is_find_buff ? to_degree(static_cast<double>(send.pitch)) : 0.0);

        // ---------- 闭环：把算法输出打给模拟器云台 ----------
        if (aim_enabled && send.is_find_buff)
        {
            UdpGimbalCommand command;
            command.yaw_deg = static_cast<float>(yaw_scale * to_degree(static_cast<double>(send.yaw)) +
                                                 yaw_offset_deg);
            command.pitch_deg = static_cast<float>(
                pitch_scale * to_degree(static_cast<double>(send.pitch)) + pitch_offset_deg);
            command.fire_advice = fire_enabled && send.is_enable_fire;
            if (command.fire_advice)
                ++fire_advices;
            const auto status = gimbal.send(command);
            if (!status.ok())
            {
                LOG_EVERY_N(WARNING, 60) << "[rune_stage] gimbal send failed: " << status.message;
            }
        }

        // ---------- 落盘单帧调试图（阶段1的轮廓可视化） ----------
        if (save_every > 0 && frames_done % save_every == 0)
        {
            cv::Mat canvas = image.clone();
            for (const auto &blade : refined.rune_blade_2D)
            {
                cv::rectangle(canvas, blade.rune_info.view_rect, cv::Scalar(128, 128, 128), 1);
                const auto draw = [&](const std::optional<std::vector<cv::Point>> &contour,
                                      const cv::Scalar &color, const char *label, bool usable)
                {
                    if (!contour.has_value())
                        return;
                    std::vector<std::vector<cv::Point>> cs{contour.value()};
                    cv::drawContours(canvas, cs, -1, usable ? color : cv::Scalar(80, 80, 80), 2);
                    cv::putText(canvas, label, contour.value()[0], cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                usable ? color : cv::Scalar(80, 80, 80), 2);
                };
                draw(blade.constrained_contours.armor_module_opt, cv::Scalar(0, 255, 0), "Armor",
                     blade.is_armor_module_usable);
                draw(blade.constrained_contours.light_arm_opt, cv::Scalar(0, 255, 255), "Light",
                     blade.is_light_arm_usable);
                draw(blade.constrained_contours.center_R_opt, cv::Scalar(255, 255, 255), "R",
                     blade.is_center_R_usable);
                for (const auto &kpt : {blade.rune_info.top, blade.rune_info.left,
                                        blade.rune_info.point_R, blade.rune_info.right,
                                        blade.rune_info.bottom})
                    cv::circle(canvas, kpt, 4, cv::Scalar(255, 0, 255), cv::FILLED);
            }
            char name[256];
            std::snprintf(name, sizeof(name), "%s/stage1/f%04d_seq%llu.png", out_dir.c_str(),
                          frames_done, static_cast<unsigned long long>(sequence));
            cv::imwrite(name, canvas);

            // 原始帧
            std::snprintf(name, sizeof(name), "%s/raw/f%04d_seq%llu.png", out_dir.c_str(),
                          frames_done, static_cast<unsigned long long>(sequence));
            cv::imwrite(name, image);

            // 各符叶视图的二值掩膜（贴回原图位置）+ 原始轮廓
            cv::Mat bin_canvas(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
            int color_index = 0;
            const cv::Scalar palette[] = {
                cv::Scalar(0, 255, 0), cv::Scalar(255, 128, 0), cv::Scalar(0, 128, 255),
                cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255)};
            for (const auto &blade : refined.rune_blade_2D)
            {
                cv::Mat mask;
                const auto raw = raw_extract_contours(blade.rune_info, -1.0, &mask);
                if (!mask.empty())
                {
                    cv::Mat roi = bin_canvas(blade.rune_info.view_rect);
                    for (int y = 0; y < mask.rows && y < roi.rows; ++y)
                        for (int x = 0; x < mask.cols && x < roi.cols; ++x)
                            if (mask.at<unsigned char>(y, x) != 0)
                                roi.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                }
                for (const auto &contour : raw)
                {
                    std::vector<std::vector<cv::Point>> cs{contour};
                    cv::drawContours(bin_canvas, cs, -1, palette[color_index % 6], 1);
                }
                cv::rectangle(bin_canvas, blade.rune_info.view_rect, cv::Scalar(90, 90, 90), 1);
                cv::putText(bin_canvas, std::to_string(color_index),
                            cv::Point(blade.rune_info.view_rect.x, blade.rune_info.view_rect.y + 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, palette[color_index % 6], 2);
                ++color_index;
            }
            std::snprintf(name, sizeof(name), "%s/bin/f%04d_seq%llu.png", out_dir.c_str(),
                          frames_done, static_cast<unsigned long long>(sequence));
            cv::imwrite(name, bin_canvas);
        }

        ++frames_done;
    }

    std::printf("\n=== summary: frames=%d find_buff_frames=%d fire_advices=%d ===\n", frames_done,
                acquisitions, fire_advices);
    return 0;
}
