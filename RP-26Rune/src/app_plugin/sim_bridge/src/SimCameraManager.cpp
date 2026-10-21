#include "SimCameraManager.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include "class_loader.hpp"
#include "json.hpp"
#include "time/time.hpp"

namespace app_plugin
{

namespace
{

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

// 算法侧硬编码约定（与 TrackerManager 的 TF 树一致，见 docs/坐标系与标定.md §5）。
constexpr M3 kVehicleFromEcs{0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};
constexpr M3 kStableToCarRdf{0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};
// A^T = FLU->RDF 的转置；算法 camera_frame/car_frame 均为 RDF(右下前) 约定，
// 模拟器的相机四元数是 FLU，须在 R_cam 之后补 A^T 才能得到 RDF->车系旋转。
constexpr M3 kStableToCarRdfTranspose{0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0};
constexpr M3 kGimbalFromCameraTranspose{0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0};

double to_degree(double rad)
{
    return rad * 57.29577951308232;
}

std::string default_ipc_directory()
{
    if (const char *configured = std::getenv("TALOS_IPC_DIR");
        configured != nullptr && configured[0] != '\0')
        return configured;

    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    const std::string base = xdg != nullptr ? xdg : "/tmp";
    return base + "/daedalus-learning-" + std::to_string(::getuid());
}

cv::Mat to_bgr_image(const daedalus::sim::sdk::v1::TcpImageFrame &frame)
{
    using daedalus::sim::sdk::v1::tcp_image::PixelFormat;

    const auto &header = frame.header;
    if (header.width == 0 || header.height == 0)
    {
        LOG_EVERY_N(ERROR, 60) << "[SimCameraManager] frame has zero size";
        return {};
    }

    const int channels = header.format == PixelFormat::Rgba32 ? 4 : 3;
    const std::size_t expected = static_cast<std::size_t>(header.width) *
                                 static_cast<std::size_t>(header.height) *
                                 static_cast<std::size_t>(channels);
    if (frame.payload.size() < expected)
    {
        LOG_EVERY_N(ERROR, 60) << "[SimCameraManager] payload too small: "
                               << frame.payload.size() << " < " << expected;
        return {};
    }

    const cv::Mat view(static_cast<int>(header.height), static_cast<int>(header.width),
                       channels == 4 ? CV_8UC4 : CV_8UC3,
                       const_cast<std::uint8_t *>(frame.payload.data()));
    cv::Mat bgr;
    cv::cvtColor(view, bgr, channels == 4 ? cv::COLOR_RGBA2BGR : cv::COLOR_RGB2BGR);
    return bgr;
}

bool write_ppm(const std::filesystem::path &path, const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    FILE *file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return false;
    std::fprintf(file, "P6\n%d %d\n255\n", rgb.cols, rgb.rows);
    const std::size_t row_bytes = static_cast<std::size_t>(rgb.cols) * 3;
    for (int row = 0; row < rgb.rows; ++row)
        std::fwrite(rgb.ptr(row), 1, row_bytes, file);
    std::fclose(file);
    return true;
}

} // namespace

struct SimCameraManager::Impl
{
    using TcpImageClient = daedalus::sim::sdk::v1::TcpImageClient;
    using TalosMetadataMapping = daedalus::sim::sdk::v1::TalosMetadataMapping;
    using EcsSubscriber = LatestChannel<ECSData>::Subscriber;
    using FramePublisher = LatestBuffer<InputFrame>::Publisher;

    void load_config();
    bool ensure_connected();

    std::string ipc_directory;
    int connect_retry_ms = 1000;
    int frame_timeout_ms = 1000;
    ECSData fallback_ecs{MyColor::Red, AimMode::BigRune, true, true, 0.0, 0.0, 0.0};

    std::unique_ptr<TcpImageClient> client;
    TalosMetadataMapping mapping;
    std::optional<EcsSubscriber> ecs_input;
    std::optional<FramePublisher> output;

    ECSData channel_ecs{};
    bool has_channel_ecs = false;
    bool mapping_open = false;
    bool connected = false;
    bool initialized = false;
    bool camera_info_logged = false;
    std::uint64_t last_sequence = 0;
    std::chrono::steady_clock::time_point next_connect_attempt{};

    // 调试用帧落盘（联调诊断，不影响正式链路）
    bool dump_enabled = false;
    int dump_every_n = 10;
    std::filesystem::path dump_dir;
    std::ofstream dump_gt_file;
    std::uint64_t dump_count = 0;
};

void SimCameraManager::Impl::load_config()
{
    J_SIM_BRIDGE.updateJson();
    if (!J_SIM_BRIDGE.config_.isOpened())
        throw std::runtime_error(
            "failed to open sim bridge config: " +
            (SIM_BRIDGE_CONFIG_DIR / "sim_bridge.json").string());

    const cv::FileStorage &config = J_SIM_BRIDGE.config_;
    const cv::FileNode sim = config["sim"];
    if (!sim.empty())
    {
        if (!sim["ipc_directory"].empty())
        {
            std::string configured;
            sim["ipc_directory"] >> configured;
            ipc_directory = configured.empty() ? default_ipc_directory() : configured;
        }
        if (!sim["connect_retry_ms"].empty())
            connect_retry_ms = static_cast<int>(sim["connect_retry_ms"]);
        if (!sim["frame_timeout_ms"].empty())
            frame_timeout_ms = static_cast<int>(sim["frame_timeout_ms"]);
    }
    if (ipc_directory.empty())
        ipc_directory = default_ipc_directory();

    const cv::FileNode aim = config["aim"];
    if (!aim.empty())
    {
        if (!aim["mode"].empty())
            fallback_ecs.mode = static_cast<AimMode>(static_cast<int>(aim["mode"]));
        if (!aim["my_color"].empty())
            fallback_ecs.my_color = static_cast<MyColor>(static_cast<int>(aim["my_color"]));
        if (!aim["is_start"].empty())
            fallback_ecs.is_start = static_cast<int>(aim["is_start"]) != 0;
        if (!aim["is_ready"].empty())
            fallback_ecs.is_ready = static_cast<int>(aim["is_ready"]) != 0;
    }

    client = std::make_unique<TcpImageClient>();

    const cv::FileNode dump = config["debug_frame_dump"];
    if (!dump.empty())
    {
        if (!dump["enabled"].empty())
            dump_enabled = static_cast<int>(dump["enabled"]) != 0;
        if (!dump["every_n"].empty())
        {
            dump_every_n = static_cast<int>(dump["every_n"]);
            if (dump_every_n < 1)
                dump_every_n = 1;
        }
        std::string dir;
        if (!dump["dir"].empty())
        {
            dump["dir"] >> dir;
            if (!dir.empty())
                dump_dir = dir;
        }
    }
    if (dump_enabled)
    {
        if (dump_dir.empty())
            dump_dir = "/tmp/rune_frames";
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        if (ec)
        {
            LOG(WARNING) << "[SimCameraManager] frame dump disabled; cannot create "
                         << dump_dir << ": " << ec.message();
            dump_enabled = false;
        }
        else
        {
            dump_gt_file.open(dump_dir / "gt.jsonl", std::ios::out | std::ios::app);
            LOG(INFO) << "[SimCameraManager] frame dump enabled; dir=" << dump_dir
                      << ", every_n=" << dump_every_n;
        }
    }
}

bool SimCameraManager::Impl::ensure_connected()
{
    if (connected)
        return true;

    const auto now = std::chrono::steady_clock::now();
    if (now < next_connect_attempt)
        return false;
    next_connect_attempt = now + std::chrono::milliseconds(connect_retry_ms);

    if (!mapping_open)
    {
        const auto status = mapping.open(ipc_directory + "/talos_ipc_meta");
        if (!status.ok())
        {
            LOG(WARNING) << "[SimCameraManager] metadata mapping open failed: " << status.message
                         << " (ipc=" << ipc_directory << ')';
            return false;
        }
        mapping_open = true;
    }

    if (!camera_info_logged)
    {
        auto reader = mapping.reader();
        if (reader.ok())
        {
            auto camera_info = reader.value->readCameraInfo();
            if (camera_info.ok())
            {
                const auto &ci = *camera_info.value;
                LOG(INFO) << "[SimCameraManager] sim camera info: fx=" << ci.fx << ", fy=" << ci.fy
                          << ", cx=" << ci.cx << ", cy=" << ci.cy
                          << ", size=" << ci.width << 'x' << ci.height;
            }
        }
        camera_info_logged = true;
    }

    const auto status = client->connect();
    if (!status.ok())
    {
        LOG(WARNING) << "[SimCameraManager] simulator connect failed: " << status.message;
        return false;
    }
    connected = true;
    last_sequence = 0;
    LOG(INFO) << "[SimCameraManager] connected to simulator; ipc=" << ipc_directory;
    return true;
}

SimCameraManager::SimCameraManager()
    : m_impl(std::make_unique<Impl>())
{
}

SimCameraManager::~SimCameraManager() = default;

void SimCameraManager::process(const app::Context &context)
{
    Impl &impl = *m_impl;

    if (!impl.initialized)
    {
        impl.load_config();
        impl.ecs_input.emplace(context.get_channel_subscriber<ECSData>(this, {}, true));
        impl.output.emplace(context.get_buffer_publisher<InputFrame>(this));
        impl.initialized = true;
        LOG(INFO) << "[SimCameraManager] initialized; ipc_directory=" << impl.ipc_directory;
    }

    if (!impl.ensure_connected())
        return;

    // 失败即返回会饿死同线程组下游插件（wait_pop 无限阻塞）；对 nextFrame 的
    // 瞬时失败原地重试——曝光历史窗口越界等瞬态在下一次尝试即可自愈。
    auto frame = impl.client->waitForLatest(
        impl.last_sequence, std::chrono::milliseconds(impl.frame_timeout_ms));
    while (!frame.ok())
    {
        LOG_EVERY_N(WARNING, 60) << "[SimCameraManager] nextFrame failed: " << frame.status.message;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        frame = impl.client->waitForLatest(
            impl.last_sequence, std::chrono::milliseconds(impl.frame_timeout_ms));
    }
    const auto &sdk_frame = *frame.value;
    const std::uint64_t sequence = sdk_frame.header.source_sequence;
    impl.last_sequence = sequence;

    auto reader = impl.mapping.reader();
    if (!reader.ok())
    {
        LOG_EVERY_N(WARNING, 60) << "[SimCameraManager] metadata reader unavailable: "
                                 << reader.status.message;
        return;
    }

    auto snapshot = reader.value->readGroundTruthForFrame(sequence);
    if (!snapshot.ok())
    {
        LOG_EVERY_N(WARNING, 60) << "[SimCameraManager] no exposure state for seq=" << sequence
                                 << ": " << snapshot.status.message;
        return;
    }
    // ABI 1 把同帧曝光位姿与验收真值放在同一快照中。此处只取
    // exposure_state 构造 ECSData，不把 ground_truth 传入识别或打符算法。
    const auto &exposure = snapshot.value->exposure_state;

    // 通道里的最新 ECS 提供 mode/my_color/is_start/is_ready 等策略位；
    // yaw/pitch/roll 必须来自与图像严格同帧的 ExposureState。
    if (const auto latest_ecs = impl.ecs_input->try_get_new())
    {
        impl.channel_ecs = *latest_ecs;
        impl.has_channel_ecs = true;
    }

    ECSData ecs = impl.has_channel_ecs ? impl.channel_ecs : impl.fallback_ecs;
    const M3 r_chassis = quat_wxyz_to_rotation(exposure.chassis_quaternion_world_wxyz);
    const M3 r_camera = quat_wxyz_to_rotation(exposure.camera_quaternion_world_wxyz);
    // Q = V^T · A · R_ch^T · R_cam · A^T · M_hc^T
    // （目标 V·RzRyRx·M_hc = A·R_ch^T·R_cam·A^T：相机 RDF -> 车系 RDF，见 docs/坐标系与标定.md）
    const M3 q = mat_mul(
        mat_mul(
            mat_mul(
                mat_mul(
                    mat_mul(mat_transpose(kVehicleFromEcs), kStableToCarRdf),
                    mat_transpose(r_chassis)),
                r_camera),
            kStableToCarRdfTranspose),
        kGimbalFromCameraTranspose);
    ecs.yaw = to_degree(std::atan2(q[3], q[0]));
    ecs.pitch = to_degree(std::atan2(-q[6], std::sqrt(q[7] * q[7] + q[8] * q[8])));
    ecs.roll = to_degree(std::atan2(q[7], q[8]));

    cv::Mat image = to_bgr_image(sdk_frame);
    if (image.empty())
        return;

    if (impl.dump_enabled && sequence % static_cast<std::uint64_t>(impl.dump_every_n) == 0)
    {
        const std::filesystem::path frame_path =
            impl.dump_dir / ("seq_" + std::to_string(sequence) + ".ppm");
        const bool written = write_ppm(frame_path, image);
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(6);
        line << "{\"seq\":" << sequence
             << ",\"dump\":" << impl.dump_count++
             << ",\"ppm_ok\":" << (written ? 1 : 0)
             << ",\"cam_pos\":[" << exposure.camera_position_world[0] << ','
             << exposure.camera_position_world[1] << ',' << exposure.camera_position_world[2]
             << "],\"cam_q_wxyz\":[" << exposure.camera_quaternion_world_wxyz[0] << ','
             << exposure.camera_quaternion_world_wxyz[1] << ','
             << exposure.camera_quaternion_world_wxyz[2] << ','
             << exposure.camera_quaternion_world_wxyz[3] << "],\"chassis_q_wxyz\":["
             << exposure.chassis_quaternion_world_wxyz[0] << ','
             << exposure.chassis_quaternion_world_wxyz[1] << ','
             << exposure.chassis_quaternion_world_wxyz[2] << ','
             << exposure.chassis_quaternion_world_wxyz[3] << "]";

        auto ground_truth = reader.value->readGroundTruthForFrame(sequence);
        if (ground_truth.ok())
        {
            const auto &batch = ground_truth.value->ground_truth;
            line << ",\"rune_count\":" << batch.rune_count << ",\"runes\":[";
            for (std::uint32_t index = 0; index < batch.rune_count; ++index)
            {
                const auto &rune = batch.runes[index];
                if (index > 0)
                    line << ',';
                line << "{\"team\":" << static_cast<int>(rune.team)
                     << ",\"rune_mode\":" << static_cast<int>(rune.rune_mode)
                     << ",\"mech\":" << static_cast<int>(rune.mechanism_state)
                     << ",\"r_center_odom\":[" << rune.r_center_odom[0] << ','
                     << rune.r_center_odom[1] << ',' << rune.r_center_odom[2] << ']'
                     << ",\"radius\":" << rune.radius
                     << ",\"angle\":" << rune.current_angle
                     << ",\"v_roll\":" << rune.v_roll
                     << ",\"dir\":" << rune.direction
                     << ",\"blade\":" << rune.blade_id
                     << ",\"acts\":[" << static_cast<int>(rune.target_activations[0]) << ','
                     << static_cast<int>(rune.target_activations[1]) << ','
                     << static_cast<int>(rune.target_activations[2]) << ','
                     << static_cast<int>(rune.target_activations[3]) << ','
                     << static_cast<int>(rune.target_activations[4]) << "]}";
            }
            line << ']';
        }
        else
        {
            line << ",\"rune_count\":-1";
        }
        line << "}\n";
        if (impl.dump_gt_file.is_open())
        {
            impl.dump_gt_file << line.str();
            impl.dump_gt_file.flush();
        }
    }

    LOG_EVERY_N(INFO, 120) << "[SimCameraManager] seq=" << sequence
                           << " ecs yaw=" << ecs.yaw << " pitch=" << ecs.pitch
                           << " roll=" << ecs.roll;

    InputFrame out;
    out.ecs_data = ecs;
    out.timestamp = timetool::now();
    out.img = std::move(image);
    impl.output->push(std::move(out));
}

} // namespace app_plugin

REGISTER_PLUGIN("CameraManager", app_plugin::SimCameraManager)
