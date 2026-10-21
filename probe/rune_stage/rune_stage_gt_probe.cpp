// 打符 GT 真值版链路探针（验证 PnP 之后的链路；不改动算法代码）：
//
// 目的：把"识别模块"的输出边界显式化，并绕过识别（NN/refiner/PnP），直接从
// 模拟器真值构造该边界，单独验证其后的全部链路：
//     InactiveTargets -> PhaseMotionEstimator -> RuneDecisionModule -> 瞄准/开火
//
// 识别模块输出契约（boundary） = InactiveTargets（power_rune_global.hpp）：
//   struct InactiveTargets {
//     struct RunePiece { Eigen::Vector3d rune_center;   // 符心（旋转中心）
//                        Eigen::Vector3d armor_center; };// 靶心（该符叶 R 环中心，轨道半径约 0.7m）
//     std::vector<RunePiece> rune_pieces;           // 未击打(亮起)符叶；大符 1~2 片
//     Eigen::Hyperplane<double,3> power_rune_plane; // 符所在平面（法向=转轴）
//     foxglove::Time capture_timestamp;             // 对应原图拍摄时间（单调，用于滤波器 dt）
//     bool is_big_rune;
//     bool is_vaild;                                // false 时上层直接 return（不喂后续模块）
//   };
//   坐标系：算法车系 car_frame（RDF：x 右 y 下 z 前），单位 m。
//
// 本探针的真值来源（与识别模块无关，独立构造）：
//   GroundTruthRune.target_activations/current_angle + 符面几何（export_labeled_frames
//   --self-test 实测常量）：
//     靶心世界坐标 = origin + Rot(轴, current_angle)·(u*w 平面内 (umid,phi) 极坐标点 + n 偏置)
//     符心 = GT r_center_odom；平面法向 = 符轴
//   再经坐标链转到算法车系：
//     world(Bevy Y-up) -> odom：o = (-w_z, -w_x, w_y)
//     odom -> car       ：p_car = A·R_chᵀ·(p_odom - P_chassis)，A = kStableToCarRdf
//   （该链与 probe_chain.cpp 同口径，实跑残差 5.7e-5，见 docs/坐标系与标定.md §4）
//
// 运行前提：
//   1) 模拟器运行中，且 app 未运行（模拟器图像流只服务单客户端）；
//   2) 二进制必须位于 RP-26Rune/output/ 下（power_rune 库内 J_POWER_RUNE 用
//      /proc/self/exe 相对路径找 config/power_rune.json）。
// usage: rune_stage_gt_probe [--frames N] [--no-aim] [--fire] [--fire-count N] [--fire-after M]
//                            [--team red|blue] [--map YSCALE YOFF PSCALE POFF]
//                            [--settle-angle DEG]
//                            [--outdir DIR] [--save-every N]
//   新 SDK 的命令空间明确定义 pitch=90° 为水平瞄准。默认映射因此为
//   cmd_yaw=-algorithm_yaw, cmd_pitch=algorithm_pitch+90°；--map 仅用于外部标定覆盖。

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>
#include <glog/logging.h>

#include "PhaseMotionEstimator.hpp"
#include "RuneDecisionModule.hpp"
#include "common/power_rune_global.hpp"
#include "common/power_rune_function.hpp"
#include "time/time.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{

using namespace daedalus::sim::sdk::v1;

// ---------- 简单线性代数（与 probe_chain.cpp 同口径） ----------
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

// 模拟器底盘(FLU) -> 算法车系(RDF)；与 probe_chain 验证一致
constexpr M3 kStableToCarRdf{0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0};

double to_degree(double rad) { return rad * 57.29577951308232; }

// 新入口的 UDP v1.3 命令要求 command_id。元数据则仍发布 ABI1，
// 因此在接口层单独适配 JSON，不猜测/复制 SHM 布局。
class TrackedGimbalSender
{
public:
    TrackedGimbalSender()
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
        endpoint_.sin_family = AF_INET;
        endpoint_.sin_port = htons(5601);
        endpoint_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }

    ~TrackedGimbalSender()
    {
        if (fd_ >= 0)
            ::close(fd_);
    }

    bool send(double yaw_deg, double pitch_deg, bool fire_advice, std::uint64_t *command_id)
    {
        if (fd_ < 0)
            return false;
        const std::uint64_t id = next_command_id_++;
        char payload[256];
        const int size = std::snprintf(
            payload, sizeof(payload),
            "{\"command_id\":%llu,\"yaw_deg\":%.7g,\"pitch_deg\":%.7g,"
            "\"fire_advice\":%s}",
            static_cast<unsigned long long>(id), yaw_deg, pitch_deg,
            fire_advice ? "true" : "false");
        if (size <= 0 || static_cast<std::size_t>(size) >= sizeof(payload))
            return false;
        const auto sent = ::sendto(fd_, payload, static_cast<std::size_t>(size), 0,
                                   reinterpret_cast<const sockaddr *>(&endpoint_),
                                   sizeof(endpoint_));
        if (sent != size)
            return false;
        if (command_id != nullptr)
            *command_id = id;
        return true;
    }

private:
    int fd_ = -1;
    sockaddr_in endpoint_{};
    std::uint64_t next_command_id_ = 1;
};

// ---------- odom(模拟器 Z-up) <-> Bevy 世界(Y-up) ----------
// odom = (-w_z, -w_x, w_y)；逆 world = (-o_y, o_z, -o_x)（export_labeled_frames 实测口径）
void worldToOdom(const double w[3], double o[3])
{
    o[0] = -w[2];
    o[1] = -w[0];
    o[2] = w[1];
}

void odomToWorld(const double o[3], double w[3])
{
    w[0] = -o[1];
    w[1] = o[2];
    w[2] = -o[0];
}

// ---------- 符面几何（export_labeled_frames --self-test 实测常量，Bevy 世界系） ----------
// 靶心（符叶 R 环中心）在符面 (u,w,n) 坐标 = 极坐标 (umid, phi_deg) 绕符面原点旋转，
// 即 u' = umid*cosφ - wmid*sinφ, w' = umid*sinφ + wmid*cosφ，另加面外偏置 n_mid。
struct FaceTargetGeom
{
    double phi_deg;
    double umid;  // 径向均值（m）
    double wmid;  // 切向均值（m）
    double n_mid; // 面外偏置（m）
};

struct FaceGeom
{
    const char *name;
    double axis[3];          // 符轴（单位，双面共用）
    double u[3], w[3], n[3]; // 符面基（单位正交）
    int lit_sign;
    FaceTargetGeom targets[5];
};

const FaceGeom kFace1{
    "FACE_1(red)",
    {-0.7071, 0.0, -0.7071},
    {-0.3073, -0.9006, 0.3073},
    {0.6368, -0.4346, -0.6368},
    {-0.7071, 0.0, -0.7071},
    +1,
    {{0.0, 0.6995, 0.0005, 0.0082},
     {72.4, 0.6985, -0.0010, 0.0074},
     {144.2, 0.6985, 0.0005, 0.0074},
     {-143.6, 0.6980, -0.0015, 0.0074},
     {-71.8, 0.6990, 0.0010, 0.0074}},
};

const FaceGeom kFace2{
    "FACE_2(blue)",
    {-0.7071, 0.0, -0.7071},
    {-0.6231, 0.4728, 0.6231},
    {-0.3343, -0.8812, 0.3343},
    {-0.7071, 0.0, -0.7071},
    -1,
    {{0.0, 0.6990, -0.0010, -0.0074},
     {-71.9, 0.6990, -0.0010, -0.0097},
     {-144.0, 0.6995, -0.0010, -0.0074},
     {144.1, 0.6985, -0.0015, -0.0074},
     {72.0, 0.6975, 0.0000, -0.0074}},
};

void facePointWorld(const FaceGeom &f, const double origin[3], double uu, double ww, double nn,
                    double out[3])
{
    for (int i = 0; i < 3; ++i)
        out[i] = origin[i] + uu * f.u[i] + ww * f.w[i] + nn * f.n[i];
}

// 绕单位轴 a（过原点 o）按右手定则旋转 angle（同 export_labeled_frames::rotateAroundAxis）
void rotateAroundAxis(const double a[3], const double o[3], double angle, double p[3])
{
    double v[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
    const double c = std::cos(angle), s = std::sin(angle);
    const double d = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    double cr[3];
    cr[0] = a[1] * v[2] - a[2] * v[1];
    cr[1] = a[2] * v[0] - a[0] * v[2];
    cr[2] = a[0] * v[1] - a[1] * v[0];
    for (int i = 0; i < 3; ++i)
        p[i] = o[i] + v[i] * c + cr[i] * s + a[i] * d * (1.0 - c);
}

// ---------- 车系变换：p_car = A·R_chᵀ·(p_odom − P_chassis) ----------
struct CarFrame
{
    Eigen::Matrix3d R;       // car <- odom（旋转）
    Eigen::Vector3d chassis; // 底盘原点（odom）

    Eigen::Vector3d point(const double p_odom[3]) const
    {
        return R * (Eigen::Vector3d(p_odom[0], p_odom[1], p_odom[2]) - chassis);
    }
    Eigen::Vector3d dir(const double d_odom[3]) const
    {
        return R * Eigen::Vector3d(d_odom[0], d_odom[1], d_odom[2]);
    }
};

CarFrame make_car_frame(const ExposureState &exposure)
{
    const M3 r_ch = quat_wxyz_to_rotation(exposure.chassis_quaternion_world_wxyz);
    const M3 r_car = mat_mul(kStableToCarRdf, mat_transpose(r_ch));
    CarFrame cf;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cf.R(i, j) = r_car[i * 3 + j];
    cf.chassis = Eigen::Vector3d(exposure.chassis_position_world[0],
                                 exposure.chassis_position_world[1],
                                 exposure.chassis_position_world[2]);
    return cf;
}

// ---------- odom 点 -> 像素（叠加验证用；口径同 export_labeled_frames::projectOdom） ----------
bool project_odom(const double p_odom[3], const ExposureState &e, const CameraInfo &cam,
                  double *px, double *py)
{
    const float *q = e.camera_quaternion_world_wxyz;
    const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    const double dx = p_odom[0] - e.camera_position_world[0];
    const double dy = p_odom[1] - e.camera_position_world[1];
    const double dz = p_odom[2] - e.camera_position_world[2];
    // v = R_cam^T·(p_o - t_o)
    const double vx = (1 - 2 * (qy * qy + qz * qz)) * dx + 2 * (qx * qy + qz * qw) * dy +
                      2 * (qx * qz - qy * qw) * dz;
    const double vy = 2 * (qx * qy - qz * qw) * dx + (1 - 2 * (qx * qx + qz * qz)) * dy +
                      2 * (qy * qz + qx * qw) * dz;
    const double vz = 2 * (qx * qz + qy * qw) * dx + 2 * (qy * qz - qx * qw) * dy +
                      (1 - 2 * (qx * qx + qy * qy)) * dz;
    // Bevy 相机前向 -Z；p_cam = M^T·v，M^T = [[0,-1,0],[0,0,1],[-1,0,0]]
    const double x = -vy;
    const double y = vz;
    const double z = -vx;
    if (z >= -0.01)
        return false;
    *px = static_cast<double>(cam.cx) - static_cast<double>(cam.fx) * x / z;
    *py = static_cast<double>(cam.cy) + static_cast<double>(cam.fy) * y / z;
    return true;
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

std::string default_ipc_directory()
{
    if (const char *configured = std::getenv("TALOS_IPC_DIR");
        configured != nullptr && configured[0] != '\0')
        return configured;

    const char *xdg = std::getenv("XDG_RUNTIME_DIR");
    const std::string base = xdg != nullptr ? xdg : "/tmp";
    return base + "/daedalus-learning-" + std::to_string(::getuid());
}

} // namespace

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    int max_frames = 0; // 0 = 无限
    bool aim_enabled = true;
    bool fire_enabled = false;
    int fire_count_max = -1; // <0 = 不限次数
    int fire_after_frame = 160; // 避免大符模型刚达 100 样本时的短暂收敛段
    int team_index = 0; // 0=red(FACE_1) 1=blue(FACE_2)
    double yaw_scale = -1.0;
    double yaw_offset_deg = 0.0;
    double pitch_scale = 1.0;
    double pitch_offset_deg = 90.0;
    double settle_angle_deg = 0.50;
    std::string out_dir = "/tmp/rune_gt";
    int save_every = 0; // 0 = 不存帧

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc)
            max_frames = std::atoi(argv[++i]);
        else if (arg == "--no-aim")
            aim_enabled = false;
        else if (arg == "--fire")
            fire_enabled = true;
        else if (arg == "--fire-count" && i + 1 < argc)
            fire_count_max = std::atoi(argv[++i]);
        else if (arg == "--fire-after" && i + 1 < argc)
            fire_after_frame = std::atoi(argv[++i]);
        else if (arg == "--team" && i + 1 < argc)
            team_index = std::string(argv[++i]) == "blue" ? 1 : 0;
        else if (arg == "--outdir" && i + 1 < argc)
            out_dir = argv[++i];
        else if (arg == "--save-every" && i + 1 < argc)
            save_every = std::atoi(argv[++i]);
        else if (arg == "--map" && i + 4 < argc)
        {
            yaw_scale = std::atof(argv[++i]);
            yaw_offset_deg = std::atof(argv[++i]);
            pitch_scale = std::atof(argv[++i]);
            pitch_offset_deg = std::atof(argv[++i]);
        }
        else if (arg == "--settle-angle" && i + 1 < argc)
            settle_angle_deg = std::atof(argv[++i]);
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    const FaceGeom &face = team_index == 0 ? kFace1 : kFace2;

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
    CameraInfo camera_info{};
    bool camera_info_ok = false;
    if (auto reader = mapping.reader(); reader.ok())
    {
        if (const auto ci = reader.value->readCameraInfo(); ci.ok())
        {
            camera_info = *ci.value;
            camera_info_ok = true;
        }
    }

    TrackedGimbalSender gimbal;

    // 算法链路对象（等价于 PowerRuneProcessor 内部成员，边界从 InactiveTargets 开始）
    PhaseMotionEstimator estimator;
    RuneDecisionModule decision;

    std::FILE *csv = std::fopen((out_dir + "/gt_feed.csv").c_str(), "w");
    if (csv != nullptr)
        std::fprintf(csv, "frame,seq,ang_rad,v_roll_rad_s,dir,acts,lit,rune_cx,rune_cy,rune_cz,"
                          "armor_cx,armor_cy,armor_cz,phase,fit_A,fit_B,fit_b,fit_C,fit_w,fit_a,"
                          "fit_delta,find,fire,settled,cmd_id,cmd_yaw,cmd_pitch,"
                          "actual_yaw,actual_pitch,yaw_error,pitch_error,rune_dh,pitch_off\n");

    LOG(INFO) << "[gt_probe] 边界=InactiveTargets{pieces(符心/靶心,车系m), plane, capture_timestamp,"
                 " is_big_rune, is_vaild}; 真值源=GT(" << face.name
              << ")+符面几何; 车系链 p_car=A·R_chᵀ·(p_odom-P_chassis)";
    LOG(INFO) << "[gt_probe] aim=" << (aim_enabled ? 1 : 0) << " fire=" << (fire_enabled ? 1 : 0)
              << " map(yaw=" << yaw_scale << "," << yaw_offset_deg
              << " pitch_scale=" << pitch_scale << ") outdir=" << out_dir;
    LOG(INFO) << "[gt_probe] SDK=" << kSdkVersion << " ABI=" << kSdkAbiRevision
              << " pitch level=" << pitch_offset_deg << "deg; fire gate angle<="
              << settle_angle_deg << "deg for 2 consecutive exposure frames";

    std::uint64_t last_sequence = 0;
    int frames_done = 0;
    int acquisitions = 0; // is_find_buff 帧数
    int fire_advices = 0;
    int lock_frame = -1;
    int first_fire_frame = -1;
    int hit_count = 0;
    int last_lit = -1;
    int no_gt_streak = 0;
    double last_cmd_pitch = 90.0;
    int aligned_frame_streak = 0;
    std::uint64_t last_command_id = 0;
    std::array<std::uint8_t, 5> previous_activations{};
    bool have_previous_activations = false;
    double last_rune_dh = 0.0; // 符心水平距离（车系 xz），供自适应俯仰补偿
    bool have_rune_dh = false;

    while (max_frames == 0 || frames_done < max_frames)
    {
        auto frame = client.waitForLatest(last_sequence, std::chrono::milliseconds(1000));
        if (!frame.ok())
        {
            LOG_EVERY_N(WARNING, 120) << "[gt_probe] nextFrame failed: " << frame.status.message;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        const std::uint64_t sequence = frame.value->header.source_sequence;
        last_sequence = sequence;

        auto reader = mapping.reader();
        if (!reader.ok())
        {
            LOG_EVERY_N(WARNING, 120) << "[gt_probe] reader unavailable: " << reader.status.message;
            continue;
        }

        // 同帧一致性：GT + 曝光状态按 source_sequence 取；历史环瞬态时原地重试（同 sim_bridge 加固）
        auto snapshot = reader.value->readGroundTruthForFrame(sequence);
        for (int retry = 0; !snapshot.ok() && retry < 40; ++retry)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            snapshot = reader.value->readGroundTruthForFrame(sequence);
        }
        if (!snapshot.ok())
        {
            LOG_EVERY_N(WARNING, 60) << "[gt_probe] no ground truth for seq=" << sequence;
            continue;
        }
        const auto &batch = snapshot.value->ground_truth;
        const auto &exposure = snapshot.value->exposure_state;

        // ABI2 要求图像、GT 和曝光状态来自同一生产者的同一帧。
        const auto &image_header = frame.value->header;
        const auto shm_header = reader.value->readHeader();
        if (!shm_header.ok() || shm_header.value->created_ns != image_header.producer_epoch ||
            batch.frame_seq != sequence || exposure.frame_seq != sequence ||
            batch.timestamp_ns != image_header.capture_timestamp_ns ||
            exposure.timestamp_ns != image_header.capture_timestamp_ns)
        {
            LOG(WARNING) << "[gt_probe] reject incoherent frame seq=" << sequence
                         << " image(epoch=" << image_header.producer_epoch
                         << ",ts=" << image_header.capture_timestamp_ns << ") snapshot(epoch="
                         << (shm_header.ok() ? shm_header.value->created_ns : 0) << ",gt_seq=" << batch.frame_seq
                         << ",gt_ts=" << batch.timestamp_ns << ",exposure_seq="
                         << exposure.frame_seq << ",exposure_ts=" << exposure.timestamp_ns << ")";
            continue;
        }

        // 本帧实际云台反馈（命令空间，与 UdpGimbalCommand 同零位）+ 相机光轴真实仰角
        float g_yaw = 0.0f, g_pitch = 0.0f;
        bool have_gimbal = false;
        {
            auto gs = reader.value->readRuntimeState();
            if (gs.ok())
            {
                have_gimbal = true;
                g_yaw = static_cast<float>(to_degree(gs.value->gimbal_yaw_rad));
                g_pitch = static_cast<float>(90.0 + to_degree(gs.value->gimbal_pitch_rad));
            }
        }
        double cam_el_deg = 0.0;
        {
            const float *q = exposure.camera_quaternion_world_wxyz;
            const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
            // 光轴 = 相机 FLU x 轴的世界方向 = R_cam 第一列
            const double ax = 1 - 2 * (qy * qy + qz * qz);
            const double ay = 2 * (qx * qy + qz * qw);
            const double az = 2 * (qx * qz - qy * qw);
            cam_el_deg = std::asin(std::max(-1.0, std::min(1.0, az))) * 180.0 / 3.14159265358979323846;
        }

        // 找本方符文（team: 0=red=FACE_1, 1=blue=FACE_2）
        const GroundTruthRune *rune = nullptr;
        for (std::uint32_t ri = 0; ri < batch.rune_count; ++ri)
            if (batch.runes[ri].team == team_index)
            {
                rune = &batch.runes[ri];
                break;
            }

        // ---------- 真值 -> InactiveTargets（识别模块输出边界） ----------
        InactiveTargets inactive;
        inactive.is_big_rune = true;
        inactive.capture_timestamp = PRF::timestamp_to_foxglove_time(
            PRF::timestamp_from_nanoseconds(image_header.capture_timestamp_ns));

        int lit_mask = 0;
        char acts_str[8] = "-----";
        double ang = 0.0;
        Eigen::Vector3d rune_center_car = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> armor_car_list;
        std::vector<double> armor_odom_list; // 叠加投影用（每次 push 3 个分量）

        if (rune != nullptr)
        {
            ang = static_cast<double>(rune->current_angle);
            for (int j = 0; j < 5; ++j)
            {
                acts_str[j] = static_cast<char>('0' + rune->target_activations[j]);
                if (rune->target_activations[j] == 1)
                    lit_mask |= (1 << j);
                if (have_previous_activations && previous_activations[j] == 1 &&
                    rune->target_activations[j] == 2)
                {
                    ++hit_count;
                    std::printf("*** HIT #%d: blade=%d seq=%llu ***\n", hit_count, j,
                                static_cast<unsigned long long>(sequence));
                }
                previous_activations[j] = rune->target_activations[j];
            }
            have_previous_activations = true;

            const CarFrame car = make_car_frame(exposure);
            const double origin_odom[3] = {rune->r_center_odom[0], rune->r_center_odom[1],
                                           rune->r_center_odom[2]};
            double origin_world[3];
            odomToWorld(origin_odom, origin_world);
            rune_center_car = car.point(origin_odom);

            for (int j = 0; j < 5 && inactive.rune_pieces.size() < 2; ++j)
            {
                if ((lit_mask & (1 << j)) == 0)
                    continue;
                const FaceTargetGeom &t = face.targets[j];
                const double phi = t.phi_deg * 3.14159265358979323846 / 180.0;
                const double uu = t.umid * std::cos(phi) - t.wmid * std::sin(phi);
                const double ww = t.umid * std::sin(phi) + t.wmid * std::cos(phi);
                double blade_world[3];
                facePointWorld(face, origin_world, uu, ww, t.n_mid, blade_world);
                rotateAroundAxis(face.axis, origin_world, ang, blade_world);
                double blade_odom[3];
                worldToOdom(blade_world, blade_odom);
                armor_car_list.push_back(car.point(blade_odom));
                armor_odom_list.push_back(blade_odom[0]);
                armor_odom_list.push_back(blade_odom[1]);
                armor_odom_list.push_back(blade_odom[2]);
            }

            if (armor_car_list.size() > 2)
                LOG(WARNING) << "[gt_probe] lit blades > 2, estimator will reject";

            for (const auto &armor_car : armor_car_list)
            {
                InactiveTargets::RunePiece piece;
                piece.rune_center = rune_center_car;
                piece.armor_center = armor_car;
                inactive.rune_pieces.push_back(piece);
            }

            // 平面：过符心、法向=符轴（车系）；取向与 PowerRunePlane 一致（z 分量 >= 0）
            double axis_odom[3];
            worldToOdom(face.axis, axis_odom);
            Eigen::Vector3d normal_car = car.dir(axis_odom).normalized();
            if (normal_car.z() < 0.0)
                normal_car = -normal_car;
            inactive.power_rune_plane = Eigen::Hyperplane<double, 3>(normal_car, rune_center_car);
            inactive.is_vaild = !inactive.rune_pieces.empty();

            if (lit_mask != last_lit)
            {
                std::printf("*** lit blade set changed: [%s] -> lit=[%d] (seq=%llu) ***\n",
                            last_lit < 0 ? "-" : std::to_string(last_lit).c_str(), lit_mask,
                            static_cast<unsigned long long>(sequence));
                last_lit = lit_mask;
            }
        }
        else
        {
            if (++no_gt_streak % 60 == 1)
                LOG(WARNING) << "[gt_probe] no rune GT for team " << team_index
                             << " (rune_count=" << batch.rune_count << ")";
        }

        // 符心水平距离（车系 xz）：自适应俯仰补偿的输入；无 GT 时沿用上一帧
        const double rune_dh = rune != nullptr
                                   ? std::hypot(rune_center_car.x(), rune_center_car.z())
                                   : 0.0;
        if (rune_dh > 0.5)
        {
            last_rune_dh = rune_dh;
            have_rune_dh = true;
        }

        // ---------- 逐帧打印 ----------
        std::printf("\n=== f%04d seq=%llu gt(%s): mode=%u mech=%u ang=%+.3frad v_roll=%+.3frad/s"
                    " dir=%d acts=[%s] | pieces=%zu valid=%d ===\n",
                    frames_done, static_cast<unsigned long long>(sequence), face.name,
                    rune != nullptr ? rune->rune_mode : 0, rune != nullptr ? rune->mechanism_state : 0,
                    ang, rune != nullptr ? static_cast<double>(rune->v_roll) : 0.0,
                    rune != nullptr ? rune->direction : 0, acts_str, inactive.rune_pieces.size(),
                    static_cast<int>(inactive.is_vaild));

        if (have_gimbal)
            std::printf("  [gimbal] actual yaw=%+.3f pitch=%+.3f | cam_axis_el_world=%+.3fdeg "
                        "[cmd-pitch-90=%+.3f]\n",
                        g_yaw, g_pitch, cam_el_deg,
                        last_cmd_pitch - 90.0);
        else
            std::printf("  [gimbal] no feedback for seq=%llu\n",
                        static_cast<unsigned long long>(sequence));

        for (std::size_t i = 0; i < inactive.rune_pieces.size(); ++i)
        {
            const auto &piece = inactive.rune_pieces[i];
            std::printf("  [feed] piece%zu rune=(%.3f,%.3f,%.3f) armor=(%.3f,%.3f,%.3f) |a-r|=%.4f\n",
                        i, piece.rune_center.x(), piece.rune_center.y(), piece.rune_center.z(),
                        piece.armor_center.x(), piece.armor_center.y(), piece.armor_center.z(),
                        (piece.armor_center - piece.rune_center).norm());
        }

        // ---------- InactiveTargets -> PhaseMotionEstimator -> RuneDecisionModule ----------
        // 与 PowerRuneProcessor 相同：无效直接跳过（不喂后续模块）
        bool has_fit = false;
        double fit_phase = 0.0, fit_A = 0.0, fit_B = 0.0, fit_b = 0.0, fit_C = 0.0, fit_w = 0.0;
        double fit_a = 0.0, fit_delta = 0.0;
        std::optional<std::vector<RuneTarget>> rune_targets;
        if (inactive.is_vaild)
        {
            estimator.estimate_phase_motion(inactive);
            rune_targets = estimator.try_get_rune_targets();
            if (!rune_targets.has_value())
            {
                std::printf("  [phase] no filtered targets yet (big rune needs >=100 samples to build)\n");
            }
            else
            {
                if (lock_frame < 0)
                {
                    lock_frame = frames_done;
                    std::printf("*** LOCKED at f%d seq=%llu ***\n", frames_done,
                                static_cast<unsigned long long>(sequence));
                }
                for (const auto &target : *rune_targets)
                {
                    const auto &mm = target.big_rune_motion_model;
                    if (!has_fit)
                    {
                        has_fit = true;
                        fit_phase = target.phase;
                        fit_A = mm.phase_cos_coefficient;
                        fit_B = mm.phase_sin_coefficient;
                        fit_b = mm.phase_linear_velocity;
                        fit_C = mm.phase_constant_offset_radians;
                        fit_w = mm.speed_angular_frequency;
                        fit_a = mm.speed_amplitude;
                        fit_delta = mm.speed_phase_shift;
                    }
                    std::printf("  [phase] phase=%+.4f armor=(%.3f,%.3f,%.3f) center=(%.3f,%.3f,%.3f)"
                                " n=%d\n",
                                target.phase, target.armor_module_center.x(),
                                target.armor_module_center.y(), target.armor_module_center.z(),
                                target.rune_center.x(), target.rune_center.y(),
                                target.rune_center.z(), target.inactivate_target_num);
                    std::printf("  [fit] A=%+.4f B=%+.4f b=%+.4f C=%+.4f w=%.4f | a=%.4f delta=%+.4f"
                                "  || GT a=%.4f w=%.4f delta=%+.4f b=%.4f dir=%d\n",
                                mm.phase_cos_coefficient, mm.phase_sin_coefficient,
                                mm.phase_linear_velocity, mm.phase_constant_offset_radians,
                                mm.speed_angular_frequency, mm.speed_amplitude, mm.speed_phase_shift,
                                rune != nullptr ? static_cast<double>(rune->sin_amplitude) : 0.0,
                                rune != nullptr ? static_cast<double>(rune->sin_omega) : 0.0,
                                rune != nullptr ? static_cast<double>(rune->sin_phase) : 0.0,
                                rune != nullptr ? static_cast<double>(rune->sin_offset) : 0.0,
                                rune != nullptr ? rune->direction : 0);
                }
                decision.update_decision_module(std::move(rune_targets.value()));
            }
        }

        const power_rune::RuneSendData send = decision.get_send_data(true);
        if (send.is_find_buff)
            ++acquisitions;
        const double pitch_offset_now = pitch_offset_deg;
        const double cmd_yaw = yaw_scale * to_degree(static_cast<double>(send.yaw)) + yaw_offset_deg;
        const double cmd_pitch =
            pitch_scale * to_degree(static_cast<double>(send.pitch)) + pitch_offset_now;
        const double yaw_error = have_gimbal ? std::remainder(cmd_yaw - g_yaw, 360.0) : 0.0;
        const double pitch_error = have_gimbal ? cmd_pitch - g_pitch : 0.0;
        const bool aligned_now = have_gimbal && std::abs(yaw_error) <= settle_angle_deg &&
                                 std::abs(pitch_error) <= settle_angle_deg;
        aligned_frame_streak = aligned_now ? aligned_frame_streak + 1 : 0;
        // ABI1 没有速度/命令回执，用连续两个曝光帧对齐作为等价稳定条件。
        const bool gimbal_settled = aligned_frame_streak >= 2;
        bool fired_now = false;
        if (send.is_find_buff)
        {
            std::printf("  [decision] find=1 fire=%d mode=%d yaw=%+.2fdeg pitch=%+.2fdeg -> cmd(yaw=%+.2f"
                        " pitch=%+.2f pitch_off=%+.2f d_h=%.2f err=(%+.3f,%+.3f) settled=%d)\n",
                        static_cast<int>(send.is_enable_fire), static_cast<int>(send.mode),
                        to_degree(static_cast<double>(send.yaw)),
                        to_degree(static_cast<double>(send.pitch)), cmd_yaw, cmd_pitch,
                        pitch_offset_now, have_rune_dh ? last_rune_dh : 0.0,
                        yaw_error, pitch_error, static_cast<int>(gimbal_settled));
        }
        else
        {
            std::printf("  [decision] find=0 fire=0 mode=%d\n", static_cast<int>(send.mode));
        }

        if (aim_enabled && send.is_find_buff)
        {
            last_cmd_pitch = cmd_pitch;
            const bool request_fire = fire_enabled && send.is_enable_fire &&
                                      gimbal_settled && frames_done >= fire_after_frame &&
                                      (fire_count_max < 0 || fire_advices < fire_count_max);
            if (request_fire)
            {
                ++fire_advices;
                fired_now = true;
                if (first_fire_frame < 0)
                    first_fire_frame = frames_done;
                std::printf("*** FIRE #%d at f%d seq=%llu cmd=(%.2f, %.2f) ***\n", fire_advices,
                            frames_done, static_cast<unsigned long long>(sequence), cmd_yaw,
                            cmd_pitch);
            }
            if (!gimbal.send(cmd_yaw, cmd_pitch, request_fire, &last_command_id))
            {
                LOG_EVERY_N(WARNING, 60) << "[gt_probe] gimbal send failed";
            }
        }

        // ---------- CSV 落盘 ----------
        if (csv != nullptr)
        {
            std::fprintf(csv, "%d,%llu,%.6f,%.6f,%d,%s,%d", frames_done,
                         static_cast<unsigned long long>(sequence), ang,
                         rune != nullptr ? static_cast<double>(rune->v_roll) : 0.0,
                         rune != nullptr ? rune->direction : 0, acts_str, lit_mask);
            if (!inactive.rune_pieces.empty())
                std::fprintf(csv, ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                             inactive.rune_pieces[0].rune_center.x(),
                             inactive.rune_pieces[0].rune_center.y(),
                             inactive.rune_pieces[0].rune_center.z(),
                             inactive.rune_pieces[0].armor_center.x(),
                             inactive.rune_pieces[0].armor_center.y(),
                             inactive.rune_pieces[0].armor_center.z());
            else
                std::fprintf(csv, ",,,,,,");
            if (has_fit)
                std::fprintf(csv, ",%.4f,%+.4f,%+.4f,%+.4f,%+.4f,%.4f,%.4f,%+.4f", fit_phase, fit_A,
                             fit_B, fit_b, fit_C, fit_w, fit_a, fit_delta);
            else
                std::fprintf(csv, ",,,,,,,,");
            std::fprintf(csv, ",%d,%d,%d,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                         static_cast<int>(send.is_find_buff), static_cast<int>(fired_now),
                         static_cast<int>(gimbal_settled),
                         static_cast<unsigned long long>(last_command_id),
                         cmd_yaw, cmd_pitch, g_yaw, g_pitch, yaw_error, pitch_error,
                         have_rune_dh ? last_rune_dh : 0.0, pitch_offset_now);
        }

        // ---------- 叠加存帧（可选；落点=真值靶心投影，用于像素级人工核对） ----------
        const bool want_save =
            (save_every > 0 && frames_done % save_every == 0) || fired_now ||
            (lock_frame == frames_done);
        if (want_save)
        {
            cv::Mat image = to_bgr_image(*frame.value);
            if (!image.empty())
            {
                if (camera_info_ok)
                {
                    for (std::size_t k = 0; k + 2 < armor_odom_list.size(); k += 3)
                    {
                        double px = 0.0, py = 0.0;
                        if (project_odom(&armor_odom_list[k], exposure, camera_info, &px, &py))
                        {
                            cv::drawMarker(image, cv::Point(cvRound(px), cvRound(py)),
                                           cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 40, 2);
                            cv::putText(image, "GT armor", cv::Point(cvRound(px) + 10, cvRound(py) - 10),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                        }
                    }
                    if (rune != nullptr)
                    {
                        const double r_odom[3] = {rune->r_center_odom[0], rune->r_center_odom[1],
                                                  rune->r_center_odom[2]};
                        double px = 0.0, py = 0.0;
                        if (project_odom(r_odom, exposure, camera_info, &px, &py))
                            cv::circle(image, cv::Point(cvRound(px), cvRound(py)), 12,
                                       cv::Scalar(255, 0, 0), 2);
                    }
                }
                char name[512];
                std::snprintf(name, sizeof(name), "%s/f%04d_seq%llu.png", out_dir.c_str(),
                              frames_done, static_cast<unsigned long long>(sequence));
                cv::imwrite(name, image);
            }
        }

        ++frames_done;
    }

    if (csv != nullptr)
        std::fclose(csv);
    const double measured_hit_rate = fire_advices > 0
                                         ? 100.0 * static_cast<double>(hit_count) / fire_advices
                                         : 0.0;
    std::printf("\n=== summary: frames=%d find_buff_frames=%d fire_advices=%d hits=%d"
                " measured_hit_rate=%.1f%% lock_frame=%d first_fire_frame=%d ===\n",
                frames_done, acquisitions, fire_advices, hit_count, measured_hit_rate,
                lock_frame, first_fire_frame);
    return 0;
}
