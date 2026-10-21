// 探针: 数值验证 "模拟器 ES → SZU 算法 TF 链" 桥接公式 + 打符链路像素校验 + 可选 reset_scene 底盘运动测试
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <array>

using namespace daedalus::sim::sdk::v1;

using V3 = std::array<double, 3>;
using M3 = std::array<double, 9>;  // row-major

static M3 mm(const M3& a, const M3& b) {
  M3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
      r[i * 3 + j] = s;
    }
  return r;
}
static M3 tr(const M3& a) { M3 r{}; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r[i * 3 + j] = a[j * 3 + i]; return r; }
static V3 mv(const M3& a, const V3& v) {
  return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
          a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
}
static double nrm(const V3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static M3 quatToR(const float q[4]) {  // wxyz
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
          2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
}
static M3 rz(double a) { return {cos(a), -sin(a), 0, sin(a), cos(a), 0, 0, 0, 1}; }
static M3 ry(double a) { return {cos(a), 0, sin(a), 0, 1, 0, -sin(a), 0, cos(a)}; }
static M3 rx(double a) { return {1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a)}; }
static void zyxFromR(const M3& R, double& yaw, double& pit, double& rol) {
  yaw = std::atan2(R[3], R[0]);
  pit = std::atan2(-R[6], std::sqrt(R[7] * R[7] + R[8] * R[8]));
  rol = std::atan2(R[7], R[8]);
}
static double maxdev(const M3& a, const M3& b) {
  double m = 0;
  for (int i = 0; i < 9; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// SZU 算法硬编码 / 我方推导的常量
static const M3 V{0, 1, 0, 0, 0, -1, -1, 0, 0};    // vehicle_from_ecs (算法硬编码)
static const M3 A{0, -1, 0, 0, 0, -1, 1, 0, 0};     // 稳定系(FLU) -> 车体系(右-下-前)
static const M3 Mhc{0, 0, -1, 1, 0, 0, 0, -1, 0};   // gimbal_from_camera (算法硬编码)
static const M3 MhcT{0, 1, 0, 0, 0, -1, -1, 0, 0};  // Mhc 转置

int main(int argc, char** argv) {
  std::string ipc = "/home/shawn/桌面/rm/完整版/runtime/talos-ipc";
  bool do_reset = false;
  int save_at = -1;
  int rounds_override = 0;
  float setcmd_yaw = 0, setcmd_pitch = 0;
  bool has_setcmd = false;
  const char* save_path = "/tmp/chain_frame.ppm";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];
    else if (std::strcmp(argv[i], "--reset") == 0) do_reset = true;
    else if (std::strcmp(argv[i], "--rounds") == 0) rounds_override = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--setcmd") == 0) { setcmd_yaw = std::atof(argv[++i]); setcmd_pitch = std::atof(argv[++i]); has_setcmd = true; }
    else if (std::strcmp(argv[i], "--save") == 0) { save_at = std::atoi(argv[++i]); }
  }

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed: %s\n", st.message.c_str()); return 1; }

  const double fx = 1303.67532368147, fy = 1303.67532368147, cx = 720.0, cy = 540.0;

  bool sent_reset = false;
  int rounds = do_reset ? 90 : 16;
  if (rounds_override > 0) rounds = rounds_override;
  V3 prev_p{};
  double prev_t = 0;
  double dR_max = 0, dp_max = 0;

  for (int it = 0; it < rounds; ++it) {
    if (it == 0 && has_setcmd) {
      UdpGimbalCommand cmd;
      cmd.yaw_deg = setcmd_yaw;
      cmd.pitch_deg = setcmd_pitch;
      auto sr = client.sendAim(cmd);
      std::printf("[setcmd] yaw=%.2f pitch=%.2f ok=%d\n", setcmd_yaw, setcmd_pitch, sr.ok());
      std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    }
    auto frame = client.nextFrame();
    if (!frame.ok()) { std::printf("it=%d: no frame (%s)\n", it, frame.status.message.c_str()); std::this_thread::sleep_for(std::chrono::milliseconds(300)); continue; }
    const auto seq = frame.value->image.header.source_sequence;
    auto reader = mapping.reader();
    auto es = reader.value->readExposureStateForFrame(seq);
    if (!es.ok()) { std::printf("it=%d seq=%llu: no ES\n", it, (unsigned long long)seq); continue; }
    const auto& E = *es.value;

    const M3 Rch = quatToR(E.chassis_quaternion_world_wxyz);
    const M3 Rcam = quatToR(E.camera_quaternion_world_wxyz);

    // 桥接公式：ECSData = ZYX(Q), Q = V^T·A·Rch^T·Rcam·MhcT
    const M3 Q = mm(mm(mm(mm(tr(V), A), tr(Rch)), Rcam), MhcT);
    double yaw, pit, rol;
    zyxFromR(Q, yaw, pit, rol);
    const M3 Qr = mm(mm(rz(yaw), ry(pit)), rx(rol));
    const M3 Ralg = mm(mm(V, Qr), Mhc);      // 算法 TF 树重建的 car_T_cam 旋转
    const M3 Rtrue = mm(mm(A, tr(Rch)), Rcam);  // 真值 car_T_cam 旋转
    const double dR = maxdev(Ralg, Rtrue);

    const V3 pc{E.camera_position_world[0], E.camera_position_world[1], E.camera_position_world[2]};
    const V3 pg{E.gimbal_position_world[0], E.gimbal_position_world[1], E.gimbal_position_world[2]};
    const V3 chp{E.chassis_position_world[0], E.chassis_position_world[1], E.chassis_position_world[2]};
    const V3 dcam{pc[0] - pg[0], pc[1] - pg[1], pc[2] - pg[2]};
    const V3 t_off = mv(Mhc, mv(tr(Rcam), dcam));       // = track.json 偏移×0.001 (m)
    const V3 p_alg = mv(mm(V, Qr), t_off);              // 算法树中相机位置(car系)
    const V3 p_true = mv(mm(A, tr(Rch)), dcam);         // 真值相机位置(car系)
    const V3 dp{p_alg[0] - p_true[0], p_alg[1] - p_true[1], p_alg[2] - p_true[2]};
    const double dpn = nrm(dp);

    dR_max = std::max(dR_max, dR);
    dp_max = std::max(dp_max, dpn);

    // 底盘运动检测（车体系速度）
    const auto tnow = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    V3 vch{0, 0, 0};
    double vm = 0;
    if (prev_t > 0) {
      const double dt = tnow - prev_t;
      if (dt > 0.05) {
        const V3 dpw{(chp[0] - prev_p[0]) / dt, (chp[1] - prev_p[1]) / dt, (chp[2] - prev_p[2]) / dt};
        vch = mv(tr(Rch), dpw);
        vm = nrm(vch);
      }
    }
    prev_p = chp; prev_t = tnow;

    // 云台状态反馈
    auto gs = reader.value->readGimbalState();

    std::printf("it=%2d seq=%llu yaw=%.3f pitch=%.3f roll=%.3f | dR=%.2e dP=%.2e | t_off_mm=(%.2f,%.2f,%.2f) | chp=(%.3f,%.3f,%.3f) chyaw=%.2f | gs=(%.2f,%.2f) | v_ch=%.3f\n",
                it, (unsigned long long)seq, yaw * 57.2957795, pit * 57.2957795, rol * 57.2957795,
                dR, dpn, t_off[0] * 1000, t_off[1] * 1000, t_off[2] * 1000,
                chp[0], chp[1], chp[2], std::atan2(Rch[3], Rch[0]) * 57.2957795,
                gs.ok() ? gs.value->yaw_deg : 0, gs.ok() ? gs.value->pitch_deg : 0, vm);
    if (vm > 0.02)
      std::printf("     >>> MOTION v_chassis=(%.3f, %.3f, %.3f) m/s  |v|=%.3f\n", vch[0], vch[1], vch[2], vm);

    // 真值：符文 + 像素投影
    auto gt = reader.value->readGroundTruthForFrame(seq);
    if (gt.ok()) {
      const auto& G = gt.value->ground_truth;
      for (std::uint32_t i = 0; i < G.rune_count && i < 4; ++i) {
        const auto& r = G.runes[i];
        const V3 Pw{r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2]};
        const V3 vw{Pw[0] - pc[0], Pw[1] - pc[1], Pw[2] - pc[2]};
        const V3 flu = mv(tr(Rcam), vw);
        const double ox = -flu[1], oy = -flu[2], oz = flu[0];
        const double px = fx * ox / oz + cx, py = fy * oy / oz + cy;
        const V3 bx = mv(Rcam, {1, 0, 0});
        const double b_az = std::atan2(bx[1], bx[0]) * 57.2957795, b_el = std::asin(bx[2]) * 57.2957795;
        const double dv = nrm(vw);
        const double r_az = std::atan2(vw[1], vw[0]) * 57.2957795, r_el = std::asin(vw[2] / dv) * 57.2957795;
        std::printf("       diag: bore_az=%.2f bore_el=%.2f | rune_az=%.2f rune_el=%.2f |d|=%.3f | flu=(%.4f,%.4f,%.4f) oz=%.3f\n",
                    b_az, b_el, r_az, r_el, dv, flu[0], flu[1], flu[2], oz);
        std::printf("       fbf: cmd_echo=(%.2f,%.2f) es_scalar=(%.4f,%.4f) camP=(%.4f,%.4f,%.4f)\n",
                    frame.value->gimbal.yaw_deg, frame.value->gimbal.pitch_deg,
                    E.gimbal_yaw_rad * 57.2957795, E.gimbal_pitch_rad * 57.2957795, pc[0], pc[1], pc[2]);
        // 算法链视角（经 car_T_cam 反投影相机系内坐标）
        const V3 car_p{mv(tr(Ralg), vw)[0], mv(tr(Ralg), vw)[1], mv(tr(Ralg), vw)[2]};
        (void)car_p;
        std::printf("     rune[%u] team=%u mode=%u mech=%u angle=%.3f v_roll=%.3f dir=%d | center_w=(%.3f,%.3f,%.3f) -> px=(%.1f,%.1f) oz=%.2f\n",
                    i, r.team, r.rune_mode, r.mechanism_state, r.current_angle, r.v_roll, r.direction,
                    Pw[0], Pw[1], Pw[2], px, py, oz);
      }
    }

    // 可选：保存一帧 PPM 供像素校验
    if (it == save_at) {
      const auto& img = frame.value->image;
      const int W = img.header.width, H = img.header.height;
      const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
      FILE* f = std::fopen(save_path, "wb");
      if (f) {
        std::fprintf(f, "P6\n%d %d\n255\n", W, H);
        for (int k = 0; k < W * H; ++k) std::fwrite(img.payload.data() + k * ch, 1, 3, f);
        std::fclose(f);
        std::printf("     saved frame -> %s\n", save_path);
      }
    }

    // 可选：reset_scene 测试底盘运动
    if (do_reset && !sent_reset && it >= 2) {
      sent_reset = true;
      SceneControlOptions so;
      so.endpoint = {"127.0.0.1", kUdpSceneControlPort};
      so.session_id = "probe-chain";
      SceneControlClient scc(so);
      auto cs = scc.createSession();
      std::printf("     [reset] createSession ok=%d msg=%s\n", cs.ok(), cs.ok() ? cs.value->message.c_str() : cs.status.message.c_str());
      auto rr = scc.resetScene();
      std::printf("     [reset] resetScene ok=%d status=%d msg=%s data=%s\n", rr.ok(),
                  rr.ok() ? (int)rr.value->status : -1,
                  rr.ok() ? rr.value->message.c_str() : rr.status.message.c_str(),
                  rr.ok() ? rr.value->data_json.c_str() : "");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(60));
  }
  std::printf("\nSUMMARY: max dR=%.3e  max |dP|=%.3e\n", dR_max, dp_max);
  return 0;
}
