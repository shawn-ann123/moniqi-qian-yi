// 探针：验证底盘/云台/相机坐标系约定、云台指令参考系、ECSData 角度映射
// 输出：
//  1) 底盘坐标轴语义（up / 运动方向在底盘系表达）
//  2) 联合旋转 R_chassis^T*R_gimbal 是否恒定（指令是关节角还是世界角）
//  3) 相机固连：R_gimbal^T*R_cam、R_gimbal^T*(Pcam-Pgimbal)、R_cam^T*(Pcam-Pgimbal)
//  4) Q = V^T*A*R_chassis^T*R_cam 的 ZYX 欧拉角（即馈给 SZU ECSData 的值）
//  5) 指令 yaw/pitch 与 θ 的线性关系（斜率/偏置）
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace daedalus::sim::sdk::v1;

using M3 = std::array<std::array<double, 3>, 3>;
using V3 = std::array<double, 3>;

static M3 matmul(const M3& a, const M3& b) {
  M3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += a[i][k] * b[k][j];
      r[i][j] = s;
    }
  return r;
}
static M3 transpose(const M3& a) {
  M3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r[i][j] = a[j][i];
  return r;
}
static V3 mv(const M3& a, const V3& v) {
  V3 r{};
  for (int i = 0; i < 3; ++i) {
    double s = 0;
    for (int k = 0; k < 3; ++k) s += a[i][k] * v[k];
    r[i] = s;
  }
  return r;
}
static V3 sub3(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
static double norm3(const V3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

static M3 quatToR(const float q[4]) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  w /= n; x /= n; y /= n; z /= n;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
           {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
           {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

static V3 zyxFromR(const M3& R) {
  const double yaw = std::atan2(R[1][0], R[0][0]);
  const double pitch = std::atan2(-R[2][0], std::sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]));
  const double roll = std::atan2(R[2][1], R[2][2]);
  return {yaw, pitch, roll};
}

static double deg(double rad) { return rad * 180.0 / M_PI; }

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

struct Sample {
  uint64_t seq = 0;
  M3 Rc{}, Rg{}, Rcam{};
  V3 Pc{}, Pg{}, Pcam{};
  double gimbal_yaw_rad = 0, gimbal_pitch_rad = 0;
  V3 q_out{};  // ZYX(Q), rad
};

static bool read_sample(ContestClient& client, TalosMetadataReader& reader, Sample& out) {
  auto frame = client.nextFrame();
  if (!frame.ok()) return false;
  const auto seq = frame.value->image.header.source_sequence;
  auto es = reader.readExposureStateForFrame(seq);
  if (!es.ok()) return false;
  const auto& e = *es.value;
  out.seq = seq;
  out.Rc = quatToR(e.chassis_quaternion_world_wxyz);
  out.Rg = quatToR(e.gimbal_quaternion_world_wxyz);
  out.Rcam = quatToR(e.camera_quaternion_world_wxyz);
  out.Pc = {e.chassis_position_world[0], e.chassis_position_world[1], e.chassis_position_world[2]};
  out.Pg = {e.gimbal_position_world[0], e.gimbal_position_world[1], e.gimbal_position_world[2]};
  out.Pcam = {e.camera_position_world[0], e.camera_position_world[1], e.camera_position_world[2]};
  out.gimbal_yaw_rad = e.gimbal_yaw_rad;
  out.gimbal_pitch_rad = e.gimbal_pitch_rad;

  // V = car_T_ecs, A = chassis->car(右前下) 轴映射
  static const M3 V = {{{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}}};
  static const M3 A = {{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}}};
  const M3 Q = matmul(transpose(V), matmul(A, matmul(transpose(out.Rc), out.Rcam)));
  out.q_out = zyxFromR(Q);
  return true;
}

int main(int argc, char** argv) {
  std::string ipc = defaultIpcDir();
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) {
    std::fprintf(stderr, "open failed: %s\n", st.message.c_str());
    return 1;
  }
  auto reader_res = mapping.reader();
  if (!reader_res.ok()) {
    std::fprintf(stderr, "no reader\n");
    return 1;
  }
  TalosMetadataReader& reader = *reader_res.value;

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) {
    std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str());
    return 1;
  }

  auto send = [&](float yaw, float pitch) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = yaw;
    cmd.pitch_deg = pitch;
    auto r = client.sendAim(cmd);
    if (!r.ok()) std::fprintf(stderr, "send failed: %s\n", r.status.message.c_str());
  };

  // ---------- 阶段 1: cmd(0,90) 静止，收集坐标轴/固连数据 ----------
  send(0.f, 90.f);
  std::this_thread::sleep_for(std::chrono::milliseconds(900));

  std::vector<Sample> samples;
  for (int i = 0; i < 200 && (int)samples.size() < 45; ++i) {
    Sample s;
    if (read_sample(client, reader, s)) samples.push_back(s);
  }
  std::printf("collected %zu frames\n", samples.size());
  if (samples.size() < 5) return 1;

  {
    const Sample& s0 = samples.front();
    const Sample& s1 = samples.back();
    // 底盘轴：世界 up 在底盘系
    const V3 up_ch = mv(transpose(s0.Rc), {0, 0, 1});
    // 运动方向在底盘系（两帧差分，遍历取平均）
    V3 mot{0, 0, 0};
    for (size_t i = 1; i < samples.size(); ++i) {
      const V3 dp = sub3(samples[i].Pc, samples[i - 1].Pc);
      const double dn = norm3(dp);
      if (dn > 1e-5) {
        const V3 m = mv(transpose(samples[i].Rc), dp);
        for (int k = 0; k < 3; ++k) mot[k] += m[k] / dn;
      }
    }
    const double mn = norm3(mot);
    for (double& v : mot) v /= (mn > 1e-9 ? mn : 1);
    std::printf("[axes] world_up in chassis = (%.3f, %.3f, %.3f)\n", up_ch[0], up_ch[1], up_ch[2]);
    std::printf("[axes] motion dir in chassis = (%.3f, %.3f, %.3f)\n", mot[0], mot[1], mot[2]);

    // 固连关系
    const M3 S = matmul(transpose(s0.Rg), s0.Rcam);
    const V3 v = mv(transpose(s0.Rg), sub3(s0.Pcam, s0.Pg));
    const V3 t_cam = mv(transpose(s0.Rcam), sub3(s0.Pcam, s0.Pg));
    double max_s = 0, max_v = 0, max_t = 0, max_joint_yaw = 0;
    V3 joint_yaw_deg{};
    for (const Sample& s : samples) {
      const M3 Si = matmul(transpose(s.Rg), s.Rcam);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) max_s = std::max(max_s, std::fabs(Si[i][j] - S[i][j]));
      const V3 vi = mv(transpose(s.Rg), sub3(s.Pcam, s.Pg));
      max_v = std::max(max_v, norm3(sub3(vi, v)));
      const V3 ti = mv(transpose(s.Rcam), sub3(s.Pcam, s.Pg));
      max_t = std::max(max_t, norm3(sub3(ti, t_cam)));
      // 关节旋转 R_chassis^T*R_gimbal 的 yaw 分量随时间变化？
      const M3 Rcg = matmul(transpose(s.Rc), s.Rg);
      const V3 ypr = zyxFromR(Rcg);
      joint_yaw_deg = {deg(ypr[0]), deg(ypr[1]), deg(ypr[2])};
      (void)ypr;
    }
    std::printf("[mount] S = Rg^T*Rcam (const? maxdev %.5f):\n", max_s);
    for (auto& row : S) std::printf("    [%.4f %.4f %.4f]\n", row[0], row[1], row[2]);
    std::printf("[mount] Pcam-Pgimbal in gimbal = (%.4f, %.4f, %.4f) (const? %.5f)\n",
                v[0], v[1], v[2], max_v);
    std::printf("[mount] Pcam-Pgimbal in cam    = (%.4f, %.4f, %.4f) (const? %.5f)\n",
                t_cam[0], t_cam[1], t_cam[2], max_t);
    std::printf("[cmd0] gs? joint Rcg yaw/pitch/roll deg latest = (%.2f, %.2f, %.2f)\n",
                joint_yaw_deg[0], joint_yaw_deg[1], joint_yaw_deg[2]);

    // 底盘 yaw 随时间（判断车是否在动）
    const V3 ch_y0 = zyxFromR(s0.Rc);
    const V3 ch_y1 = zyxFromR(s1.Rc);
    std::printf("[chassis] ZYX yaw deg: first=%.2f last=%.2f  |P drift|=%.3f m\n",
                deg(ch_y0[0]), deg(ch_y1[0]), norm3(sub3(s1.Pc, s0.Pc)));
  }

  // 每帧的 Q 输出（θ）与 ES 反馈
  std::printf("\n[per-frame] seq gimbal_yaw/pitch(rad,ES)  theta_yaw/pitch/roll(deg,ZYX)\n");
  for (size_t i = 0; i < samples.size(); i += 6) {
    const Sample& s = samples[i];
    std::printf("  %llu  es=(%.4f, %.4f)  th=(%.2f, %.2f, %.2f)\n",
                (unsigned long long)s.seq, s.gimbal_yaw_rad, s.gimbal_pitch_rad,
                deg(s.q_out[0]), deg(s.q_out[1]), deg(s.q_out[2]));
  }

  // ---------- 阶段 2: 指令与 θ 的映射 ----------
  struct CmdCase { float yaw, pitch; };
  const CmdCase cases[] = {{0, 90}, {10, 90}, {-10, 90}, {0, 100}, {0, 80}, {20, 75}};
  std::printf("\n[cmd-map] cmd(yaw,pitch) -> mean theta(yaw,pitch) & gs feedback\n");
  for (const CmdCase& c : cases) {
    send(c.yaw, c.pitch);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    // 取 5 帧平均
    double ty = 0, tp = 0, fy = 0, fp = 0;
    int n = 0;
    for (int i = 0; i < 200 && n < 5; ++i) {
      Sample s;
      if (read_sample(client, reader, s)) {
        ty += deg(s.q_out[0]);
        tp += deg(s.q_out[1]);
        fy += s.gimbal_yaw_rad;
        fp += s.gimbal_pitch_rad;
        ++n;
      }
    }
    const auto gs = reader.readGimbalState();
    std::printf("  cmd=(%6.1f,%6.1f) -> th=(%7.2f,%7.2f)  es_rad=(%7.4f,%7.4f)  gs_deg=(%6.2f,%6.2f)\n",
                c.yaw, c.pitch, ty / n, tp / n, fy / n, fp / n,
                gs.ok() ? gs.value->yaw_deg : 0.0, gs.ok() ? gs.value->pitch_deg : 0.0);
  }

  // ---------- 阶段 3: 指令 (0,90) 持续，观察关节/世界参考系 ----------
  send(0.f, 90.f);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  std::printf("\n[drift] cmd=(0,90) held; chassis yaw vs theta_yaw vs joint yaw\n");
  for (int i = 0; i < 80; ++i) {
    Sample s;
    if (read_sample(client, reader, s)) {
      const V3 ch = zyxFromR(s.Rc);
      const V3 jt = zyxFromR(matmul(transpose(s.Rc), s.Rg));
      std::printf("  seq=%llu  chassis_yaw=%.2f  theta_yaw=%.2f  joint_yaw(Rcg)=%.2f  theta_pitch=%.2f\n",
                  (unsigned long long)s.seq, deg(ch[0]), deg(s.q_out[0]), deg(jt[0]), deg(s.q_out[1]));
    }
  }
  return 0;
}
