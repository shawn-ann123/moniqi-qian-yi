// 探针3：用真值验证相机内外参与坐标系约定
// 1) 读取相机世界位姿 + 符心真值(odom)
// 2) 把符心变换到相机系，用内参投影，核对是否落在图像中心附近
// 3) 验证 相机系 x右 y下 z前
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

// q(wxyz) 把世界向量 v 转到该坐标系: v_frame = q^-1 * v
static void quatConjRotate(const float q[4], const float v[3], float out[3]) {
  const float w=q[0], x=-q[1], y=-q[2], z=-q[3];
  float tx = 2*(y*v[2]-z*v[1]);
  float ty = 2*(z*v[0]-x*v[2]);
  float tz = 2*(x*v[1]-y*v[0]);
  out[0] = v[0] + w*tx + (y*tz - z*ty);
  out[1] = v[1] + w*ty + (z*tx - x*tz);
  out[2] = v[2] + w*tz + (x*ty - y*tx);
}

int main(int argc, char** argv) {
  std::string ipc = defaultIpcDir();
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed: %s\n", st.message.c_str()); return 1; }

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }

  // 相机内参(来自 camera-calibration.json)
  const double fx = 1303.67532368147, fy = 1303.67532368147, cx = 720.0, cy = 540.0;

  struct { float yaw; float pitch; const char* tag; } cmds[] = {
    {0.0f, 90.0f, "yaw0_pitch90"},
    {-15.0f, 95.0f, "yaw-15_pitch95"},
    {15.0f, 85.0f, "yaw15_pitch85"},
  };

  for (auto& c : cmds) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = c.yaw; cmd.pitch_deg = c.pitch;
    auto sr = client.sendAim(cmd);
    if (!sr.ok()) { std::fprintf(stderr, "sendAim failed\n"); return 1; }
    std::this_thread::sleep_for(std::chrono::milliseconds(900));

    auto frame = client.nextFrame();
    if (!frame.ok()) { std::printf("%s: no frame\n", c.tag); continue; }
    const auto seq = frame.value->image.header.source_sequence;

    auto reader = mapping.reader();
    auto es = reader.value->readExposureStateForFrame(seq);
    auto gt = reader.value->readGroundTruthForFrame(seq);
    if (!es.ok() || !gt.ok()) { std::printf("%s: no es/gt\n", c.tag); continue; }

    const auto& E = *es.value;
    const auto& G = gt.value->ground_truth;
    std::printf("== %s (gs yaw=%.2f pitch=%.2f) world_frame=%u ==\n", c.tag,
                frame.value->gimbal.yaw_deg, frame.value->gimbal.pitch_deg, E.world_frame);
    std::printf("  cam_pos_w=(%.4f,%.4f,%.4f) cam_q=(%.4f,%.4f,%.4f,%.4f)\n",
                E.camera_position_world[0], E.camera_position_world[1],
                E.camera_position_world[2],
                E.camera_quaternion_world_wxyz[0], E.camera_quaternion_world_wxyz[1],
                E.camera_quaternion_world_wxyz[2], E.camera_quaternion_world_wxyz[3]);
    std::printf("  chassis_pos_w=(%.4f,%.4f,%.4f) chassis_rpy=(%.4f,%.4f,%.4f) gimbal_yaw=%.4f pitch=%.4f\n",
                E.chassis_position_world[0], E.chassis_position_world[1],
                E.chassis_position_world[2],
                E.chassis_rpy_world[0], E.chassis_rpy_world[1],
                E.chassis_rpy_world[2],
                E.gimbal_yaw_rad, E.gimbal_pitch_rad);

    for (std::uint32_t i = 0; i < G.rune_count && i < 2; ++i) {
      const auto& r = G.runes[i];
      // 符心世界坐标
      float d[3] = {
        r.r_center_odom[0] - E.camera_position_world[0],
        r.r_center_odom[1] - E.camera_position_world[1],
        r.r_center_odom[2] - E.camera_position_world[2]};
      float pc[3];
      quatConjRotate(E.camera_quaternion_world_wxyz, d, pc);
      // 假设相机系 x右 y下 z前
      const double u = fx * pc[0] / pc[2] + cx;
      const double v = fy * pc[1] / pc[2] + cy;
      std::printf("  rune[%u] team=%u mode=%u center_odom=(%.4f,%.4f,%.4f) radius=%.3f angle=%.3f v_roll=%.3f\n",
                  i, r.team, r.rune_mode, r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2],
                  r.radius, r.current_angle, r.v_roll);
      std::printf("    -> cam frame (x右y下z前): (%.4f,%.4f,%.4f) dist=%.3f\n",
                  pc[0], pc[1], pc[2], std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]));
      std::printf("    -> project: (u=%.1f, v=%.1f)  [图像中心=(720,540)]\n", u, v);
    }
  }
  return 0;
}
