// 探针2：通过 ContestClient 取帧 + 曝光姿态，确定 yaw/pitch 相对世界系的方向约定
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

// 四元数(wxyz)旋转向量
static void quatRotate(const float q[4], const float v[3], float out[3]) {
  const float w=q[0], x=q[1], y=q[2], z=q[3];
  float tx = 2*(y*v[2]-z*v[1]);
  float ty = 2*(z*v[0]-x*v[2]);
  float tz = 2*(x*v[1]-y*v[0]);
  out[0] = v[0] + w*tx + (y*tz - z*ty);
  out[1] = v[1] + w*ty + (z*tx - x*tz);
  out[2] = v[2] + w*tz + (x*ty - y*tx);
}

static void probe(ContestClient& client, TalosMetadataMapping& mapping, const char* tag) {
  auto frame = client.nextFrame();
  if (!frame.ok()) { std::printf("%s: no frame\n", tag); return; }
  auto reader = mapping.reader();
  auto es = reader.value->readExposureStateForFrame(frame.value->image.header.source_sequence);
  if (!es.ok()) { std::printf("%s: no exposure state\n", tag); return; }
  const float* gq = es.value->gimbal_quaternion_world_wxyz;
  const float* cq = es.value->camera_quaternion_world_wxyz;
  // 世界系：up=+Y（右手渲染系）。相机前向=相机系+Z
  float fwd_c[3], gz[3], gx[3], gy[3];
  const float z_axis[3] = {0,0,1};
  const float x_axis[3] = {1,0,0};
  const float y_axis[3] = {0,1,0};
  quatRotate(cq, z_axis, fwd_c);
  quatRotate(gq, z_axis, gz);
  quatRotate(gq, x_axis, gx);
  quatRotate(gq, y_axis, gy);
  std::printf("%s: cmd_fb(yaw=%.2fdeg pitch=%.2fdeg) | cam_fwd_w=(%.3f,%.3f,%.3f) | gimbal x_w=(%.3f,%.3f,%.3f) y_w=(%.3f,%.3f,%.3f) z_w=(%.3f,%.3f,%.3f) | es yaw_rad=%.4f pitch_rad=%.4f\n",
              tag, frame.value->gimbal.yaw_deg, frame.value->gimbal.pitch_deg,
              fwd_c[0], fwd_c[1], fwd_c[2],
              gx[0], gx[1], gx[2], gy[0], gy[1], gy[2], gz[0], gz[1], gz[2],
              es.value->gimbal_yaw_rad, es.value->gimbal_pitch_rad);
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

  struct { float yaw; float pitch; const char* tag; } cmds[] = {
    {0.0f, 90.0f, "zero"},
    {30.0f, 90.0f, "yaw+30"},
    {0.0f, 90.0f, "yaw->0"},
    {0.0f, 110.0f, "pitch=110"},
    {0.0f, 70.0f, "pitch=70"},
    {0.0f, 90.0f, "restore"},
  };
  for (auto& c : cmds) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = c.yaw; cmd.pitch_deg = c.pitch;
    client.sendAim(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    probe(client, mapping, c.tag);
  }
  return 0;
}
