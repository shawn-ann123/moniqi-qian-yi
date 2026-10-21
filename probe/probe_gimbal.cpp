// 探针：读取模拟器云台状态并发送指令，确定 yaw/pitch 零点与方向约定
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

static void printState(const char* tag, TalosMetadataMapping& mapping) {
  auto reader = mapping.reader();
  if (!reader.ok()) { std::printf("%s: no reader\n", tag); return; }
  auto gs = reader.value->readGimbalState();
  auto rs = reader.value->readRuntimeState();
  if (gs.ok())
    std::printf("%s: gimbal yaw_deg=%.3f pitch_deg=%.3f (yaw_vel=%.2f pitch_vel=%.2f)\n",
                tag, gs.value->yaw_deg, gs.value->pitch_deg,
                gs.value->yaw_velocity_deg_s, gs.value->pitch_velocity_deg_s);
  if (rs.ok())
    std::printf("%s: runtime yaw_rad=%.4f pitch_rad=%.4f\n",
                tag, rs.value->gimbal_yaw_rad, rs.value->gimbal_pitch_rad);
  // 曝光姿态：云台/相机世界四元数
  auto img = reader.value->readLatestImageMeta();
  if (img.ok()) {
    auto es = reader.value->readExposureStateForFrame(img.value->seq);
    if (es.ok()) {
      const auto& q = es.value->gimbal_quaternion_world_wxyz;
      const auto& c = es.value->camera_quaternion_world_wxyz;
      std::printf("%s: gimbal_world_q(wxyz)=(%.4f,%.4f,%.4f,%.4f) camera_world_q(wxyz)=(%.4f,%.4f,%.4f,%.4f)\n",
                  tag, q[0], q[1], q[2], q[3], c[0], c[1], c[2], c[3]);
      std::printf("%s: gimbal_yaw_rad=%.4f gimbal_pitch_rad=%.4f flags=%u\n",
                  tag, es.value->gimbal_yaw_rad, es.value->gimbal_pitch_rad, es.value->state_flags);
    }
  }
}

int main(int argc, char** argv) {
  std::string ipc = defaultIpcDir();
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed: %s\n", st.message.c_str()); return 1; }

  printState("initial", mapping);

  UdpGimbalClient gimbal;
  struct { float yaw; float pitch; } cmds[] = {
    {0.0f, 90.0f},    // 归零
    {10.0f, 90.0f},   // yaw +10
    {0.0f, 100.0f},   // pitch 100 (>90)
    {0.0f, 80.0f},    // pitch 80 (<90)
    {0.0f, 90.0f},    // 回零
  };
  for (auto& c : cmds) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = c.yaw; cmd.pitch_deg = c.pitch;
    auto r = gimbal.send(cmd);
    if (!r.ok()) { std::fprintf(stderr, "send failed: %s\n", r.message.c_str()); return 1; }
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    char tag[64]; std::snprintf(tag, sizeof(tag), "after yaw=%.1f pitch=%.1f", c.yaw, c.pitch);
    printState(tag, mapping);
  }
  return 0;
}
