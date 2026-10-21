// 探针10：批量采集 (图像, ES, GT) 三元组，供离线 numpy 穷举坐标系假设
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <thread>
#include <chrono>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

int main(int argc, char** argv) {
  std::string ipc = defaultIpcDir();
  std::string outdir = "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/dump";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];
    else if (std::strcmp(argv[i], "--out") == 0) outdir = argv[++i];
  }

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed\n"); return 1; }

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed\n"); return 1; }

  FILE* dump = std::fopen((outdir + "/dump.txt").c_str(), "w");
  std::fprintf(dump, "# yaw pitch seq cam_px cam_py cam_pz qw qx qy qz ch_px ch_py ch_pz ch_r ch_p ch_y g_yaw g_pitch rune_count [center(3) radius angle team mode]... tgt_count [tpos(3) yaw vyaw team]...\n");

  for (int yaw = -180; yaw < 180; yaw += 15) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = (float)yaw; cmd.pitch_deg = 90.0f;
    client.sendAim(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(450));
    auto frame = client.nextFrame();
    if (!frame.ok()) { std::printf("yaw=%+4d: no frame\n", yaw); continue; }
    const auto& img = frame.value->image;
    const int W = img.header.width, H = img.header.height;
    const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
    const auto seq = img.header.source_sequence;

    char path[256];
    std::snprintf(path, sizeof(path), "%s/f_%+04d.ppm", outdir.c_str(), yaw);
    FILE* f = std::fopen(path, "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; ++i)
      std::fwrite(img.payload.data() + i*ch, 1, 3, f);
    std::fclose(f);

    auto reader = mapping.reader();
    auto es = reader.value->readExposureStateForFrame(seq);
    auto gt = reader.value->readGroundTruthForFrame(seq);
    if (!es.ok() || !gt.ok()) { std::printf("yaw=%+4d: es/gt missing\n", yaw); continue; }
    const auto& E = *es.value;
    const auto& G = gt.value->ground_truth;

    std::fprintf(dump, "%d 90 %llu", yaw, (unsigned long long)seq);
    for (int i = 0; i < 3; ++i) std::fprintf(dump, " %.6f", E.camera_position_world[i]);
    for (int i = 0; i < 4; ++i) std::fprintf(dump, " %.6f", E.camera_quaternion_world_wxyz[i]);
    for (int i = 0; i < 3; ++i) std::fprintf(dump, " %.6f", E.chassis_position_world[i]);
    for (int i = 0; i < 3; ++i) std::fprintf(dump, " %.6f", E.chassis_rpy_world[i]);
    std::fprintf(dump, " %.6f %.6f", E.gimbal_yaw_rad, E.gimbal_pitch_rad);
    std::fprintf(dump, " %u", G.rune_count);
    for (std::uint32_t i = 0; i < G.rune_count; ++i) {
      const auto& r = G.runes[i];
      std::fprintf(dump, " %.6f %.6f %.6f %.6f %.6f %u %u",
                   r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2],
                   r.radius, r.current_angle, r.team, r.rune_mode);
    }
    std::fprintf(dump, " %u", G.target_count);
    for (std::uint32_t i = 0; i < G.target_count && i < 4; ++i) {
      const auto& t = G.targets[i];
      std::fprintf(dump, " %.6f %.6f %.6f %.6f %.6f %u %u",
                   t.position[0], t.position[1], t.position[2], t.yaw, t.vyaw, t.team, t.armor_count);
    }
    std::fprintf(dump, "\n");
    std::printf("yaw=%+4d seq=%llu runes=%u targets=%u\n", yaw, (unsigned long long)seq, G.rune_count, G.target_count);
  }
  std::fclose(dump);
  std::printf("dumped to %s/dump.txt\n", outdir.c_str());
  return 0;
}
