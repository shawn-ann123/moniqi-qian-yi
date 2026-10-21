// 探针8：带位姿的偏航扫描 —— 每帧同时记录 ES+GT 并预测符心像素，与图像对照
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
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed\n"); return 1; }

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed\n"); return 1; }

  const double fx=1303.67532368147, fy=1303.67532368147, cx=720.0, cy=540.0;

  for (int yaw = -180; yaw < 180; yaw += 15) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = (float)yaw; cmd.pitch_deg = 90.0f;
    client.sendAim(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto frame = client.nextFrame();
    if (!frame.ok()) continue;
    const auto& img = frame.value->image;
    const int W = img.header.width, H = img.header.height;
    const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
    const auto seq = img.header.source_sequence;

    char path[256];
    std::snprintf(path, sizeof(path), "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/sw2_%+04d.ppm", yaw);
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
    const float* q = E.camera_quaternion_world_wxyz;

    auto bodyFromWorld = [&](const float dw[3], double out[3], int conv) {
      const double w=q[0], x=q[1], y=q[2], z=q[3];
      double tx = 2*(y*dw[2]-z*dw[1]);
      double ty = 2*(z*dw[0]-x*dw[2]);
      double tz = 2*(x*dw[1]-y*dw[0]);
      double bx = dw[0] - w*tx - (y*tz - z*ty);
      double by = dw[1] - w*ty - (z*tx - x*tz);
      double bz = dw[2] - w*tz - (x*ty - y*tx);
      if (conv == 0) { out[0]=bx; out[1]=by; out[2]=bz; }
      else { out[0]=bz; out[1]=-bx; out[2]=-by; }
    };

    std::printf("yaw=%+4d cam=(%.2f,%.2f,%.2f)", yaw,
                E.camera_position_world[0], E.camera_position_world[1], E.camera_position_world[2]);
    if (G.rune_count > 0) {
      const auto& r = G.runes[0];
      float w0[3] = {r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2]};
      float w1[3] = {r.r_center_odom[0], r.r_center_odom[2], -r.r_center_odom[1]};
      const float* cand[2] = {w0, w1};
      for (int cw = 0; cw < 2; ++cw) {
        float d[3] = {cand[cw][0]-E.camera_position_world[0],
                      cand[cw][1]-E.camera_position_world[1],
                      cand[cw][2]-E.camera_position_world[2]};
        for (int conv = 0; conv < 2; ++conv) {
          double b[3];
          bodyFromWorld(d, b, conv);
          if (b[2] > 0.01) {
            double u = fx*b[0]/b[2] + cx;
            double v = fy*b[1]/b[2] + cy;
            std::printf("  w%d c%d:(%.0f,%.0f)%s", cw, conv, u, v,
                        (u>-720&&u<2160&&v>-540&&v<1620)?"*":"");
          } else {
            std::printf("  w%d c%d:behind", cw, conv);
          }
        }
      }
    }
    std::printf("\n");
  }
  return 0;
}
