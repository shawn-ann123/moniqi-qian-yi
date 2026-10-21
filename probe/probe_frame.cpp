// 探针7：符已可见，验证坐标系假设 —— 用各假设预测符心像素位置，与图像对照
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
  float cy = 40.0f, cp = 90.0f;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];
    else if (std::strcmp(argv[i], "--yaw") == 0) cy = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--pitch") == 0) cp = std::atof(argv[++i]);
  }

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed\n"); return 1; }

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed\n"); return 1; }

  UdpGimbalCommand cmd;
  cmd.yaw_deg = cy; cmd.pitch_deg = cp;
  client.sendAim(cmd);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));

  auto frame = client.nextFrame();
  if (!frame.ok()) { std::fprintf(stderr, "no frame\n"); return 1; }
  const auto& img = frame.value->image;
  const int W = img.header.width, H = img.header.height;
  const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
  const auto seq = img.header.source_sequence;

  char path[256];
  std::snprintf(path, sizeof(path), "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/frame_yaw%+.0f_pitch%.0f.ppm", cy, cp);
  FILE* f = std::fopen(path, "wb");
  std::fprintf(f, "P6\n%d %d\n255\n", W, H);
  for (int i = 0; i < W*H; ++i)
    std::fwrite(img.payload.data() + i*ch, 1, 3, f);
  std::fclose(f);

  auto reader = mapping.reader();
  auto es = reader.value->readExposureStateForFrame(seq);
  auto gt = reader.value->readGroundTruthForFrame(seq);
  if (!es.ok() || !gt.ok()) { std::fprintf(stderr, "no es/gt\n"); return 1; }
  const auto& E = *es.value;
  const auto& G = gt.value->ground_truth;

  const double fx=1303.67532368147, fy=1303.67532368147, cx=720.0, cyc=540.0;
  const float* q = E.camera_quaternion_world_wxyz;
  std::printf("cmd yaw=%.1f pitch=%.1f\n", cy, cp);
  std::printf("cam_pos=(%.4f,%.4f,%.4f) q_wxyz=(%.4f,%.4f,%.4f,%.4f)\n",
              E.camera_position_world[0], E.camera_position_world[1], E.camera_position_world[2],
              q[0], q[1], q[2], q[3]);
  std::printf("rune_count=%u\n", G.rune_count);

  // 四元数旋转(R^T): world->body。body 约定候选:
  //  A: body = optical (x右 y下 z前)
  //  B: body = 机体系 (x前 y左 z上)  对应 optical=(x_o,y_o,z_o) 的映射 body=(z_o,-x_o,-y_o)
  // 世界位置候选: odom 直接当世界系 (W0)
  auto bodyFromWorld = [&](const float dw[3], double out[3], int conv) {
    // 先用 q^-1 得伪 body 帧 (数学坐标系 wxyz 直接反转)
    const double w=q[0], x=q[1], y=q[2], z=q[3];
    double tx = 2*(y*dw[2]-z*dw[1]);
    double ty = 2*(z*dw[0]-x*dw[2]);
    double tz = 2*(x*dw[1]-y*dw[0]);
    double bx = dw[0] - w*tx - (y*tz - z*ty);
    double by = dw[1] - w*ty - (z*tx - x*tz);
    double bz = dw[2] - w*tz - (x*ty - y*tx);
    if (conv == 0) { out[0]=bx; out[1]=by; out[2]=bz; }           // 直接当 optical
    else { out[0]=bz; out[1]=-bx; out[2]=-by; }                   // 当机体系再转 optical
  };

  for (std::uint32_t i = 0; i < G.rune_count; ++i) {
    const auto& r = G.runes[i];
    // 世界系候选: (x,y,z) 与 (x,z,-y)
    const float* cand[2] = { nullptr, nullptr };
    float w0[3] = {r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2]};
    float w1[3] = {r.r_center_odom[0], r.r_center_odom[2], -r.r_center_odom[1]};
    cand[0]=w0; cand[1]=w1;
    for (int cw = 0; cw < 2; ++cw) {
      float d[3] = {cand[cw][0]-E.camera_position_world[0],
                    cand[cw][1]-E.camera_position_world[1],
                    cand[cw][2]-E.camera_position_world[2]};
      double dist = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
      for (int conv = 0; conv < 2; ++conv) {
        double b[3];
        bodyFromWorld(d, b, conv);
        if (b[2] <= 0.01) { std::printf("  rune[%u] world%d conv%d: behind (z=%.2f) dist=%.2f\n", i, cw, conv, b[2], dist); continue; }
        double u = fx*b[0]/b[2] + cx;
        double v = fy*b[1]/b[2] + cyc;
        std::printf("  rune[%u] world%d conv%d: dist=%.2f proj=(%.0f, %.0f) %s\n",
                    i, cw, conv, dist, u, v,
                    (u>0&&u<W&&v>0&&v<H) ? "<-- 在画面内" : "");
      }
    }
  }
  return 0;
}
