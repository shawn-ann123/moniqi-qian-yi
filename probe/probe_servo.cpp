// 探针9：目视伺服瞄准符心，用对准帧解算坐标系约定
// 1) 检测图像中符的发光目标(橙色/蓝色)质心
// 2) 迭代调整 yaw/pitch 使质心到图像中心
// 3) 对准后读取 ES+GT，输出: 相机q、各列向量、与 rune 方向的夹角(多假设)
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
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
  float yaw = 60.0f, pitch = 90.0f;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--ipc-dir") == 0) ipc = argv[++i];
    else if (std::strcmp(argv[i], "--yaw") == 0) yaw = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--pitch") == 0) pitch = std::atof(argv[++i]);
  }

  ContestClientOptions options;
  options.ipc_directory = ipc;
  ContestClient client(options);
  auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed\n"); return 1; }

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed\n"); return 1; }

  const double fx=1303.67532368147, fy=1303.67532368147, cx=720.0, cy=540.0;

  struct Blob { bool found=false; double u=0, v=0; int n=0; bool orange=false; };
  auto detect = [&](const TcpImageFrame& img) -> Blob {
    Blob b;
    const int W = img.header.width, H = img.header.height;
    const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
    double so=0, sv=0; int no=0;
    double sb=0, svb=0; int nb=0;
    for (int y = 0; y < H; y += 2)
      for (int x = 0; x < W; x += 2) {
        const std::uint8_t* p = img.payload.data() + (y*W + x)*ch;
        int r=p[0], g=p[1], bl=p[2];
        if (r > 180 && r-bl > 80 && r-g > 40) { so+=x; sv+=y; ++no; }
        if (bl > 180 && bl-r > 80 && bl-g > 40) { sb+=x; svb+=y; ++nb; }
      }
    if (no >= 10 && no >= nb) { b.found=true; b.u=so/no; b.v=sv/no; b.n=no; b.orange=true; }
    else if (nb >= 10) { b.found=true; b.u=sb/nb; b.v=svb/nb; b.n=nb; b.orange=false; }
    return b;
  };

  UdpGimbalCommand cmd;
  cmd.yaw_deg = yaw; cmd.pitch_deg = pitch;
  client.sendAim(cmd);
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  for (int iter = 0; iter < 40; ++iter) {
    auto frame = client.nextFrame();
    if (!frame.ok()) { std::this_thread::sleep_for(std::chrono::milliseconds(300)); continue; }
    Blob b = detect(frame.value->image);
    std::printf("iter%02d cmd=(%+7.2f,%6.2f) blob=%s", iter, yaw, pitch,
                b.found ? (b.orange ? "橙" : "蓝") : "无");
    if (b.found) std::printf(" (%4.0f,%4.0f) n=%d", b.u, b.v, b.n);
    std::printf("\n");
    if (!b.found) {
      yaw -= 15.0f;
      if (yaw < -180.0f) yaw += 360.0f;
      cmd.yaw_deg = yaw; cmd.pitch_deg = pitch;
      client.sendAim(cmd);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
    const double eu = b.u - cx, ev = b.v - cy;
    if (std::fabs(eu) < 12 && std::fabs(ev) < 12) break;
    yaw   -= 0.045f * (float)eu;
    pitch += 0.045f * (float)ev;
    if (yaw < -180.0f) yaw += 360.0f;
    if (yaw > 180.0f) yaw -= 360.0f;
    cmd.yaw_deg = yaw; cmd.pitch_deg = pitch;
    client.sendAim(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
  }

  // 对准后：抓当前帧 + ES + GT，解算坐标系
  auto frame = client.nextFrame();
  if (!frame.ok()) { std::fprintf(stderr, "final frame missing\n"); return 1; }
  const auto seq = frame.value->image.header.source_sequence;
  auto reader = mapping.reader();
  auto es = reader.value->readExposureStateForFrame(seq);
  auto gt = reader.value->readGroundTruthForFrame(seq);
  if (!es.ok() || !gt.ok()) { std::fprintf(stderr, "es/gt missing\n"); return 1; }
  const auto& E = *es.value;
  const auto& G = gt.value->ground_truth;
  const float* q = E.camera_quaternion_world_wxyz;

  std::printf("\n== 对准结果 ==\n");
  std::printf("cmd=(%+.2f, %.2f) ES: gimbal_yaw=%.4f pitch=%.4f\n", yaw, pitch, E.gimbal_yaw_rad, E.gimbal_pitch_rad);
  std::printf("cam_pos=(%.4f,%.4f,%.4f)\n", E.camera_position_world[0], E.camera_position_world[1], E.camera_position_world[2]);
  std::printf("q_wxyz=(%.6f,%.6f,%.6f,%.6f)\n", q[0],q[1],q[2],q[3]);
  std::printf("chassis_pos=(%.4f,%.4f,%.4f) chassis_rpy=(%.4f,%.4f,%.4f)\n",
              E.chassis_position_world[0], E.chassis_position_world[1], E.chassis_position_world[2],
              E.chassis_rpy_world[0], E.chassis_rpy_world[1], E.chassis_rpy_world[2]);

  // R 的列向量（q 把某body系的轴映射到世界）
  auto col = [&](int c) {
    double w=q[0], x=q[1], y=q[2], z=q[3];
    double R[3][3] = {
      {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)},
      {2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)},
      {2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}};
    return std::array<double,3>{R[0][c], R[1][c], R[2][c]};
  };
  auto c0 = col(0), c1 = col(1), c2 = col(2);
  std::printf("R cols: c0=(%.4f,%.4f,%.4f) c1=(%.4f,%.4f,%.4f) c2=(%.4f,%.4f,%.4f)\n",
              c0[0],c0[1],c0[2], c1[0],c1[1],c1[2], c2[0],c2[1],c2[2]);

  for (std::uint32_t i = 0; i < G.rune_count; ++i) {
    const auto& r = G.runes[i];
    float w0[3] = {r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2]};
    float w1[3] = {r.r_center_odom[0], r.r_center_odom[2], -r.r_center_odom[1]};
    const float* wc[2] = {w0, w1};
    for (int k = 0; k < 2; ++k) {
      double d[3] = {wc[k][0]-E.camera_position_world[0],
                     wc[k][1]-E.camera_position_world[1],
                     wc[k][2]-E.camera_position_world[2]};
      double L = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
      std::printf("rune[%u]%s: dist=%.3f dir=(%.4f,%.4f,%.4f)\n", i, k?"(x,z,-y)":"", L, d[0]/L, d[1]/L, d[2]/L);
      for (int c = 0; c < 3; ++c) {
        auto cc = col(c);
        double dot = (d[0]*cc[0]+d[1]*cc[1]+d[2]*cc[2])/L;
        std::printf("   angle(c%d, dir)=%.2f°  %s\n", c, std::acos(std::max(-1.0,std::min(1.0,dot)))*180.0/M_PI,
                    std::fabs(dot) > 0.998 ? "<== 同向" : (dot < -0.998 ? "<== 反向" : ""));
      }
    }
  }
  return 0;
}
