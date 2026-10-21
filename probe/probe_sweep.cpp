// 探针6：偏航扫描，找到符在视野中的方向
#include <daedalus_sim_sdk/contest_client.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

  for (int yaw = -180; yaw < 180; yaw += 30) {
    UdpGimbalCommand cmd;
    cmd.yaw_deg = (float)yaw; cmd.pitch_deg = 90.0f;
    client.sendAim(cmd);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    auto frame = client.nextFrame();
    if (!frame.ok()) continue;
    const auto& img = frame.value->image;
    const int W = img.header.width, H = img.header.height;
    const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;

    char path[256];
    std::snprintf(path, sizeof(path), "/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/sweep_%+04d.ppm", yaw);
    FILE* f = std::fopen(path, "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; ++i)
      std::fwrite(img.payload.data() + i*ch, 1, 3, f);
    std::fclose(f);

    // 亮斑质心（饱和亮色）
    double sx=0, sy=0; int n=0;
    for (int y = 0; y < H; y += 2)
      for (int x = 0; x < W; x += 2) {
        const std::uint8_t* p = img.payload.data() + (y*W + x)*ch;
        int r=p[0], g=p[1], b=p[2];
        int mx = std::max(r, std::max(g,b)), mn = std::min(r, std::min(g,b));
        if (mx > 180 && mx - mn > 100) { sx+=x; sy+=y; ++n; }
      }
    std::printf("yaw=%+4d: blob n=%d centroid=(%.0f,%.0f)\n", yaw, n, n>0?sx/n:0, n>0?sy/n:0);
  }
  return 0;
}
