// 探针4：直接看图像，用亮斑检测找到符在图像中的位置，
// 再用射线-真值匹配确定相机系轴向约定
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

// v_world = q * v_frame （四元数 wxyz 正向旋转）
static void quatRotate(const float q[4], const float v[3], float out[3]) {
  const float w=q[0], x=q[1], y=q[2], z=q[3];
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

  UdpGimbalCommand cmd;
  cmd.yaw_deg = 0.0f; cmd.pitch_deg = 90.0f;
  client.sendAim(cmd);
  std::this_thread::sleep_for(std::chrono::milliseconds(900));

  auto frame = client.nextFrame();
  if (!frame.ok()) { std::fprintf(stderr, "no frame\n"); return 1; }
  const auto& img = frame.value->image;
  const auto seq = img.header.source_sequence;
  const int W = img.header.width, H = img.header.height;
  const int ch = img.header.format == tcp_image::PixelFormat::Rgba32 ? 4 : 3;
  std::printf("frame %llux%llu ch=%d seq=%llu\n", (unsigned long long)W, (unsigned long long)H, ch,
              (unsigned long long)seq);

  // 保存 PPM 供人工查看
  {
    FILE* f = std::fopen("/home/shawn/桌面/rm/日志/迁移（模拟器）/probe/frame.ppm", "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W*H; ++i)
      std::fwrite(img.payload.data() + i*ch, 1, 3, f);
    std::fclose(f);
    std::printf("saved frame.ppm\n");
  }

  // 亮斑检测：饱和且亮（符激活发光体）
  double sx=0, sy=0; int n=0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const std::uint8_t* p = img.payload.data() + (y*W + x)*ch;
      int r=p[0], g=p[1], b=p[2];
      int mx = std::max(r, std::max(g,b)), mn = std::min(r, std::min(g,b));
      if (mx > 180 && mx - mn > 100) { sx+=x; sy+=y; ++n; }
    }
  if (n > 0) std::printf("bright blob: n=%d centroid=(%.1f, %.1f)\n", n, sx/n, sy/n);
  else std::printf("no bright blob found\n");

  auto reader = mapping.reader();
  auto es = reader.value->readExposureStateForFrame(seq);
  auto gt = reader.value->readGroundTruthForFrame(seq);
  if (!es.ok() || !gt.ok()) { std::fprintf(stderr, "no es/gt\n"); return 1; }
  const auto& E = *es.value;
  const auto& G = gt.value->ground_truth;

  const double fx=1303.67532368147, fy=1303.67532368147, cx=720.0, cy=540.0;

  if (n > 0) {
    const double u = sx/n, v = sy/n;
    // 4 种轴向假设下，像素射线（相机系）
    struct Hyp { const char* name; double sx, sy, sz; } hyps[] = {
      {"x右 y下 z前", 1, 1, 1},
      {"x左 y下 z前", -1, 1, 1},
      {"x右 y上 z前(GL-style z后)", 1, -1, -1},
      {"x右 y上 z前", 1, -1, 1},
      {"x左 y上 z后", -1, -1, -1},
    };
    for (auto& h : hyps) {
      float ray_c[3] = {(float)(h.sx*(u-cx)/fx), (float)(h.sy*(v-cy)/fy), (float)(h.sz)};
      float ray_w[3];
      quatRotate(E.camera_quaternion_world_wxyz, ray_c, ray_w);
      float L = std::sqrt(ray_w[0]*ray_w[0]+ray_w[1]*ray_w[1]+ray_w[2]*ray_w[2]);
      for (int k=0;k<3;++k) ray_w[k]/=L;
      std::printf("hyp %-24s ray_w=(%.4f,%.4f,%.4f)\n", h.name, ray_w[0], ray_w[1], ray_w[2]);
      for (std::uint32_t i = 0; i < G.rune_count && i < 2; ++i) {
        const auto& r = G.runes[i];
        float d[3] = {r.r_center_odom[0]-E.camera_position_world[0],
                      r.r_center_odom[1]-E.camera_position_world[1],
                      r.r_center_odom[2]-E.camera_position_world[2]};
        float t = d[0]*ray_w[0]+d[1]*ray_w[1]+d[2]*ray_w[2];
        float perp[3] = {d[0]-t*ray_w[0], d[1]-t*ray_w[1], d[2]-t*ray_w[2]};
        float dist = std::sqrt(perp[0]*perp[0]+perp[1]*perp[1]+perp[2]*perp[2]);
        std::printf("    rune[%u]: t=%.3f 垂距=%.4f m\n", i, t, dist);
      }
    }
  }

  // 世界系各轴的“上”验证：相机 z 轴在世界系中的方向
  float zc[3]; quatRotate(E.camera_quaternion_world_wxyz, (float[]){0,0,1}, zc);
  float xc[3]; quatRotate(E.camera_quaternion_world_wxyz, (float[]){1,0,0}, xc);
  float yc[3]; quatRotate(E.camera_quaternion_world_wxyz, (float[]){0,1,0}, yc);
  std::printf("cam axes in world: X=(%.4f,%.4f,%.4f) Y=(%.4f,%.4f,%.4f) Z=(%.4f,%.4f,%.4f)\n",
              xc[0],xc[1],xc[2], yc[0],yc[1],yc[2], zc[0],zc[1],zc[2]);
  std::printf("cam pos world: (%.4f,%.4f,%.4f)\n",
              E.camera_position_world[0], E.camera_position_world[1], E.camera_position_world[2]);
  return 0;
}
