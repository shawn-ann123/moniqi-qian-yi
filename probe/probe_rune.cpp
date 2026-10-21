// 探针5：激活大符场景，再验证真值与图像的对应关系
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
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }

  auto sc = client.selectScene(ContestScene::Energy);
  std::printf("selectScene Energy: ok=%d msg=%s\n", sc.ok(), sc.ok() ? "" : sc.status.message.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(2500));

  RuneScenario scenario;
  scenario.mode = RuneMode::Large;
  scenario.motion = RuneMotion::RuleDriven;
  auto rs = client.setRuneScenario(scenario);
  std::printf("setRuneScenario Large/RuleDriven: ok=%d", rs.ok());
  if (rs.ok()) std::printf(" status=%d msg=%s data=%s", (int)rs.value->status, rs.value->message.c_str(), rs.value->data_json.c_str());
  else std::printf(" msg=%s", rs.status.message.c_str());
  std::printf("\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  TalosMetadataMapping mapping;
  auto st = mapping.open(ipc + "/talos_ipc_meta");
  if (!st.ok()) { std::fprintf(stderr, "open failed: %s\n", st.message.c_str()); return 1; }

  for (int k = 0; k < 3; ++k) {
    auto frame = client.nextFrame();
    if (!frame.ok()) { std::fprintf(stderr, "no frame\n"); return 1; }
    const auto seq = frame.value->image.header.source_sequence;
    auto reader = mapping.reader();
    auto gt = reader.value->readGroundTruthForFrame(seq);
    if (!gt.ok()) continue;
    const auto& G = gt.value->ground_truth;
    std::printf("[frame seq=%llu] rune_count=%u target_count=%u\n",
                (unsigned long long)seq, G.rune_count, G.target_count);
    for (std::uint32_t i = 0; i < G.rune_count && i < 4; ++i) {
      const auto& r = G.runes[i];
      std::printf("  rune[%u] team=%u mode=%u mech=%u center=(%.4f,%.4f,%.4f) radius=%.3f angle=%.3f v_roll=%.3f dir=%d blade=%d\n",
                  i, r.team, r.rune_mode, r.mechanism_state,
                  r.r_center_odom[0], r.r_center_odom[1], r.r_center_odom[2],
                  r.radius, r.current_angle, r.v_roll, r.direction, r.blade_id);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
  }
  return 0;
}
