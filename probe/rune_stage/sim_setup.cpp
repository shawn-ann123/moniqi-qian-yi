#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace daedalus::sim::sdk::v1;

int main(int argc, char **argv)
{
    const char *configured = std::getenv("TALOS_IPC_DIR");
    if (configured == nullptr || configured[0] == '\0')
    {
        std::fprintf(stderr, "TALOS_IPC_DIR is not set\n");
        return 2;
    }

    TalosMetadataMapping mapping;
    const auto mapped = mapping.open(std::string(configured) + "/talos_ipc_meta");
    if (!mapped.ok())
    {
        std::fprintf(stderr, "metadata: %s\n", mapped.message.c_str());
        return 1;
    }
    auto reader = mapping.reader();
    if (!reader.ok())
    {
        std::fprintf(stderr, "reader: %s\n", reader.status.message.c_str());
        return 1;
    }
    auto camera = reader.value->readCameraInfo();
    if (!camera.ok())
    {
        std::fprintf(stderr, "camera: %s\n", camera.status.message.c_str());
        return 1;
    }

    if (argc > 1 && std::string(argv[1]) == "health")
    {
        std::printf("ok sdk=%s camera=%ux%u fx=%.3f fy=%.3f\n", kSdkVersion,
                    camera.value->width, camera.value->height, camera.value->fx, camera.value->fy);
        return 0;
    }

    SceneControlOptions options;
    options.session_id = "rune-migration-setup";
    SceneControlClient scene(options);
    const auto session = scene.createSession();
    if (!session.ok())
    {
        std::fprintf(stderr, "createSession: %s\n", session.status.message.c_str());
        return 1;
    }
    const auto selected = scene.setScene(SceneMode::Energy);
    if (!selected.ok())
    {
        std::fprintf(stderr, "setScene: %s\n", selected.status.message.c_str());
        return 1;
    }

    RuneState rune;
    rune.mode = RuneMode::Large;
    rune.pending_targets = {1};
    rune.activated_targets = {0, 2};
    const auto state = scene.setRuneState(rune);
    if (!state.ok())
    {
        std::fprintf(stderr, "setRuneState: %s\n", state.status.message.c_str());
        return 1;
    }

    // 新公共 SDK 定义 pitch=90° 为水平瞄准。大符中心约低 20°。
    UdpGimbalClient gimbal;
    UdpGimbalCommand aim;
    aim.yaw_deg = 0.0F;
    aim.pitch_deg = 70.0F;
    aim.fire_advice = false;
    ClientStatus aimed = ClientStatus::success();
    for (int attempt = 0; attempt < 20; ++attempt)
    {
        aimed = gimbal.send(aim);
        if (!aimed.ok())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!aimed.ok())
    {
        std::fprintf(stderr, "aim: %s\n", aimed.message.c_str());
        return 1;
    }

    std::printf("ok scene=energy rune=large pending=1 activated=0,2 aim=(0,70) sdk=%s\n",
                kSdkVersion);
    return 0;
}
