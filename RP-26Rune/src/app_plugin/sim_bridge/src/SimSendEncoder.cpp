#include "SimSendEncoder.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>

#include <glog/logging.h>

#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

#include "class_loader.hpp"
#include "json.hpp"

namespace app_plugin
{

struct SimSendEncoder::Impl
{
    using FireSubscriber = LatestBuffer<FireResult>::Subscriber;
    using UdpGimbalClient = daedalus::sim::sdk::v1::UdpGimbalClient;

    void load_config();

    std::optional<FireSubscriber> input;
    std::unique_ptr<UdpGimbalClient> client;
    double yaw_offset_deg = 0.0;
    double yaw_scale = 1.0;
    double pitch_scale = 1.0;
    double pitch_offset_deg = 65.0;
    bool fire_enabled = true;
    long long fire_interval_ms = 100;
    bool send_when_not_found = false;
    bool initialized = false;
    bool had_target = false;
    bool has_fired = false;
    std::chrono::steady_clock::time_point last_fire_sent{};
};

void SimSendEncoder::Impl::load_config()
{
    J_SIM_BRIDGE.updateJson();
    if (!J_SIM_BRIDGE.config_.isOpened())
        throw std::runtime_error(
            "failed to open sim bridge config: " +
            (SIM_BRIDGE_CONFIG_DIR / "sim_bridge.json").string());

    const cv::FileNode send = J_SIM_BRIDGE.config_["send"];
    if (send.empty())
        return;

    if (!send["yaw_offset_deg"].empty())
        yaw_offset_deg = static_cast<double>(send["yaw_offset_deg"]);
    if (!send["yaw_scale"].empty())
        yaw_scale = static_cast<double>(send["yaw_scale"]);
    if (!send["pitch_scale"].empty())
        pitch_scale = static_cast<double>(send["pitch_scale"]);
    if (!send["pitch_offset_deg"].empty())
        pitch_offset_deg = static_cast<double>(send["pitch_offset_deg"]);
    if (!send["fire_enabled"].empty())
        fire_enabled = static_cast<int>(send["fire_enabled"]) != 0;
    if (!send["fire_interval_ms"].empty())
        fire_interval_ms = static_cast<int>(send["fire_interval_ms"]);
    if (!send["send_when_not_found"].empty())
        send_when_not_found = static_cast<int>(send["send_when_not_found"]) != 0;
}

SimSendEncoder::SimSendEncoder()
    : m_impl(std::make_unique<Impl>())
{
}

SimSendEncoder::~SimSendEncoder() = default;

void SimSendEncoder::process(const app::Context &context)
{
    Impl &impl = *m_impl;

    if (!impl.initialized)
    {
        impl.load_config();
        impl.input.emplace(context.get_buffer_subscriber<FireResult>(this));
        impl.client = std::make_unique<Impl::UdpGimbalClient>();
        impl.initialized = true;
        LOG(INFO) << "[SimSendEncoder] initialized; pitch_offset_deg=" << impl.pitch_offset_deg
                  << ", yaw_offset_deg=" << impl.yaw_offset_deg
                  << ", fire_enabled=" << impl.fire_enabled
                  << ", fire_interval_ms=" << impl.fire_interval_ms;
    }

    const FireResult result = impl.input->wait_pop();
    if (!std::isfinite(result.yaw) || !std::isfinite(result.pitch))
    {
        LOG_EVERY_N(ERROR, 20) << "[SimSendEncoder] rejected non-finite result: yaw="
                               << result.yaw << ", pitch=" << result.pitch;
        return;
    }

    if (!result.is_find_buff && !impl.send_when_not_found)
    {
        if (impl.had_target)
        {
            impl.had_target = false;
            LOG(INFO) << "[SimSendEncoder] target lost; hold current aim";
        }
        return;
    }
    if (!impl.had_target)
    {
        impl.had_target = true;
        LOG(INFO) << "[SimSendEncoder] target acquired";
    }

    daedalus::sim::sdk::v1::UdpGimbalCommand command;
    const float yaw_deg = static_cast<float>(impl.yaw_scale * result.yaw + impl.yaw_offset_deg);
    const float pitch_deg = static_cast<float>(impl.pitch_scale * result.pitch + impl.pitch_offset_deg);
    command.yaw_deg = yaw_deg;
    command.pitch_deg = pitch_deg;

    const auto now = std::chrono::steady_clock::now();
    const bool fire_due =
        impl.fire_enabled && result.is_enable_fire &&
        (!impl.has_fired ||
         now - impl.last_fire_sent >= std::chrono::milliseconds(impl.fire_interval_ms));
    command.fire_advice = fire_due;

    const auto status = impl.client->send(command);
    if (!status.ok())
    {
        LOG_EVERY_N(WARNING, 20) << "[SimSendEncoder] udp send failed: " << status.message;
        return;
    }

    if (fire_due)
    {
        impl.last_fire_sent = now;
        impl.has_fired = true;
        LOG(INFO) << "[SimSendEncoder] fire advice: yaw=" << yaw_deg
                  << ", pitch=" << pitch_deg;
    }
    else
    {
        LOG_EVERY_N(INFO, 120) << "[SimSendEncoder] aim sent: yaw=" << yaw_deg
                               << ", pitch=" << pitch_deg
                               << ", find_buff=" << result.is_find_buff
                               << ", enable_fire=" << result.is_enable_fire;
    }
}

} // namespace app_plugin

REGISTER_PLUGIN("SendEncoder", app_plugin::SimSendEncoder)
