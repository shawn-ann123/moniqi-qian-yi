#pragma once

#include <memory>

#include "CameraManager.hpp"

namespace app_plugin
{

// 模拟器桥接：从 Daedalus 模拟器 TCP 图像流取帧，同一帧的
// ExposureState 换算成算法约定的 ECSData 后发布 InputFrame。
class SimCameraManager final : public CameraManager
{
public:
    SimCameraManager();
    ~SimCameraManager() override;

    void process(const app::Context &context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace app_plugin
