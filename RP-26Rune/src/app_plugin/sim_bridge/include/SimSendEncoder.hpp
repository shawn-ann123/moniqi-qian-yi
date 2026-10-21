#pragma once

#include <memory>

#include "SendEncoder.hpp"

namespace app_plugin
{

// 模拟器桥接：消费 FireResult，转换成模拟器云台指令（UDP）发送。
class SimSendEncoder final : public SendEncoder
{
public:
    SimSendEncoder();
    ~SimSendEncoder() override;

    void process(const app::Context &context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace app_plugin
