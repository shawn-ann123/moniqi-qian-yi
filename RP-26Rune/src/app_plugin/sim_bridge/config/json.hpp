#pragma once

#include <filesystem>

#include "json/ReJson.hpp"

inline const std::filesystem::path SIM_BRIDGE_CONFIG_DIR =
    std::filesystem::path(__FILE__).parent_path();
inline ReJson J_SIM_BRIDGE((SIM_BRIDGE_CONFIG_DIR / "sim_bridge.json").string());
