# Public framework examples keep the original plugin names and topology.
add_subdirectory(single_camera_manager)
add_subdirectory(detector)
add_subdirectory(fire_control_system)
add_subdirectory(receive_decoder)
add_subdirectory(send_encoder)
add_subdirectory(tracker_manager)
# 模拟器桥接插件（工作空间新增，不修改示例插件本身）。
add_subdirectory(sim_bridge)
