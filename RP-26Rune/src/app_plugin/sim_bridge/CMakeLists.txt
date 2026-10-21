set(SIM_BRIDGE_PLUGIN_TARGET sim_bridge_plugin_lib)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(DAEDALUS_SIM_SDK_ROOT "$ENV{DAEDALUS_SIM_SDK_ROOT}"
    CACHE PATH "Daedalus simulator SDK root")
if(NOT DAEDALUS_SIM_SDK_ROOT)
    message(FATAL_ERROR
        "DAEDALUS_SIM_SDK_ROOT is required; point it at the simulator's installed sdk directory")
endif()
unset(DAEDALUS_SIM_SDK_LIBRARY CACHE)
find_library(DAEDALUS_SIM_SDK_LIBRARY
    NAMES daedalus_sim_sdk
    PATHS "${DAEDALUS_SIM_SDK_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(${SIM_BRIDGE_PLUGIN_TARGET} SHARED
    src/SimCameraManager.cpp
    src/SimSendEncoder.cpp
)

target_include_directories(${SIM_BRIDGE_PLUGIN_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    "${DAEDALUS_SIM_SDK_ROOT}/include"
)

target_link_libraries(${SIM_BRIDGE_PLUGIN_TARGET} PUBLIC
    app_interface_lib
    class_loader_lib
    config_loader_lib
    google_logger_lib
    opencv_core
    opencv_imgproc
    "${DAEDALUS_SIM_SDK_LIBRARY}"
    pthread
)
