# External dependency prefix for RP-26Rune.
# Override the default with: RP26_DEPS_ROOT=/path/to/prefix cmake ...
set(_RP26_DEPS_ROOT "$ENV{RP26_DEPS_ROOT}")
if(NOT _RP26_DEPS_ROOT)
    set(_RP26_DEPS_ROOT "$ENV{HOME}/.local/opt/rp26-deps")
endif()
set(_RP26_DEPS_USR "${_RP26_DEPS_ROOT}/usr")

if(NOT EXISTS "${_RP26_DEPS_USR}/include")
    message(FATAL_ERROR
        "RP-26Rune dependency prefix not found: ${_RP26_DEPS_ROOT}. "
        "Set RP26_DEPS_ROOT to the external dependency prefix.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${_RP26_DEPS_USR}")
list(PREPEND CMAKE_LIBRARY_PATH
    "${_RP26_DEPS_USR}/lib/x86_64-linux-gnu/blas"
    "${_RP26_DEPS_USR}/lib/x86_64-linux-gnu/lapack"
    "${_RP26_DEPS_USR}/lib/x86_64-linux-gnu"
    "${_RP26_DEPS_USR}/lib")
list(PREPEND CMAKE_INCLUDE_PATH "${_RP26_DEPS_USR}/include")
list(PREPEND CMAKE_MODULE_PATH "${_RP26_DEPS_USR}/share/glog/cmake")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "${_RP26_DEPS_USR}/include")
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "${_RP26_DEPS_USR}/include")
