#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RP="$WS/RP-26Rune"
SIM="${DAEDALUS_SIM_ROOT:-/home/shawn/桌面/rm/daedalus-intel-optimized}"
BUILD="${RP26_BUILD_DIR:-$HOME/.cache/rp26-rune/build}"
DEPS_ROOT="${RP26_DEPS_ROOT:-$HOME/.local/opt/rp26-deps}"
SDK_INSTALL="${DAEDALUS_SIM_SDK_ROOT:-$SIM/sdk}"

# 模拟器 SDK 由已安装的模拟器提供，通用三方库放在仓库外。
if [[ ! -d "$DEPS_ROOT/usr/include" ]]; then
  echo "未找到外部依赖目录：$DEPS_ROOT" >&2
  echo "请设置 RP26_DEPS_ROOT 指向外部依赖前缀。" >&2
  exit 1
fi
if [[ ! -f "$SDK_INSTALL/include/daedalus_sim_sdk/talos_v1.hpp" || \
      ! -f "$SDK_INSTALL/lib/libdaedalus_sim_sdk.a" ]]; then
  echo "未找到新模拟器公共 SDK：$SDK_INSTALL" >&2
  exit 1
fi

cmake -S "$RP" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$WS/cmake/rp26-user-deps.cmake" \
  -DDAEDALUS_SIM_SDK_ROOT="$SDK_INSTALL"
cmake --build "$BUILD" -j"$(nproc)"
RP26_DEPS_ROOT="$DEPS_ROOT" DAEDALUS_SIM_SDK_ROOT="$SDK_INSTALL" \
  "$WS/probe/rune_stage/build.sh" all

echo "已构建本地 RP26 与两条打符链路：$RP/output"
