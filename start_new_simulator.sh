#!/usr/bin/env bash
set -euo pipefail

SIM="${DAEDALUS_SIM_ROOT:-/home/shawn/桌面/rm/daedalus-intel-optimized}"
START_SCRIPT="$SIM/start-simulator.sh"

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "找不到新模拟器启动脚本：$START_SCRIPT" >&2
  exit 1
fi

export DAEDALUS_TALOS_IMAGE_TRANSPORT=tcp
export TALOS_IPC_DIR="${TALOS_IPC_DIR:-$SIM/runtime/talos-ipc}"
# 新入口默认关闭外部自瞄接收（等价于未按 F5），关闭时 5601
# 会收包但不驱动云台。真值/识别两条链路都需要在启动时显式开启。
export DAEDALUS_AUTO_AIM_ON_START="${DAEDALUS_AUTO_AIM_ON_START:-1}"
cd "$SIM"
exec ./start-simulator.sh "$@"
