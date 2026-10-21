#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RP="$WS/RP-26Rune"
SIM="${DAEDALUS_SIM_ROOT:-/home/shawn/桌面/rm/daedalus-intel-optimized}"
DEPS_ROOT="${RP26_DEPS_ROOT:-$HOME/.local/opt/rp26-deps}"
IPC="${TALOS_IPC_DIR:-$SIM/runtime/talos-ipc}"
MODE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
  cat <<'EOF'
用法：
  ./run_rune.sh vision [-- 传给 app 的参数]
  ./run_rune.sh ground-truth [-- 传给 rune_stage_gt_probe 的参数]
  ./run_rune.sh stage [-- 传给 rune_stage_probe 的参数]

vision       完整链路：模拟器图像 -> NN 识别 -> 深大打符 -> 云台/开火
ground-truth 真值链路：模拟器 GT -> InactiveTargets -> 后续打符 -> 云台/开火
stage        带分阶段输出的识别链路诊断程序

运行前请先启动新模拟器：
  ./start_new_simulator.sh
EOF
}

if [[ "$MODE" != "vision" && "$MODE" != "ground-truth" && "$MODE" != "stage" ]]; then
  usage
  exit 2
fi

if [[ ! -x "$RP/output/app" || ! -x "$RP/output/rune_stage_probe" || ! -x "$RP/output/rune_stage_gt_probe" || ! -x "$RP/output/sim_setup" ]]; then
  "$WS/build_all.sh"
fi

export TALOS_IPC_DIR="$IPC"
export DAEDALUS_TALOS_IMAGE_TRANSPORT=tcp
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="$RP/output:$DEPS_ROOT/usr/lib/x86_64-linux-gnu:$DEPS_ROOT/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SETUP="$RP/output/sim_setup"
if ! "$SETUP" health >/dev/null 2>&1; then
  echo "新模拟器未就绪，或 IPC 目录不匹配：$IPC" >&2
  echo "请先运行：'$WS/start_new_simulator.sh'" >&2
  exit 1
fi

# 通过公共 SDK 切换大符场景和亮叶状态，不触碰模拟器内部实体。
"$SETUP"

case "$MODE" in
  vision)
    exec "$RP/output/app" "$@"
    ;;
  ground-truth)
    exec "$RP/output/rune_stage_gt_probe" "$@"
    ;;
  stage)
    exec "$RP/output/rune_stage_probe" "$@"
    ;;
esac
