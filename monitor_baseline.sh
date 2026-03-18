#!/bin/bash
# ============================================================================
# 训练监控脚本 (监控 baseline 实验)
#
# 使用方法:
#   ./monitor_train.sh              # 监控最新的训练日志
#   ./monitor_train.sh nohup        # 监控 nohup 输出
# ============================================================================

cd "$(dirname "$0")"

LOG_DIR="logs/df40_style"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "=============================================="
    echo "训练监控脚本 (baseline)"
    echo "=============================================="
    echo ""
    echo "用法: ./monitor_train.sh [nohup]"
    echo ""
    echo "参数:"
    echo "  (无参数)    监控最新的训练日志"
    echo "  nohup       监控最新的 nohup 输出文件"
    echo ""
    echo "示例:"
    echo "  ./monitor_train.sh"
    echo "  ./monitor_train.sh nohup"
    echo ""
    exit 0
fi

if [ "$1" = "nohup" ]; then
    # 监控 nohup 输出
    LATEST_LOG=$(ls -t nohup_baseline_*.out 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        echo "未找到 nohup 输出文件！"
        echo "请先运行 ./run_train.sh 启动训练"
        exit 1
    fi
else
    # 监控最新的训练日志
    LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)

    # 如果子目录没找到，在 logs 根目录查找
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(find logs -name "*.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    fi
fi

if [ -z "$LATEST_LOG" ]; then
    echo "=============================================="
    echo "未找到日志文件！"
    echo "=============================================="
    echo ""
    echo "可能的原因:"
    echo "  1. 训练尚未启动"
    echo ""
    echo "启动训练:"
    echo "  ./run_train.sh"
    echo ""
    exit 1
fi

# 显示训练状态
echo "=============================================="
echo "训练监控 (baseline)"
echo "=============================================="
echo "日志文件: $LATEST_LOG"
echo ""

# 显示训练进程
TRAIN_PID=$(pgrep -f "train.py" 2>/dev/null | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "训练进程: PID $TRAIN_PID (运行中)"
else
    echo "训练进程: (未检测到运行中的训练)"
fi

echo ""
echo "按 Ctrl+C 退出监控（不会停止训练）"
echo "=============================================="
echo ""

tail -f "$LATEST_LOG"
