#!/bin/bash
# ============================================================================
# 后台训练启动脚本 (使用 baseline.yaml 配置)
# 
# 特性：
#   1. 使用 nohup 后台运行，断开 SSH 连接不会中断训练
#   2. 自动将输出保存到日志文件
#   3. 提供监控命令
#
# 使用方法:
#   chmod +x run_train.sh
#   
#   # 直接启动训练
#   ./run_train.sh
#   
#   # 指定 GPU
#   ./run_train.sh --device 1
#   
#   # 覆盖其他参数
#   ./run_train.sh --batch_size 64 --lr 3e-5
#   
#   # 启用 wandb
#   ./run_train.sh --use_wandb
#
# 监控训练:
#   ./monitor_train.sh                      # 使用监控脚本
#
# 停止训练:
#   kill $(cat logs/train_pid.txt)
#   # 或者
#   ps aux | grep train.py
#   kill <PID>
# ============================================================================

set -e

# 进入项目目录
cd "$(dirname "$0")"

# 固定配置文件
CONFIG_FILE="configs/baseline.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建 nohup 输出文件
NOHUP_LOG="nohup_baseline_${TIMESTAMP}.out"

echo "=============================================="
echo "启动后台训练"
echo "=============================================="
echo "配置文件: $CONFIG_FILE"
echo "额外参数: $@"
echo "Nohup 日志: $NOHUP_LOG"
echo "=============================================="

# 启动训练（后台运行）
# -u: unbuffered，确保输出实时写入
nohup python -u tools/train.py \
    --config "$CONFIG_FILE" \
    "$@" \
    > "$NOHUP_LOG" 2>&1 &

# 获取进程 ID
PID=$!
mkdir -p logs
echo "$PID" > logs/train_pid.txt

echo ""
echo "=============================================="
echo "训练已在后台启动!"
echo "=============================================="
echo ""
echo "进程 PID: $PID"
echo ""

# 等待几秒让日志文件创建
sleep 3

# 尝试找到训练日志
echo "监控命令:"
echo "  tail -f $NOHUP_LOG"
echo "  ./monitor_train.sh"

echo ""
echo "查看进程:"
echo "  ps aux | grep train.py"
echo ""
echo "查看 GPU:"
echo "  nvidia-smi"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止训练:"
echo "  kill $PID"
echo ""
echo "=============================================="

# 显示前几行输出确认启动成功
echo ""
echo "启动输出 (前 20 行):"
echo "----------------------------------------------"
sleep 2
head -20 "$NOHUP_LOG" 2>/dev/null || echo "(等待输出...)"
echo "----------------------------------------------"
echo ""
echo "使用 'tail -f $NOHUP_LOG' 查看完整输出"
