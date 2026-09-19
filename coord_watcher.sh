#!/bin/bash
# 协调 watcher: 等 E5(S11) 结束后, 在 queue4 resume 2B 之前插空跑 ResNet, 再恢复队列
set -u
LOG=/root/coord.log
log() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }

log "coord watcher 启动, 等 E5 结束..."
while pgrep -f 'e5_multi_horizon' > /dev/null 2>&1; do
  sleep 120
done
log "E5 结束, 暂停 queue4 (阻止其抢跑 2B)"

# 暂停 queue4 (抢在它 resume 2B 之前)
pkill -f 'queue4_watcher' 2>/dev/null
sleep 3

# 清理显存 (确保干净)
pkill -9 -f 'resnet_snn' 2>/dev/null
sleep 2

# 跑 ResNet
log "跑 ResNet SNN..."
cd /root/human-brain-simulation || exit 1
nohup /root/M2/.venv/bin/python resnet_snn.py > logs/resnet_snn.log 2>&1 &
RESNET_PID=$!
log "ResNet PID=$RESNET_PID"

# 等 ResNet 结束
while kill -0 "$RESNET_PID" 2>/dev/null; do
  sleep 60
done
log "ResNet 结束 (exit 检查)"

# 恢复队列 (重启 queue4, 让它 resume 2B)
log "恢复队列, 重启 queue4_watcher..."
nohup bash /root/queue4_watcher.sh > /dev/null 2>&1 &
log "queue4 已重启, 协调完成 $(date -u '+%F %T')"
