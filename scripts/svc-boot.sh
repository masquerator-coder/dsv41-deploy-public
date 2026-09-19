#!/usr/bin/env bash
# svc-boot.sh — 开机自启包装（由 systemd unit dsv41.service 调用）
#   1) 等本机 docker 与两台 worker 的 docker/ssh 就绪（重启后 worker 往往慢一拍）
#   2) 若服务已在跑 → 直接退出（幂等）；若容器在但 health 不通（半死状态）→ 先 stop
#   3) 交给 svc.sh start（预检 → share → serve → 等就绪 → 三层验证）
set -u
cd "$(dirname "$0")" || exit 1
LOG="boot-autostart-$(date +%Y%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date '+%F %T') dsv41 开机自启开始（日志 $LOG）==="
[ -f .env ] || { echo "缺 .env"; exit 1; }
set -a; . ./.env; set +a
PORT=${PORT:-8888}

wait_ready() {
  local i ok
  for i in $(seq 1 60); do
    ok=1
    docker info >/dev/null 2>&1 || ok=0
    for h in ${WORKER_HOSTS:-}; do
      ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$h" \
        'sudo -n true >/dev/null 2>&1 && docker info >/dev/null 2>&1' || ok=0
    done
    [ "$ok" = 1 ] && { echo "  就绪（第 ${i} 次探测）"; return 0; }
    echo "  等待本机 docker / worker 就绪…（$i/60）"; sleep 10
  done
  return 1
}
wait_ready || { echo "✗ 10 分钟内未能就绪，放弃自动启动（可手动 ./svc.sh start）"; exit 1; }

if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "  服务已在运行（/health OK），无需启动"; exit 0
fi
if docker ps --format '{{.Names}}' | grep -qE '^dsv41-(head|worker)$'; then
  echo "  发现残留容器但 /health 不通 → 先停干净再起"
  ./svc.sh stop || true
fi

echo "  交给 svc.sh start（含预检/share/等就绪/三层验证，最长 30 分钟）"
./svc.sh start 1800
rc=$?
echo "=== $(date '+%F %T') 结束 rc=$rc ==="
exit $rc
