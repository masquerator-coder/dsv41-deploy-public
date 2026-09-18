#!/usr/bin/env bash
# =============================================================================
# svc.sh — DeepSeek-V4.1-Flash（3× DGX Spark / TP=3）服务一键启停与体检
#   位置：head 的 ~/dsv41-3xspark/svc.sh（与 start.sh 同目录，依赖其 .env）
#
# 用法：
#   ./svc.sh status            只读体检（不碰服务）
#   ./svc.sh preflight         只跑启动前预检
#   ./svc.sh start [超时秒]     预检 → share → serve → 等就绪 → 三层验证（默认 1500s）
#   ./svc.sh stop              停服 → 三台逐台核对容器/显存 → 残留清理指引
#   ./svc.sh restart [超时秒]   stop + start
#   ./svc.sh logs [-f]         看引擎日志（-f 跟随）
#
# 固化的铁律：
#   1) 重启机器后必须 share 再 serve，否则 worker 的 NFS 检查会静默卡十几分钟
#   2) stop 常超时并留下 worker 容器 → 必须逐台核对
#   3) 绝不用 docker ps -q | xargs docker rm -f（引擎容器也在列表里）
#   4) 起服务前必须确认三台 GPU 空闲：上一轮残留会让这一轮假失败
#   5) 同一处反复失败 → 先重启三台再重试
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1
[ -f .env ] || { echo "找不到 .env（必须在本脚本所在目录执行）" >&2; exit 1; }
set -a; . ./.env; set +a

WANT_KERNEL=${WANT_KERNEL:-6.17.0-1031-nvidia}
WANT_DRIVER=${WANT_DRIVER:-580.173.02}
LOG_WAIT_DEFAULT=1500
PORT=${PORT:-8888}
MODEL=${SERVED_MODEL_NAME:-deepseek-v4.1-flash}
SSH="ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"

WORKERS=(); read -r -a WORKERS <<<"${WORKER_HOSTS:-}"
NODES=(localhost "${WORKERS[@]:-}")          # localhost 在前，远端逐个在后
SELF=$(hostname)

c_r=$'\033[31m'; c_g=$'\033[32m'; c_y=$'\033[33m'; c_b=$'\033[36m'; c_0=$'\033[0m'
ok()   { echo "  ${c_g}✓${c_0} $*"; }
bad()  { echo "  ${c_r}✗${c_0} $*"; FAILED=1; }
warn() { echo "  ${c_y}!${c_0} $*"; }
info() { echo "  ${c_b}·${c_0} $*"; }
hr()   { echo "───────────────────────────────────────────────────────────────"; }
die()  { echo "${c_r}[x]${c_0} $*" >&2; exit 1; }

# 本地/远端统一执行（避免依赖“自己 ssh 自己”）
node_do() { local h=$1; shift
  if [ "$h" = "localhost" ] || [ "$h" = "$SELF" ]; then bash -c "$*"; else $SSH "$h" "$*"; fi; }
newest_log() { ls -t serve-*.log 2>/dev/null | head -1; }
ib_mode() { [ "${NCCL_NET:-Socket}" = "IB" ] && echo 1 || echo 0; }

# =============================================================================
preflight() {
  hr; echo "${c_b}[1/5] 环境预检${c_0}"; FAILED=0; SERVICE_RUNNING=0

  k=$(uname -r)
  [ "$k" = "$WANT_KERNEL" ] && ok "内核 $k" \
    || bad "内核 $k ≠ $WANT_KERNEL（7.0.0-1019 有 CMA 回归 → RoCE 内存注册 ENOMEM）"
  cma=$(grep -m1 '^CmaTotal' /proc/meminfo | awk '{print $2}')
  [ "${cma:-0}" != "0" ] && ok "CmaTotal=${cma} kB" \
    || bad "CmaTotal=0 kB（CMA 回归特征）—— 先 kernel_switch2.sh 换内核，再启动"
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')
  [ "$drv" = "$WANT_DRIVER" ] && ok "驱动 $drv" || warn "驱动 $drv ≠ $WANT_DRIVER（与内核配对）"
  docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "镜像 $IMAGE" || bad "镜像 $IMAGE 不存在"

  if [ "$(ib_mode)" = 1 ]; then
    g=$(grep -c 'NCCL_IB_USE_INLINE=${NCCL_IB_USE_INLINE:-0}' start.sh 2>/dev/null || true)
    p=$(grep -c 'NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS=${NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS:-0}' start.sh 2>/dev/null || true)
    if [ "${g:-0}" -ge 1 ] && [ "${p:-0}" -ge 1 ]; then ok "start.sh 已透传 USE_INLINE=0 / PREPOST=0"
    else bad "start.sh 未补透传（python3 patch_startsh_envvar.py NCCL_IB_USE_INLINE 0 等）—— pynccl 会冻死"; fi
  else
    warn ".env 是 socket 档（NCCL_NET=Socket），吞吐约为 RoCE 档的 55%"
  fi

  for h in "${NODES[@]}"; do
    out=$(node_do "$h" 'ls -1 /etc/nvidia/ 2>/dev/null | grep -qi cx7-hotplug-enabled && echo X=1 || echo X=0; \
      echo A=$(sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l); \
      echo U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d " "); \
      echo D=$(docker ps --format {{.Names}} 2>/dev/null | grep -c dsv41- || true)')
    get() { echo "$out" | sed -n "s/^$1=//p" | head -1; }
    [ "$(get X)" = "1" ] && bad "$h: /etc/nvidia/cx7-hotplug-enabled 存在（必须移走）" || ok "$h: 无 cx7 hotplug 文件"
    if [ "$(get D)" != "0" ]; then
      warn "$h: 服务正在运行（$(get D) 个 dsv41 容器；$(get A) 个计算进程属于它）"
      SERVICE_RUNNING=1
    else
      [ "$(get A)" = "0" ] && ok "$h: 无 GPU 计算进程" \
        || bad "$h: 有 $(get A) 个 GPU 计算进程残留（会让本轮假失败 → 重启该机）"
      info "$h: GPU $(get U)"
    fi
    if [ "$(ib_mode)" = 1 ]; then
      st=$(node_do "$h" 'for d in /sys/class/infiniband/*; do cat $d/ports/1/state; done' 2>/dev/null | grep -c '4: ACTIVE')
      [ "${st:-0}" = "4" ] && ok "$h: 四个 IB 口全 ACTIVE" || bad "$h: 仅 ${st:-0}/4 个 IB 口 ACTIVE（查接线）"
    fi
  done

  [ "$FAILED" = 0 ] || die "预检未过：先修掉 ✗ 项（比启动后等 15 分钟才发现便宜得多）"
  [ "$SERVICE_RUNNING" = 1 ] && echo "  预检通过（注：服务当前已在运行）。" || echo "  预检通过。"
}

# =============================================================================
do_share() {
  hr; echo "${c_b}[2/5] share（NFS 权重导出）${c_0}"
  out=$(timeout 600 ./start.sh share 2>&1)
  n=$(echo "$out" | grep -c 'dsv41-weights has config.json')
  [ "${n:-0}" -ge 2 ] && ok "两个 worker 都看到 config.json（$n/2）" \
    || { echo "$out" | tail -6; die "share 失败（worker 卷陈旧时先在 worker 上 docker volume rm -f dsv41-weights 再 share）"; }
}

# =============================================================================
do_serve() {
  local wait_s=${1:-$LOG_WAIT_DEFAULT} log t0 el
  hr; echo "${c_b}[3/5] serve（后台启动，日志落盘）${c_0}"
  log="serve-$(date +%m%d-%H%M).log"
  (setsid nohup ./start.sh serve >"$log" 2>&1 &)
  sleep 5
  [ -f "$log" ] || die "serve 未能启动（未生成 $log）"
  ok "已后台启动：$log（预计 13–15 分钟：权重 99.4GB/节点 ≈7.5min → 图捕获 → warm-up 14s）"

  hr; echo "${c_b}[4/5] 等待就绪（上限 ${wait_s}s）${c_0}"
  t0=$(date +%s)
  while :; do
    el=$(( $(date +%s) - t0 ))
    if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo; ok "health=200（用时 ${el}s）"; LOG_FILE=$log; break
    fi
    if grep -q 'Scheduler hit an exception' "$log" 2>/dev/null; then
      echo; grep -nE 'Scheduler hit an exception|ibv_reg_mr_iova2|ncclSystemError|RuntimeError' "$log" | tail -5
      die "启动中崩溃（见上）。若同一处反复出现：先重启三台再重试"
    fi
    if [ "$el" -gt 40 ] && ! docker ps --format '{{.Names}}' | grep -q '^dsv41-head$'; then
      die "head 容器已退出；日志尾部：$(tail -3 "$log" | cut -c1-140)"
    fi
    printf '\r   等待中… %4ss（权重加载阶段日志长时间不动是正常的）' "$el"
    [ "$el" -ge "$wait_s" ] && { echo; die "超时未见 health=200；尾部：$(tail -3 "$log" | cut -c1-140)"; }
    sleep 15
  done
}

# =============================================================================
verify() {
  local log=${1:-$(newest_log)}
  hr; echo "${c_b}[5/5] 三层验证${c_0}"; FAILED=0
  local ibl=$(curl -s -m 5 "http://127.0.0.1:${PORT}/health" -w ' %{http_code}' 2>/dev/null)
  echo "$ibl" | grep -q '200' && ok "① /health = 200" || bad "① /health 异常：$ibl"

  m=$(curl -s -m 8 "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"][0];print(d["id"],d.get("max_model_len"))' 2>/dev/null)
  [ -n "$m" ] && ok "② 模型 $m" || bad "② /v1/models 无响应"

  local healthy=0 s h
  for h in "${NODES[@]}"; do
    s=$(node_do "$h" 'docker ps --format "{{.Names}} {{.Status}}" 2>/dev/null | grep dsv41 | tr "\n" " "')
    if echo "$s" | grep -q '(healthy)'; then healthy=$((healthy+1)); ok "③ $h: $s"
    else bad "③ $h 容器异常：${s:-无}"; fi
  done
  [ "$healthy" -ge 3 ] || bad "③ 仅 $healthy/3 台 healthy"

  if [ "$(ib_mode)" = 1 ]; then
    local ib rm
    ib=$(grep -c 'via NET/IB' "$log" 2>/dev/null || true); rm=$(grep -c 'ibv_reg_mr_iova2' "$log" 2>/dev/null || true)
    [ "${ib:-0}" -gt 0 ] && ok "④ 传输 RoCE：via NET/IB ${ib} 条" || bad "④ 未见 via NET/IB（是否落到 socket）"
    [ "${rm:-0}" = "0" ] && ok "④ reg_mr 失败 0 次" || bad "④ ibv_reg_mr_iova2 失败 ${rm} 次（内核 CMA！）"
  else
    ok "④ 传输 socket（按 .env）"
  fi

  local n r
  n=$(date +%s)
  r=$(curl -s -m 90 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"只回答数字：$n\"}],\"max_tokens\":24,\"temperature\":0}")
  echo "$r" | grep -q "$n" && ok "⑤ nonce 生成正确" \
    || warn "⑤ nonce 未原样出现（模型可能改写数字）：$(echo "$r" | cut -c1-140)"

  hr
  if [ "$FAILED" = 0 ]; then
    echo "${c_g}服务就绪${c_0}  API http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}   日志 $log"
    echo "        基准：python3 bench_decode.py 300   ｜   python3 bench_conc.py http://127.0.0.1:${PORT} 4 200"
  else
    echo "${c_r}验证未全通过${c_0}（见 ✗）；日志 $log"; return 1
  fi
}

# =============================================================================
cmd_status() {
  hr; echo "${c_b}服务体检${c_0}"
  echo "  /health : $(curl -s -m 5 "http://127.0.0.1:${PORT}/health" -w ' %{http_code}' 2>/dev/null || echo 无响应)"
  local log; log=$(newest_log)
  if [ -n "${log:-}" ]; then
    echo "  最新日志: $log"
    echo "  传输    : $(grep -q 'Using network IB' "$log" && echo 'RoCE/IB' || echo 'socket')  ｜ via NET/IB=$(grep -c 'via NET/IB' "$log") ｜ reg_mr 失败=$(grep -c 'ibv_reg_mr_iova2' "$log")"
  fi
  echo "  内核/驱动: $(uname -r) / $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) ｜ CmaTotal=$(grep -m1 '^CmaTotal' /proc/meminfo | awk '{print $2}')kB"
  for h in "${NODES[@]}"; do
    echo "  $h: $(node_do "$h" 'docker ps --format "{{.Names}} {{.Status}}" 2>/dev/null | grep dsv41 | tr "\n" " "; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader | tr "\n" " "')"
  done
}

cmd_stop() {
  hr; echo "${c_b}停服${c_0}"
  timeout 420 ./start.sh stop 2>&1 | tail -5 || warn "start.sh stop 超时（常见），继续逐台核对"
  sleep 5
  local left=0 h s a
  for h in "${NODES[@]}"; do
    s=$(node_do "$h" 'docker ps --format {{.Names}} 2>/dev/null | grep dsv41 | tr "\n" " "')
    a=$(node_do "$h" 'sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l')
    if [ -n "${s// /}" ]; then bad "$h 仍有容器：$s"; left=1; else ok "$h 无 dsv41-* 容器"; fi
    if [ "${a:-0}" = "0" ]; then ok "$h GPU 无计算进程"; else bad "$h 仍有 $a 个 GPU 计算进程"; left=1; fi
  done
  if [ "$left" = 1 ]; then
    echo; echo "  残留清理（${c_r}按确切名字${c_0}；绝不要 docker ps -q | xargs docker rm -f）："
    for h in "${NODES[@]}"; do echo "    $h : docker rm -f dsv41-head dsv41-worker   # 该机存在哪个删哪个"; done
    echo "  显存不释放（挂起 CUDA 上下文）→ 重启该机，然后 ./svc.sh share && ./svc.sh start"
    return 1
  fi
  ok "已全部停止干净（dsv41-nfs 仍在属正常：它就是权重共享容器）"
}

case "${1:-status}" in
  status)    cmd_status ;;
  preflight) preflight ;;
  start)
    preflight
    [ "${SERVICE_RUNNING:-0}" = 1 ] && die "服务已在运行。要重启请用 ./svc.sh restart；要先停请用 ./svc.sh stop"
    do_share; do_serve "${2:-$LOG_WAIT_DEFAULT}"; verify "$LOG_FILE" ;;
  stop)      cmd_stop ;;
  restart)   cmd_stop || true; preflight; do_share; do_serve "${2:-$LOG_WAIT_DEFAULT}"; verify "$LOG_FILE" ;;
  logs)      shift; ./start.sh logs "$@" ;;
  -h|--help|help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "未知子命令：$1（status|preflight|start|stop|restart|logs）" ;;
esac
