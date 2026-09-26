#!/usr/bin/env bash
# =============================================================================
# probe_indexer_variant.sh — 判定本集群引擎（da64c5cbb）的 dense prefill indexer
#                            属于哪个变体，从而决定能否移植 indexer_chunked(v1/v3)
#
# 性质：**只读**。只做 import + inspect.getsource 的字符串判断，不 patch、不写文件、
#       不加载模型权重、不改服务状态。可在服务运行中执行，也可在服务停止后执行。
#
# 背景（证据出处均为上游 MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks）：
#   adapter/indexer_chunked.py:16-19     目标镜像 e087e662 —— candidate masks 在
#                                        self.candidate_masks（list），helper 名 _mask_topk_scores
#   adapter/indexer_chunked_v3.py:9-10   目标分支 f80c91a4b+ —— masks 在
#                                        forward_metadata.candidate_metadata（CandidateMasks）
#   本集群引擎是 da64c5cbb（见 docs/ADAPTER-MIGRATION-PLAN.md:17），**两者都不是**，
#   所以必须先探测，不能猜。
#
# 用法（在 head 上，与 start.sh/.env 同目录）：
#   bash probe_indexer_variant.sh              # 探本机 head 容器
#   HEAD_CTN=dsv41-head bash probe_indexer_variant.sh
#   bash probe_indexer_variant.sh --dry-run    # 只看将要执行什么，不真跑
#
# 退出码：0 = V1/V3/NATIVE（有明确可行结论）｜2 = 容器不在/不可用｜3 = import 失败
#         4 = NEITHER/REFUSE（不可移植）｜5 = AMBIGUOUS（需人工）｜6 = 未判定
# =============================================================================
set -u

HEAD_CTN="${HEAD_CTN:-dsv41-head}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

c_r=$'\033[31m'; c_g=$'\033[32m'; c_y=$'\033[33m'; c_b=$'\033[36m'; c_0=$'\033[0m'
hr() { echo "───────────────────────────────────────────────────────────────"; }

hr
echo "${c_b}probe_indexer_variant.sh${c_0}  容器=${HEAD_CTN}  模式=$([ $DRY = 1 ] && echo dry-run || echo live)"
hr

# ---------------------------------------------------------------------------
# 探测脚本本体。写成 heredoc 是为了避免多层引号转义把判据改坏。
# 判据直接抄自两个 adapter 的 install() 守卫，保证"能装上"与"探测通过"等价。
# ---------------------------------------------------------------------------
read -r -d '' PROBE <<'PYEOF'
import inspect, json, sys, traceback

OUT = {}
def rec(k, v):
    OUT[k] = v
    return v          # 返回写入值，便于在判据里直接用

# --- 1) 目标模块与类是否存在 ------------------------------------------------
try:
    from sglang.srt.layers.attention import deepseek_v4_backend as B
except Exception:
    traceback.print_exc()
    print("PROBE_IMPORT_FAILED")
    sys.exit(3)

rec("module_file", getattr(B, "__file__", None))
cls = getattr(B, "DeepseekV4AttnBackend", None)
rec("has_backend_class", cls is not None)

# 引擎自带 build 指纹（尽力而为，拿不到就是 None）
for modname, attr in (("sglang", "__version__"), ("sglang.srt", "__version__")):
    try:
        m = __import__(modname, fromlist=["x"])
        rec("version:%s" % modname, getattr(m, attr, None))
    except Exception:
        rec("version:%s" % modname, None)
try:
    import importlib.metadata as md
    rec("dist:sglang", md.version("sglang"))
except Exception:
    rec("dist:sglang", None)

# --- 2) 关键方法源码（两个变体都靠它判别） ----------------------------------
src = None
if cls is not None and hasattr(cls, "_low_ratio_index_topk_dense"):
    try:
        src = inspect.getsource(cls._low_ratio_index_topk_dense)
    except Exception as e:
        rec("getsource_error", repr(e))
rec("has_low_ratio_index_topk_dense",
    cls is not None and hasattr(cls, "_low_ratio_index_topk_dense"))

if src is not None:
    rec("inspect:PyCF_ONLY_AST" if False else "uses_self_candidate_masks", "self.candidate_masks" in src)
    rec("uses_candidate_metadata", "candidate_metadata" in src)
    rec("uses_publish_or_consume", "_publish_or_consume_candidates" in src)
    # v3 的"已经原生带 #39187"判据：命中则无需移植
    rec("already_has_39187_budget", "_DENSE_INDEXER_LOGITS_BUDGET_BYTES" in src)
    rec("source_len", len(src))

# --- 3) v1 所需符号 ---------------------------------------------------------
NEEDED_V1 = ("_dense_fp4_mqa_logits", "topk_transform_ragged_v2", "_mask_topk_scores",
             "select_candidate_blocks", "ceil_align", "_as_int_list")
rec("v1_missing", [n for n in NEEDED_V1 if not hasattr(B, n)])

# --- 4) v3 所需符号 ---------------------------------------------------------
NEEDED_V3 = ("_dense_fp4_mqa_logits", "topk_transform_ragged_v2", "mask_topk_scores",
             "select_candidate_blocks", "ceil_align", "_as_int_list", "CandidateMasks",
             "published_masks", "get_parallel")
rec("v3_missing", [n for n in NEEDED_V3 if not hasattr(B, n)])

# --- 5) v3 额外依赖：fp4_indexer 里的量化函数 -------------------------------
try:
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
    rec("has_quantize_fp4_indexer_tensor", True)
except Exception as e:
    rec("has_quantize_fp4_indexer_tensor", False)
    rec("quantize_import_error", repr(e))

# --- 6) 复刻两个 adapter 的 install() 守卫，给出"装上会怎样" ------------------
verdict = "UNKNOWN"
reason = ""

if cls is None or src is None:
    verdict, reason = "REFUSE", "DeepseekV4AttnBackend 或其 _low_ratio_index_topk_dense 不存在"
else:
    v1_missing = OUT["v1_missing"]
    v3_missing = OUT["v3_missing"]
    native = OUT["already_has_39187_budget"]
    # 与 adapter 守卫逐字对应（indexer_chunked.py:217-238 / indexer_chunked_v3.py:352-360）
    v1_guard = (not v1_missing) and ("self.candidate_masks" in src
                                     or "_publish_or_consume_candidates" in src)
    v3_guard = (not v3_missing) and ("candidate_metadata" in src
                                     and "self.candidate_masks" not in src)
    rec("v1_guard_passes", v1_guard)
    rec("v3_guard_passes", v3_guard)
    if native:
        verdict, reason = "NATIVE", "stock 已原生携带 #39187（v3 会自行退出，不需要移植）"
    elif v1_guard and not v3_guard:
        verdict, reason = "V1", "命中 v1 守卫（self.candidate_masks 变体）；v3 会 refuse"
    elif v3_guard and not v1_guard:
        verdict, reason = "V3", "命中 v3 守卫（candidate_metadata 变体）；v1 会 refuse"
    elif v1_guard and v3_guard:
        verdict, reason = "AMBIGUOUS", "两个守卫都过：异常情况，需人工读源码确认"
    else:
        verdict, reason = "NEITHER", "两个 adapter 的守卫都不满足 —— 该函数已被上游改写，不可直接移植"

rec("verdict", verdict)
rec("verdict_reason", reason)

# 机器可读部分强制 ASCII：容器/宿主 locale 非 UTF-8 时中文 json 会解码失败。
# 中文理由单独走 stderr，供人阅读。
print("===PROBE_JSON_BEGIN===")
print(json.dumps(OUT, indent=2, ensure_ascii=True, sort_keys=True))
print("===PROBE_JSON_END===")
sys.stderr.write("verdict_reason(zh): %s\n" % reason)
PYEOF

# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
if [ $DRY = 1 ]; then
  echo "将执行："
  echo "  docker exec -i $HEAD_CTN python3 - <<'PY'  （探针 ${#PROBE} 字节，只读）"
  echo "  # 若容器不在，回退：docker run --rm --entrypoint python3 <IMAGE> -"
  hr; exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "${c_r}docker 不可用${c_0}" >&2; exit 2; }

if docker ps --format '{{.Names}}' | grep -qx "$HEAD_CTN"; then
  echo "· 容器在跑，直接 exec（不影响服务）"
  RAW=$(printf '%s' "$PROBE" | docker exec -i "$HEAD_CTN" python3 - 2>&1); RC=$?
elif docker ps -a --format '{{.Names}}' | grep -qx "$HEAD_CTN"; then
  echo "${c_y}! 容器存在但未运行；服务需处于运行状态才能 exec。${c_0}"
  echo "  启动服务后再跑：./svc.sh start" ; hr; exit 2
else
  echo "${c_y}! 找不到容器 $HEAD_CTN。${c_0}"
  echo "  服务在跑时容器名应为 dsv41-head（见 fleet/start.sh:90）。"
  echo "  若服务未起，可用镜像直接跑：" 
  echo "    docker run --rm -i --entrypoint python3 <IMAGE> - < probe 正文"
  hr; exit 2
fi

echo "$RAW" | sed -n '/===PROBE_JSON_BEGIN===/,/===PROBE_JSON_END===/p' | sed '1d;$d'

# 提取 verdict 行。用 awk 而非 grep -o：
#   在 head（fq-dgx-01, GNU grep 3.11）上实测，任何含双引号的 grep -o/-oE 模式都返回 0 命中
#   （'"verdict": *"[A-Z]*"'、'"verdict":[[:space:]]*"[A-Z]+"' 均失败），而同一份输入
#   awk -F'"' 与 grep+cut 都正确。此差异 2026-09-26 在本机复现，故改用 awk（不依赖引号转义）。
#   注意必须匹配整行 "^  \"verdict\":"，否则会命中 "verdict_reason"。
V=$(printf '%s\n' "$RAW" | awk -F'"' '/^[[:space:]]*"verdict":/ {print $4; exit}')

hr
case "${V:-}" in
  V1)   echo "${c_g}结论：V1${c_0}  → 可移植 indexer_chunked.py（v1）" ;;
  V3)   echo "${c_g}结论：V3${c_0}  → 可移植 indexer_chunked_v3.py（注意它附带 SG18 prefill TP split 依赖）" ;;
  NATIVE) echo "${c_g}结论：NATIVE${c_0} → 引擎已原生带 #39187，**无需移植**" ;;
  NEITHER) echo "${c_r}结论：NEITHER${c_0} → 两个 adapter 都不能直接移植，停手上报" ;;
  AMBIGUOUS) echo "${c_y}结论：AMBIGUOUS${c_0} → 需人工读源码确认" ;;
  *)    echo "${c_r}结论：未能判定${c_0}（rc=$RC）—— 原始输出尾部：" ; echo "$RAW" | tail -15 ;;
esac
hr
echo "提醒：本脚本只判定「能否装上」，不判定「装上后是否正确」。"
echo "      真实长上下文（~200k）压测与 qeval 质量门仍需另行执行。"

# 给出机器可用的退出码，便于上游自动化
case "${V:-}" in
  V1|V3|NATIVE) exit 0 ;;      # 有明确可行结论
  NEITHER|REFUSE) exit 4 ;;    # 不可移植 / 结构不符
  AMBIGUOUS) exit 5 ;;         # 需人工判断
  *) exit 6 ;;                 # 未判定
esac