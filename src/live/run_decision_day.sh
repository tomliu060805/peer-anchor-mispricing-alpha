#!/usr/bin/env bash
# 决策日自动运行入口。
#
# 逻辑:
#   1. 判断今天是否为决策日(自锚定日起每 5 个交易日)。非决策日直接退出。
#   2. 数据新鲜度检查: 源数据必须已更新到今天, 否则等待/退出(不用陈旧数据下单)。
#   3. 增量跑流水线, 生成当期持仓与订单。
#   4. 结果写 output/live/, 并追加 append-only 日志。
#
# 建议 crontab (交易日 16:30, 收盘后数据落地):
#   30 16 * * 1-5  /path/to/src/live/run_decision_day.sh >> /path/to/output/live/cron.log 2>&1
#
# 环境变量: 见 .env.example。必须提供 STOCK_ROOT / INDEX_ROOT / TICK_ROOT。

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$(dirname "$HERE")")"
cd "$PROJ"

[ -f .env ] && set -a && . ./.env && set +a
PY="${PYTHON_BIN:-python}"
LIVE="$PROJ/output/live"
mkdir -p "$LIVE"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- 互斥锁 ----------
# 缓存是全局共享状态: 一个进程重建 npz 的同时另一个进程读它, 会读到写了一半的
# 文件 (zipfile.BadZipFile)。研究期与本次生产化都因此白跑过。靠纪律不够, 上锁。
LOCKFILE="$PROJ/output/live/.pipeline.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "另一个流水线任务正在运行(锁 $LOCKFILE), 本次退出"
  exit 0
fi

# ---------- 1. 决策日判断 ----------
IS_DAY=$("$PY" - <<'EOF'
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.environ.get('PROJ', '.'), 'src'))
from paths import CACHE_DIR, STOCK_ROOT
import datetime as dt
# 用源数据的交易日历判断(不依赖缓存, 因为缓存可能还没更新)
files = sorted(f[:10] for f in os.listdir(f'{STOCK_ROOT}/price/price_daily') if f.endswith('.parquet'))
today = dt.date.today().isoformat()
if files[-1] != today:
    print(f'NOT_READY {files[-1]}')
    raise SystemExit
anchor = os.environ.get('ANCHOR_DATE', '')
if not anchor:
    print('DECISION')          # 未设锚定日: 首次运行即为锚定
    raise SystemExit
if anchor not in files:
    print('DECISION')
    raise SystemExit
gap = files.index(today) - files.index(anchor)
print('DECISION' if gap % 5 == 0 else f'SKIP gap={gap}')
EOF
) || true

log "决策日判断: $IS_DAY"
case "$IS_DAY" in
  NOT_READY*) log "源数据尚未更新到今日(${IS_DAY#NOT_READY }), 退出"; exit 0 ;;
  SKIP*)      log "非决策日, 退出"; exit 0 ;;
  DECISION)   log "今日为决策日, 开始运行" ;;
  *)          log "判断异常, 退出"; exit 1 ;;
esac

# ---------- 2. 流水线 ----------
log "S1-S5 增量构建..."
"$PY" src/live/pipeline.py --stage all

log "一致性检查..."
if ! "$PY" src/live/pipeline.py --check; then
  log "缓存一致性未通过, 中止(不在不一致的数据上下单)"
  exit 1
fi

# ---------- 3. 组合 ----------
log "生成当期持仓..."
"$PY" src/live/paper_ledger.py

TODAY=$(date '+%F')
log "完成: output/live/portfolio_${TODAY}.csv / orders_${TODAY}.csv"

# ---------- 4. 执行提示 ----------
cat <<TIP

执行提示 (路径B, 项目默认):
  T+1 上午 11:30 前  按 orders_${TODAY}.csv 中 SELL 腿卖出
  T+1 13:30-14:30    按 BUY 腿买入
  禁止开盘集中下单 (val 段实测开盘买入比午后贵 13-17bp/笔)

TIP
