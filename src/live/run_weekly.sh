#!/usr/bin/env bash
# 每周固定时间跑一次: 增量建数据 -> 出 Top-50 与目标持仓 -> 发邮件。
#
# 默认周日 20:00 运行(信号取上周五收盘数据), 周一执行。
# crontab 见文件末尾注释。
#
# 与 run_decision_day.sh 的区别: 那个按"每5个交易日"的网格判断是否为决策日,
# 适合严格贴合回测调仓节奏; 本脚本按自然周固定运行, 更好记也更好对账。
# 两者不要同时挂——调仓节奏只能有一个。

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$(dirname "$HERE")")"
cd "$PROJ"
[ -f .env ] && set -a && . ./.env && set +a
PY="${PYTHON_BIN:-python}"
mkdir -p output/live
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

fail() { log "失败: $*"; 
  if [ -n "${SMTP_USER:-}" ]; then
    "$PY" - <<PYEOF 2>/dev/null || true
import os,sys; sys.path.insert(0,'$PROJ/src/live'); sys.path.insert(0,'$PROJ/src')
from send_report import send
send('[联动锚定策略] 周任务失败', '''$*

定时任务未产出本周信号, 请登录服务器查看 output/live/weekly_cron.log。''')
PYEOF
  fi
  exit 1; }

log "===== 周任务开始 ====="

# 1. 源数据新鲜度: 必须有本周最后一个交易日的数据
LATEST=$(ls "${STOCK_ROOT}/price/price_daily" | grep -oP '^\d{4}-\d{2}-\d{2}' | tail -1)
AGE=$(( ( $(date +%s) - $(date -d "$LATEST" +%s) ) / 86400 ))
log "源数据最新交易日: $LATEST (距今 ${AGE} 天)"
[ "$AGE" -gt 5 ] && fail "源数据已 ${AGE} 天未更新(最新 $LATEST), 拒绝在陈旧数据上出信号"

# 2. 增量建数据
log "流水线 S1-S5..."
"$PY" src/live/pipeline.py --stage all || fail "流水线报错"

log "一致性检查..."
"$PY" src/live/pipeline.py --check || fail "缓存一致性未通过, 不在不一致的数据上出信号"

# 3. 出信号
log "生成目标持仓与账本..."
"$PY" src/live/paper_ledger.py || fail "持仓生成报错"
log "生成 Top-50..."
"$PY" src/live/weekly_top50.py --top 50 || fail "Top-50 生成报错"

# 4. 发邮件(缺凭证只警告, 文件已落地不算失败)
if [ -n "${SMTP_USER:-}" ] && [ -n "${MAIL_TO:-}" ]; then
  log "发送邮件..."
  "$PY" src/live/send_report.py --top 50 || log "警告: 邮件发送失败, 报告已存于 output/live/"
else
  log "未配置 SMTP_USER/MAIL_TO, 跳过发信; 报告在 output/live/"
fi

log "===== 周任务完成 ====="

# ---------------------------------------------------------------------------
# crontab -e 加入(周日 20:00):
#   0 20 * * 0  /path/to/src/live/run_weekly.sh >> /path/to/output/live/weekly_cron.log 2>&1
# 执行: 周一 11:30 前卖出腿, 13:30-14:30 买入腿(路径B)
# ---------------------------------------------------------------------------
