#!/usr/bin/env bash
# 提交前净化检查: 扫描所有受版本控制的文件是否含有不应公开的内容。
#
# 起因: 项目公开时人工清理过一轮, 但后续新增内容又把不该公开的字样带了回来。
# 靠每次记得去查不可靠, 所以做成机械化闸门。
#
# 规则表放在 .sanitize_rules (已 gitignore)。规则里必然包含要禁的字样本身,
# 写进脚本会让检查器命中自己; 那份清单本身也不适合出现在公开仓库里。
# 因此脚本本身不含任何模式, 缺规则表时直接拒绝放行。
#
# 用法:  bash sanitize_check.sh
# 装成 hook: ln -sf ../../sanitize_check.sh .git/hooks/pre-commit
# 退出码 1 = 有命中, 不应提交。

set -uo pipefail
# 必须切到仓库根: 作为 .git/hooks/pre-commit 运行时 BASH_SOURCE 指向 .git/hooks/,
# 在那里 git ls-files 列不出任何文件——检查会静默全过, 等于没装。
cd "$(git rev-parse --show-toplevel)" || exit 1

RULES=.sanitize_rules
if [ ! -f "$RULES" ]; then
  echo "缺少规则表 $RULES —— 无法检查, 拒绝放行。"
  echo "格式: 每行一条  类别名称|正则1|正则2|..."
  exit 1
fi

fail=0
n=$(git ls-files | wc -l)
echo "净化检查 (扫描 $n 个受版本控制的文件)"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  name="${line%%|*}"
  pat="${line#*|}"
  hits=$(git ls-files | while read -r f; do
    case "$f" in *.png|*.jpg|*.jpeg|*.npz|*.npy|*.pkl|*.parquet|*.pdf) continue;; esac
    [ "$f" = "$RULES" ] && continue
    grep -HnE "$pat" "$f" 2>/dev/null
  done)
  if [ -n "$hits" ]; then
    echo "✗ $name"
    echo "$hits" | sed 's/^/    /'
    fail=1
  else
    echo "✓ $name"
  fi
done < "$RULES"

if [ $fail -ne 0 ]; then
  echo
  echo "有命中——请先清理再提交。"
  exit 1
fi
echo
echo "全部通过。"
