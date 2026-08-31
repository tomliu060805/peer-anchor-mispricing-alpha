# -*- coding: utf-8 -*-
"""项目路径解析。

优先级:
  1) 环境变量 PROJ_ROOT
  2) 由本文件位置向上推导(src/paths.py -> 项目根)

外部数据源(行情库/逐笔库)通过环境变量覆盖, 未设置时为空字符串, 需按 .env.example 配置。
"""
import os

PROJ_ROOT = os.environ.get(
    'PROJ_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ---- 项目内目录 ----
CACHE_DIR = os.path.join(PROJ_ROOT, 'cache')
DATA_DIR = os.path.join(PROJ_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJ_ROOT, 'output')
CHART_DIR = os.path.join(PROJ_ROOT, 'charts')
CONFIG_DIR = os.path.join(PROJ_ROOT, 'config')

# ---- 外部数据源(需按自有环境覆盖) ----
STOCK_ROOT = os.environ.get('STOCK_ROOT', '')
INDEX_ROOT = os.environ.get('INDEX_ROOT', '')
TICK_ROOT = os.environ.get('TICK_ROOT', '')
BARRA_ROOT = os.environ.get('BARRA_ROOT', '')
MONEYFLOW_ROOT = os.environ.get('MONEYFLOW_ROOT', '')

# ---- 可选: 外部对照项目(仅少数诊断脚本使用, 缺失时相关脚本不可运行) ----
EXT_REGIME_ROOT = os.environ.get('EXT_REGIME_ROOT', '')      # 多空头+regime/ThemePoolTailDrop
EXT_THEME_ROOT = os.environ.get('EXT_THEME_ROOT', '')        # ThemeDetection 代码库
