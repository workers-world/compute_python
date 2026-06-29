---
name: strategy-optimizer-engine
description: >-
  Strategy Optimizer Python 计算引擎（compute_python）：FastAPI 回测/寻优/WFO/scenario-table。
  在 compute_python 仓库改 engine/optimizer/wfo、对接 reconciliation 前端、或扩展策略回测能力时使用。
---

# Strategy Optimizer Python 引擎

## 仓库与协作

| 仓库 | 路径 | 分支（参考） | 角色 |
|------|------|-------------|------|
| **compute_python** | `reconciliation/compute_python/` | `dev_00_02_00` | FastAPI 回测引擎 |
| **reconciliation** | `reconciliation/reconciliation/` | `dev_00_14_00` | Java 代理 + 前端 |

**协作顺序**：Python 引擎先行 → reconciliation 更新契约/前端 → 本地验证。

| 文档 | 位置 |
|------|------|
| 接口契约 | `reconciliation/docs/strategy-optimizer-engine-contract.md` |
| 投资方法论 | `reconciliation/docs/策略优化.md` |
| 前端页面 | `reconciliation/src/main/resources/static/strategy-optimizer.html` |
| Java 代理 | `StrategyOptimizerClient` → `compute.python.url` |

## 目录结构（6 个源文件）

```
compute_python/
├── main.py       # FastAPI 端点
├── engine.py     # simulate / metrics / scenario-table（核心）
├── models.py     # Pydantic 请求/响应
├── optimizer.py  # Grid + Optuna TPE
├── wfo.py        # Walk-Forward
├── Dockerfile    # uvicorn main:app :8000
└── venv/         # 本地 Python 3.14 + numpy/optuna/joblib
```

## 策略语义

**下跌卖出 / 再跌补回**（均值回归）：

- 持有中：从入场后 **high** 更新最高点；**low** 相对最高点跌 ≥ `d` → 卖出 `s` 比例
- 已卖出：从卖出参考价 **low** 再跌 ≥ `r` → 全仓买回
- **同根 K 线**：卖出后本 Bar 禁止再买（`sold_this_bar`）

与 **资产试点积存金建议**（`/asset-pilot`）不同：本引擎是参数实验室，不可直接当实盘指令。

## 成本模型（1.1.0+）

```python
# engine.execution_sell_price / execution_buy_price
sell = ref * (1 - spread - slip) * (1 - commission)
buy  = ref * (1 + spread + slip) * (1 + commission)
# spread = bid_ask_spread, slip = slippage_bps / 10000
```

前端分别传 `commission_rate`、`bid_ask_spread`、`slippage_bps`，**不要**再把点差合并进 slippage。

## API 端点

| 方法 | 路径 | 核心函数 |
|------|------|----------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/optimize` | `optimizer.optimize` + 可选 `wfo.walk_forward` |
| POST | `/api/simulate` | `engine.simulate` + `compute_metrics` |
| POST | `/api/scenario-table` | `compute_scenario_table`（静态网格，不跑历史） |
| POST | `/api/advice` | `_build_instruction`（即时操作建议） |

Bar 字段：`date, close` 必填；`high/low/open` 可选。`main._bars_arrays()` 提取 `closes/highs/lows`。

## 关键函数与调用链

### simulate（改动最频繁）

```python
simulate(closes, highs, lows, d, r, s, initial_shares,
         commission_rate, bid_ask_spread, slippage_bps, dates)
# → (shares, cash, daily_returns, cycles, instruction, equity_curve, benchmark_curve)
```

优化器通过 `make_simulate_adapter(lows, dates, bid_ask_spread)` 包装为 `(closes, highs, d, r, s, ...) → simulate` 签名。

### optimize

- `GRID`：全组合遍历（joblib 并行）
- `TPE`：Optuna，`n_trials` 次采样
- `min_cycles > 0`：交易轮次不足 → 目标 `-inf`
- 过拟合检测：最优参数邻域 Z 矩阵陡降 >30%

### WFO

- `wfo_windows >= 2` 启用；扩展窗口训练 + 固定 chunk 测试
- 每窗计算 **train_sharpe** 与 **test_sharpe**（勿再写死 0）
- `WfoReport.oos_metrics`：avg_train/test sharpe 等汇总

## 改动检查清单

改 engine/optimizer/wfo 后：

```bash
cd compute_python
./venv/bin/python -m py_compile main.py engine.py models.py optimizer.py wfo.py
# 可选 smoke test
./venv/bin/python -c "
from engine import simulate, make_simulate_adapter
import numpy as np
c=h=l=np.linspace(100,110,30); d=[f'2025-01-{i+1:02d}' for i in range(30)]
print(simulate(c,h,l,0.1,0.1,0.5,1000,dates=d)[0])
"
```

reconciliation 侧：`mvn compile -q`；页面 `/strategy-optimizer` 验证 B&H、WFO、场景 Tab。

## 常见陷阱（已踩过）

1. **simulate 返回 7 元组** — optimizer 只取前 4 项给 metrics
2. **scenario-table 必须传成本** — 否则净增克数偏乐观
3. **WFO 需传 lows + bid_ask_spread** — 各窗 `make_simulate_adapter` 独立绑定
4. **CycleDetail 日期** — 由 `simulate(dates=...)` 写入，main 勿再覆盖
5. **正弦波样本** — 仅 close+high，缺 low 时 fallback 到 close，回测偏乐观

## 待实现（契约已预留）

- `regime_filter`（趋势过滤）
- `buy_tranches`（分批补仓）
- `POST /api/monte-carlo`
- K 线买入触发改用 high（当前买卖触发均用 low）

## 版本

- **1.1.0**：bid_ask_spread、low 路径、equity/benchmark curve、WFO train_sharpe、min_cycles
- **1.0.x**：仅 commission + slippage，close 触发

详细字段见 [reference.md](reference.md)。
