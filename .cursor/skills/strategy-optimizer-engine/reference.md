# Strategy Optimizer — 接口与模型速查

## OptimizeRequest（节选）

| 字段 | 默认 | 说明 |
|------|------|------|
| `bars` | — | `[{date, close, high?, low?, open?}]` |
| `param_d` / `param_r` | — | `{min, max, step}` |
| `sell_ratio` | 0.5 | 固定 s（param_s 为空时） |
| `commission_rate` | 0.0001 | 单边手续费 |
| `bid_ask_spread` | 0.004 | 单边买卖点差 |
| `slippage_bps` | 0 | 额外滑点 bps |
| `min_cycles` | 0 | 寻优最少完整轮次，0=不约束 |
| `objective` | SHARPE_RATIO | 见 models.Objective |
| `wfo_windows` | 0 | ≥2 启用 WFO |
| `n_trials` | 200 | TPE 次数 |

## OptimizeResult / SimulateResult（节选）

| 字段 | 说明 |
|------|------|
| `metrics` | OptimizationMetrics |
| `best_params` | `{d, r, s}` |
| `equity_curve` | `[{date, value}]` |
| `benchmark_curve` | B&H 每日市值 |
| `wfo_report` | windows + oos_metrics |
| `oos_metrics` | avg_train/test sharpe 等 |
| `overfit_warning` | bool |

## OptimizationMetrics

`final_shares`, `total_asset`, `annual_return`, `sharpe_ratio`, `calmar_ratio`, `sortino_ratio`, `max_drawdown`, `win_rate`, `profit_factor`, `total_cycles`, `insufficient_trades`

## ScenarioTableRequest（节选）

`current_price`, `current_shares`, `avg_cost`, `highest_since`, `d_values[]`, `r_values[]`, `sell_ratio`, `sell_direction` (`drop`|`rise`), `max_sell_price`, `min_buy_price`, `commission_rate`, `bid_ask_spread`, `slippage_bps`

## Docker / 配置

- 镜像 CMD：`uvicorn main:app --host 0.0.0.0 --port 8000`
- reconciliation：`compute.python.url` → `http://strategy-optimizer:8000`（docker-compose）

## reconciliation Java 路由

| Java | Python |
|------|--------|
| `POST /api/strategy-optimizer/optimize` | `/api/optimize` |
| `POST /api/strategy-optimizer/simulate` | `/api/simulate` |
| `POST /api/strategy-optimizer/scenario-table` | `/api/scenario-table` |
| `POST /api/strategy-optimizer/advice` | `/api/advice` |

Payload 经 `StrategyOptimizerX1.payload` Map 透传，字段名 snake_case 与 Python 一致。
