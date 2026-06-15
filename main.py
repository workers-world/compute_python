"""策略参数优化计算服务 — FastAPI HTTP API"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from engine import _build_instruction, compute_metrics, compute_scenario_table, simulate
from models import (
    AdviceRequest,
    Objective,
    OptimizationMetrics,
    OptimizeRequest,
    OptimizeResult,
    ScenarioTableRequest,
    SimulateRequest,
    SimulateResult,
    SurfaceData,
    TradeInstruction,
)
from optimizer import optimize
from wfo import walk_forward

logging.basicConfig(
    filename="strategy_optimizer.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("strategy-optimizer started")
    yield
    logger.info("strategy-optimizer stopped")


app = FastAPI(title="Strategy Optimizer", version="1.0.0", lifespan=lifespan)

# ── 健康检查 ──

@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── 即时操作建议：根据实际持仓 + 参数，告诉用户现在该做什么 ──

@app.post("/api/advice")
async def advice_endpoint(req: AdviceRequest):
    highest = req.highest_since_entry or max(req.current_price, req.entry_price)
    if req.status == "holding":
        instruction = _build_instruction(
            0, req.current_price, highest, req.entry_price, None,
            req.current_shares, req.current_cash,
            req.current_shares * req.current_price + req.current_cash,
            req.d, req.r,
        )
    else:
        sp = req.sell_price or req.entry_price
        instruction = _build_instruction(
            1, req.current_price, 0, 0, sp,
            req.current_shares, req.current_cash,
            req.current_shares * req.current_price + req.current_cash,
            req.d, req.r,
        )
    return {"instruction": instruction}


# ── 场景分析表格：每个 (d, r) 组合的卖出/买入数值明细 ──

@app.post("/api/scenario-table")
async def scenario_table_endpoint(req: ScenarioTableRequest):
    highest = req.highest_since or max(req.current_price, req.avg_cost)
    rows = compute_scenario_table(
        req.current_price, req.current_shares, req.avg_cost, highest,
        req.d_values, req.r_values, req.sell_ratio, req.sell_direction,
    )
    return {"rows": rows}


# ── 单参数模拟：用指定 d/r/s 在完整历史上回放策略 ──

@app.post("/api/simulate", response_model=SimulateResult)
async def simulate_endpoint(req: SimulateRequest):
    closes = np.array([b.close for b in req.bars], dtype=float)
    highs = np.array([b.high or b.close for b in req.bars], dtype=float)

    final_shares, cash, daily_returns, cycles, instruction = simulate(
        closes, highs, req.d, req.r, req.s,
        req.initial_shares, req.commission_rate, req.slippage_bps,
    )

    # Fill cycle dates from bars
    dates = [b.date for b in req.bars]
    for idx, c in enumerate(cycles):
        c.entry_date = dates[0]
        c.sell_date = None
        c.buy_date = None

    metrics = compute_metrics(
        final_shares, cash, closes[-1], req.initial_shares, closes[0],
        daily_returns, cycles,
    )
    return SimulateResult(
        metrics=OptimizationMetrics(**metrics),
        cycles=cycles,
        instruction=TradeInstruction(**instruction) if instruction else None,
    )


# ── 参数优化：TPE 贝叶斯搜索或网格遍历，找最优 d/r ──

@app.post("/api/optimize", response_model=OptimizeResult)
async def optimize_endpoint(req: OptimizeRequest):
    if len(req.bars) < 20:
        raise HTTPException(400, "至少需要 20 条价格数据")

    closes = np.array([b.close for b in req.bars], dtype=float)
    highs = np.array([b.high or b.close for b in req.bars], dtype=float)
    dates = [b.date for b in req.bars]

    best, metrics, dv, rv, z, elapsed, overfit = optimize(
        simulate, closes, highs,
        req.param_d, req.param_r, req.param_s, req.sell_ratio,
        req.initial_shares, req.commission_rate, req.slippage_bps,
        req.objective, req.optimizer.value, req.n_trials, req.parallel,
    )

    result = OptimizeResult(
        best_params=best,
        metrics=OptimizationMetrics(**metrics),
        surface=SurfaceData(d_values=dv, r_values=rv, z_matrix=z) if dv and rv else None,
        overfit_warning=overfit,
        elapsed_seconds=round(elapsed, 3),
    )

    # WFO
    if req.wfo_windows >= 2:
        try:
            result.wfo_report = walk_forward(
                simulate, closes, highs, dates,
                req.param_d, req.param_r, req.param_s, req.sell_ratio,
                req.initial_shares, req.commission_rate, req.slippage_bps,
                req.objective, req.wfo_windows, req.n_trials, req.parallel,
            )
        except Exception as e:
            logger.warning("WFO failed: %s", e)
            result.wfo_report = None

    logger.info(
        "optimize done best=%s sharpe=%s elapsed=%.3fs",
        best, metrics.get("sharpe_ratio"), elapsed,
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
