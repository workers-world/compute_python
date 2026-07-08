"""策略参数优化计算服务 — FastAPI HTTP API"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from engine import (
    _build_instruction,
    compute_metrics,
    compute_scenario_table,
    make_simulate_adapter,
    simulate,
)
from models import (
    AdviceRequest,
    EquityPoint,
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


app = FastAPI(title="Strategy Optimizer", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    print(
        f"[DOCKER] {request.method} {request.url.path} → {response.status_code} ({elapsed:.3f}s)",
        flush=True,
    )
    logger.info(
        "%s %s → %s (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


def _bars_arrays(bars):
    closes = np.array([b.close for b in bars], dtype=float)
    highs = np.array([b.high or b.close for b in bars], dtype=float)
    lows = np.array([b.low or b.close for b in bars], dtype=float)
    dates = [b.date for b in bars]
    return closes, highs, lows, dates


def _curve_points(raw_curve):
    return [EquityPoint(**p) for p in raw_curve]


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/advice")
async def advice_endpoint(req: AdviceRequest):
    highest = req.highest_since_entry or max(req.current_price, req.entry_price)
    if req.status == "holding":
        instruction = _build_instruction(
            0,
            req.current_price,
            highest,
            req.entry_price,
            None,
            req.current_shares,
            req.current_cash,
            req.current_shares * req.current_price + req.current_cash,
            req.d,
            req.r,
        )
    else:
        sp = req.sell_price or req.entry_price
        instruction = _build_instruction(
            1,
            req.current_price,
            0,
            0,
            sp,
            req.current_shares,
            req.current_cash,
            req.current_shares * req.current_price + req.current_cash,
            req.d,
            req.r,
        )
    return {"instruction": instruction}


@app.post("/api/scenario-table")
async def scenario_table_endpoint(req: ScenarioTableRequest):
    highest = req.highest_since or max(req.current_price, req.avg_cost)
    rows = compute_scenario_table(
        req.current_price,
        req.current_shares,
        req.avg_cost,
        highest,
        req.d_values,
        req.r_values,
        req.sell_ratio,
        req.sell_direction,
        req.max_sell_price,
        req.min_buy_price,
        req.commission_rate,
        req.bid_ask_spread,
        req.slippage_bps,
    )
    return {"rows": rows}


@app.post("/api/simulate", response_model=SimulateResult)
async def simulate_endpoint(req: SimulateRequest):
    closes, highs, lows, dates = _bars_arrays(req.bars)

    (
        final_shares,
        cash,
        daily_returns,
        cycles,
        instruction,
        equity_curve,
        benchmark_curve,
    ) = simulate(
        closes,
        highs,
        lows,
        req.d,
        req.r,
        req.s,
        req.initial_shares,
        req.commission_rate,
        req.bid_ask_spread,
        req.slippage_bps,
        dates,
    )

    metrics = compute_metrics(
        final_shares,
        cash,
        closes[-1],
        req.initial_shares,
        closes[0],
        daily_returns,
        cycles,
    )
    return SimulateResult(
        metrics=OptimizationMetrics(**metrics),
        cycles=cycles,
        instruction=TradeInstruction(**instruction) if instruction else None,
        equity_curve=_curve_points(equity_curve),
        benchmark_curve=_curve_points(benchmark_curve),
    )


@app.post("/api/optimize", response_model=OptimizeResult)
async def optimize_endpoint(req: OptimizeRequest):
    if len(req.bars) < 20:
        raise HTTPException(400, "至少需要 20 条价格数据")

    closes, highs, lows, dates = _bars_arrays(req.bars)
    simulate_fn = make_simulate_adapter(lows, dates, req.bid_ask_spread)

    best, metrics, dv, rv, z, elapsed, overfit = optimize(
        simulate_fn,
        closes,
        highs,
        req.param_d,
        req.param_r,
        req.param_s,
        req.sell_ratio,
        req.initial_shares,
        req.commission_rate,
        req.slippage_bps,
        req.objective,
        req.optimizer.value,
        req.n_trials,
        req.parallel,
        req.min_cycles,
    )

    (
        final_shares,
        cash,
        daily_returns,
        _cycles,
        _instruction,
        equity_curve,
        benchmark_curve,
    ) = simulate(
        closes,
        highs,
        lows,
        best["d"],
        best["r"],
        best.get("s", req.sell_ratio),
        req.initial_shares,
        req.commission_rate,
        req.bid_ask_spread,
        req.slippage_bps,
        dates,
    )
    metrics = compute_metrics(
        final_shares,
        cash,
        closes[-1],
        req.initial_shares,
        closes[0],
        daily_returns,
        _cycles,
        req.min_cycles,
    )

    result = OptimizeResult(
        best_params=best,
        metrics=OptimizationMetrics(**metrics),
        surface=SurfaceData(d_values=dv, r_values=rv, z_matrix=z) if dv and rv else None,
        overfit_warning=overfit,
        elapsed_seconds=round(elapsed, 3),
        equity_curve=_curve_points(equity_curve),
        benchmark_curve=_curve_points(benchmark_curve),
    )

    if req.wfo_windows >= 2:
        try:
            result.wfo_report = walk_forward(
                simulate_fn,
                closes,
                highs,
                lows,
                dates,
                req.param_d,
                req.param_r,
                req.param_s,
                req.sell_ratio,
                req.initial_shares,
                req.commission_rate,
                req.bid_ask_spread,
                req.slippage_bps,
                req.objective,
                req.wfo_windows,
                req.n_trials,
                req.parallel,
                req.min_cycles,
            )
            if result.wfo_report and result.wfo_report.oos_metrics:
                result.oos_metrics = result.wfo_report.oos_metrics
        except Exception as e:
            logger.warning("WFO failed: %s", e)
            result.wfo_report = None

    logger.info(
        "optimize done best=%s sharpe=%s elapsed=%.3fs",
        best,
        metrics.get("sharpe_ratio"),
        elapsed,
    )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
