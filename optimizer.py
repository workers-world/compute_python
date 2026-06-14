"""
策略参数优化模块 —— 对策略模拟进行参数空间搜索，找出最优参数组合。

本模块实现了两种最常用的优化器：
1. 网格搜索 (Grid search) —— 遍历所有参数组合，计算确定性的最优解
2. TPE (Tree‑structured Parzen Estimator) —— 使用 Optuna 库进行贝叶斯优化
   与 Optuna 的集成是合理的，因为 TPE 是业界目前公认的 SOTA 贝叶斯优化器，
   其智能采样能力使得在相同计算预算下可以探索更大的参数空间，
   避免了手动实现复杂优化算法的必要。

两种优化器通过统一的 Python 函数接口暴露给 FastAPI 服务层，
并且支持多线程 / 多进程并行执行 trial，满足性能需求。
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from models import Objective, ParamRange

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def _build_grid(param: ParamRange) -> list:
    values = []
    v = param.min
    while v <= param.max + 1e-10:
        values.append(round(v, 6))
        v += param.step
    return values


def _objective_value(metrics: dict, objective: Objective) -> float:
    """将优化目标映射为标量 (最大化方向)。"""
    key_map = {
        Objective.FINAL_SHARES: "final_shares",
        Objective.TOTAL_ASSET: "total_asset",
        Objective.SHARPE_RATIO: "sharpe_ratio",
        Objective.CALMAR_RATIO: "calmar_ratio",
        Objective.SORTINO_RATIO: "sortino_ratio",
        Objective.MAX_DRAWDOWN: "max_drawdown",
    }
    key = key_map[objective]
    val = metrics.get(key)
    if val is None:
        return float("-inf")

    if objective == Objective.MAX_DRAWDOWN:
        return -abs(val)  # 回撤越小越好
    return float(val)


def grid_search(
    simulate_fn: Callable,
    closes: np.ndarray,
    highs: np.ndarray,
    param_d: ParamRange,
    param_r: ParamRange,
    param_s: ParamRange | None,
    sell_ratio: float,
    initial_shares: float,
    commission_rate: float,
    slippage_bps: float,
    objective: Objective,
    parallel: bool = False,
) -> tuple[dict, dict, list, list, list]:
    """
    网格搜索优化。返回 (best_params, best_metrics, d_vals, r_vals, z)。

    z 是一个二维 list，z[i][j] = objective_value(d_vals[i], r_vals[j])。
    用于前端绘制热力图 / 参数曲面。
    """
    d_values = _build_grid(param_d)
    r_values = _build_grid(param_r)
    if param_s is not None:
        s_values = _build_grid(param_s)
    else:
        s_values = [sell_ratio]

    best_obj = float("-inf")
    best_params = {}
    best_metrics = {}
    z = [[0.0] * len(r_values) for _ in range(len(d_values))]

    # Flatten all combinations
    combos = [(di, ri, si) for di in d_values for ri in r_values for si in s_values]

    def _eval_one(di, ri, si):
        final_shares, cash, daily_returns, cycles, _ = simulate_fn(
            closes, highs, di, ri, si, initial_shares, commission_rate, slippage_bps
        )
        from engine import compute_metrics  # circular-safe
        metrics = compute_metrics(
            final_shares, cash, closes[-1], initial_shares, closes[0],
            daily_returns, cycles,
        )
        obj_val = _objective_value(metrics, objective)
        return obj_val, metrics, {"d": di, "r": ri, "s": si}

    if parallel and len(combos) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(_eval_one)(di, ri, si) for di, ri, si in combos
        )
    else:
        results = [_eval_one(di, ri, si) for di, ri, si in combos]

    # Fill Z matrix and track best
    idx = 0
    for i, di in enumerate(d_values):
        for j, rj in enumerate(r_values):
            # Take first s value for surface (or use the best across s)
            si = s_values[0]
            obj_val, _, _ = results[idx]
            z[i][j] = obj_val
            idx += len(s_values)

    for obj_val, metrics, params in results:
        if obj_val > best_obj or (math.isnan(obj_val) and math.isnan(best_obj)):
            best_obj = obj_val
            best_params = params
            best_metrics = metrics

    return best_params, best_metrics, d_values, r_values, z


def tpe_search(
    simulate_fn: Callable,
    closes: np.ndarray,
    highs: np.ndarray,
    param_d: ParamRange,
    param_r: ParamRange,
    param_s: ParamRange | None,
    sell_ratio: float,
    initial_shares: float,
    commission_rate: float,
    slippage_bps: float,
    objective: Objective,
    n_trials: int = 200,
    parallel: bool = False,
) -> tuple[dict, dict, list, list, list]:
    """TPE 贝叶斯优化。返回同 grid_search。"""
    if not HAS_OPTUNA:
        raise RuntimeError("optuna 未安装，请 pip install optuna")

    def _objective(trial):
        di = trial.suggest_float("d", param_d.min, param_d.max)
        ri = trial.suggest_float("r", param_r.min, param_r.max)
        if param_s is not None:
            si = trial.suggest_float("s", param_s.min, param_s.max)
        else:
            si = sell_ratio

        final_shares, cash, daily_returns, cycles, _ = simulate_fn(
            closes, highs, di, ri, si, initial_shares, commission_rate, slippage_bps
        )
        from engine import compute_metrics
        metrics = compute_metrics(
            final_shares, cash, closes[-1], initial_shares, closes[0],
            daily_returns, cycles,
        )
        return _objective_value(metrics, objective)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=-1 if parallel else 1)

    # Run a final eval with best params to get full metrics
    bp = study.best_params
    di, ri = bp["d"], bp["r"]
    si = bp.get("s", sell_ratio)
    final_shares, cash, daily_returns, cycles, _ = simulate_fn(
        closes, highs, di, ri, si, initial_shares, commission_rate, slippage_bps
    )
    from engine import compute_metrics
    best_metrics = compute_metrics(
        final_shares, cash, closes[-1], initial_shares, closes[0],
        daily_returns, cycles,
    )

    # 生成插值后的参数曲面（用于前端热力图）
    d_values = _build_grid(param_d)
    r_values = _build_grid(param_r)
    z = [[0.0] * len(r_values) for _ in range(len(d_values))]
    best_obj = _objective_value(best_metrics, objective)

    # 用 TPE 的采样器对网格点做估值（基于已采样的 trial 插值）
    if study.trials:
        all_trials = [(t.params["d"], t.params.get("r", 0), t.params.get("s", sell_ratio), t.value)
                      for t in study.trials if t.value is not None]
        for i, dv in enumerate(d_values):
            for j, rv in enumerate(r_values):
                # Nearest-neighbor interpolation from sampled trials
                best_dist = float("inf")
                best_val = float("-inf")
                for td, tr, ts, tv in all_trials:
                    dist = (td - dv) ** 2 + (tr - rv) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_val = tv
                z[i][j] = best_val if best_dist < 0.01 else float("nan")

    return (
        {"d": di, "r": ri, "s": si},
        best_metrics,
        d_values, r_values, z,
    )


def optimize(
    simulate_fn, closes, highs, param_d, param_r, param_s, sell_ratio,
    initial_shares, commission_rate, slippage_bps, objective,
    optimizer_type, n_trials=200, parallel=False,
):
    """统一入口，根据 optimizer_type 分发到 grid_search 或 tpe_search。"""
    t0 = time.perf_counter()

    if optimizer_type == "GRID":
        best, metrics, dv, rv, z = grid_search(
            simulate_fn, closes, highs, param_d, param_r, param_s,
            sell_ratio, initial_shares, commission_rate, slippage_bps,
            objective, parallel,
        )
    else:
        best, metrics, dv, rv, z = tpe_search(
            simulate_fn, closes, highs, param_d, param_r, param_s,
            sell_ratio, initial_shares, commission_rate, slippage_bps,
            objective, n_trials, parallel,
        )

    elapsed = time.perf_counter() - t0

    # Overfit detection: 检查最优参数是否附近目标值陡降
    overfit_warning = False
    if z:
        n_rows, n_cols = len(z), len(z[0])
        peak_val = float("-inf")
        peak_i = peak_j = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if z[i][j] > peak_val:
                    peak_val = z[i][j]
                    peak_i, peak_j = i, j
        # 检查周边 4 邻域目标值下降幅度
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = peak_i + di, peak_j + dj
            if 0 <= ni < n_rows and 0 <= nj < n_cols and not math.isnan(z[ni][nj]):
                neighbors.append(z[ni][nj])
        if neighbors:
            avg_neighbor = sum(neighbors) / len(neighbors)
            if peak_val > 0 and (peak_val - avg_neighbor) / peak_val > 0.3:
                overfit_warning = True

    return best, metrics, dv, rv, z, elapsed, overfit_warning
