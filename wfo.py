"""
Walk-Forward 优化模块 — 将历史数据切分为多个训练/测试窗口，
在每个窗口上独立优化参数，然后用最优参数在后续样本外区间上评估，
最后汇总各窗口的参数稳定性。
"""

from __future__ import annotations

import numpy as np

from models import Objective, ParamRange, WfoReport, WfoWindow


def walk_forward(
    simulate_fn,
    closes: np.ndarray,
    highs: np.ndarray,
    dates: list,
    param_d: ParamRange,
    param_r: ParamRange,
    param_s: ParamRange | None,
    sell_ratio: float,
    initial_shares: float,
    commission_rate: float,
    slippage_bps: float,
    objective: Objective,
    n_windows: int = 4,
    n_trials: int = 100,
    parallel: bool = False,
) -> WfoReport:
    if n_windows < 2:
        n_windows = 2

    n = len(closes)
    chunk = n // n_windows
    if chunk < 20:
        raise ValueError("数据不足，每窗口至少需要 20 条记录")

    windows: list[WfoWindow] = []
    all_best_params = []

    for w in range(1, n_windows):
        split = w * chunk
        train_closes = closes[:split]
        train_highs = highs[:split]
        test_closes = closes[split:split + chunk]
        test_highs = highs[split:split + chunk]
        train_dates = dates[:split]
        test_dates = dates[split:split + chunk]

        from optimizer import tpe_search
        best, _, _, _, _ = tpe_search(
            simulate_fn, train_closes, train_highs,
            param_d, param_r, param_s,
            sell_ratio, initial_shares, commission_rate, slippage_bps,
            objective, n_trials, parallel,
        )
        all_best_params.append(best)

        final_shares, cash, daily_returns, _, _2 = simulate_fn(
            test_closes, test_highs,
            best["d"], best["r"], best.get("s", sell_ratio),
            initial_shares, commission_rate, slippage_bps,
        )
        from engine import compute_metrics
        test_metrics = compute_metrics(
            final_shares, cash, test_closes[-1], initial_shares, test_closes[0],
            daily_returns, [],
        )
        test_sharpe = test_metrics.get("sharpe_ratio") or 0.0

        windows.append(WfoWindow(
            train_start=train_dates[0] if train_dates else None,
            train_end=train_dates[-1] if train_dates else None,
            test_start=test_dates[0] if test_dates else None,
            test_end=test_dates[-1] if test_dates else None,
            best_params=best,
            train_sharpe=0.0,
            test_sharpe=test_sharpe,
        ))

    if len(all_best_params) >= 2:
        d_vals = [p["d"] for p in all_best_params]
        r_vals = [p["r"] for p in all_best_params]
        d_range = param_d.max - param_d.min
        r_range = param_r.max - param_r.min
        d_cv = np.std(d_vals) / d_range if d_range > 0 else 0
        r_cv = np.std(r_vals) / r_range if r_range > 0 else 0
        stability = max(0.0, 1.0 - (d_cv + r_cv) / 2)
    else:
        stability = 1.0

    return WfoReport(
        windows=windows,
        param_stability_score=round(stability, 4),
    )
