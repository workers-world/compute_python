from __future__ import annotations

import numpy as np

from models import Objective, ParamRange, WfoReport, WfoWindow


def _avg_metric(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def walk_forward(
    simulate_fn,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    dates: list,
    param_d: ParamRange,
    param_r: ParamRange,
    param_s: ParamRange | None,
    sell_ratio: float,
    initial_shares: float,
    commission_rate: float,
    bid_ask_spread: float,
    slippage_bps: float,
    objective: Objective,
    n_windows: int = 4,
    n_trials: int = 100,
    parallel: bool = False,
    min_cycles: int = 0,
) -> WfoReport:
    if n_windows < 2:
        n_windows = 2

    n = len(closes)
    chunk = n // n_windows
    if chunk < 20:
        raise ValueError("数据不足，每窗口至少需要 20 条记录")

    windows: list[WfoWindow] = []
    all_best_params = []
    train_sharpes = []
    test_sharpes = []
    test_annual_returns = []

    for w in range(1, n_windows):
        split = w * chunk
        train_closes = closes[:split]
        train_highs = highs[:split]
        train_lows = lows[:split]
        test_closes = closes[split : split + chunk]
        test_highs = highs[split : split + chunk]
        test_lows = lows[split : split + chunk]
        train_dates = dates[:split]
        test_dates = dates[split : split + chunk]

        from engine import make_simulate_adapter
        from optimizer import tpe_search

        train_sim = make_simulate_adapter(train_lows, train_dates, bid_ask_spread)
        test_sim = make_simulate_adapter(test_lows, test_dates, bid_ask_spread)

        best, _, _, _, _ = tpe_search(
            train_sim,
            train_closes,
            train_highs,
            param_d,
            param_r,
            param_s,
            sell_ratio,
            initial_shares,
            commission_rate,
            slippage_bps,
            objective,
            n_trials,
            parallel,
            min_cycles,
        )
        all_best_params.append(best)

        from engine import compute_metrics

        train_result = train_sim(
            train_closes,
            train_highs,
            best["d"],
            best["r"],
            best.get("s", sell_ratio),
            initial_shares,
            commission_rate,
            slippage_bps,
        )
        train_metrics = compute_metrics(
            train_result[0],
            train_result[1],
            train_closes[-1],
            initial_shares,
            train_closes[0],
            train_result[2],
            train_result[3],
            min_cycles,
        )
        train_sharpe = train_metrics.get("sharpe_ratio") or 0.0

        test_result = test_sim(
            test_closes,
            test_highs,
            best["d"],
            best["r"],
            best.get("s", sell_ratio),
            initial_shares,
            commission_rate,
            slippage_bps,
        )
        test_metrics = compute_metrics(
            test_result[0],
            test_result[1],
            test_closes[-1],
            initial_shares,
            test_closes[0],
            test_result[2],
            test_result[3],
            min_cycles,
        )
        test_sharpe = test_metrics.get("sharpe_ratio") or 0.0
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
        test_annual_returns.append(test_metrics.get("annual_return"))

        windows.append(
            WfoWindow(
                train_start=train_dates[0] if train_dates else None,
                train_end=train_dates[-1] if train_dates else None,
                test_start=test_dates[0] if test_dates else None,
                test_end=test_dates[-1] if test_dates else None,
                best_params=best,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
            )
        )

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

    oos_metrics = {
        "avg_train_sharpe": _avg_metric(train_sharpes),
        "avg_test_sharpe": _avg_metric(test_sharpes),
        "avg_test_annual_return": _avg_metric(test_annual_returns),
        "windows": len(windows),
    }

    return WfoReport(
        windows=windows,
        param_stability_score=round(stability, 4),
        oos_metrics=oos_metrics,
    )
