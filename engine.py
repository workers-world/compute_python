from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from models import CycleDetail

TRADING_DAYS_PER_YEAR = 252


def _slippage_ratio(slippage_bps: float) -> float:
    return slippage_bps / 10000.0


def execution_sell_price(
    ref: float,
    commission_rate: float,
    bid_ask_spread: float,
    slippage_bps: float,
) -> float:
    slip = _slippage_ratio(slippage_bps)
    return ref * (1 - bid_ask_spread - slip) * (1 - commission_rate)


def execution_buy_price(
    ref: float,
    commission_rate: float,
    bid_ask_spread: float,
    slippage_bps: float,
) -> float:
    slip = _slippage_ratio(slippage_bps)
    return ref * (1 + bid_ask_spread + slip) * (1 + commission_rate)


def make_simulate_adapter(
    lows: np.ndarray,
    dates: Optional[List],
    bid_ask_spread: float,
):
    def adapter(
        closes: np.ndarray,
        highs: np.ndarray,
        d: float,
        r: float,
        s: float,
        initial_shares: float = 1000.0,
        commission_rate: float = 0.0001,
        slippage_bps: float = 0.0,
    ):
        return simulate(
            closes,
            highs,
            lows,
            d,
            r,
            s,
            initial_shares,
            commission_rate,
            bid_ask_spread,
            slippage_bps,
            dates,
        )

    return adapter


def simulate(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    d: float,
    r: float,
    s: float,
    initial_shares: float = 1000.0,
    commission_rate: float = 0.0001,
    bid_ask_spread: float = 0.004,
    slippage_bps: float = 0.0,
    dates: Optional[List] = None,
) -> Tuple[float, float, List[float], List[CycleDetail], dict, List[dict], List[dict]]:
    n = len(closes)
    if n == 0:
        return initial_shares, 0.0, [], [], {}, [], []

    shares = float(initial_shares)
    cash = 0.0
    cycles: List[CycleDetail] = []
    daily_returns = np.zeros(n)
    equity_curve: List[dict] = []

    state = 0
    highest = -1.0
    entry_price = closes[0]
    entry_idx = 0
    sell_ref_price = 0.0
    sell_idx = -1
    shares_at_sell = 0.0
    cash_after_sell = 0.0
    prev_total = shares * closes[0]

    for i in range(n):
        sold_this_bar = False
        close_i = closes[i]
        high_i = highs[i] if highs[i] is not None and not math.isnan(highs[i]) else close_i
        low_i = lows[i] if lows[i] is not None and not math.isnan(lows[i]) else close_i

        if state == 0:
            if high_i > highest:
                highest = high_i

            if highest > 0 and (highest - low_i) / highest >= d:
                sell_qty = shares * s
                shares_at_sell = shares
                exec_sell = execution_sell_price(
                    low_i, commission_rate, bid_ask_spread, slippage_bps
                )
                cash += sell_qty * exec_sell
                cash_after_sell = cash
                shares -= sell_qty
                sell_ref_price = low_i
                state = 1
                sell_idx = i
                sold_this_bar = True

        elif not sold_this_bar:
            if sell_ref_price > 0 and (sell_ref_price - low_i) / sell_ref_price >= r:
                exec_buy = execution_buy_price(
                    low_i, commission_rate, bid_ask_spread, slippage_bps
                )
                qty = cash / exec_buy if exec_buy > 0 else 0.0
                shares_before_cycle = shares_at_sell
                entry_asset = shares_before_cycle * entry_price
                shares += qty
                cash = 0.0
                cycle_asset = shares * close_i
                return_pct = (
                    (cycle_asset - entry_asset) / entry_asset if entry_asset > 0 else None
                )

                cycles.append(
                    CycleDetail(
                        entry_date=_idx_to_date(dates, entry_idx),
                        entry_price=entry_price,
                        sell_date=_idx_to_date(dates, sell_idx),
                        sell_price=execution_sell_price(
                            sell_ref_price,
                            commission_rate,
                            bid_ask_spread,
                            slippage_bps,
                        ),
                        buy_date=_idx_to_date(dates, i),
                        buy_price=exec_buy,
                        completed=True,
                        shares_before=shares_before_cycle,
                        shares_after=shares,
                        cash_after_sell=cash_after_sell,
                        return_pct=return_pct,
                    )
                )

                state = 0
                highest = close_i
                entry_price = close_i
                entry_idx = i
                sell_ref_price = 0.0
                sell_idx = -1

        total_now = shares * close_i + cash
        equity_curve.append(
            {
                "date": _idx_to_date(dates, i),
                "value": round(total_now, 2),
            }
        )
        if prev_total > 0:
            daily_returns[i] = total_now / prev_total - 1
        prev_total = total_now

    if state == 1:
        cycles.append(
            CycleDetail(
                entry_date=_idx_to_date(dates, entry_idx),
                entry_price=entry_price,
                sell_date=_idx_to_date(dates, sell_idx),
                sell_price=execution_sell_price(
                    sell_ref_price,
                    commission_rate,
                    bid_ask_spread,
                    slippage_bps,
                ) if sell_ref_price > 0 else None,
                buy_date=None,
                buy_price=None,
                completed=False,
                shares_before=shares_at_sell,
                shares_after=shares,
                cash_after_sell=cash_after_sell,
                return_pct=None,
            )
        )

    final_close = closes[-1]
    total_value = shares * final_close + cash
    instruction = _build_instruction(
        state,
        final_close,
        highest,
        entry_price,
        sell_ref_price,
        shares,
        cash,
        total_value,
        d,
        r,
    )
    benchmark_curve = compute_benchmark_curve(
        closes, dates, initial_shares, commission_rate, bid_ask_spread, slippage_bps
    )

    return (
        shares,
        cash,
        daily_returns.tolist(),
        cycles,
        instruction,
        equity_curve,
        benchmark_curve,
    )


def compute_benchmark_curve(
    closes: np.ndarray,
    dates: Optional[List],
    initial_shares: float,
    commission_rate: float = 0.0,
    bid_ask_spread: float = 0.0,
    slippage_bps: float = 0.0,
) -> List[dict]:
    if len(closes) == 0:
        return []
    entry_cost = execution_buy_price(
        closes[0], commission_rate, bid_ask_spread, slippage_bps
    )
    shares = initial_shares
    curve = []
    for i, close_i in enumerate(closes):
        value = shares * close_i
        curve.append(
            {
                "date": _idx_to_date(dates, i),
                "value": round(value, 2),
            }
        )
    if curve:
        curve[0]["value"] = round(shares * entry_cost, 2)
    return curve


def _build_instruction(
    state,
    current_price,
    highest,
    entry_price,
    sell_price,
    shares,
    cash,
    total_value,
    d,
    r,
):
    if state == 0:
        sell_trigger = highest * (1 - d) if highest > 0 else 0
        drop_pct = (highest - current_price) / highest * 100 if highest > 0 else 0
        if drop_pct >= 0:
            drop_text = f"已回落 {drop_pct:.1f}%"
        else:
            drop_text = f"较最高点上涨 {abs(drop_pct):.1f}%"
        return {
            "status": "holding",
            "status_label": "持有中",
            "current_price": round(current_price, 2),
            "highest_since_entry": round(highest, 2),
            "sell_trigger_price": round(sell_trigger, 2),
            "position_shares": round(shares, 4),
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "hint": f"{drop_text}，触发卖出需跌至≤{sell_trigger:.2f}（从最高点跌{d*100:.0f}%）",
        }
    if state == 1:
        buy_trigger = sell_price * (1 - r) if sell_price > 0 else 0
        drop_pct = (sell_price - current_price) / sell_price * 100 if sell_price > 0 else 0
        if drop_pct >= 0:
            drop_text = f"已回落 {drop_pct:.1f}%"
        else:
            drop_text = f"较卖出价上涨 {abs(drop_pct):.1f}%（已高于卖出价）"
        return {
            "status": "waiting_to_buy",
            "status_label": "等待补仓",
            "current_price": round(current_price, 2),
            "sell_price_at": round(sell_price, 2),
            "buy_trigger_price": round(buy_trigger, 2),
            "position_shares": round(shares, 4),
            "cash": round(cash, 2),
            "total_value": round(total_value, 2),
            "hint": f"卖出价 {sell_price:.2f}，{drop_text}，触发买入需跌至≤{buy_trigger:.2f}（从卖出价再跌{r*100:.0f}%）",
        }
    return {
        "status": "idle",
        "status_label": "无持仓",
        "current_price": round(current_price, 2),
        "position_shares": 0,
        "cash": 0,
        "total_value": 0,
        "hint": "",
    }


def _idx_to_date(dates: Optional[List], idx: int) -> Optional[str]:
    if dates is not None and 0 <= idx < len(dates):
        return str(dates[idx])
    return None


def compute_scenario_table(
    current_price: float,
    current_shares: float,
    avg_cost: float,
    highest_since: float,
    d_values: list,
    r_values: list,
    sell_ratio: float = 0.2,
    sell_direction: str = "drop",
    max_sell_price: float | None = None,
    min_buy_price: float | None = None,
    commission_rate: float = 0.0001,
    bid_ask_spread: float = 0.004,
    slippage_bps: float = 0.0,
) -> list:
    invested = round(current_shares * avg_cost, 2)
    gap_per_g = round(current_price - avg_cost, 2)
    loss_rate = round((current_price - avg_cost) / avg_cost * 100, 2)
    total_loss = round(current_shares * (current_price - avg_cost), 2)

    rows = []
    for d in d_values:
        for r in r_values:
            if sell_direction == "rise":
                sell_trigger = round(current_price * (1 + d), 2)
            else:
                sell_trigger = round(highest_since * (1 - d), 2)
            buy_trigger = round(sell_trigger * (1 - r), 2) if sell_trigger > 0 else 0
            if max_sell_price and sell_trigger > max_sell_price:
                continue
            if min_buy_price and buy_trigger < min_buy_price:
                continue

            sell_qty = round(current_shares * sell_ratio, 4)
            sell_fill = execution_sell_price(
                sell_trigger, commission_rate, bid_ask_spread, slippage_bps
            )
            recovered = round(sell_qty * sell_fill, 2)
            remain_shares = current_shares - sell_qty
            buy_fill = execution_buy_price(
                buy_trigger, commission_rate, bid_ask_spread, slippage_bps
            )
            buy_qty = round(recovered / buy_fill, 4) if buy_fill > 0 else 0
            buy_cash = round(buy_qty * buy_fill, 2)
            final_shares = round(remain_shares + buy_qty, 4)
            net_gain = round(final_shares - current_shares, 4)
            final_value = round(final_shares * current_price, 2)
            remain_cash = round(recovered - buy_cash, 2)

            rows.append(
                {
                    "d": round(d, 4),
                    "r": round(r, 4),
                    "current_shares": current_shares,
                    "avg_cost": avg_cost,
                    "invested": invested,
                    "current_price": current_price,
                    "gap_per_g": gap_per_g,
                    "loss_rate": loss_rate,
                    "total_loss": total_loss,
                    "sell_ratio": sell_ratio,
                    "sell_trigger": sell_trigger,
                    "sell_qty": sell_qty,
                    "recovered_cash": recovered,
                    "remain_shares": remain_shares,
                    "buy_trigger": buy_trigger,
                    "buy_ratio": 1.0,
                    "buy_qty": buy_qty,
                    "buy_cash": buy_cash,
                    "final_shares": final_shares,
                    "net_gain": net_gain,
                    "final_value": final_value,
                    "remain_cash": remain_cash,
                }
            )
    return rows


def compute_metrics(
    final_shares: float,
    cash: float,
    final_price: float,
    initial_shares: float,
    initial_price: float,
    daily_returns: List[float],
    cycles: List[CycleDetail],
    min_cycles: int = 0,
) -> dict:
    dr = np.array(daily_returns)
    dr_nonzero = dr[dr != 0]

    total_asset = final_shares * final_price + cash
    initial_asset = initial_shares * initial_price

    n_days = len(daily_returns)
    if n_days > 0 and initial_asset > 0:
        annual_return = (total_asset / initial_asset) ** (
            TRADING_DAYS_PER_YEAR / max(n_days, 1)
        ) - 1
    else:
        annual_return = None

    if len(dr_nonzero) > 1 and dr_nonzero.std() > 0:
        sharpe = dr_nonzero.mean() / dr_nonzero.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = None

    if len(dr_nonzero) > 0:
        cumulative = (1 + dr).cumprod() if len(dr) > 0 else np.ones(len(dr))
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else None
    else:
        max_dd = None

    calmar = (
        annual_return / abs(max_dd)
        if (annual_return is not None and max_dd is not None and max_dd != 0)
        else None
    )

    if len(dr_nonzero) > 1:
        downside = dr_nonzero[dr_nonzero < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = dr_nonzero.mean() / downside.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino = None
    else:
        sortino = None

    completed = [c for c in cycles if c.completed and c.return_pct is not None]
    wins = sum(1 for c in completed if c.return_pct and c.return_pct > 0)
    win_rate = wins / len(completed) if completed else None

    total_gain = sum(c.return_pct for c in completed if c.return_pct and c.return_pct > 0)
    total_loss_val = abs(
        sum(c.return_pct for c in completed if c.return_pct and c.return_pct < 0)
    )
    profit_factor = (
        total_gain / total_loss_val
        if total_loss_val > 0
        else (None if not completed else float("inf"))
    )

    total_cycles = len(completed)

    return {
        "final_shares": round(final_shares, 4),
        "total_asset": round(total_asset, 2),
        "annual_return": round(annual_return, 6) if annual_return is not None else None,
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6) if max_dd is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "total_cycles": total_cycles,
        "insufficient_trades": min_cycles > 0 and total_cycles < min_cycles,
    }
