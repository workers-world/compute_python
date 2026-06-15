"""
交易模拟引擎 —— 支撑本模块的交易策略模拟功能。

本文件以纯函数方式实现了自定义交易策略的模拟计算：
根据历史价格序列，模拟多次买卖循环，并在给定的参数（d, r, s）下
输出最终持仓股数、总资产、收益率序列以及每一笔交易的明细。

该模块是优化搜索和回测评估的核心依赖，追求高计算效率与无副作用设计，
以便在参数扫描、WFO 等场景中被大量调用。

引入本文件是合理的，它是需求规格说明中核心策略的算法实现，
没有它就无法进行后续的优化、回测和参数分析。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from models import CycleDetail

# 一年约 252 个交易日（用于计算年化收益率）
TRADING_DAYS_PER_YEAR = 252


def simulate(
    closes: np.ndarray,
    highs: np.ndarray,
    d: float,
    r: float,
    s: float,
    initial_shares: float = 1000.0,
    commission_rate: float = 0.0003,
    slippage_bps: float = 2.0,
) -> Tuple[float, float, List[float], List[CycleDetail]]:
    """
    在给定价格序列上模拟"下跌卖出 / 再跌补回"策略。

    Parameters
    ----------
    closes : ndarray of shape (N,)
        每日收盘价。
    highs : ndarray of shape (N,)
        每日最高价，与 closes 等长。用于更新「入场后最高点」。
    d : float
        首次卖出触发阈值。从入场后最高点下跌 d 即卖出。
    r : float
        补仓再跌幅度。卖出后，从卖出价再次下跌 r 即买入。
    s : float
        卖出比例，0 < s <= 1。
    initial_shares : float
        初始持仓股数。
    commission_rate : float
        手续费率，默认万分之三（0.0003）。
    slippage_bps : float
        滑点（基点），默认 2 bps。实际成交价向不利方向偏移。

    Returns
    -------
    (final_shares, cash, daily_returns, cycles)
        - final_shares : 最终持仓股数
        - cash : 最终剩余现金
        - daily_returns : 日收益率序列（长度同 closes，无交易日为 0）
        - cycles : 各次买卖循环明细
    """
    n = len(closes)
    if n == 0:
        return initial_shares, 0.0, [], []

    shares = float(initial_shares)
    cash = 0.0
    cycles: List[CycleDetail] = []
    daily_returns = np.zeros(n)

    # 状态机: 0=持有(等待卖出), 1=已卖出(等待买入)
    state = 0
    highest = -1.0            # 入场后最高点
    entry_price = closes[0]   # 本次循环入场价
    entry_idx = 0
    sell_price = 0.0
    sell_idx = -1
    prev_total = shares * closes[0]  # 追踪日收益率用

    for i in range(n):
        high_i = highs[i] if highs[i] is not None and not math.isnan(highs[i]) else closes[i]
        close_i = closes[i]

        if state == 0:
            # 更新最高点（前视偏差保护：high_i 是当天数据，允许当天使用）
            if high_i > highest:
                highest = high_i

            # 检查卖出条件：从最高点下跌 >= d
            if highest > 0 and (highest - close_i) / highest >= d:
                # 执行卖出
                sell_qty = shares * s
                sell_price = close_i * (1 - slippage_bps / 10000)
                cash += sell_qty * sell_price * (1 - commission_rate)
                shares -= sell_qty
                state = 1
                sell_idx = i

        else:  # state == 1，已卖出，等待买入
            # 检查买入条件：从卖出价再次下跌 >= r
            if sell_price > 0 and (sell_price - close_i) / sell_price >= r:
                # 全部现金买入
                buy_price = close_i * (1 + slippage_bps / 10000)
                qty = cash * (1 - commission_rate) / buy_price
                shares += qty
                cash = 0.0

                # 记录循环
                cycles.append(CycleDetail(
                    entry_date=_idx_to_date(None, entry_idx),
                    entry_price=entry_price,
                    sell_date=_idx_to_date(None, sell_idx),
                    sell_price=sell_price,
                    buy_date=_idx_to_date(None, i),
                    buy_price=buy_price,
                    completed=True,
                    shares_before=shares - qty + sell_qty,
                    shares_after=shares,
                    cash_after_sell=cash + qty * buy_price,  # 卖出后现金（买入前）
                    return_pct=(shares * close_i + cash) / (initial_shares * closes[0]) - 1,
                ))

                # 重置状态，准备下一循环
                state = 0
                highest = close_i
                entry_price = close_i
                entry_idx = i

        # 日收益率
        total_now = shares * close_i + cash
        if prev_total > 0:
            daily_returns[i] = total_now / prev_total - 1
        prev_total = total_now

    # 若最终仍在等待买入，以收盘价结算
    if state == 1 and cycles:
        cycles[-1].buy_date = None
        cycles[-1].buy_price = None
        cycles[-1].completed = False

    # 构建当前交易指令
    final_close = closes[-1]
    total_value = shares * final_close + cash
    instruction = _build_instruction(
        state, final_close, highest, entry_price, sell_price,
        shares, cash, total_value, d, r,
    )

    return shares, cash, daily_returns.tolist(), cycles, instruction


def _build_instruction(state, current_price, highest, entry_price, sell_price,
                       shares, cash, total_value, d, r):
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
    elif state == 1:
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
    else:
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
    commission_rate: float = 0.0,
    slippage_bps: float = 0.0,
) -> list:
    """按用户图片格式，为每个 (d, r) 组合生成场景分析行。"""
    invested = round(current_shares * avg_cost, 2)
    gap_per_g = round(current_price - avg_cost, 2)
    loss_rate = round((current_price - avg_cost) / avg_cost * 100, 2)
    total_loss = round(current_shares * (current_price - avg_cost), 2)

    rows = []
    for d in d_values:
        for r in r_values:
            sell_trigger = round(highest_since * (1 - d), 2)
            buy_trigger = round(sell_trigger * (1 - r), 2) if sell_trigger > 0 else 0

            # 卖出行
            sell_qty = round(current_shares * sell_ratio, 4)
            sell_price = sell_trigger  # 触发价即卖价
            recovered = round(sell_qty * sell_price, 2)

            # 买入后结果
            remain_shares = current_shares - sell_qty
            buy_qty = round(recovered / buy_trigger, 4) if buy_trigger > 0 else 0
            buy_cash = round(buy_qty * buy_trigger, 2)
            final_shares = round(remain_shares + buy_qty, 4)
            net_gain = round(final_shares - current_shares, 4)
            final_value = round(final_shares * current_price, 2)
            remain_cash = round(recovered - buy_cash, 2)

            rows.append({
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
            })
    return rows


def compute_metrics(
    final_shares: float,
    cash: float,
    final_price: float,
    initial_shares: float,
    initial_price: float,
    daily_returns: List[float],
    cycles: List[CycleDetail],
) -> dict:
    """根据模拟结果计算各项绩效指标。"""
    dr = np.array(daily_returns)
    dr_nonzero = dr[dr != 0]

    total_asset = final_shares * final_price + cash
    initial_asset = initial_shares * initial_price

    # 年化收益
    n_days = len(daily_returns)
    if n_days > 0 and initial_asset > 0:
        annual_return = (total_asset / initial_asset) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1
    else:
        annual_return = None

    # Sharpe
    if len(dr_nonzero) > 1 and dr_nonzero.std() > 0:
        daily_rf = 0.0  # 假设无风险利率 0
        sharpe = (dr_nonzero.mean() - daily_rf) / dr_nonzero.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = None

    # Max drawdown
    if len(dr_nonzero) > 0:
        cumulative = (1 + dr).cumprod() if len(dr) > 0 else np.ones(len(dr))
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else None
    else:
        max_dd = None

    # Calmar
    calmar = annual_return / abs(max_dd) if (annual_return is not None and max_dd is not None and max_dd != 0) else None

    # Sortino
    if len(dr_nonzero) > 1:
        downside = dr_nonzero[dr_nonzero < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (dr_nonzero.mean() - 0) / downside.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino = None
    else:
        sortino = None

    # Win rate & profit factor
    completed = [c for c in cycles if c.completed and c.return_pct is not None]
    wins = sum(1 for c in completed if c.return_pct and c.return_pct > 0)
    win_rate = wins / len(completed) if completed else None

    total_gain = sum(c.return_pct for c in completed if c.return_pct and c.return_pct > 0)
    total_loss = abs(sum(c.return_pct for c in completed if c.return_pct and c.return_pct < 0))
    profit_factor = total_gain / total_loss if total_loss > 0 else (None if not completed else float("inf"))

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
        "total_cycles": len(completed),
    }
