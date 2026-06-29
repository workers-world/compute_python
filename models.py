from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Objective(str, Enum):
    FINAL_SHARES = "FINAL_SHARES"
    TOTAL_ASSET = "TOTAL_ASSET"
    SHARPE_RATIO = "SHARPE_RATIO"
    CALMAR_RATIO = "CALMAR_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"


class OptimizerType(str, Enum):
    TPE = "TPE"
    GRID = "GRID"


# ── request ──────────────────────────────────────────────

class Bar(BaseModel):
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float


class ParamRange(BaseModel):
    min: float = Field(..., gt=0)
    max: float = Field(..., gt=0)
    step: float = Field(..., gt=0)


class OptimizeRequest(BaseModel):
    bars: List[Bar]
    initial_shares: float = 1000.0
    param_d: ParamRange
    param_r: ParamRange
    param_s: Optional[ParamRange] = None
    sell_ratio: float = 0.5
    commission_rate: float = 0.0001
    bid_ask_spread: float = 0.004
    slippage_bps: float = 0.0
    min_cycles: int = 0
    objective: Objective = Objective.SHARPE_RATIO
    optimizer: OptimizerType = OptimizerType.TPE
    n_trials: int = 200
    wfo_windows: int = 0
    parallel: bool = True


class ScenarioTableRequest(BaseModel):
    current_price: float = Field(..., gt=0)
    current_shares: float = Field(..., gt=0)
    avg_cost: float = Field(..., gt=0)
    highest_since: Optional[float] = None
    d_values: List[float]
    r_values: List[float]
    sell_ratio: float = 0.2
    sell_direction: str = "drop"
    max_sell_price: Optional[float] = None
    min_buy_price: Optional[float] = None
    commission_rate: float = 0.0001
    bid_ask_spread: float = 0.004
    slippage_bps: float = 0.0


class AdviceRequest(BaseModel):
    """当前持仓分析：根据实际仓位 + 最优参数 + 最新价格，给出即时操作建议"""
    current_shares: float = Field(..., gt=0)          # 当前持仓股数
    current_cash: float = 0.0                          # 当前现金
    entry_price: float = Field(..., gt=0)              # 你的入场/成本价
    highest_since_entry: Optional[float] = None        # 入场后见过的最高价，不填默认取 current_price
    status: str = "holding"                            # "holding" 持有中 | "waiting_to_buy" 已卖出
    sell_price: Optional[float] = None                 # 如已卖出，填卖出成交价
    current_price: float = Field(..., gt=0)            # 最新价格
    d: float = Field(..., gt=0, le=1)                  # 卖出触发阈值
    r: float = Field(..., gt=0, le=1)                  # 补仓再跌幅度
    s: float = Field(0.5, gt=0, le=1)                 # 卖出比例


class SimulateRequest(BaseModel):
    bars: List[Bar]
    initial_shares: float = 1000.0
    d: float = Field(..., gt=0, le=1)
    r: float = Field(..., gt=0, le=1)
    s: float = Field(0.5, gt=0, le=1)
    commission_rate: float = 0.0001
    bid_ask_spread: float = 0.004
    slippage_bps: float = 0.0


# ── response ─────────────────────────────────────────────

class CycleDetail(BaseModel):
    entry_date: Optional[date] = None
    entry_price: float
    sell_date: Optional[date] = None
    sell_price: Optional[float] = None
    buy_date: Optional[date] = None
    buy_price: Optional[float] = None
    completed: bool = False
    shares_before: float
    shares_after: Optional[float] = None
    cash_after_sell: Optional[float] = None
    return_pct: Optional[float] = None


class OptimizationMetrics(BaseModel):
    final_shares: float
    total_asset: float
    annual_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_cycles: int = 0
    insufficient_trades: bool = False


class EquityPoint(BaseModel):
    date: Optional[str] = None
    value: float


class SurfaceData(BaseModel):
    d_values: List[float]
    r_values: List[float]
    z_matrix: List[List[float]]     # d × r → objective value


class WfoWindow(BaseModel):
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    test_start: Optional[date] = None
    test_end: Optional[date] = None
    best_params: dict
    train_sharpe: float
    test_sharpe: float


class WfoReport(BaseModel):
    windows: List[WfoWindow]
    param_stability_score: float
    oos_metrics: Optional[dict] = None


class TradeInstruction(BaseModel):
    status: str = ""                        # "holding" | "waiting_to_buy" | "idle"
    status_label: str = ""                  # 中文状态描述
    current_price: Optional[float] = None
    highest_since_entry: Optional[float] = None     # 入场后最高点
    sell_trigger_price: Optional[float] = None      # 触发卖出价格
    sell_price_at: Optional[float] = None           # 上次卖出价
    buy_trigger_price: Optional[float] = None       # 触发买入价格
    position_shares: float = 0.0
    cash: float = 0.0
    total_value: Optional[float] = None
    hint: str = ""                          # 一句话操作建议


class SimulateResult(BaseModel):
    metrics: OptimizationMetrics
    cycles: List[CycleDetail]
    instruction: Optional[TradeInstruction] = None
    equity_curve: List[EquityPoint] = []
    benchmark_curve: List[EquityPoint] = []


class OptimizeResult(BaseModel):
    best_params: dict
    metrics: OptimizationMetrics
    surface: Optional[SurfaceData] = None
    wfo_report: Optional[WfoReport] = None
    overfit_warning: bool = False
    elapsed_seconds: float = 0.0
    equity_curve: List[EquityPoint] = []
    benchmark_curve: List[EquityPoint] = []
    oos_metrics: Optional[dict] = None
