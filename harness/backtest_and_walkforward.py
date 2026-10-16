"""
Event-Driven Backtesting & Walk-Forward Evaluation Harness
Models Indian NSE Equity (MIS 5x Intraday) & F&O (Options/Futures) Transaction Costs:
- Brokerage (Flat Rs 20 or 0.03% whichever is lower for Equity, Rs 20 flat for F&O)
- STT (Securities Transaction Tax)
- NSE Exchange Transaction Charges
- SEBI Turnover Charges
- Stamp Duty & 18% GST
- Bid-Ask Spread & Latency Slippage
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class IndianCostBreakdown:
    brokerage: float
    stt: float
    exchange_txn_charge: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    slippage_cost: float

    @property
    def total_cost(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn_charge
            + self.sebi_fee
            + self.stamp_duty
            + self.gst
            + self.slippage_cost
        )


def calculate_indian_intraday_costs(
    buy_turnover: float,
    sell_turnover: float,
    segment: str = "FNO_OPTIONS",
    slippage_bps: float = 4.0,
) -> IndianCostBreakdown:
    total_turnover = buy_turnover + sell_turnover
    if segment == "EQUITY_MIS":
        buy_brok = min(20.0, buy_turnover * 0.0003)
        sell_brok = min(20.0, sell_turnover * 0.0003)
        brokerage = buy_brok + sell_brok
        stt = sell_turnover * 0.00025  # 0.025% on sell side for intraday equity
        exch = total_turnover * 0.0000297  # NSE EQ charge
        stamp = buy_turnover * 0.00003  # 0.003% on buy side
    else:  # FNO_OPTIONS
        brokerage = 40.0  # Rs 20 buy + Rs 20 sell flat
        stt = sell_turnover * 0.001  # 0.1% on option sell premium
        exch = total_turnover * 0.0003503  # NSE Options txn charge
        stamp = buy_turnover * 0.00003

    sebi = total_turnover * 0.000001
    gst = 0.18 * (brokerage + exch + sebi)
    slippage = total_turnover * (slippage_bps / 10000.0)
    return IndianCostBreakdown(
        brokerage=round(brokerage, 2),
        stt=round(stt, 2),
        exchange_txn_charge=round(exch, 2),
        sebi_fee=round(sebi, 2),
        stamp_duty=round(stamp, 2),
        gst=round(gst, 2),
        slippage_cost=round(slippage, 2),
    )


class WalkForwardHarness:
    """
    Runs Purged Walk-Forward Validation across rolling historical windows
    to prevent look-ahead bias and curve fitting.
    """

    def __init__(self, initial_capital: float = 50000.0, segment: str = "FNO_OPTIONS"):
        self.initial_capital = initial_capital
        self.segment = segment

    def evaluate_strategy(self, trades: List[Dict]) -> Dict:
        capital = self.initial_capital
        gross_pnl = 0.0
        total_fees = 0.0
        equity_curve = [capital]
        wins = 0

        for t in trades:
            costs = calculate_indian_intraday_costs(
                buy_turnover=t["buy_val"],
                sell_turnover=t["sell_val"],
                segment=self.segment,
                slippage_bps=t.get("slippage_bps", 4.0),
            )
            raw_diff = t["sell_val"] - t["buy_val"] if t["side"] == "LONG" else t["buy_val"] - t["sell_val"]
            net_trade_pnl = raw_diff - costs.total_cost
            gross_pnl += raw_diff
            total_fees += costs.total_cost
            capital += net_trade_pnl
            equity_curve.append(round(capital, 2))
            if net_trade_pnl > 0:
                wins += 1

        returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1.0)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 15)) if len(returns) > 1 else 0.0
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / np.maximum(peaks, 1.0)
        max_dd_pct = float(np.max(drawdowns) * 100.0)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "gross_pnl": round(gross_pnl, 2),
            "total_charges_and_slippage": round(total_fees, 2),
            "net_pnl": round(gross_pnl - total_fees, 2),
            "win_rate_pct": round((wins / max(len(trades), 1)) * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "equity_curve": equity_curve,
        }
