"""
Unified Paper Trading Simulator & Broker Promotion Gatekeeper
Enforces pre-trade risk circuit breakers and validates promotion criteria before live API execution.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RiskLimits:
    max_daily_drawdown_pct: float = 3.0
    max_risk_per_trade_pct: float = 1.5
    max_consecutive_losses: int = 3
    auto_square_off_ist: str = "15:15"


class PromotionGatekeeper:
    """
    Verifies whether a strategy has passed all Walk-Forward Backtest & Live Paper Trading
    thresholds before unlocking real-money Broker API order routing.
    """

    @staticmethod
    def evaluate_readiness(metrics: Dict) -> Dict:
        checks = [
            {
                "gate": "Walk-Forward Out-of-Sample Sharpe >= 1.50",
                "passed": metrics.get("sharpe_ratio", 0.0) >= 1.50,
                "actual": f"{metrics.get('sharpe_ratio', 0.0):.2f}",
            },
            {
                "gate": "Net Profit Positive After STT, Brokerage & Slippage",
                "passed": metrics.get("net_pnl", 0.0) > 0.0,
                "actual": f"Rs {metrics.get('net_pnl', 0.0):,.2f}",
            },
            {
                "gate": "Max Intraday Drawdown <= 6.0%",
                "passed": metrics.get("max_drawdown_pct", 100.0) <= 6.0,
                "actual": f"{metrics.get('max_drawdown_pct', 0.0):.2f}%",
            },
            {
                "gate": "INT8 Quantization KL-Divergence Drift < 0.005",
                "passed": metrics.get("kl_drift", 1.0) < 0.005,
                "actual": f"{metrics.get('kl_drift', 0.0):.5f}",
            },
            {
                "gate": "Paper Trading Sample >= 30 Multi-Regime Sessions",
                "passed": metrics.get("paper_sessions", 0) >= 30,
                "actual": f"{metrics.get('paper_sessions', 0)} sessions",
            },
        ]
        all_passed = all(c["passed"] for c in checks)
        return {"ready_for_live_capital": all_passed, "gates": checks}
