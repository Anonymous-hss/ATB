"""
FastAPI Backend Entrypoint for QuantPulse AI Intraday Trading SaaS
Run locally with: uvicorn api.server:app --reload --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
from quantization.int8_quantizer import IntradayModelQuantizer
from harness.backtest_and_walkforward import WalkForwardHarness, calculate_indian_intraday_costs
from execution.paper_and_live_router import PromotionGatekeeper

app = FastAPI(title="QuantPulse AI Intraday Trading & Quantization Harness API", version="1.0.0")
quantizer = IntradayModelQuantizer()


class CostRequest(BaseModel):
    buy_turnover: float
    sell_turnover: float
    segment: str = "FNO_OPTIONS"
    slippage_bps: float = 4.0


@app.get("/health")
def health():
    return {"status": "ok", "engine": "INT8-Calibrated-Execution-Harness"}


@app.get("/quantization/benchmark")
def benchmark_quantization():
    res = quantizer.benchmark()
    return {k: v.__dict__ for k, v in res.items()}


@app.post("/harness/indian-costs")
def compute_costs(req: CostRequest):
    costs = calculate_indian_intraday_costs(
        req.buy_turnover, req.sell_turnover, req.segment, req.slippage_bps
    )
    return {**costs.__dict__, "total_cost": costs.total_cost}
