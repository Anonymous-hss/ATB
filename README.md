# ATB — QuantPulse AI v5.2 (24/7 Multi-Market Capital Rotator & Glass-Box INT8 Trading Engine)

**Remote Repository:** `https://github.com/Anonymous-hss/ATB.git`

## What's Inside This Repository
1. **`index.html` / `quant_ai_trading_saas.html` — Live 24/7 Glass-Box Visual Witness Terminal (Zero Blackbox)**:
   - Connects directly to Binance's real-time 24/7 WebSocket (`wss://stream.binance.com:9443/ws/btcusdt@trade`) and streams the 24-hour Follow-the-Sun Multi-Market Capital Rotation (`NSE -> MCX -> US -> Crypto`).
   - Displays all **8 causal ATR(10)-normalized features**, the **176-byte INT8 student neural network probabilities**, the **5 pre-trade risk gates**, and a plain-English **Glass-Box Decision Stream** on every tick.
2. **`continual_learning_rotator.py` — 24-Hour Multi-Market Capital Rotator & Online INT8 Continual Learner (`v5.2`)**:
   - Fixes `BUG-07` (`Low-VIX Zombie Trade & Single-Index Capital Lockup`) via a **60-Minute (12-Bar) Velocity Time-Stop (`TIME_STOP_ROTATE_60M`)**.
   - Feeds stagnant/false-breakout bars back into the FP32 master network with hard-negative labels (`Class 2 = Hold/Chop`), runs regularized gradient descent, and re-quantizes to 99.95% clipped INT8 (`KL-divergence = 0.000007 < 0.005`).
3. **`stage1_free_broker_daemon.py` — 100% Free Indian Broker Live Paper Daemon (`₹0/Month`)**:
   - Connects to Angel One SmartAPI (`SmartWebSocketV2` + `getCandleData` + headless `pyotp` TOTP) and Upstox API v2/v3 with the **50% High-Water-Mark Profit-Sweep Vault**.
4. **`.github/workflows/atb_24_7_rotator.yml` — Automated GitHub Actions Runner**:
   - Runs `continual_learning_rotator.py` automatically on GitHub's cloud runners and commits updated audit logs & retrained INT8 weights directly to `main`.

## 1-Command Push to Your GitHub Repository
This folder is already initialized with `git init`, `branch -M main`, and `remote origin https://github.com/Anonymous-hss/ATB.git`. Just run:
```bash
./push_to_github.sh
# or: git push -u origin main
```
