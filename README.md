# RocketPaper

Automated Nifty 0-DTE credit spread paper trader. Runs every Tuesday on GitHub Actions — fully unattended.

---

<!-- STATS_START -->
| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |
|:------:|:--------:|:---------:|:-------:|:------:|:------:|
| 30 | 67% (20W/10L) | Rs -35,441 | Rs -1,181 | Rs 54,024 | 2W |
<!-- STATS_END -->

---

## Live Dashboard

![Paper Trading Dashboard](paper_dashboard.png)

---

## Last Trade

<!-- LAST_TRADE_START -->
| Field | Value |
|-------|-------|
| Date | 2026-10-13 (Tuesday) |
| Direction | Bullish |
| Spread | PE 22550/22350 (4-wide) |
| Credit | Rs 1,414 |
| Risk | Rs 11,586 |
| Exit | Time Exit at 15:15 |
| Nifty | 22557 -> 22702 |
| Result | **W Rs +1,273 (+11.0%)** |
<!-- LAST_TRADE_END -->

---

## Commands

Trigger via **Actions → RocketPaper → Run workflow** — results sent to Telegram.

| Command | What it does |
|---------|-------------|
| `run` | Full trade cycle (auto every Tuesday) |
| `test` | Dry run — check prices, no trade |
| `status` | Current trade status or next trade time |
| `lasttrade` | Last trade details → Telegram |
| `portfolio` | Overall P&L summary → Telegram |
| `alltrades` | All trades table → Telegram |
| `dashboard` | Regenerate dashboard PNG |
