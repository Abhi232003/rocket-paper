# RocketPaper

Automated Nifty 0-DTE credit spread paper trader. Runs every Tuesday on GitHub Actions — fully unattended.

---

<!-- STATS_START -->
| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |
|:------:|:--------:|:---------:|:-------:|:------:|:------:|
| 15 | 53% (8W/7L) | Rs -31,094 | Rs -2,073 | Rs 41,327 | 4L |
<!-- STATS_END -->

---

## Live Dashboard

![Paper Trading Dashboard](paper_dashboard.png)

---

## Last Trade

<!-- LAST_TRADE_START -->
| Field | Value |
|-------|-------|
| Date | 2026-06-30 (Tuesday) |
| Direction | Bullish |
| Spread | PE 23950/23750 (4-wide) |
| Credit | Rs 1,554 |
| Risk | Rs 11,446 |
| Exit | Time Exit at 15:15 |
| Nifty | 23942 -> 23866 |
| Result | **L Rs -3,982 (-34.8%)** |
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
