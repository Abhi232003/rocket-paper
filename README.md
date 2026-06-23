# RocketPaper

Automated Nifty 0-DTE credit spread paper trader. Runs every Tuesday on GitHub Actions — fully unattended.

---

<!-- STATS_START -->
| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |
|:------:|:--------:|:---------:|:-------:|:------:|:------:|
| 14 | 57% (8W/6L) | Rs -27,112 | Rs -1,937 | Rs 37,345 | 3L |
<!-- STATS_END -->

---

## Live Dashboard

![Paper Trading Dashboard](paper_dashboard.png)

---

## Last Trade

<!-- LAST_TRADE_START -->
| Field | Value |
|-------|-------|
| Date | 2026-06-23 (Tuesday) |
| Direction | Bearish |
| Spread | CE 23900/23700 (4-wide) |
| Credit | Rs -12,015 |
| Risk | Rs 25,015 |
| Exit | Time Exit at 15:15 |
| Nifty | 23912 -> 23826 |
| Result | **L Rs -12,266 (-49.0%)** |
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
