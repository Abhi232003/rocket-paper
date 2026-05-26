# RocketPaper

Automated Nifty 0-DTE credit spread paper trader. Runs every Tuesday on GitHub Actions — fully unattended.

---

<!-- STATS_START -->
| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |
|:------:|:--------:|:---------:|:-------:|:------:|:------:|
| 10 | 70% (7W/3L) | Rs -14,438 | Rs -1,444 | Rs 24,671 | 2L |
<!-- STATS_END -->

---

## Live Dashboard

![Paper Trading Dashboard](paper_dashboard.png)

---

## Last Trade

<!-- LAST_TRADE_START -->
| Field | Value |
|-------|-------|
| Date | 2026-05-26 (Tuesday) |
| Direction | Bearish |
| Spread | CE 23900/23700 (4-wide) |
| Credit | Rs -11,931 |
| Risk | Rs 24,931 |
| Exit | Time Exit at 15:15 |
| Nifty | 23905 -> 23911 |
| Result | **L Rs -12,178 (-48.9%)** |
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
