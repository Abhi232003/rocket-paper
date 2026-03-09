# -*- coding: utf-8 -*-
"""
RocketPaper — Nifty 0-DTE Credit Spread Paper Trader
=====================================================
Runs on GitHub Actions every Tuesday. Single long-running job:
  1. 13:35 IST: Enter trade (sell credit spread, direction from DAX)
  2. 13:36–15:15 IST: Monitor every 5 min, exit on TP/SL
  3. 15:15 IST: Time-exit if neither TP/SL hit
  4. Commit results back to repo

Strategy matches RocketData backtest exactly:
  - Direction: DAX opening gap direction (no threshold)
  - Spread: 4-wide (200pt), ATM
  - TP: 20% of max risk → exit immediately
  - SL: 80% of max risk → exit immediately
  - Lot size: 65

Usage:
  python paper_trader.py run          # Full run: entry → monitor → exit
  python paper_trader.py test         # Dry run: check prices, no trade
  python paper_trader.py summary      # Print trade summary
  python paper_trader.py dashboard    # Regenerate dashboard chart
"""

import os
import sys
import json
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import pyotp
import yfinance as yf
from SmartApi import SmartConnect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
API_KEY      = os.environ.get("ANGEL_API_KEY", "")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
MPIN         = os.environ.get("ANGEL_MPIN", "")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")

# Strategy — matches backtest exactly
WIDTH          = 4             # number of strikes
SPREAD_WIDTH   = WIDTH * 50    # 200pt
TP_PCT         = 20.0          # exit at +20% of max risk (gross, before costs)
SL_PCT         = 80.0          # exit at -80% of max risk (gross, before costs)
STRIKE_STEP    = 50
LOT_SIZE       = 65

ENTRY_TIME     = "13:35"
EXIT_TIME      = "15:15"
MONITOR_INTERVAL = 60          # seconds between price checks (1 min, matches backtest)

EXPIRY_WEEKDAY = 1             # Tuesday

IST = ZoneInfo("Asia/Kolkata")

# Paths
BASE_DIR       = Path(__file__).parent
TRADES_CSV     = BASE_DIR / "paper_trades.csv"
CHECKS_CSV     = BASE_DIR / "paper_checks.csv"
DASHBOARD_PNG  = BASE_DIR / "paper_dashboard.png"

# Costs (same as backtest)
BROKERAGE_PER_SIDE = 20.0
STT_PCT            = 0.05 / 100
GST_PCT            = 0.18
SLIPPAGE_PCT       = 0.005

# Telegram notifications
TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("rocket_paper")


# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════
def notify(message: str):
    """Send a Telegram push notification. Silently fails if not configured."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TG_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log.warning(f"Telegram notification failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  TIME
# ═══════════════════════════════════════════════════════════════════
def now_ist() -> datetime:
    return datetime.now(IST)

def today_ist() -> date:
    return now_ist().date()

def is_expiry_day(d: date = None) -> bool:
    if d is None:
        d = today_ist()
    return d.weekday() == EXPIRY_WEEKDAY

def get_expiry_string(d: date = None) -> str:
    if d is None:
        d = today_ist()
    return d.strftime("%d%b%Y").upper()

def wait_until_ist(hour: int, minute: int):
    """Wait until the specified IST time."""
    while True:
        now = now_ist()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        log.info(f"Waiting until {hour:02d}:{minute:02d} IST... ({remaining:.0f}s remaining)")
        time.sleep(min(remaining, 60))


# ═══════════════════════════════════════════════════════════════════
#  ANGEL ONE API
# ═══════════════════════════════════════════════════════════════════
def angel_login() -> SmartConnect:
    if not API_KEY or not CLIENT_ID:
        raise RuntimeError("Angel One credentials not set. "
                           "Set ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_MPIN, ANGEL_TOTP_SECRET.")
    obj = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    data = obj.generateSession(CLIENT_ID, MPIN, totp)
    if not data or data.get("message") != "SUCCESS":
        raise RuntimeError(f"Angel One login failed: {data}")
    log.info(f"Angel One login OK (Client: {CLIENT_ID})")
    return obj


def get_nifty_ltp(obj: SmartConnect) -> Optional[float]:
    try:
        data = obj.ltpData("NSE", "Nifty 50", "99926000")
        if data and data.get("status"):
            return float(data["data"]["ltp"])
    except Exception as e:
        log.error(f"Error fetching Nifty LTP: {e}")
    return None


def get_option_greeks(obj: SmartConnect, expiry_str: str) -> Optional[pd.DataFrame]:
    try:
        data = obj.optionGreek({"name": "NIFTY", "expirydate": expiry_str})
        if data and data.get("status") and data.get("data"):
            rows = []
            for item in data["data"]:
                rows.append({
                    "strike":   float(item.get("strikeprice", 0)),
                    "CE_ltp":   float(item.get("ce_ltp", 0)),
                    "PE_ltp":   float(item.get("pe_ltp", 0)),
                    "CE_bid":   float(item.get("ce_bestbid", 0)),
                    "PE_bid":   float(item.get("pe_bestbid", 0)),
                    "CE_ask":   float(item.get("ce_bestask", 0)),
                    "PE_ask":   float(item.get("pe_bestask", 0)),
                    "CE_iv":    float(item.get("ce_iv", 0)),
                    "PE_iv":    float(item.get("pe_iv", 0)),
                })
            return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"Error fetching option chain: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════
#  DAX DIRECTION
# ═══════════════════════════════════════════════════════════════════
def get_dax_direction() -> tuple:
    """Get DAX opening direction. No threshold — always returns a direction."""
    dax = yf.download("^GDAXI", period="5d", progress=False, auto_adjust=True)
    if dax.empty or len(dax) < 2:
        log.warning("Could not fetch DAX data")
        return None, 0.0

    if isinstance(dax.columns, pd.MultiIndex):
        dax.columns = [c[0] for c in dax.columns]

    today_open = float(dax["Open"].iloc[-1])
    prev_close = float(dax["Close"].iloc[-2])
    gap_pct = ((today_open - prev_close) / prev_close) * 100.0
    direction = "Bullish" if gap_pct >= 0 else "Bearish"

    log.info(f"DAX: PrevClose={prev_close:.0f}, Open={today_open:.0f}, "
             f"Gap={gap_pct:+.2f}%, Direction={direction}")
    return direction, gap_pct


# ═══════════════════════════════════════════════════════════════════
#  SPREAD
# ═══════════════════════════════════════════════════════════════════
def build_spread(nifty_price: float, direction: str,
                 chain: Optional[pd.DataFrame]) -> Dict:
    atm = round(nifty_price / STRIKE_STEP) * STRIKE_STEP

    if direction == "Bullish":
        opt_type = "PE"
        sold_strike   = atm
        bought_strike = atm - SPREAD_WIDTH
    else:
        opt_type = "CE"
        sold_strike   = atm
        bought_strike = atm + SPREAD_WIDTH

    spread = {
        "direction": direction, "option_type": opt_type,
        "sold_strike": int(sold_strike), "bought_strike": int(bought_strike),
        "spread_width": SPREAD_WIDTH, "nifty_price": round(nifty_price, 2),
    }

    if chain is not None and not chain.empty:
        bid_col = f"{opt_type}_bid"
        ask_col = f"{opt_type}_ask"
        ltp_col = f"{opt_type}_ltp"
        iv_col  = f"{opt_type}_iv"

        sold_row   = chain[chain["strike"] == sold_strike]
        bought_row = chain[chain["strike"] == bought_strike]

        if not sold_row.empty and not bought_row.empty:
            sold_bid = float(sold_row[bid_col].iloc[0])
            sold_ltp = float(sold_row[ltp_col].iloc[0])
            bought_ask = float(bought_row[ask_col].iloc[0])
            bought_ltp = float(bought_row[ltp_col].iloc[0])

            spread["sold_premium"]   = sold_bid if sold_bid > 0 else sold_ltp
            spread["bought_premium"] = bought_ask if bought_ask > 0 else bought_ltp
            spread["sold_iv"]   = float(sold_row[iv_col].iloc[0])
            spread["bought_iv"] = float(bought_row[iv_col].iloc[0])

            net_credit = spread["sold_premium"] - spread["bought_premium"]
            spread["net_credit"]    = round(net_credit, 2)
            spread["net_credit_rs"] = round(net_credit * LOT_SIZE, 2)
            spread["max_risk_rs"]   = round((SPREAD_WIDTH - net_credit) * LOT_SIZE, 2)
            spread["prices_source"] = "live_chain"
        else:
            log.warning(f"Strikes {sold_strike}/{bought_strike} not in chain")
            spread["prices_source"] = "missing"
    else:
        spread["prices_source"] = "no_chain"

    return spread


def get_spread_value(obj: SmartConnect, expiry_str: str, spread: Dict) -> Optional[float]:
    """Get current spread value (sold - bought) from live chain."""
    chain = get_option_greeks(obj, expiry_str)
    if chain is None or chain.empty:
        return None

    opt_type = spread["option_type"]
    ltp_col = f"{opt_type}_ltp"

    sold_row   = chain[chain["strike"] == spread["sold_strike"]]
    bought_row = chain[chain["strike"] == spread["bought_strike"]]

    if sold_row.empty or bought_row.empty:
        return None

    sold_ltp   = float(sold_row[ltp_col].iloc[0])
    bought_ltp = float(bought_row[ltp_col].iloc[0])

    return sold_ltp - bought_ltp


def get_exit_premiums(obj: SmartConnect, expiry_str: str, spread: Dict) -> tuple:
    """Get individual sold/bought exit LTPs from live chain."""
    chain = get_option_greeks(obj, expiry_str)
    if chain is None or chain.empty:
        return None, None

    opt_type = spread["option_type"]
    ltp_col = f"{opt_type}_ltp"

    sold_row   = chain[chain["strike"] == spread["sold_strike"]]
    bought_row = chain[chain["strike"] == spread["bought_strike"]]

    if sold_row.empty or bought_row.empty:
        return None, None

    return float(sold_row[ltp_col].iloc[0]), float(bought_row[ltp_col].iloc[0])


def compute_costs(sold_prem: float, bought_prem: float) -> float:
    brokerage = BROKERAGE_PER_SIDE * 4
    gst = brokerage * GST_PCT
    stt = (sold_prem + bought_prem) * LOT_SIZE * STT_PCT
    slippage = (sold_prem + bought_prem) * LOT_SIZE * SLIPPAGE_PCT * 2
    return round(brokerage + gst + stt + slippage, 2)


# ═══════════════════════════════════════════════════════════════════
#  TRADE LOG
# ═══════════════════════════════════════════════════════════════════
COLUMNS = [
    "date", "day", "direction", "gap_pct",
    "entry_time", "nifty_entry", "nifty_exit",
    "option_type", "sold_strike", "bought_strike", "spread_width",
    "sold_entry_prem", "bought_entry_prem", "sold_iv_entry", "bought_iv_entry",
    "sold_exit_prem", "bought_exit_prem",
    "net_credit", "net_credit_rs", "max_risk_rs",
    "exit_type", "exit_time", "gross_pl_rs", "cost_rs", "net_pl_rs", "net_pl_pct",
    "prices_source", "status", "notes",
]

def log_trade(data: Dict):
    row = {col: data.get(col, "") for col in COLUMNS}
    df_new = pd.DataFrame([row])
    if TRADES_CSV.exists():
        df = pd.concat([pd.read_csv(TRADES_CSV), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(TRADES_CSV, index=False)
    log.info(f"Trade logged → {TRADES_CSV.name}")

def load_trades() -> pd.DataFrame:
    if TRADES_CSV.exists():
        return pd.read_csv(TRADES_CSV)
    return pd.DataFrame(columns=COLUMNS)


def log_check(trade_date: str, check_time: str, check_num: int,
              nifty: float, spread_val: float, gross_pct: float):
    """Append a monitoring check row to paper_checks.csv."""
    row = pd.DataFrame([{
        "date": trade_date, "time": check_time, "check": check_num,
        "nifty": round(nifty, 2), "spread_value": round(spread_val, 4),
        "gross_pl_pct": round(gross_pct, 2),
    }])
    if CHECKS_CSV.exists():
        row.to_csv(CHECKS_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(CHECKS_CSV, index=False)


# ═══════════════════════════════════════════════════════════════════
#  STATS HELPERS
# ═══════════════════════════════════════════════════════════════════
def get_streak(pls) -> str:
    """Return current streak string e.g. '5W' or '2L'."""
    if len(pls) == 0:
        return "—"
    streak = 0
    last_sign = None
    for pl in reversed(pls):
        sign = "W" if pl > 0 else "L"
        if last_sign is None:
            last_sign = sign
        if sign == last_sign:
            streak += 1
        else:
            break
    return f"{streak}{last_sign}"


def get_portfolio_stats() -> dict:
    """Compute portfolio stats from trades CSV."""
    df = load_trades()
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return {"n": 0}
    pls = completed["net_pl_rs"].astype(float).values
    n = len(pls)
    wins = int((pls > 0).sum())
    losses = n - wins
    total = float(pls.sum())
    avg = float(pls.mean())
    best = float(pls.max())
    worst = float(pls.min())
    cum = np.cumsum(pls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum))
    streak = get_streak(pls)
    return {
        "n": n, "wins": wins, "losses": losses,
        "wr": wins / n * 100,
        "total": total, "avg": avg,
        "best": best, "worst": worst,
        "max_dd": max_dd, "streak": streak,
    }


def next_trade_date() -> str:
    """Return the next Tuesday date string."""
    d = today_ist()
    days_ahead = (EXPIRY_WEEKDAY - d.weekday()) % 7
    if days_ahead == 0 and now_ist().hour >= 15:
        days_ahead = 7
    if days_ahead == 0:
        return f"{d.strftime('%A, %d %b %Y')} (today!)"
    nxt = d + timedelta(days=days_ahead)
    return f"{nxt.strftime('%A, %d %b %Y')} at 13:35 IST"


# ═══════════════════════════════════════════════════════════════════
#  README — AUTO-UPDATE
# ═══════════════════════════════════════════════════════════════════
_README = Path(__file__).parent / "README.md"

def _replace_block(text: str, marker_start: str, marker_end: str, content: str) -> str:
    s = text.find(marker_start)
    e = text.find(marker_end)
    if s == -1 or e == -1:
        return text
    return text[:s + len(marker_start)] + "\n" + content + text[e:]


def update_readme(row: dict = None):
    """Update both stats table and last-trade block in README."""
    if not _README.exists():
        log.warning("README.md not found — skipping update")
        return

    text = _README.read_text(encoding="utf-8")

    # ── Stats table ──
    s = get_portfolio_stats()
    if s["n"] > 0:
        stats_block = (
            f"| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |\n"
            f"|:------:|:--------:|:---------:|:-------:|:------:|:------:|\n"
            f"| {s['n']} | {s['wr']:.0f}% ({s['wins']}W/{s['losses']}L) "
            f"| Rs {s['total']:+,.0f} | Rs {s['avg']:+,.0f} "
            f"| Rs {s['max_dd']:,.0f} | {s['streak']} |\n"
        )
    else:
        stats_block = (
            "| Trades | Win Rate | Total P&L | Avg P&L | Max DD | Streak |\n"
            "|:------:|:--------:|:---------:|:-------:|:------:|:------:|\n"
            "| 0 | — | Rs 0 | — | — | — |\n"
        )
    text = _replace_block(text, "<!-- STATS_START -->", "<!-- STATS_END -->", stats_block)

    # ── Last trade ──
    if row:
        pl = float(row.get("net_pl_rs", 0))
        pct = float(row.get("net_pl_pct", 0))
        pl_icon = "W" if pl > 0 else "L"
        sold   = row.get("sold_strike", "")
        bought = row.get("bought_strike", "")
        opt    = row.get("option_type", "")
        spread_label = f"{opt} {sold}/{bought}" if sold and bought else "—"

        trade_block = (
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Date | {row.get('date', '—')} ({row.get('day', '')}) |\n"
            f"| Direction | {row.get('direction', '—')} |\n"
            f"| Spread | {spread_label} (4-wide) |\n"
            f"| Credit | Rs {float(row.get('net_credit_rs', 0)):,.0f} |\n"
            f"| Risk | Rs {float(row.get('max_risk_rs', 0)):,.0f} |\n"
            f"| Exit | {row.get('exit_type', '—')} at {row.get('exit_time', '—')} |\n"
            f"| Nifty | {row.get('nifty_entry', '—')} -> {row.get('nifty_exit', '—')} |\n"
            f"| Result | **{pl_icon} Rs {pl:+,.0f} ({pct:+.1f}%)** |\n"
        )
        text = _replace_block(text, "<!-- LAST_TRADE_START -->", "<!-- LAST_TRADE_END -->", trade_block)

    _README.write_text(text, encoding="utf-8")
    log.info("README.md updated (stats + last trade)")


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def generate_dashboard():
    df = load_trades()
    completed = df[df["status"] == "completed"].copy()

    if completed.empty:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        ax.text(0.5, 0.58, "[ RocketPaper ]", transform=ax.transAxes,
                fontsize=26, fontweight="bold", ha="center", va="center", color="#2196F3")
        ax.text(0.5, 0.44, "No trades logged yet.", transform=ax.transAxes,
                fontsize=16, ha="center", va="center", color="#555555")
        ax.text(0.5, 0.33, "First auto-run: next Tuesday at 13:35 IST", transform=ax.transAxes,
                fontsize=11, ha="center", va="center", color="#888888", style="italic")
        ax.text(0.5, 0.18,
                "Backtest reference: 88 trades · 87.5% WR · Rs +1,23,266 total · MaxDD Rs 5,292",
                transform=ax.transAxes, fontsize=9, ha="center", va="center",
                color="#aaaaaa", fontfamily="monospace")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(DASHBOARD_PNG, bbox_inches="tight")
        plt.close()
        log.info(f"Placeholder dashboard saved → {DASHBOARD_PNG.name}")
        return

    completed["date_dt"] = pd.to_datetime(completed["date"])
    completed = completed.sort_values("date_dt")

    net_pls = completed["net_pl_rs"].astype(float).values
    dates = completed["date_dt"].values
    cum_pl = np.cumsum(net_pls)

    n = len(completed)
    wins = int((net_pls > 0).sum())
    wr = wins / n * 100
    total_pl = cum_pl[-1]
    avg_pl = np.mean(net_pls)
    best = np.max(net_pls)
    worst = np.min(net_pls)
    peak = np.maximum.accumulate(cum_pl)
    max_dd = np.max(peak - cum_pl)

    completed["month"] = completed["date_dt"].dt.to_period("M")
    monthly = completed.groupby("month").agg(
        pl_rs=("net_pl_rs", "sum"),
        trades=("net_pl_rs", "count"),
        wins=("net_pl_rs", lambda x: int((x > 0).sum())),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), dpi=120,
                             gridspec_kw={"height_ratios": [1.2, 0.8, 0.6]})

    # Panel 1: Equity curve
    ax = axes[0]
    ax.set_title("RocketPaper — Live Equity Curve (TP=20%, SL=80%, 4×50pt, Every Tuesday)",
                 fontsize=13, fontweight="bold")
    ax.fill_between(dates, cum_pl, where=(cum_pl >= 0), alpha=0.3, color="#4CAF50")
    ax.fill_between(dates, cum_pl, where=(cum_pl < 0), alpha=0.3, color="#F44336")
    ax.plot(dates, cum_pl, color="#2196F3", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Cumulative P&L (Rs)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    stats = (
        f"Trades: {n}  |  Wins: {wins}  |  WR: {wr:.0f}%\n"
        f"Total P&L: Rs {total_pl:+,.0f}  |  Avg: Rs {avg_pl:+,.0f}\n"
        f"Best: Rs {best:+,.0f}  |  Worst: Rs {worst:+,.0f}  |  MaxDD: Rs {max_dd:,.0f}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # Panel 2: Per-trade bars
    ax = axes[1]
    ax.set_title("Per-Trade Net P&L", fontsize=12, fontweight="bold")
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in net_pls]
    ax.bar(range(n), net_pls, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    labels = [pd.Timestamp(d).strftime("%m/%d") for d in dates]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Net P&L (Rs)")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Monthly
    ax = axes[2]
    ax.set_title("Monthly Summary", fontsize=12, fontweight="bold")
    if not monthly.empty:
        m_labels = [str(m) for m in monthly.index]
        m_vals = monthly["pl_rs"].values
        m_colors = ["#4CAF50" if v > 0 else "#F44336" for v in m_vals]
        bars = ax.bar(range(len(m_labels)), m_vals, color=m_colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for i, (bar, tc, wc) in enumerate(zip(bars, monthly["trades"], monthly["wins"])):
            wr_m = wc / tc * 100 if tc > 0 else 0
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (30 if y >= 0 else -50),
                    f"{tc}t/{wr_m:.0f}%", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=8)
        ax.set_xticks(range(len(m_labels)))
        ax.set_xticklabels(m_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Net P&L (Rs)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(DASHBOARD_PNG, bbox_inches="tight")
    plt.close()
    log.info(f"Dashboard saved → {DASHBOARD_PNG.name}")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY (console)
# ═══════════════════════════════════════════════════════════════════
def print_summary():
    df = load_trades()
    completed = df[df["status"] == "completed"]

    print(f"\n{'='*60}")
    print(f"  RocketPaper — PAPER TRADING SUMMARY")
    print(f"{'='*60}")

    if completed.empty:
        print("  No completed trades yet.")
        print(f"{'='*60}\n")
        return

    n = len(completed)
    pls = completed["net_pl_rs"].astype(float)
    wins = int((pls > 0).sum())
    total = pls.sum()

    print(f"  Trades:    {n}")
    print(f"  Wins:      {wins} ({wins/n*100:.0f}%)")
    print(f"  Total P&L: Rs {total:+,.0f}")
    print(f"  Average:   Rs {pls.mean():+,.0f}")
    print()

    recent = completed.tail(10)
    print(f"  {'Date':<12} {'Dir':<6} {'Nifty':>7} {'Credit':>7} {'Exit':<9} {'P&L Rs':>8} {'P&L%':>7}")
    print(f"  {'-'*62}")
    for _, t in recent.iterrows():
        nf = f"{float(t.get('nifty_entry',0)):>7.0f}" if t.get("nifty_entry") else "    N/A"
        cr = f"{float(t.get('net_credit_rs',0)):>7.0f}" if t.get("net_credit_rs") else "    N/A"
        pct = f"{float(t.get('net_pl_pct',0)):>+6.1f}%" if t.get("net_pl_pct") else "    N/A"
        print(f"  {t['date']:<12} {str(t.get('direction',''))[:4]:<6} {nf} {cr} "
              f"{str(t.get('exit_type','')):<9} {float(t.get('net_pl_rs',0)):>+7,.0f} {pct}")

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM COMMANDS
# ═══════════════════════════════════════════════════════════════════
def cmd_status():
    """Send current status / next trade info to Telegram."""
    s = get_portfolio_stats()
    today = today_ist()
    is_exp = is_expiry_day(today)
    nxt = next_trade_date()

    lines = ["<b>STATUS</b>"]
    if is_exp and 13 <= now_ist().hour < 16:
        lines.append("Trade may be ACTIVE right now")
    elif is_exp:
        lines.append(f"Expiry day — trade window: 13:35-15:15 IST")
    else:
        lines.append(f"Next trade: {nxt}")

    if s["n"] > 0:
        lines.append(f"\nTrades: {s['n']} | WR: {s['wr']:.0f}%")
        lines.append(f"P&L: Rs {s['total']:+,.0f} | Streak: {s['streak']}")
    else:
        lines.append("\nNo trades yet.")

    notify("\n".join(lines))
    print("Status sent to Telegram.")


def cmd_lasttrade():
    """Send last trade details to Telegram."""
    df = load_trades()
    completed = df[df["status"] == "completed"]

    if completed.empty:
        notify("<b>LAST TRADE</b>\n\nNo completed trades yet.")
        print("Last trade sent to Telegram.")
        return

    t = completed.iloc[-1]
    pl = float(t.get("net_pl_rs", 0))
    pct = float(t.get("net_pl_pct", 0))
    icon = "W" if pl > 0 else "L"

    lines = [
        f"<b>LAST TRADE</b>",
        f"",
        f"Date: {t.get('date', '—')} ({t.get('day', '')})",
        f"Direction: {t.get('direction', '—')}",
        f"Spread: {t.get('option_type', '')} {t.get('sold_strike', '')}/{t.get('bought_strike', '')} (4-wide)",
        f"Credit: Rs {float(t.get('net_credit_rs', 0)):,.0f}",
        f"Risk: Rs {float(t.get('max_risk_rs', 0)):,.0f}",
        f"Nifty: {t.get('nifty_entry', '—')} -> {t.get('nifty_exit', '—')}",
        f"Exit: {t.get('exit_type', '—')} at {t.get('exit_time', '—')}",
        f"",
        f"Gross P&L: Rs {float(t.get('gross_pl_rs', 0)):+,.0f}",
        f"Costs: Rs {float(t.get('cost_rs', 0)):,.0f}",
        f"<b>Net P&L: {icon} Rs {pl:+,.0f} ({pct:+.1f}%)</b>",
    ]
    notify("\n".join(lines))
    print("Last trade sent to Telegram.")


def cmd_portfolio():
    """Send portfolio summary to Telegram."""
    s = get_portfolio_stats()
    if s["n"] == 0:
        notify("<b>PORTFOLIO</b>\n\nNo completed trades yet.")
        print("Portfolio sent to Telegram.")
        return

    lines = [
        f"<b>PORTFOLIO SUMMARY</b>",
        f"",
        f"Trades: {s['n']} ({s['wins']}W / {s['losses']}L)",
        f"Win Rate: {s['wr']:.0f}%",
        f"Streak: {s['streak']}",
        f"",
        f"Total P&L: Rs {s['total']:+,.0f}",
        f"Avg P&L: Rs {s['avg']:+,.0f}",
        f"Best: Rs {s['best']:+,.0f}",
        f"Worst: Rs {s['worst']:+,.0f}",
        f"Max Drawdown: Rs {s['max_dd']:,.0f}",
        f"",
        f"Next trade: {next_trade_date()}",
    ]
    notify("\n".join(lines))
    print("Portfolio sent to Telegram.")


def cmd_alltrades():
    """Send all trades table to Telegram."""
    df = load_trades()
    completed = df[df["status"] == "completed"]

    if completed.empty:
        notify("<b>ALL TRADES</b>\n\nNo completed trades yet.")
        print("All trades sent to Telegram.")
        return

    lines = ["<b>ALL TRADES</b>", ""]
    for i, (_, t) in enumerate(completed.iterrows(), 1):
        pl = float(t.get("net_pl_rs", 0))
        pct = float(t.get("net_pl_pct", 0))
        icon = "W" if pl > 0 else "L"
        d = t.get("date", "—")
        ext = str(t.get("exit_type", "—"))[:4]
        lines.append(f"{i}. {d} | {ext} | {icon} Rs {pl:+,.0f} ({pct:+.1f}%)")

    s = get_portfolio_stats()
    lines.append(f"\nTotal: Rs {s['total']:+,.0f} | WR: {s['wr']:.0f}% | Streak: {s['streak']}")

    # Telegram has a 4096 char limit; split if needed
    msg = "\n".join(lines)
    if len(msg) > 4000:
        # Send the last 50 trades only
        lines_short = ["<b>ALL TRADES (recent)</b>", ""]
        for i, (_, t) in enumerate(completed.tail(50).iterrows(), 1):
            pl = float(t.get("net_pl_rs", 0))
            pct = float(t.get("net_pl_pct", 0))
            icon = "W" if pl > 0 else "L"
            lines_short.append(f"{i}. {t.get('date','—')} | {str(t.get('exit_type','—'))[:4]} | {icon} Rs {pl:+,.0f} ({pct:+.1f}%)")
        lines_short.append(f"\nTotal: Rs {s['total']:+,.0f} | WR: {s['wr']:.0f}% | Streak: {s['streak']}")
        msg = "\n".join(lines_short)

    notify(msg)
    print("All trades sent to Telegram.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN RUN — single long-running job
# ═══════════════════════════════════════════════════════════════════
def run_bot():
    """Full lifecycle: enter → monitor (TP/SL) → exit."""
    today = today_ist()
    ist_now = now_ist()
    log.info(f"{'='*60}")
    log.info(f"RocketPaper — {today.strftime('%A, %d %b %Y')} {ist_now.strftime('%H:%M')} IST")
    log.info(f"{'='*60}")

    notify(f"\U0001f680 <b>RocketPaper Activated</b>\n{today.strftime('%A, %d %b %Y')} {ist_now.strftime('%H:%M')} IST")

    # Expiry check
    if not is_expiry_day(today):
        log.info(f"Today is {today.strftime('%A')} — not expiry day (Tuesday). No trade.")
        notify(f"\u274c <b>No Trade</b> — {today.strftime('%A')} is not expiry day")
        return

    log.info("Today IS expiry day.")

    # DAX direction (no threshold — always trade)
    direction, gap_pct = get_dax_direction()
    if direction is None:
        log.info("Could not determine DAX direction. Using Bullish default.")
        direction = "Bullish"
        gap_pct = 0.0

    log.info(f"Direction: {direction} (DAX gap: {gap_pct:+.2f}%)")

    # Login and get entry prices
    obj = angel_login()

    nifty = get_nifty_ltp(obj)
    if nifty is None:
        log.error("Could not get Nifty LTP. Aborting.")
        log_trade({"date": str(today), "day": today.strftime("%A"),
                   "direction": direction, "gap_pct": round(gap_pct, 3),
                   "status": "no_nifty_ltp"})
        notify(f"\u26a0\ufe0f <b>No Trade</b> — Could not fetch Nifty LTP")
        return

    log.info(f"Nifty at entry: {nifty:.2f}")

    expiry_str = get_expiry_string(today)
    chain = get_option_greeks(obj, expiry_str)
    spread = build_spread(nifty, direction, chain)

    if spread.get("sold_premium") is None or spread.get("bought_premium") is None:
        log.warning("Could not get option prices. Aborting.")
        log_trade({"date": str(today), "day": today.strftime("%A"),
                   "direction": direction, "gap_pct": round(gap_pct, 3),
                   "nifty_entry": nifty, "status": "no_option_prices",
                   "prices_source": spread.get("prices_source", "unknown")})
        notify(f"\u26a0\ufe0f <b>No Trade</b> — Option prices unavailable\nNifty: {nifty:.0f}")
        return

    net_credit = spread["net_credit"]
    net_credit_rs = spread["net_credit_rs"]
    max_risk_rs = spread["max_risk_rs"]

    log.info(f"ENTRY: Sell {spread['sold_strike']} {spread['option_type']}, "
             f"Buy {spread['bought_strike']} {spread['option_type']}")
    log.info(f"  Sold:   Rs {spread['sold_premium']:.2f} (IV: {spread.get('sold_iv',0):.1f}%)")
    log.info(f"  Bought: Rs {spread['bought_premium']:.2f} (IV: {spread.get('bought_iv',0):.1f}%)")
    log.info(f"  Credit: Rs {net_credit:.2f}/unit = Rs {net_credit_rs:.0f}")
    log.info(f"  Risk:   Rs {max_risk_rs:.0f}")
    log.info(f"  TP at:  Rs {max_risk_rs * TP_PCT / 100:.0f} profit")
    log.info(f"  SL at:  Rs {max_risk_rs * SL_PCT / 100:.0f} loss")

    entry_time_str = now_ist().strftime("%H:%M")

    notify(
        f"\U0001f4c8 <b>TRADE ENTRY</b>\n"
        f"Direction: {direction} (DAX {gap_pct:+.2f}%)\n"
        f"Nifty: {nifty:.0f}\n"
        f"Sell {spread['sold_strike']} {spread['option_type']} | "
        f"Buy {spread['bought_strike']} {spread['option_type']}\n"
        f"Credit: Rs {net_credit_rs:,.0f} | Risk: Rs {max_risk_rs:,.0f}\n"
        f"TP: Rs {max_risk_rs * TP_PCT / 100:,.0f} | SL: Rs {max_risk_rs * SL_PCT / 100:,.0f}"
    )

    # ── MONITORING LOOP ──
    log.info(f"\n{'='*60}")
    log.info("MONITORING — checking for TP/SL every 1 min")
    log.info(f"{'='*60}")

    exit_type = "Time Exit"
    exit_time_str = ""
    sold_exit = spread["sold_premium"]
    bought_exit = spread["bought_premium"]
    nifty_exit = nifty
    check_count = 0

    while True:
        ist_now = now_ist()
        exit_hour, exit_min = 15, 15

        # Check if past exit time
        if ist_now.hour > exit_hour or (ist_now.hour == exit_hour and ist_now.minute >= exit_min):
            log.info(f"Reached {EXIT_TIME} IST — time exit.")
            exit_time_str = ist_now.strftime("%H:%M")
            break

        # Wait between checks (skip wait on first iteration)
        if check_count > 0:
            time.sleep(MONITOR_INTERVAL)

        check_count += 1

        # Re-login if needed (sessions can expire)
        try:
            nifty_now = get_nifty_ltp(obj)
        except Exception:
            try:
                obj = angel_login()
                nifty_now = get_nifty_ltp(obj)
            except Exception as e:
                log.warning(f"Check #{check_count}: API error: {e}")
                continue

        if nifty_now is None:
            log.warning(f"Check #{check_count}: No Nifty price")
            continue

        spread_value = get_spread_value(obj, expiry_str, spread)
        if spread_value is None:
            log.warning(f"Check #{check_count}: No option prices")
            continue

        nifty_exit = nifty_now
        # Current spread value = what we'd pay to close
        spread_value = max(0, min(spread_value, SPREAD_WIDTH))

        pl_per_unit = net_credit - spread_value
        gross_pl_rs = pl_per_unit * LOT_SIZE
        # TP/SL checked on GROSS P&L (matching backtest — costs deducted at final exit)
        gross_pl_pct = (gross_pl_rs / max_risk_rs * 100) if max_risk_rs > 0 else 0

        log.info(f"Check #{check_count} [{ist_now.strftime('%H:%M')}]: "
                 f"Nifty={nifty_now:.0f}, SpreadVal={spread_value:.2f}, "
                 f"Gross P&L=Rs {gross_pl_rs:+,.0f} ({gross_pl_pct:+.1f}%)")

        log_check(str(today), ist_now.strftime("%H:%M:%S"), check_count,
                  nifty_now, spread_value, gross_pl_pct)

        # TP hit? (gross, before costs — matches backtest)
        if gross_pl_pct > 0 and gross_pl_pct >= TP_PCT:
            exit_type = "TP"
            exit_time_str = ist_now.strftime("%H:%M")
            log.info(f"TP HIT at {exit_time_str}! Gross P&L = Rs {gross_pl_rs:+,.0f} ({gross_pl_pct:+.1f}%)")
            notify(
                f"\u2705 <b>TP HIT!</b> at {exit_time_str} IST\n"
                f"Gross P&L: Rs {gross_pl_rs:+,.0f} ({gross_pl_pct:+.1f}%)\n"
                f"Nifty: {nifty_now:.0f} | Check #{check_count}"
            )
            break

        # SL hit? (gross, before costs — matches backtest)
        if gross_pl_pct < 0 and abs(gross_pl_pct) >= SL_PCT:
            exit_type = "SL"
            exit_time_str = ist_now.strftime("%H:%M")
            log.info(f"SL HIT at {exit_time_str}! Gross P&L = Rs {gross_pl_rs:+,.0f} ({gross_pl_pct:+.1f}%)")
            notify(
                f"\U0001f6d1 <b>SL HIT!</b> at {exit_time_str} IST\n"
                f"Gross P&L: Rs {gross_pl_rs:+,.0f} ({gross_pl_pct:+.1f}%)\n"
                f"Nifty: {nifty_now:.0f} | Check #{check_count}"
            )
            break

    # ── FINAL EXIT PRICING ──
    log.info(f"\n{'='*60}")
    log.info(f"EXIT — {exit_type} at {exit_time_str or EXIT_TIME}")
    log.info(f"{'='*60}")

    # Get final prices
    try:
        nifty_exit = get_nifty_ltp(obj) or nifty_exit
        final_spread_value = get_spread_value(obj, expiry_str, spread)
        sold_exit_real, bought_exit_real = get_exit_premiums(obj, expiry_str, spread)
    except Exception:
        final_spread_value = None
        sold_exit_real, bought_exit_real = None, None

    if final_spread_value is not None:
        final_spread_value = max(0, min(final_spread_value, SPREAD_WIDTH))
        pl_per_unit = net_credit - final_spread_value
        gross_pl_rs = pl_per_unit * LOT_SIZE
        cost_rs = compute_costs(spread["sold_premium"], spread["bought_premium"])
        net_pl_rs = gross_pl_rs - cost_rs
        net_pl_pct = (net_pl_rs / max_risk_rs * 100) if max_risk_rs > 0 else 0

        sold_exit = sold_exit_real if sold_exit_real is not None else 0.0
        bought_exit = bought_exit_real if bought_exit_real is not None else 0.0

        log.info(f"  Final spread value: {final_spread_value:.2f}")
        log.info(f"  Gross P&L: Rs {gross_pl_rs:+,.0f}")
        log.info(f"  Costs:     Rs {cost_rs:.0f}")
        log.info(f"  Net P&L:   Rs {net_pl_rs:+,.0f} ({net_pl_pct:+.1f}%)")
        status = "completed"
    else:
        log.warning("Could not get exit prices — using entry prices as estimate")
        gross_pl_rs = 0
        cost_rs = compute_costs(spread["sold_premium"], spread["bought_premium"])
        net_pl_rs = -cost_rs
        net_pl_pct = (-cost_rs / max_risk_rs * 100) if max_risk_rs > 0 else 0
        status = "incomplete"

    # Log trade
    log_trade({
        "date": str(today), "day": today.strftime("%A"),
        "direction": direction, "gap_pct": round(gap_pct, 3),
        "entry_time": entry_time_str,
        "nifty_entry": nifty, "nifty_exit": nifty_exit,
        "option_type": spread["option_type"],
        "sold_strike": spread["sold_strike"],
        "bought_strike": spread["bought_strike"],
        "spread_width": spread["spread_width"],
        "sold_entry_prem": spread["sold_premium"],
        "bought_entry_prem": spread["bought_premium"],
        "sold_iv_entry": spread.get("sold_iv", 0),
        "bought_iv_entry": spread.get("bought_iv", 0),
        "sold_exit_prem": round(sold_exit, 2) if isinstance(sold_exit, float) else "",
        "bought_exit_prem": round(bought_exit, 2) if isinstance(bought_exit, float) else "",
        "net_credit": net_credit,
        "net_credit_rs": net_credit_rs,
        "max_risk_rs": max_risk_rs,
        "exit_type": exit_type,
        "exit_time": exit_time_str or EXIT_TIME,
        "gross_pl_rs": round(gross_pl_rs, 2),
        "cost_rs": round(cost_rs, 2),
        "net_pl_rs": round(net_pl_rs, 2),
        "net_pl_pct": round(net_pl_pct, 2),
        "prices_source": spread["prices_source"],
        "status": status,
        "notes": f"{check_count} checks",
    })

    # README + Dashboard
    update_readme({
        "date": str(today), "day": today.strftime("%A"),
        "direction": direction,
        "option_type": spread["option_type"],
        "sold_strike": spread["sold_strike"],
        "bought_strike": spread["bought_strike"],
        "net_credit_rs": net_credit_rs,
        "max_risk_rs": max_risk_rs,
        "exit_type": exit_type,
        "exit_time": exit_time_str or EXIT_TIME,
        "nifty_entry": f"{nifty:.0f}",
        "nifty_exit": f"{nifty_exit:.0f}",
        "net_pl_rs": round(net_pl_rs, 2),
        "net_pl_pct": round(net_pl_pct, 2),
    })
    generate_dashboard()
    print_summary()

    log.info(f"\nRESULT: {exit_type} — Rs {net_pl_rs:+,.0f} ({net_pl_pct:+.1f}%)")

    s = get_portfolio_stats()
    emoji = "\u2705" if net_pl_rs > 0 else "\U0001f534"
    notify(
        f"{emoji} <b>TRADE CLOSED — {exit_type}</b>\n"
        f"Net P&L: Rs {net_pl_rs:+,.0f} ({net_pl_pct:+.1f}%)\n"
        f"Gross: Rs {gross_pl_rs:+,.0f} | Costs: Rs {cost_rs:,.0f}\n"
        f"Nifty: {nifty:.0f} \u2192 {nifty_exit:.0f}\n"
        f"Checks: {check_count} | Status: {status}\n"
        f"\nStreak: {s['streak']} | WR: {s['wr']:.0f}% | Total: Rs {s['total']:+,.0f}"
    )


# ═══════════════════════════════════════════════════════════════════
#  TEST / DRY RUN
# ═══════════════════════════════════════════════════════════════════
def run_test():
    today = today_ist()
    print(f"--- RocketPaper DRY RUN ---")
    print(f"Date: {today} ({today.strftime('%A')})")
    print(f"IST now: {now_ist().strftime('%H:%M:%S')}")
    print(f"Is expiry (Tuesday): {is_expiry_day(today)}")

    direction, gap = get_dax_direction()
    print(f"DAX direction: {direction} (gap: {gap:+.2f}%)")

    msg_lines = [
        f"🧪 <b>RocketPaper DRY RUN</b>",
        f"Date: {today} ({today.strftime('%A')})",
        f"Expiry day: {is_expiry_day(today)}",
        f"DAX: {direction} ({gap:+.2f}%)",
    ]

    try:
        obj = angel_login()
        nifty = get_nifty_ltp(obj)
        if nifty:
            print(f"Nifty LTP: {nifty:.2f}")
            expiry_str = get_expiry_string(today)
            chain = get_option_greeks(obj, expiry_str)
            spread = build_spread(nifty, direction or "Bullish", chain)
            print(f"Would sell: {spread['sold_strike']} {spread['option_type']}")
            print(f"Would buy:  {spread['bought_strike']} {spread['option_type']}")
            msg_lines.append(f"Nifty LTP: {nifty:.0f}")
            msg_lines.append(f"Would sell: {spread['sold_strike']} {spread['option_type']}")
            msg_lines.append(f"Would buy:  {spread['bought_strike']} {spread['option_type']}")
            if spread.get("net_credit_rs"):
                print(f"Credit: Rs {spread['net_credit_rs']:.0f}")
                print(f"Risk:   Rs {spread['max_risk_rs']:.0f}")
                msg_lines.append(f"Credit: Rs {spread['net_credit_rs']:.0f} | Risk: Rs {spread['max_risk_rs']:.0f}")
            msg_lines.append("✅ All systems OK")
        else:
            msg_lines.append("⚠️ Could not fetch Nifty LTP")
    except Exception as e:
        print(f"API error: {e}")
        msg_lines.append(f"❌ API error: {e}")

    notify("\n".join(msg_lines))


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"

    if cmd == "run":
        run_bot()
    elif cmd == "test":
        run_test()
    elif cmd == "summary":
        print_summary()
    elif cmd == "dashboard":
        generate_dashboard()
        update_readme()
        print("Dashboard regenerated.")
    elif cmd == "status":
        cmd_status()
    elif cmd == "lasttrade":
        cmd_lasttrade()
    elif cmd == "portfolio":
        cmd_portfolio()
    elif cmd == "alltrades":
        cmd_alltrades()
    else:
        print("Usage: python paper_trader.py <run|test|status|lasttrade|portfolio|alltrades|summary|dashboard>")
