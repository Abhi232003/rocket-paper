# -*- coding: utf-8 -*-
"""
RocketData — Nifty 0-DTE Credit Spread Backtest
================================================
Single finalized strategy using real Nifty 1-min candles from Angel One.

Strategy: Sell ATM credit spread every expiry day (Tuesday).
  - Direction: DAX opening gap direction (no threshold — trade every expiry)
  - Spread: 4-wide (200pt)
  - TP: 20% of max risk → exit immediately
  - SL: 80% of max risk → exit immediately
  - Time exit: 15:15 IST if neither TP/SL hit
  - Monitoring: every 1-min bar (~100 bars per trade)

What's real vs estimated:
  ✅ Real Nifty index prices minute-by-minute (Angel One Historical API)
  ✅ Real DAX gap direction (yfinance)
  ✅ Path-dependent TP/SL with 1-min granularity
  ✅ Intrinsic-only exit for last 15 min (0-DTE theta collapse)
  ⚠️ Entry credit estimated via Black-Scholes (no real option prices available)
  ⚠️ IV assumed static at 15% (real 0-DTE IV can be 15-30%)
"""

import os, sys, time, math
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from SmartApi import SmartConnect
import pyotp
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
API_KEY      = os.environ.get("ANGEL_API_KEY", "")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
MPIN         = os.environ.get("ANGEL_MPIN", "")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")
NIFTY_TOKEN  = "99926000"

# Strategy params
ENTRY_TIME     = "13:35"
EXIT_TIME      = "15:15"
STRIKE_STEP    = 50
LOT_SIZE       = 65
TP_PCT         = 20
SL_PCT         = 80
WIDTH          = 4
SPREAD_WIDTH   = WIDTH * STRIKE_STEP  # 200pt

# BS pricing
SESSION_MINUTES = 375
IV_DEFAULT      = 0.15
RISK_FREE       = 0.065

# Costs
BROKERAGE_PER_SIDE = 20.0
STT_PCT            = 0.05 / 100
GST_PCT            = 0.18
SLIPPAGE_PCT       = 0.005

# Date range
START_DATE = date(2024, 6, 1)
END_DATE   = date(2026, 3, 8)
EXPIRY_CHANGE_DATE = date(2024, 11, 20)

CACHE_DIR = Path(__file__).parent / "nifty_cache"


# ═══════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES
# ═══════════════════════════════════════════════════════════════════
def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
    sq_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq_T)
    d2 = d1 - sigma * sq_T
    if opt_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def intrinsic_value(S, K, opt_type):
    return max(S - K, 0) if opt_type == "CE" else max(K - S, 0)


# ═══════════════════════════════════════════════════════════════════
#  ANGEL ONE API
# ═══════════════════════════════════════════════════════════════════
def login_angel():
    obj = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    data = obj.generateSession(CLIENT_ID, MPIN, totp)
    if not data or data.get("message") != "SUCCESS":
        raise RuntimeError(f"Login failed: {data}")
    print(f"✓ Angel One login OK (Client: {CLIENT_ID})")
    return obj


def fetch_nifty_1min(obj, dt: date, max_retries=2) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"nifty_{dt.strftime('%Y%m%d')}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["timestamp"]).set_index("timestamp")

    for attempt in range(max_retries):
        try:
            params = {
                "exchange": "NSE", "symboltoken": NIFTY_TOKEN,
                "interval": "ONE_MINUTE",
                "fromdate": f"{dt.strftime('%Y-%m-%d')} 09:15",
                "todate":   f"{dt.strftime('%Y-%m-%d')} 15:30",
            }
            resp = obj.getCandleData(params)
            if resp and resp.get("data") and len(resp["data"]) > 0:
                df = pd.DataFrame(resp["data"],
                                  columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                CACHE_DIR.mkdir(exist_ok=True)
                df.to_csv(cache_file)
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  ⚠ Error fetching {dt}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def get_price_at_time(df: pd.DataFrame, target_time: str):
    if df.empty:
        return None
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    target = datetime.combine(idx[0].date(), datetime.strptime(target_time, "%H:%M").time())
    for delta in [2, 5]:
        mask = (idx >= target - timedelta(minutes=delta)) & (idx <= target + timedelta(minutes=1))
        subset = df[mask]
        if not subset.empty:
            sub_idx = subset.index.tz_localize(None) if subset.index.tz is not None else subset.index
            closest = (sub_idx - target).map(abs).argmin()
            return float(subset.iloc[closest]["close"])
    return None


def get_all_prices_in_window(df: pd.DataFrame, start_time: str, end_time: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    dt = idx[0].date()
    t_start = datetime.combine(dt, datetime.strptime(start_time, "%H:%M").time())
    t_end   = datetime.combine(dt, datetime.strptime(end_time,   "%H:%M").time())
    result = df.loc[(idx >= t_start) & (idx <= t_end), "close"].copy()
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    return result


# ═══════════════════════════════════════════════════════════════════
#  DAX DIRECTION (no threshold — just direction)
# ═══════════════════════════════════════════════════════════════════
def fetch_dax_data(start: date, end: date) -> pd.DataFrame:
    cache_file = CACHE_DIR / "dax_daily.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=[0])
        if df.index[-1].date() >= end - timedelta(days=5):
            return df

    print("Downloading DAX data...")
    dax = yf.download("^GDAXI", start=start - timedelta(days=10),
                       end=end + timedelta(days=1), progress=False)
    if dax.empty:
        raise RuntimeError("Failed to download DAX data")
    if isinstance(dax.columns, pd.MultiIndex):
        dax.columns = [c[0] for c in dax.columns]
    CACHE_DIR.mkdir(exist_ok=True)
    dax.to_csv(cache_file)
    return dax


def compute_dax_direction(dax_df: pd.DataFrame, trade_date: date):
    """Returns (direction, gap_pct). Direction is always Bullish/Bearish (no threshold)."""
    dax_dates = dax_df.index.date if hasattr(dax_df.index, 'date') else dax_df.index

    today_idx = -1
    for i, d in enumerate(dax_dates):
        if d == trade_date:
            today_idx = i
            break

    if today_idx < 1:
        return None, 0.0

    prev_close = float(dax_df.iloc[today_idx - 1]["Close"])
    today_open = float(dax_df.iloc[today_idx]["Open"])

    if prev_close == 0:
        return None, 0.0

    gap_pct = ((today_open - prev_close) / prev_close) * 100.0
    direction = "Bullish" if gap_pct >= 0 else "Bearish"
    return direction, gap_pct


# ═══════════════════════════════════════════════════════════════════
#  EXPIRY DATES
# ═══════════════════════════════════════════════════════════════════
def generate_expiry_dates(start: date, end: date) -> list:
    expiries = []
    current = start
    while current <= end:
        target_weekday = 3 if current < EXPIRY_CHANGE_DATE else 1
        days_ahead = target_weekday - current.weekday()
        if days_ahead < 0:
            days_ahead += 7
        expiry = current + timedelta(days=days_ahead)
        if start <= expiry <= end and expiry not in expiries:
            expiries.append(expiry)
        current += timedelta(days=7)
    return sorted(set(expiries))


# ═══════════════════════════════════════════════════════════════════
#  COSTS
# ═══════════════════════════════════════════════════════════════════
def compute_costs(sold_premium, bought_premium):
    brokerage = BROKERAGE_PER_SIDE * 4  # 2 legs × 2 sides
    gst = brokerage * GST_PCT
    stt = (sold_premium + bought_premium) * LOT_SIZE * STT_PCT
    slippage = (sold_premium + bought_premium) * LOT_SIZE * SLIPPAGE_PCT * 2
    return brokerage + gst + stt + slippage


def _time_to_minutes(time_str):
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


# ═══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION (path-dependent TP/SL, 1-min bars)
# ═══════════════════════════════════════════════════════════════════
def simulate_trade(nifty_prices: pd.Series, entry_nifty: float, signal: str) -> dict:
    """
    Simulate a single 0-DTE credit spread trade.
    Checks TP/SL at every 1-min bar (path-dependent).
    """
    atm = round(entry_nifty / STRIKE_STEP) * STRIKE_STEP

    if signal == "Bullish":
        opt_type = "PE"
        sold_strike   = atm
        bought_strike = atm - SPREAD_WIDTH
    else:
        opt_type = "CE"
        sold_strike   = atm
        bought_strike = atm + SPREAD_WIDTH

    # Entry pricing via BS
    T_entry = 115.0 / (SESSION_MINUTES * 252)  # ~115 min to close
    sold_premium   = _bs_price(entry_nifty, sold_strike,   T_entry, RISK_FREE, IV_DEFAULT, opt_type)
    bought_premium = _bs_price(entry_nifty, bought_strike, T_entry, RISK_FREE, IV_DEFAULT, opt_type)

    net_credit = sold_premium - bought_premium
    if net_credit <= 0:
        return None

    max_loss_per_unit = SPREAD_WIDTH - net_credit
    max_risk_rs = max_loss_per_unit * LOT_SIZE
    net_credit_rs = net_credit * LOT_SIZE

    if max_risk_rs < 200:
        return None

    max_profit_pct = (net_credit_rs / max_risk_rs) * 100.0

    # Path-dependent monitoring — check every 1-min bar
    exit_type = "Time Exit"
    exit_pl_pct = 0.0
    exit_nifty = entry_nifty
    exit_bar = 0

    prices_list = list(nifty_prices.items())
    n_bars = len(prices_list)

    for i, (ts, nifty_now) in enumerate(prices_list):
        current_min = _time_to_minutes(ts.strftime("%H:%M"))
        close_min = _time_to_minutes("15:30")
        mins_left = close_min - current_min
        is_last_bar = (i == n_bars - 1)

        T_now = max(mins_left / (SESSION_MINUTES * 252), 0.00001)
        use_intrinsic = is_last_bar or mins_left <= 15

        if use_intrinsic:
            sold_now   = intrinsic_value(nifty_now, sold_strike, opt_type)
            bought_now = intrinsic_value(nifty_now, bought_strike, opt_type)
        else:
            sold_now   = _bs_price(nifty_now, sold_strike,   T_now, RISK_FREE, IV_DEFAULT, opt_type)
            bought_now = _bs_price(nifty_now, bought_strike, T_now, RISK_FREE, IV_DEFAULT, opt_type)

        spread_value = max(0, min(sold_now - bought_now, SPREAD_WIDTH))
        pl_per_unit = net_credit - spread_value
        pl_pct = (pl_per_unit * LOT_SIZE / max_risk_rs) * 100.0

        # TP check
        if pl_pct > 0 and (pl_pct >= TP_PCT or pl_pct >= max_profit_pct):
            exit_type = "TP"
            exit_pl_pct = min(pl_pct, max_profit_pct)
            exit_nifty = nifty_now
            exit_bar = i
            break

        # SL check
        if pl_pct < 0 and abs(pl_pct) >= SL_PCT:
            exit_type = "SL"
            exit_pl_pct = -SL_PCT
            exit_nifty = nifty_now
            exit_bar = i
            break

        # Last bar — time exit
        if is_last_bar:
            exit_type = "Time Exit"
            exit_pl_pct = max(min(pl_pct, max_profit_pct), -100.0)
            exit_nifty = nifty_now
            exit_bar = i
            break

    # Costs
    cost_rs = compute_costs(sold_premium, bought_premium)
    cost_pct = (cost_rs / max_risk_rs) * 100.0

    if cost_pct > 30:
        return None

    gross_pl_rs = exit_pl_pct / 100 * max_risk_rs
    net_pl_pct = exit_pl_pct - cost_pct
    net_pl_pct = max(net_pl_pct, -100.0)
    net_pl_rs = net_pl_pct / 100 * max_risk_rs

    return {
        "exit_type":       exit_type,
        "gross_pl_pct":    round(exit_pl_pct, 2),
        "gross_pl_rs":     round(gross_pl_rs, 2),
        "net_pl_pct":      round(net_pl_pct, 2),
        "net_pl_rs":       round(net_pl_rs, 2),
        "net_credit":      round(net_credit, 2),
        "net_credit_rs":   round(net_credit_rs, 2),
        "max_risk_rs":     round(max_risk_rs, 2),
        "sold_strike":     sold_strike,
        "bought_strike":   bought_strike,
        "entry_nifty":     round(entry_nifty, 2),
        "exit_nifty":      round(exit_nifty, 2),
        "opt_type":        opt_type,
        "cost_rs":         round(cost_rs, 2),
        "cost_pct":        round(cost_pct, 2),
        "n_bars":          n_bars,
        "exit_bar":        exit_bar,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("RocketData — Nifty 0-DTE Credit Spread Backtest")
    print("=" * 70)
    print(f"Strategy:  TP={TP_PCT}%, SL={SL_PCT}%, Width={WIDTH}×{STRIKE_STEP}pt, Every Expiry Day")
    print(f"Period:    {START_DATE} to {END_DATE}")
    print(f"Entry:     {ENTRY_TIME} IST | Exit: {EXIT_TIME} IST | Lot: {LOT_SIZE}")
    print(f"Direction: DAX opening gap (no threshold)")
    print()

    obj = login_angel()
    time.sleep(0.5)

    dax_df = fetch_dax_data(START_DATE, END_DATE)

    expiry_dates = generate_expiry_dates(
        START_DATE - timedelta(days=7), END_DATE + timedelta(days=14))
    valid_expiries = [e for e in expiry_dates if START_DATE <= e <= END_DATE]
    print(f"✓ {len(valid_expiries)} expiry dates in range")

    # DAX directions
    dax_dates_arr = dax_df.index.date if hasattr(dax_df.index, 'date') else [d.date() for d in dax_df.index]
    dax_dirs = {}
    for dt in dax_dates_arr:
        if dt < START_DATE or dt > END_DATE:
            continue
        direction, gap_pct = compute_dax_direction(dax_df, dt)
        if direction:
            dax_dirs[dt] = (direction, gap_pct)

    print(f"✓ DAX direction data for {len(dax_dirs)} days")

    # Download Nifty 1-min data
    print(f"\n--- Downloading Nifty 1-min data for {len(valid_expiries)} expiry dates ---")
    nifty_data = {}
    for i, dt in enumerate(valid_expiries):
        df = fetch_nifty_1min(obj, dt)
        if not df.empty:
            nifty_data[dt] = df
        if (i + 1) % 20 == 0 or i == len(valid_expiries) - 1:
            print(f"  Progress: {i+1}/{len(valid_expiries)} ({len(nifty_data)} loaded)")
        time.sleep(0.1)

    valid_expiries = [e for e in valid_expiries if e in nifty_data]
    print(f"✓ {len(valid_expiries)} valid expiry dates with data")

    # ── Run simulation ──
    print(f"\n{'='*70}")
    print("RUNNING SIMULATION")
    print(f"{'='*70}\n")

    trades = []

    for dt in valid_expiries:
        gap_info = dax_dirs.get(dt)
        if gap_info is None:
            continue

        direction, gap_pct = gap_info
        nifty_df = nifty_data[dt]

        entry_price = get_price_at_time(nifty_df, ENTRY_TIME)
        if entry_price is None:
            continue

        prices = get_all_prices_in_window(nifty_df, ENTRY_TIME, EXIT_TIME)
        if len(prices) < 5:
            continue

        result = simulate_trade(prices, entry_price, direction)
        if result is None:
            continue

        result["date"] = dt.strftime("%Y-%m-%d")
        result["day"] = dt.strftime("%A")
        result["signal"] = direction
        result["gap_pct"] = round(gap_pct, 2)
        trades.append(result)

    # ── Results ──
    print_results(trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(trades, ts)
    plot_equity_curve(trades, ts)
    plot_monthly_returns(trades, ts)

    print(f"\n✓ All done! {len(trades)} trades simulated.")


# ═══════════════════════════════════════════════════════════════════
#  RESULTS PRINTING
# ═══════════════════════════════════════════════════════════════════
def print_results(trades):
    print(f"\n{'='*70}")
    print("RocketData — BACKTEST RESULTS")
    print(f"{'='*70}")

    if not trades:
        print("  No trades!")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t["net_pl_pct"] > 0)
    losses = n - wins
    wr = wins / n * 100

    avg_pl = np.mean([t["net_pl_pct"] for t in trades])
    total_pl_rs = sum(t["net_pl_rs"] for t in trades)
    avg_credit = np.mean([t["net_credit_rs"] for t in trades])
    avg_risk = np.mean([t["max_risk_rs"] for t in trades])

    cum_pl = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum_pl += t["net_pl_rs"]
        peak = max(peak, cum_pl)
        max_dd = max(max_dd, peak - cum_pl)

    tp = sum(1 for t in trades if t["exit_type"] == "TP")
    sl = sum(1 for t in trades if t["exit_type"] == "SL")
    te = sum(1 for t in trades if t["exit_type"] == "Time Exit")

    print(f"\n  Trades:       {n}")
    print(f"  Wins:         {wins} ({wr:.1f}%)")
    print(f"  Losses:       {losses}")
    print(f"  TP exits:     {tp}")
    print(f"  SL exits:     {sl}")
    print(f"  Time exits:   {te}")
    print(f"  Avg P&L%:     {avg_pl:+.2f}%")
    print(f"  Total P&L:    Rs {total_pl_rs:+,.0f}")
    print(f"  Avg Credit:   Rs {avg_credit:,.0f}")
    print(f"  Avg Risk:     Rs {avg_risk:,.0f}")
    print(f"  Max Drawdown: Rs {max_dd:,.0f}")

    # Detailed trade log
    print(f"\n{'='*70}")
    print("TRADE LOG")
    print(f"{'='*70}")
    print(f"\n{'#':>3} {'Date':<12} {'Day':<4} {'Dir':<5} {'Gap%':>5} "
          f"{'Nifty':>7} {'Type':<3} {'Sold':>6} {'Bght':>6} "
          f"{'Credit':>7} {'Risk':>7} {'Exit':<6} {'Gross':>7} {'Cost':>5} {'Net':>7} {'Net%':>7} {'Bar':>4}")
    print("-" * 120)

    cum = 0.0
    for i, t in enumerate(trades, 1):
        cum += t["net_pl_rs"]
        print(f"{i:>3} {t['date']:<12} {t['day'][:3]:<4} {t['signal'][:4]:<5} {t['gap_pct']:>+4.1f}% "
              f"{t['entry_nifty']:>7.0f} {t['opt_type']:<3} {t['sold_strike']:>6} {t['bought_strike']:>6} "
              f"{t['net_credit_rs']:>6,.0f} {t['max_risk_rs']:>6,.0f} {t['exit_type']:<6} "
              f"{t['gross_pl_rs']:>+6,.0f} {t['cost_rs']:>5,.0f} {t['net_pl_rs']:>+6,.0f} "
              f"{t['net_pl_pct']:>+6.1f}% {t['exit_bar']:>4}")

    print(f"\n  Cumulative P&L: Rs {cum:+,.0f}")

    # Monthly breakdown
    print(f"\n{'='*70}")
    print("MONTHLY BREAKDOWN")
    print(f"{'='*70}")
    print(f"\n  {'Month':<10} {'Trds':>5} {'W':>3} {'L':>3} {'WR%':>6} {'P&L Rs':>9} {'Cum Rs':>10}")
    print(f"  {'-'*52}")

    monthly = {}
    for t in trades:
        key = t["date"][:7]
        if key not in monthly:
            monthly[key] = {"trades": 0, "wins": 0, "pl_rs": 0}
        monthly[key]["trades"] += 1
        if t["net_pl_pct"] > 0:
            monthly[key]["wins"] += 1
        monthly[key]["pl_rs"] += t["net_pl_rs"]

    cum_mo = 0
    for key in sorted(monthly):
        m = monthly[key]
        cum_mo += m["pl_rs"]
        wr_m = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        print(f"  {key:<10} {m['trades']:>5} {m['wins']:>3} {m['trades']-m['wins']:>3} "
              f"{wr_m:>5.0f}% {m['pl_rs']:>+8,.0f} {cum_mo:>+9,.0f}")


def save_results(trades, ts):
    if not trades:
        return
    df = pd.DataFrame(trades)
    out = Path(__file__).parent / f"rocket_backtest_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved {len(trades)} trades to {out.name}")


# ═══════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════
def plot_equity_curve(trades, ts):
    if not trades:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=120,
                             gridspec_kw={"height_ratios": [1.3, 0.7]})

    dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in trades]
    net_pls = np.array([t["net_pl_rs"] for t in trades])
    cum_rs = np.cumsum(net_pls)
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pl_pct"] > 0)
    wr = wins / n * 100
    total = cum_rs[-1]

    peak = np.maximum.accumulate(cum_rs)
    max_dd = np.max(peak - cum_rs)

    tp = sum(1 for t in trades if t["exit_type"] == "TP")
    sl = sum(1 for t in trades if t["exit_type"] == "SL")
    te = sum(1 for t in trades if t["exit_type"] == "Time Exit")

    # Panel 1: Equity curve
    ax = axes[0]
    ax.set_title("RocketData — Equity Curve (TP=20%, SL=80%, 4×50pt, Every Expiry Day)",
                 fontsize=13, fontweight='bold')
    ax.fill_between(dates, cum_rs, where=(cum_rs >= 0), alpha=0.3, color="#4CAF50")
    ax.fill_between(dates, cum_rs, where=(cum_rs < 0), alpha=0.3, color="#F44336")
    ax.plot(dates, cum_rs, color="#2196F3", linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel("Cumulative P&L (Rs)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    stats = (
        f"Trades: {n}  |  Wins: {wins}  |  WR: {wr:.0f}%  |  TP:{tp} SL:{sl} TE:{te}\n"
        f"Total P&L: Rs {total:+,.0f}  |  Avg: Rs {np.mean(net_pls):+,.0f}\n"
        f"Best: Rs {np.max(net_pls):+,.0f}  |  Worst: Rs {np.min(net_pls):+,.0f}  "
        f"|  MaxDD: Rs {max_dd:,.0f}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # Panel 2: Per-trade bars
    ax = axes[1]
    ax.set_title("Per-Trade Net P&L (Rs)", fontsize=12, fontweight='bold')
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in net_pls]
    trade_labels = [d.strftime("%m/%d") for d in dates]
    ax.bar(range(n), net_pls, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    step = max(1, n // 25)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([trade_labels[i] for i in range(0, n, step)], rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Net P&L (Rs)")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / f"rocket_equity_{ts}.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved equity curve to {out.name}")


def plot_monthly_returns(trades, ts):
    if not trades:
        return

    monthly = {}
    for t in trades:
        dt = datetime.strptime(t["date"], "%Y-%m-%d")
        key = (dt.year, dt.month)
        if key not in monthly:
            monthly[key] = {"pl_rs": 0, "trades": 0, "wins": 0}
        monthly[key]["pl_rs"] += t["net_pl_rs"]
        monthly[key]["trades"] += 1
        if t["net_pl_pct"] > 0:
            monthly[key]["wins"] += 1

    months = sorted(monthly.keys())
    labels = [f"{m[0]}-{m[1]:02d}" for m in months]
    values = [monthly[m]["pl_rs"] for m in months]
    bar_colors = ['#4CAF50' if v > 0 else '#F44336' for v in values]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.set_title("RocketData — Monthly Net Returns (Every Expiry Day, No DAX Filter)",
                 fontsize=14, fontweight='bold')

    bars = ax.bar(range(len(labels)), values, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    for i, (bar, m) in enumerate(zip(bars, months)):
        tc = monthly[m]["trades"]
        wc = monthly[m]["wins"]
        wr_m = wc / tc * 100 if tc > 0 else 0
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                y + (60 if y >= 0 else -150),
                f"{tc}t/{wr_m:.0f}%", ha='center',
                va='bottom' if y >= 0 else 'top', fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Net P&L (Rs)", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / f"rocket_monthly_{ts}.png"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved monthly chart to {out.name}")


if __name__ == "__main__":
    main()
