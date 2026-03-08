# -*- coding: utf-8 -*-
"""
DAX Gap -> Nifty Options Intraday Backtest  (v4 - Option Selling + Smart Filters)
===================================================================================
Strategy:
  1. Calculate DAX opening gap each morning.
  2. Optionally confirm gap with first DAX candle after open.
  3. If gap >= +0.25% (confirmed):
       BUY mode  -> buy Nifty ATM CE at 13:35 IST.
       SELL mode -> sell Nifty OTM PE at 13:35 IST (collect premium, theta+).
     If gap <= -0.25% (confirmed):
       BUY mode  -> buy Nifty ATM PE at 13:35 IST.
       SELL mode -> sell Nifty OTM CE at 13:35 IST.
     Else -> no trade.
  4. Exit at first of: TP hit / SL hit / 15:15 IST.

v4 changes from v3:
  1. SELL (option writing) mode: collect premium, profit from theta decay
     - Bullish signal -> sell OTM PE (win if Nifty stays up/flat)
     - Bearish signal -> sell OTM CE (win if Nifty stays down/flat)
  2. OTM strike offset support (0=ATM, 1=1-OTM, 2=2-OTM)
  3. Smart trade filters: IV cap, gap size limits
  4. Correct signal-aware Nifty move -> option mapping for cross-type trades
  5. 6-scenario matrix: 4 SELL variants + 2 optimised BUY variants

Author : GitHub Copilot
"""

# -- standard library ---------------------------------------------------------
import warnings
import logging
import math
import os
from datetime import datetime, time, timedelta, date as date_type
from typing import Optional, Tuple, List, Dict

# -- third-party --------------------------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf

# -- optional: TVDatafeed ------------------------------------------------------
try:
    from tvDatafeed import TvDatafeed, Interval as TvInterval
    _HAS_TVDATAFEED = True
except ImportError:
    _HAS_TVDATAFEED = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURATION
# =============================================================================

CFG = {
    # -- date range ------------------------------------------------------------
    "start_date": "2024-01-01",
    "end_date":   "2026-03-08",

    # -- strategy parameters ---------------------------------------------------
    "gap_threshold_pct": 0.15,      # lowered for more signals
    "entry_time_ist":    "13:35",
    "exit_time_ist":     "15:15",
    "nifty_strike_step": 50,

    # -- scenario matrix (v7 - expiry-day credit spreads) ----------------------
    # KEY INSIGHT: expiry-day spreads had 100% WR because theta crush is extreme.
    # All scenarios now focus on expiry day vs all days for comparison.
    # TP/SL are % of max_risk. Wider spreads = better cost efficiency.
    "scenarios": [
        # -- A: 2-wide (100pt), expiry-only, both signals -------------------
        {"name": "A1", "tp_pct": 25, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 2,
         "expiry_only": True},
        # -- B: 3-wide (150pt), expiry-only, both signals -------------------
        {"name": "B1", "tp_pct": 20, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 3,
         "expiry_only": True},
        {"name": "B2", "tp_pct": 25, "sl_type": "premium", "sl_pct": 100,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 3,
         "expiry_only": True},
        # -- C: 4-wide (200pt), expiry-only, both signals -------------------
        {"name": "C1", "tp_pct": 20, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 4,
         "expiry_only": True},
        # -- D: 2-wide, expiry-only, BEARISH-only (best signal) -------------
        {"name": "D1", "tp_pct": 25, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 2,
         "expiry_only": True, "signal_filter": ["Bearish"]},
        # -- E: 3-wide, expiry-only, BEARISH-only ---------------------------
        {"name": "E1", "tp_pct": 20, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 3,
         "expiry_only": True, "signal_filter": ["Bearish"]},
        # -- F: 4-wide ALL days (reference, was profitable in prev run) ------
        {"name": "F1", "tp_pct": 20, "sl_type": "premium", "sl_pct": 80,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 4},
        # -- G: 4-wide, expiry-only, no SL (let theta do the work) ----------
        {"name": "G1", "tp_pct": 20, "sl_type": "premium", "sl_pct": 100,
         "trade_mode": "SPREAD", "strike_offset": 0, "spread_width_strikes": 4,
         "expiry_only": True},
    ],

    # -- cost model ------------------------------------------------------------
    "brokerage_per_side_rs": 20.0,      # discount broker (Zerodha/Groww)
    "stt_pct_of_premium":    0.05,
    "gst_pct_on_brokerage":  18.0,
    "sebi_charges_per_crore": 10.0,
    "slippage_pct_per_side":  0.5,
    "lot_size": 25,

    # -- option pricing defaults -----------------------------------------------
    "implied_vol":     0.16,
    "risk_free_rate":  0.065,
    "session_hours":   6.25,

    # -- dynamic IV estimation -------------------------------------------------
    "iv_rolling_window": 20,
    "iv_premium_factor": 1.15,
    "iv_open_boost":     1.08,
    "iv_close_decay":    0.95,
    "iv_min":            0.08,
    "iv_max":            0.40,

    # -- filters ---------------------------------------------------------------
    "min_move_pct_filter":   0.1,          # lowered for OTM sensitivity
    "min_option_premium_rs": 5.0,          # allow very cheap options
    "candle_sl_fallback_points": 40,

    # -- trade filters (v6) ----------------------------------------------------
    "filters": {
        "iv_max": 0.30,          # more permissive for OTM buying
        "gap_max_pct": 3.00,     # allow bigger gaps for more signals
    },

    # -- DAX signal confirmation -----------------------------------------------
    "dax_confirmation_enabled": False,     # disabled: more trades

    # -- stochastic simulation -------------------------------------------------
    "sim_num_paths": 200,

    # -- selling parameters ----------------------------------------------------
    "sell_margin_per_lot_rs": 150000,

    # -- data sources ----------------------------------------------------------
    "intraday_csv_dir": "data/nifty_5min",
    "data_source": "auto",
}

# =============================================================================
#  MATH HELPERS (module-level for reuse)
# =============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# =============================================================================
#  MODULE 1 -- DATA FETCHING
# =============================================================================

def fetch_dax_daily(start: str, end: str) -> pd.DataFrame:
    log.info("Fetching DAX daily data from yfinance ...")
    raw = yf.download("^GDAXI", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError("No DAX data returned. Check your internet connection.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw.index = pd.to_datetime(raw.index).normalize()
    log.info(f"  DAX rows fetched: {len(raw)}")
    return raw


def fetch_nifty_daily(start: str, end: str) -> pd.DataFrame:
    log.info("Fetching Nifty daily data from yfinance ...")
    raw = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError("No Nifty data returned.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw.index = pd.to_datetime(raw.index).normalize()
    log.info(f"  Nifty daily rows fetched: {len(raw)}")
    return raw


def _clamp_start_date(start: str, max_days_back: int) -> str:
    earliest = (datetime.now() - timedelta(days=max_days_back)).strftime("%Y-%m-%d")
    return max(start, earliest)


def fetch_dax_intraday(start: str, end: str) -> pd.DataFrame:
    log.info("Fetching DAX intraday data for signal confirmation ...")
    for interval, label, max_days in [("1h", "1h", 729), ("5m", "5m", 59)]:
        clamped = _clamp_start_date(start, max_days)
        if clamped >= end:
            continue
        try:
            raw = yf.download("^GDAXI", start=clamped, end=end,
                              interval=interval, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            raw.index = raw.index.tz_convert("Europe/Berlin")
            log.info(f"  DAX {label} rows fetched: {len(raw)}")
            return raw
        except Exception as exc:
            log.debug(f"  DAX {label} fetch failed: {exc}")
    log.warning("  No DAX intraday data available -- signal confirmation disabled.")
    return pd.DataFrame()


def load_nifty_intraday_from_csv(start: str, end: str) -> pd.DataFrame:
    csv_dir = CFG.get("intraday_csv_dir", "data/nifty_5min")
    if not os.path.isdir(csv_dir):
        log.info(f"  Local CSV directory not found: {csv_dir} (skipping)")
        return pd.DataFrame()

    all_dfs = []
    for f in sorted(os.listdir(csv_dir)):
        if not f.lower().endswith(".csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(csv_dir, f), parse_dates=True, index_col=0)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        log.info("  No CSV files loaded from local directory.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs).sort_index()
    combined.index = pd.to_datetime(combined.index)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("Asia/Kolkata")
    else:
        combined.index = combined.index.tz_convert("Asia/Kolkata")

    start_ts = pd.Timestamp(start, tz="Asia/Kolkata")
    end_ts   = pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1)
    combined = combined[(combined.index >= start_ts) & (combined.index < end_ts)]
    log.info(f"  Local CSV Nifty rows loaded: {len(combined)}")
    return combined


def fetch_nifty_intraday_tvdatafeed(start: str, end: str) -> pd.DataFrame:
    if not _HAS_TVDATAFEED:
        return pd.DataFrame()
    log.info("Fetching Nifty 5-min data from TVDatafeed ...")
    try:
        tv = TvDatafeed()
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        days = (end_dt - start_dt).days
        n_bars = min(days * 75, 50000)

        df = tv.get_hist("NIFTY", "NSE", interval=TvInterval.in_5_minute, n_bars=n_bars)
        if df is None or df.empty:
            log.warning("  TVDatafeed returned no data.")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")

        col_map = {"open": "Open", "high": "High", "low": "Low",
                   "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)

        start_ts = pd.Timestamp(start, tz="Asia/Kolkata")
        end_ts   = pd.Timestamp(end, tz="Asia/Kolkata") + pd.Timedelta(days=1)
        df = df[(df.index >= start_ts) & (df.index < end_ts)]
        log.info(f"  TVDatafeed Nifty rows: {len(df)}")
        return df
    except Exception as exc:
        log.warning(f"  TVDatafeed failed: {exc}")
        return pd.DataFrame()


def fetch_nifty_intraday_1h(start: str, end: str) -> pd.DataFrame:
    clamped_start = _clamp_start_date(start, 729)
    if clamped_start >= end:
        log.info("  1h data: date range entirely outside 730-day yfinance window.")
        return pd.DataFrame()
    log.info(f"Fetching Nifty 1-hour data from yfinance ({clamped_start} -> {end}) ...")
    try:
        raw = yf.download("^NSEI", start=clamped_start, end=end, interval="1h",
                          progress=False, auto_adjust=True)
        if raw.empty:
            log.warning("  yfinance returned no 1h data.")
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert("Asia/Kolkata")
        log.info(f"  Nifty 1h rows fetched: {len(raw)}")
        return raw
    except Exception as exc:
        log.warning(f"  yfinance 1h fetch failed: {exc}")
        return pd.DataFrame()


def fetch_nifty_intraday_5m(start: str, end: str) -> pd.DataFrame:
    clamped_start = _clamp_start_date(start, 59)
    if clamped_start >= end:
        log.info("  5m data: date range entirely outside 60-day yfinance window.")
        return pd.DataFrame()
    log.info(f"Fetching Nifty 5-min data from yfinance ({clamped_start} -> {end}) ...")
    try:
        raw = yf.download("^NSEI", start=clamped_start, end=end, interval="5m",
                          progress=False, auto_adjust=True)
        if raw.empty:
            log.warning("  yfinance returned no 5m data.")
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert("Asia/Kolkata")
        log.info(f"  Nifty 5m rows fetched: {len(raw)}")
        return raw
    except Exception as exc:
        log.warning(f"  yfinance 5m fetch failed: {exc}")
        return pd.DataFrame()


_bhavcopy_cache: dict = {}


def fetch_nse_bhavcopy_option(trade_date: date_type, symbol: str = "NIFTY") -> Optional[pd.DataFrame]:
    import requests
    import zipfile
    import io

    if trade_date in _bhavcopy_cache:
        return _bhavcopy_cache[trade_date]

    try:
        fmt = trade_date.strftime("%Y%m%d")
        url = (
            f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{fmt}_F_0000.csv.zip"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer":    "https://www.nseindia.com",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            _bhavcopy_cache[trade_date] = None
            return None

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        df = pd.read_csv(zf.open(zf.namelist()[0]))

        result = df[
            (df["TckrSymb"] == symbol) &
            (df["OptnTp"].isin(["CE", "PE"]))
        ].copy()

        _bhavcopy_cache[trade_date] = result if not result.empty else None
        return _bhavcopy_cache[trade_date]

    except Exception as exc:
        log.debug(f"  NSE bhavcopy fetch failed for {trade_date}: {exc}")
        _bhavcopy_cache[trade_date] = None
        return None


def build_intraday_map(intraday_df: pd.DataFrame) -> dict:
    if intraday_df.empty:
        return {}
    grouped = {}
    for day, grp in intraday_df.groupby(intraday_df.index.date):
        grouped[day] = grp.sort_index()
    return grouped


# =============================================================================
#  MODULE 2 -- SIGNAL GENERATION
# =============================================================================

def compute_dax_gap(dax_df: pd.DataFrame) -> pd.DataFrame:
    df = dax_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df["DAX_Prev_Close"] = df["Close"].shift(1)
    df["DAX_Open"]       = df["Open"]
    df["DAX_High"]       = df["High"]
    df["DAX_Low"]        = df["Low"]
    df["DAX_Gap_Pct"]    = (df["DAX_Open"] - df["DAX_Prev_Close"]) / df["DAX_Prev_Close"] * 100

    threshold = CFG["gap_threshold_pct"]
    df["Signal"] = "No Trade"
    df.loc[df["DAX_Gap_Pct"] >=  threshold, "Signal"] = "Bullish"
    df.loc[df["DAX_Gap_Pct"] <= -threshold, "Signal"] = "Bearish"

    df = df.dropna(subset=["DAX_Prev_Close"])
    return df[["DAX_Prev_Close", "DAX_Open", "DAX_High", "DAX_Low", "DAX_Gap_Pct", "Signal"]]


def confirm_dax_signal(signal: str, dax_day_bars: Optional[pd.DataFrame],
                       dax_open: float) -> bool:
    if not CFG.get("dax_confirmation_enabled", True):
        return True
    if dax_day_bars is None or dax_day_bars.empty:
        return True

    first_bar = dax_day_bars.iloc[0]

    if len(dax_day_bars) >= 2:
        bar_diff_min = (dax_day_bars.index[1] - dax_day_bars.index[0]).total_seconds() / 60
    else:
        bar_diff_min = 60

    if bar_diff_min <= 10:
        tolerance = dax_open * 0.001
        if signal == "Bullish":
            return float(first_bar["Low"]) >= (dax_open - tolerance)
        elif signal == "Bearish":
            return float(first_bar["High"]) <= (dax_open + tolerance)
    else:
        if signal == "Bullish":
            return float(first_bar["Close"]) >= dax_open
        elif signal == "Bearish":
            return float(first_bar["Close"]) <= dax_open
    return True


def get_atm_strike(spot_price: float, step: int = 50) -> int:
    return int(round(spot_price / step) * step)


def get_otm_strike(spot_price: float, option_type: str,
                   offset: int = 1, step: int = 50) -> int:
    """Get OTM strike: CE goes higher, PE goes lower."""
    atm = get_atm_strike(spot_price, step)
    if option_type == "CE":
        return atm + offset * step
    else:
        return atm - offset * step


# =============================================================================
#  MODULE 3 -- OPTION PRICING & IMPLIED VOLATILITY
# =============================================================================

def estimate_implied_volatility(nifty_daily: pd.DataFrame,
                                trade_date: pd.Timestamp) -> float:
    try:
        idx = nifty_daily.index.get_loc(trade_date)
    except KeyError:
        return CFG["implied_vol"]

    window = CFG.get("iv_rolling_window", 20)
    if idx < window:
        return CFG["implied_vol"]

    closes = nifty_daily["Close"].iloc[max(0, idx - window):idx].values.astype(float)
    if len(closes) < 5:
        return CFG["implied_vol"]

    log_returns = np.diff(np.log(closes))
    if len(log_returns) < 3:
        return CFG["implied_vol"]

    realized_vol = float(np.std(log_returns, ddof=1) * np.sqrt(252))
    iv = realized_vol * CFG.get("iv_premium_factor", 1.15)
    iv *= CFG.get("iv_open_boost", 1.08)

    iv = max(CFG.get("iv_min", 0.08), min(iv, CFG.get("iv_max", 0.40)))
    return round(iv, 4)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "CE") -> float:
    if T <= 0 or sigma <= 0:
        if option_type == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sq_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq_T)
    d2 = d1 - sigma * sq_T

    if option_type == "CE":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(price, 0.0)


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "CE") -> dict:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.5 if option_type == "CE" else -0.5,
                "gamma": 0.0, "theta": 0.0}

    sq_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq_T)
    d2 = d1 - sigma * sq_T

    Nd1  = _norm_cdf(d1)
    Nnd1 = _norm_pdf(d1)
    gamma = Nnd1 / (S * sigma * sq_T)

    if option_type == "CE":
        delta = Nd1
        Nd2 = _norm_cdf(d2)
        theta = (-(S * Nnd1 * sigma) / (2.0 * sq_T)
                 - r * K * math.exp(-r * T) * Nd2) / 252.0
    else:
        delta = Nd1 - 1.0
        Nmd2 = _norm_cdf(-d2)
        theta = (-(S * Nnd1 * sigma) / (2.0 * sq_T)
                 + r * K * math.exp(-r * T) * Nmd2) / 252.0

    return {"delta": delta, "gamma": gamma, "theta": theta}


def estimate_option_price_at_entry(
    S: float, K: float, T: float, sigma: float, r: float,
    option_type: str, bhav_ref_price: Optional[float] = None,
) -> float:
    bs_price = _bs_price(S, K, T, r, sigma, option_type)
    bs_price = max(bs_price, 0.1)

    if bhav_ref_price is not None and bhav_ref_price > 0:
        deviation = abs(bs_price - bhav_ref_price) / bhav_ref_price
        if deviation > 0.50:
            blended = 0.6 * bs_price + 0.4 * bhav_ref_price
            return blended
    return bs_price


def simulate_option_premium_move_pct(
    nifty_entry: float,
    nifty_move: float,
    option_entry_price: float,
    option_type: str = "CE",
    days_to_expiry: float = 7.0,
    time_held_fraction: float = 0.5,
    sigma: Optional[float] = None,
) -> float:
    if option_entry_price is None or option_entry_price <= 0:
        option_entry_price = max(nifty_entry * 0.005, 50.0)

    if sigma is None:
        sigma = CFG["implied_vol"]
    r = CFG["risk_free_rate"]

    T = max(days_to_expiry / 252.0, 0.0001)
    K = get_atm_strike(nifty_entry, CFG["nifty_strike_step"])

    greeks = _bs_greeks(nifty_entry, K, T, r, sigma, option_type)
    delta  = greeks["delta"]
    gamma  = greeks["gamma"]
    theta  = greeks["theta"]

    delta_premium = delta * nifty_move + 0.5 * gamma * nifty_move ** 2
    theta_decay   = theta * time_held_fraction
    delta_premium_net = delta_premium + theta_decay

    new_premium  = max(option_entry_price + delta_premium_net, 0.0)
    new_premium  = min(new_premium, option_entry_price * 10.0)
    premium_move = new_premium - option_entry_price

    return (premium_move / option_entry_price) * 100


# =============================================================================
#  MODULE 4 -- PATH SIMULATION & ENTRY DATA
# =============================================================================

def simulate_intraday_path(
    open_: float, high_: float, low_: float, close_: float,
    entry_price: float, signal: str,
    num_paths: int = 200, num_steps: int = 20,
) -> dict:
    if high_ <= low_ or num_paths < 1:
        fav = max(high_ - entry_price, 0) if signal == "Bullish" else max(entry_price - low_, 0)
        adv = max(entry_price - low_, 0) if signal == "Bullish" else max(high_ - entry_price, 0)
        return {"max_favorable": fav, "max_adverse": adv}

    seed = int(abs(entry_price * 100 + open_ * 10 + close_)) % (2**31)
    rng = np.random.default_rng(seed)
    favs = np.empty(num_paths)
    advs = np.empty(num_paths)

    for p in range(num_paths):
        path = np.empty(num_steps + 1)
        path[0] = entry_price
        path[-1] = close_

        vol_per_step = (high_ - low_) * 0.04
        for i in range(1, num_steps):
            remaining = num_steps - i
            drift = (close_ - path[i - 1]) / remaining
            noise = rng.normal(0, vol_per_step)
            path[i] = np.clip(path[i - 1] + drift + noise, low_, high_)

        path_max = np.max(path)
        path_min = np.min(path)

        if signal == "Bullish":
            favs[p] = max(path_max - entry_price, 0.0)
            advs[p] = max(entry_price - path_min, 0.0)
        else:
            favs[p] = max(entry_price - path_min, 0.0)
            advs[p] = max(path_max - entry_price, 0.0)

    return {
        "max_favorable": float(np.median(favs)),
        "max_adverse":   float(np.median(advs)),
    }


def get_entry_data_from_intraday(
    day_bars: pd.DataFrame,
    entry_time_str: str,
    signal: str,
    bar_interval_min: int = 5,
) -> Optional[dict]:
    ref_date = day_bars.index[0]
    entry_time = pd.Timestamp(
        datetime.strptime(entry_time_str, "%H:%M").replace(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
        )
    ).tz_localize("Asia/Kolkata")

    exit_time = pd.Timestamp(
        datetime.strptime(CFG["exit_time_ist"], "%H:%M").replace(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
        )
    ).tz_localize("Asia/Kolkata")

    valid_bars = day_bars[day_bars.index <= entry_time]
    if valid_bars.empty:
        valid_bars = day_bars[
            (day_bars.index >= entry_time) &
            (day_bars.index < entry_time + pd.Timedelta(minutes=bar_interval_min + 1))
        ]
        if valid_bars.empty:
            return None

    entry_bar = valid_bars.iloc[-1]
    bar_start = valid_bars.index[-1]

    if bar_interval_min <= 5:
        entry_price = float(entry_bar["Close"])
    else:
        minutes_into_bar = max((entry_time - bar_start).total_seconds() / 60.0, 0)
        bar_fraction = min(minutes_into_bar / bar_interval_min, 1.0)
        bar_open  = float(entry_bar["Open"])
        bar_close = float(entry_bar["Close"])
        entry_price = bar_open + (bar_close - bar_open) * bar_fraction

    candle_high = float(entry_bar["High"])
    candle_low  = float(entry_bar["Low"])

    post_entry = day_bars[
        (day_bars.index > bar_start) &
        (day_bars.index <= exit_time)
    ]

    if post_entry.empty:
        return {
            "entry_price": entry_price, "candle_high": candle_high,
            "candle_low": candle_low, "max_favorable": 0.0, "max_adverse": 0.0,
            "post_highs": np.array([]), "post_lows": np.array([]),
            "is_daily_approx": False,
            "bar_interval": bar_interval_min,
        }

    post_highs = post_entry["High"].values.astype(float)
    post_lows  = post_entry["Low"].values.astype(float)

    if signal == "Bullish":
        max_favorable = float(np.max(post_highs) - entry_price)
        max_adverse   = float(entry_price - np.min(post_lows))
    else:
        max_favorable = float(entry_price - np.min(post_lows))
        max_adverse   = float(np.max(post_highs) - entry_price)

    return {
        "entry_price":   entry_price,
        "candle_high":   candle_high,
        "candle_low":    candle_low,
        "max_favorable": max(max_favorable, 0.0),
        "max_adverse":   max(max_adverse,   0.0),
        "post_highs":    post_highs,
        "post_lows":     post_lows,
        "is_daily_approx": False,
        "bar_interval":  bar_interval_min,
    }


def get_entry_data_from_daily(
    nifty_daily_row: pd.Series,
    signal: str,
) -> dict:
    try:
        open_  = float(nifty_daily_row["Open"])
        high_  = float(nifty_daily_row["High"])
        low_   = float(nifty_daily_row["Low"])
        close_ = float(nifty_daily_row["Close"])
    except Exception:
        open_  = float(nifty_daily_row.iloc[0])
        high_  = float(nifty_daily_row.iloc[1])
        low_   = float(nifty_daily_row.iloc[2])
        close_ = float(nifty_daily_row.iloc[3])

    ENTRY_FRACTION = 260.0 / 375.0
    entry_price = open_ + (close_ - open_) * ENTRY_FRACTION
    entry_price = max(low_, min(entry_price, high_))

    sim = simulate_intraday_path(
        open_, high_, low_, close_, entry_price, signal,
        num_paths=CFG.get("sim_num_paths", 200),
    )
    max_favorable = sim["max_favorable"]
    max_adverse   = sim["max_adverse"]

    fallback_pts = CFG["candle_sl_fallback_points"]

    return {
        "entry_price":     entry_price,
        "candle_high":     entry_price + fallback_pts,
        "candle_low":      entry_price - fallback_pts,
        "max_favorable":   max_favorable,
        "max_adverse":     max_adverse,
        "post_highs":      np.array([high_]),
        "post_lows":       np.array([low_]),
        "is_daily_approx": True,
        "daily_open":      open_,
        "daily_close":     close_,
        "bar_interval":    375,
    }


def get_option_entry_price_from_bhavcopy(
    bhav_df: pd.DataFrame,
    strike: int,
    option_type: str,
    trade_date: Optional[date_type] = None,
) -> Tuple[Optional[float], Optional[date_type]]:
    if bhav_df is None or bhav_df.empty:
        return None, None

    sub = bhav_df[
        (bhav_df["StrkPric"].astype(float).round(0).astype(int) == strike) &
        (bhav_df["OptnTp"] == option_type)
    ].copy()

    if sub.empty:
        return None, None

    sub["XpryDt"] = pd.to_datetime(sub["XpryDt"], errors="coerce")
    if trade_date is not None:
        trade_ts = pd.Timestamp(trade_date)
        sub = sub[sub["XpryDt"] >= trade_ts]
    if sub.empty:
        return None, None

    sub = sub.sort_values("XpryDt")
    row         = sub.iloc[0]
    expiry_date = row["XpryDt"].date() if pd.notna(row["XpryDt"]) else None

    price = float(row["OpnPric"])
    if price <= 0:
        price = float(row["ClsPric"])
    return (price if price > 0 else None), expiry_date


# =============================================================================
#  MODULE 5 -- TRADE SIMULATION
# =============================================================================

def compute_transaction_costs_pct(option_entry_price: float, lot_size: int) -> float:
    premium  = option_entry_price
    qty      = lot_size
    capital  = premium * qty

    brokerage = CFG["brokerage_per_side_rs"] * 2
    stt       = (CFG["stt_pct_of_premium"] / 100.0) * capital
    gst       = (CFG["gst_pct_on_brokerage"] / 100.0) * brokerage
    sebi      = (CFG["sebi_charges_per_crore"] / 1e7) * capital * 2
    slippage  = (CFG["slippage_pct_per_side"] / 100.0) * capital * 2

    total_cost_rs = brokerage + stt + gst + sebi + slippage
    cost_pct = (total_cost_rs / capital) * 100.0 if capital > 0 else 5.0
    return round(cost_pct, 4)


def compute_spread_transaction_costs_rs(
    sold_premium: float, bought_premium: float, lot_size: int,
) -> float:
    """Transaction costs in Rs for a 2-leg spread (4 sides total)."""
    brokerage = CFG["brokerage_per_side_rs"] * 4  # 2 legs × 2 sides
    capital_sold   = sold_premium * lot_size
    capital_bought = bought_premium * lot_size
    stt = (CFG["stt_pct_of_premium"] / 100.0) * (capital_sold + capital_bought)
    gst = (CFG["gst_pct_on_brokerage"] / 100.0) * brokerage
    total_capital = capital_sold + capital_bought
    sebi = (CFG["sebi_charges_per_crore"] / 1e7) * total_capital * 2
    slippage = (CFG["slippage_pct_per_side"] / 100.0) * total_capital * 2
    return brokerage + stt + gst + sebi + slippage


def simulate_spread_trade(
    entry_data: dict,
    signal: str,
    sold_premium: float,
    bought_premium: float,
    sold_strike: int,
    bought_strike: int,
    scenario: dict,
    option_type: str = "CE",
    days_to_expiry: float = 7.0,
    sigma: Optional[float] = None,
) -> dict:
    """
    Credit spread simulator.
    Sells near-strike, buys far-strike to cap risk.
    P&L computed as % of max_risk (= spread_width*lot - net_credit*lot).
    """
    tp_pct  = scenario["tp_pct"]
    sl_pct  = scenario["sl_pct"]
    lot_size = CFG["lot_size"]
    step = CFG["nifty_strike_step"]

    nifty_entry   = entry_data["entry_price"]
    max_favorable = entry_data["max_favorable"]
    max_adverse   = entry_data["max_adverse"]

    if sigma is None:
        sigma = CFG["implied_vol"]
    r = CFG["risk_free_rate"]
    T = max(days_to_expiry / 252.0, 0.0001)
    TIME_HELD = 100.0 / 375.0

    # Net credit collected per unit
    net_credit = sold_premium - bought_premium
    if net_credit <= 0:
        return {
            "gross_pl_pct": 0.0, "net_pl_pct": 0.0,
            "exit_type": "Skipped_NoCredit",
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "sl_trigger_pct": 0.0, "est_premium": 0.0,
            "transaction_cost_pct": 0.0,
            "net_credit_rs": 0.0, "max_risk_rs": 0.0,
            "spread_capital_rs": 0.0,
        }

    spread_width = abs(bought_strike - sold_strike)
    max_loss_per_unit = spread_width - net_credit  # max loss per option unit
    max_risk_rs = max_loss_per_unit * lot_size     # total capital at risk per lot
    net_credit_rs = net_credit * lot_size

    if max_risk_rs < 200.0:
        return {
            "gross_pl_pct": 0.0, "net_pl_pct": 0.0,
            "exit_type": "Skipped_TinyRisk",
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "sl_trigger_pct": 0.0, "est_premium": 0.0,
            "transaction_cost_pct": 0.0,
            "net_credit_rs": round(net_credit_rs, 2),
            "max_risk_rs": round(max_risk_rs, 2),
            "spread_capital_rs": round(max_risk_rs, 2),
        }

    # Convert signal-relative moves to absolute Nifty moves
    if signal == "Bullish":
        actual_nifty_rise = max_favorable
        actual_nifty_fall = max_adverse
    else:
        actual_nifty_rise = max_adverse
        actual_nifty_fall = max_favorable

    # Compute both legs at favorable & adverse Nifty extremes
    # For CE spread (bear call): sold CE + bought CE (higher strike)
    #   favorable = Nifty falls -> both CEs lose value -> spread gains
    #   adverse = Nifty rises -> both CEs gain value -> spread loses
    # For PE spread (bull put): sold PE + bought PE (lower strike)
    #   favorable = Nifty rises -> both PEs lose value -> spread gains
    #   adverse = Nifty falls -> both PEs gain value -> spread loses

    if option_type == "CE":
        fav_move = -actual_nifty_fall    # CE profits (seller) when nifty falls
        adv_move = actual_nifty_rise     # CE loses (seller) when nifty rises
    else:
        fav_move = actual_nifty_rise     # PE profits (seller) when nifty rises
        adv_move = -actual_nifty_fall    # PE loses (seller) when nifty falls

    # Price both legs at entry, favorable exit, adverse exit
    T_exit = max((days_to_expiry - TIME_HELD) / 252.0, 0.0001)
    sigma_exit = sigma * CFG.get("iv_close_decay", 0.95)

    # Favorable scenario
    S_fav = nifty_entry + fav_move
    sold_fav  = _bs_price(S_fav, sold_strike,   T_exit, r, sigma_exit, option_type)
    bought_fav = _bs_price(S_fav, bought_strike, T_exit, r, sigma_exit, option_type)
    spread_fav = sold_fav - bought_fav
    # Clamp spread values to [0, spread_width] (theoretical bounds)
    spread_fav = max(0.0, min(spread_fav, spread_width))
    fav_pl_per_unit = net_credit - spread_fav  # positive = profit

    # Adverse scenario
    S_adv = nifty_entry + adv_move
    sold_adv  = _bs_price(S_adv, sold_strike,   T_exit, r, sigma_exit, option_type)
    bought_adv = _bs_price(S_adv, bought_strike, T_exit, r, sigma_exit, option_type)
    spread_adv = sold_adv - bought_adv
    spread_adv = max(0.0, min(spread_adv, spread_width))
    adv_pl_per_unit = net_credit - spread_adv  # negative = loss

    # Time exit (no move, just theta)
    sold_time  = _bs_price(nifty_entry, sold_strike,   T_exit, r, sigma_exit, option_type)
    bought_time = _bs_price(nifty_entry, bought_strike, T_exit, r, sigma_exit, option_type)
    spread_time = sold_time - bought_time
    spread_time = max(0.0, min(spread_time, spread_width))
    time_pl_per_unit = net_credit - spread_time

    # Convert to % of max_risk
    fav_pl_pct  = (fav_pl_per_unit * lot_size / max_risk_rs) * 100.0
    adv_pl_pct  = (adv_pl_per_unit * lot_size / max_risk_rs) * 100.0 if adv_pl_per_unit < 0 else 0.0
    time_pl_pct = (time_pl_per_unit * lot_size / max_risk_rs) * 100.0

    favorable_pct = max(fav_pl_pct, 0.0)
    adverse_pct   = abs(min(adv_pl_pct, 0.0))

    # Cap favorable at max possible (net_credit / max_risk * 100)
    max_profit_pct = (net_credit_rs / max_risk_rs) * 100.0
    favorable_pct = min(favorable_pct, max_profit_pct)

    # Cap adverse at 100% (max_risk)
    adverse_pct = min(adverse_pct, 100.0)

    # TP / SL check
    tp_hit = favorable_pct >= tp_pct
    sl_hit = adverse_pct >= sl_pct

    if tp_hit and sl_hit:
        if favorable_pct / max(adverse_pct, 0.001) >= tp_pct / max(sl_pct, 0.001):
            exit_type = "TP"
            gross_pl  = min(tp_pct, max_profit_pct)
        else:
            exit_type = "SL"
            gross_pl  = -sl_pct
    elif tp_hit:
        exit_type = "TP"
        gross_pl  = min(tp_pct, max_profit_pct)
    elif sl_hit:
        exit_type = "SL"
        gross_pl  = -sl_pct
    else:
        exit_type = "Time Exit"
        gross_pl  = time_pl_pct

    # Clamp
    gross_pl = max(gross_pl, -100.0)
    gross_pl = min(gross_pl, max_profit_pct)

    # Transaction costs as % of max_risk
    cost_rs = compute_spread_transaction_costs_rs(sold_premium, bought_premium, lot_size)
    cost_pct = (cost_rs / max_risk_rs) * 100.0

    # Skip trades where transaction costs eat too much of the capital
    if cost_pct > 30.0:
        return {
            "gross_pl_pct": 0.0, "net_pl_pct": 0.0,
            "exit_type": "Skipped_HighCost",
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "sl_trigger_pct": 0.0, "est_premium": 0.0,
            "transaction_cost_pct": round(cost_pct, 4),
            "net_credit_rs": round(net_credit_rs, 2),
            "max_risk_rs": round(max_risk_rs, 2),
            "spread_capital_rs": round(max_risk_rs, 2),
        }

    net_pl = gross_pl - cost_pct
    # Clamp net_pl: can never lose more than 100% of max_risk
    net_pl = max(net_pl, -100.0)

    return {
        "gross_pl_pct":         round(gross_pl, 2),
        "net_pl_pct":           round(net_pl, 2),
        "exit_type":            exit_type,
        "max_favorable_pct":    round(favorable_pct, 2),
        "max_adverse_pct":      round(adverse_pct, 2),
        "sl_trigger_pct":       round(sl_pct, 2),
        "est_premium":          round(net_credit, 2),
        "transaction_cost_pct": round(cost_pct, 4),
        "net_credit_rs":        round(net_credit_rs, 2),
        "max_risk_rs":          round(max_risk_rs, 2),
        "spread_capital_rs":    round(max_risk_rs, 2),
    }


def simulate_trade(
    entry_data: dict,
    signal: str,
    option_entry_price: float,
    scenario: dict,
    option_type: str = "CE",
    days_to_expiry: float = 7.0,
    sigma: Optional[float] = None,
    trade_mode: str = "BUY",
) -> dict:
    """
    Core trade simulator (v4).
    Supports both BUY and SELL modes.
    For SELL mode: computes buyer's P&L then mirrors it (seller = -buyer).
    """
    tp_pct  = scenario["tp_pct"]
    sl_type = scenario["sl_type"]
    sl_pct  = scenario["sl_pct"]

    nifty_entry   = entry_data["entry_price"]
    max_favorable = entry_data["max_favorable"]
    max_adverse   = entry_data["max_adverse"]
    candle_high   = entry_data["candle_high"]
    candle_low    = entry_data["candle_low"]
    post_highs    = entry_data["post_highs"]
    post_lows     = entry_data["post_lows"]

    est_premium = option_entry_price if (option_entry_price and option_entry_price > 0) \
                  else max(nifty_entry * 0.005, 50.0)
    lot_size = CFG["lot_size"]
    if sigma is None:
        sigma = CFG["implied_vol"]

    # -- Volatility filter -----------------------------------------------------
    total_range_pct = (max_favorable + max_adverse) / nifty_entry * 100.0
    if total_range_pct < CFG["min_move_pct_filter"]:
        return {
            "gross_pl_pct": 0.0, "net_pl_pct": 0.0,
            "exit_type": "Skipped_LowVol",
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "sl_trigger_pct": 0.0, "est_premium": round(est_premium, 2),
            "transaction_cost_pct": 0.0,
        }

    # -- Minimum premium filter ------------------------------------------------
    if est_premium < CFG.get("min_option_premium_rs", 30.0):
        return {
            "gross_pl_pct": 0.0, "net_pl_pct": 0.0,
            "exit_type": "Skipped_LowPrem",
            "max_favorable_pct": 0.0, "max_adverse_pct": 0.0,
            "sl_trigger_pct": 0.0, "est_premium": round(est_premium, 2),
            "transaction_cost_pct": 0.0,
        }

    TIME_HELD_FRACTION = 100.0 / 375.0  # 13:35->15:15 = 100 min / 375 min

    # -- Convert signal-relative moves to absolute Nifty spot moves ------------
    # max_favorable/max_adverse are from the SIGNAL's perspective:
    #   Bullish: max_favorable = nifty rose, max_adverse = nifty fell
    #   Bearish: max_favorable = nifty fell, max_adverse = nifty rose
    # We need to map to option-specific moves regardless of signal direction.
    if signal == "Bullish":
        actual_nifty_rise = max_favorable   # absolute: how much Nifty went UP
        actual_nifty_fall = max_adverse     # absolute: how much Nifty went DOWN
    else:  # Bearish
        actual_nifty_rise = max_adverse     # Nifty went UP (against bearish signal)
        actual_nifty_fall = max_favorable   # Nifty went DOWN (with bearish signal)

    # Map to signed nifty moves for option pricing
    if option_type == "CE":
        fav_nifty_move = actual_nifty_rise      # CE gains from nifty rising
        adv_nifty_move = -actual_nifty_fall     # CE loses from nifty falling
    else:  # PE
        fav_nifty_move = -actual_nifty_fall     # PE gains from nifty falling
        adv_nifty_move = actual_nifty_rise      # PE loses from nifty rising

    favorable_prem_pct = simulate_option_premium_move_pct(
        nifty_entry, fav_nifty_move, est_premium,
        option_type=option_type, days_to_expiry=days_to_expiry,
        time_held_fraction=TIME_HELD_FRACTION, sigma=sigma,
    )
    adverse_prem_pct = simulate_option_premium_move_pct(
        nifty_entry, adv_nifty_move, est_premium,
        option_type=option_type, days_to_expiry=days_to_expiry,
        time_held_fraction=TIME_HELD_FRACTION, sigma=sigma,
    )
    adverse_prem_pct = abs(adverse_prem_pct)

    # -- SELL mode: swap favorable/adverse (seller = -buyer) -------------------
    if trade_mode == "SELL":
        favorable_prem_pct, adverse_prem_pct = adverse_prem_pct, favorable_prem_pct

    # -- Compute actual delta for Nifty-to-premium conversions -----------------
    T = max(days_to_expiry / 252.0, 0.0001)
    K = get_atm_strike(nifty_entry, CFG["nifty_strike_step"])
    greeks = _bs_greeks(nifty_entry, K, T, CFG["risk_free_rate"], sigma, option_type)
    actual_delta = abs(greeks["delta"])
    actual_delta = max(actual_delta, 0.01)

    # -- Determine SL level ----------------------------------------------------
    if sl_type == "premium":
        sl_trigger_pct = sl_pct
    else:
        # Candle SL only applies to BUY mode
        if trade_mode == "SELL":
            sl_trigger_pct = sl_pct if sl_pct else 100.0
        else:
            if signal == "Bullish":
                candle_sl_nifty = max(nifty_entry - candle_low, 0.0)
            else:
                candle_sl_nifty = max(candle_high - nifty_entry, 0.0)

            if option_type == "CE":
                sl_nifty_signed = -candle_sl_nifty
            else:
                sl_nifty_signed = candle_sl_nifty
            sl_prem_pct_raw = simulate_option_premium_move_pct(
                nifty_entry, sl_nifty_signed, est_premium,
                option_type=option_type, days_to_expiry=days_to_expiry,
                time_held_fraction=0.1, sigma=sigma,
            )
            sl_trigger_pct = max(abs(sl_prem_pct_raw), 1.0)

    # -- Bar-by-bar candle SL check (BUY mode only) ----------------------------
    candle_sl_hit_first = False
    actual_candle_sl_pct = None

    if (sl_type == "candle" and trade_mode == "BUY"
            and len(post_lows) > 0 and len(post_highs) > 0):
        tp_nifty_equiv = tp_pct / 100.0 * est_premium / actual_delta

        for h, l in zip(post_highs, post_lows):
            if signal == "Bullish":
                if l < candle_low:
                    breach_dist = nifty_entry - l
                    sl_signed = -breach_dist if option_type == "CE" else breach_dist
                    actual_candle_sl_pct = abs(simulate_option_premium_move_pct(
                        nifty_entry, sl_signed, est_premium,
                        option_type=option_type, days_to_expiry=days_to_expiry,
                        time_held_fraction=0.1, sigma=sigma,
                    ))
                    candle_sl_hit_first = True
                    break
                if h >= nifty_entry + tp_nifty_equiv:
                    break
            else:
                if h > candle_high:
                    breach_dist = h - nifty_entry
                    sl_signed = breach_dist if option_type == "PE" else -breach_dist
                    actual_candle_sl_pct = abs(simulate_option_premium_move_pct(
                        nifty_entry, sl_signed, est_premium,
                        option_type=option_type, days_to_expiry=days_to_expiry,
                        time_held_fraction=0.1, sigma=sigma,
                    ))
                    candle_sl_hit_first = True
                    break
                if l <= nifty_entry - tp_nifty_equiv:
                    break

    if candle_sl_hit_first and actual_candle_sl_pct is not None:
        sl_trigger_pct = max(actual_candle_sl_pct, 1.0)

    # Cap SL: BUY max loss = 100% of premium; SELL capped at scenario SL
    if trade_mode == "BUY":
        sl_trigger_pct = min(sl_trigger_pct, 100.0)

    # -- Determine exit outcome ------------------------------------------------
    tp_hit = favorable_prem_pct >= tp_pct
    sl_hit = (adverse_prem_pct >= sl_trigger_pct) or candle_sl_hit_first

    if tp_hit and sl_hit:
        if favorable_prem_pct / max(adverse_prem_pct, 0.001) >= tp_pct / max(sl_trigger_pct, 0.001):
            exit_type = "TP"
            gross_pl  = tp_pct
        else:
            exit_type = "SL"
            gross_pl  = -sl_trigger_pct
    elif tp_hit:
        exit_type = "TP"
        gross_pl  = tp_pct
    elif sl_hit:
        exit_type = "SL"
        gross_pl  = -sl_trigger_pct
    else:
        exit_type = "Time Exit"
        gross_pl  = min(favorable_prem_pct, tp_pct)
        gross_pl  = gross_pl if gross_pl > adverse_prem_pct else -adverse_prem_pct

    # -- Transaction costs -----------------------------------------------------
    friction = compute_transaction_costs_pct(est_premium, lot_size)
    # Cap gross P&L: BUY max loss = 100%, SELL limited by SL
    if trade_mode == "BUY":
        gross_pl = max(gross_pl, -100.0)
    net_pl = gross_pl - friction

    return {
        "gross_pl_pct":         round(gross_pl,   2),
        "net_pl_pct":           round(net_pl,     2),
        "exit_type":            exit_type,
        "max_favorable_pct":    round(favorable_prem_pct, 2),
        "max_adverse_pct":      round(adverse_prem_pct,   2),
        "sl_trigger_pct":       round(sl_trigger_pct,     2),
        "est_premium":          round(est_premium, 2),
        "transaction_cost_pct": round(friction,   4),
    }


# =============================================================================
#  MODULE 6 -- PERFORMANCE ANALYTICS
# =============================================================================

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    return float(drawdown.min()) if len(drawdown) > 0 else 0.0


def compute_sharpe_ratio(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def compute_sortino_ratio(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_daily
    downside = returns[returns < 0]
    if len(downside) < 1:
        return float("inf") if excess.mean() > 0 else 0.0
    down_std = downside.std()
    if down_std == 0 or np.isnan(down_std):
        return 0.0
    return float(excess.mean() / down_std * np.sqrt(252))


def compute_max_consecutive_losses(returns: pd.Series) -> int:
    max_streak = 0
    current = 0
    for r in returns:
        if r <= 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_monthly_returns(trades_df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    _excl = ["No Trade", "Signal_Rejected"]
    df = trades_df[
        (trades_df["Scenario"] == scenario_name) &
        (~trades_df["Exit_Type"].isin(_excl)) &
        (~trades_df["Exit_Type"].str.startswith("Skipped", na=False))
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    monthly = df.groupby("Month").agg(
        Total_PL_Pct=("Net_PL_Pct", "sum"),
        Trades=("Net_PL_Pct", "count"),
        Avg_PL_Pct=("Net_PL_Pct", "mean"),
        Wins=("Net_PL_Pct", lambda x: (x > 0).sum()),
    ).reset_index()
    monthly["Win_Rate_%"] = (monthly["Wins"] / monthly["Trades"] * 100).round(1)
    return monthly


def expiry_vs_nonexpiry_stats(trades_df: pd.DataFrame, scenario_name: str) -> dict:
    _excl = ["No Trade", "Signal_Rejected"]
    df = trades_df[
        (trades_df["Scenario"] == scenario_name) &
        (~trades_df["Exit_Type"].isin(_excl)) &
        (~trades_df["Exit_Type"].str.startswith("Skipped", na=False))
    ].copy()

    result = {}
    for label, mask in [("Expiry", df["Is_Expiry_Day"] == True),
                        ("Non-Expiry", df["Is_Expiry_Day"] == False)]:
        sub = df[mask]
        n = len(sub)
        if n == 0:
            result[label] = {"Trades": 0, "Win_%": 0, "Avg_PL_%": 0}
            continue
        wins = (sub["Net_PL_Pct"] > 0).sum()
        result[label] = {
            "Trades":   n,
            "Win_%":    round(wins / n * 100, 1),
            "Avg_PL_%": round(sub["Net_PL_Pct"].mean(), 2),
            "Total_PL_%": round(sub["Net_PL_Pct"].sum(), 2),
        }
    return result


def performance_summary(trades_df: pd.DataFrame, scenario_name: str) -> dict:
    df = trades_df[trades_df["Scenario"] == scenario_name].copy()
    _excluded = ["No Trade", "Signal_Rejected"]
    trade_rows = df[
        (~df["Exit_Type"].isin(_excluded)) &
        (~df["Exit_Type"].str.startswith("Skipped", na=False))
    ]
    skipped   = df[df["Exit_Type"].str.startswith("Skipped", na=False)]
    rejected  = df[df["Exit_Type"] == "Signal_Rejected"]

    total   = len(trade_rows)
    wins    = len(trade_rows[trade_rows["Net_PL_Pct"] > 0])
    losses  = len(trade_rows[trade_rows["Net_PL_Pct"] <= 0])
    win_rate = wins / total * 100 if total > 0 else 0.0
    avg_profit = trade_rows[trade_rows["Net_PL_Pct"] > 0]["Net_PL_Pct"].mean() if wins > 0 else 0.0
    avg_loss   = trade_rows[trade_rows["Net_PL_Pct"] <= 0]["Net_PL_Pct"].mean() if losses > 0 else 0.0

    gross_profit  = trade_rows[trade_rows["Net_PL_Pct"] > 0]["Net_PL_Pct"].sum()
    gross_loss    = abs(trade_rows[trade_rows["Net_PL_Pct"] <= 0]["Net_PL_Pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve   = trade_rows["Net_PL_Pct"].cumsum().reset_index(drop=True)
    max_drawdown   = compute_max_drawdown(equity_curve)
    net_profit_pct = trade_rows["Net_PL_Pct"].sum()

    returns = trade_rows["Net_PL_Pct"].reset_index(drop=True)
    sharpe  = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    max_consec_losses = compute_max_consecutive_losses(returns)

    lot_size = CFG["lot_size"]
    has_spread = "Spread_Capital_Rs" in trade_rows.columns

    if has_spread and trade_rows["Spread_Capital_Rs"].notna().any():
        # Credit spread mode: capital per trade = max_risk (Spread_Capital_Rs)
        rs_pl = trade_rows.apply(
            lambda r: (r["Spread_Capital_Rs"] if pd.notna(r.get("Spread_Capital_Rs"))
                       else 1000.0) * r["Net_PL_Pct"] / 100.0,
            axis=1,
        )
        total_pl_rs      = rs_pl.sum()
        avg_pl_per_trade = rs_pl.mean() if len(rs_pl) > 0 else 0.0
        avg_capital      = trade_rows["Spread_Capital_Rs"].dropna().mean()
    elif "Option_Price" in trade_rows.columns:
        rs_pl = trade_rows.apply(
            lambda r: (r["Option_Price"] if pd.notna(r["Option_Price"]) else 50.0)
                      * lot_size * r["Net_PL_Pct"] / 100.0,
            axis=1,
        )
        total_pl_rs      = rs_pl.sum()
        avg_pl_per_trade = rs_pl.mean() if len(rs_pl) > 0 else 0.0
        avg_capital      = trade_rows["Option_Price"].dropna().mean() * lot_size
    else:
        total_pl_rs = avg_pl_per_trade = avg_capital = 0.0

    return {
        "Scenario":             scenario_name,
        "Active Trades":        total,
        "Skipped":              len(skipped),
        "Rejected":             len(rejected),
        "Wins":                 wins,
        "Losses":               losses,
        "Win Rate %":           round(win_rate, 2),
        "Avg Win %":            round(avg_profit, 2),
        "Avg Loss %":           round(avg_loss, 2),
        "Profit Factor":        round(profit_factor, 3),
        "Sharpe Ratio":         round(sharpe, 3),
        "Sortino Ratio":        round(sortino, 3),
        "Max Consec Losses":    max_consec_losses,
        "Avg P&L/Trade %":      round(net_profit_pct / total, 2) if total > 0 else 0.0,
        "Max Drawdown %":       round(max_drawdown, 2),
        "Avg Capital/Trade Rs": round(avg_capital),
        "Total P&L Rs":         round(total_pl_rs),
        "Avg P&L/Trade Rs":     round(avg_pl_per_trade),
    }


def print_summary_table(summaries: list) -> None:
    df = pd.DataFrame(summaries)
    print("\n" + "=" * 110)
    print("  BACKTEST SUMMARY - ALL SCENARIOS")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110 + "\n")


def print_monthly_table(monthly_df: pd.DataFrame, scenario_name: str) -> None:
    if monthly_df.empty:
        return
    print(f"\n--- MONTHLY RETURNS: {scenario_name} ---")
    print(monthly_df.to_string(index=False))


def print_expiry_stats(trades_df: pd.DataFrame, scenario_names: list) -> None:
    print("\n--- EXPIRY vs NON-EXPIRY PERFORMANCE ---")
    for scen in scenario_names:
        stats = expiry_vs_nonexpiry_stats(trades_df, scen)
        print(f"  {scen}:")
        for label, s in stats.items():
            print(f"    {label:12s}: {s.get('Trades', 0)} trades, "
                  f"Win {s.get('Win_%', 0)}%, "
                  f"Avg P&L {s.get('Avg_PL_%', 0):.2f}%, "
                  f"Total P&L {s.get('Total_PL_%', 0):.2f}%")
    print()


# =============================================================================
#  MODULE 7 -- MAIN BACKTEST RUNNER
# =============================================================================

def run_backtest() -> pd.DataFrame:
    start = CFG["start_date"]
    end   = CFG["end_date"]

    # -- Step 1: Fetch all data sources ----------------------------------------
    dax_daily    = fetch_dax_daily(start, end)
    nifty_daily  = fetch_nifty_daily(start, end)

    if isinstance(nifty_daily.columns, pd.MultiIndex):
        nifty_daily.columns = [col[0] for col in nifty_daily.columns]

    # Intraday data: priority chain
    intraday_5m = load_nifty_intraday_from_csv(start, end)

    if intraday_5m.empty:
        intraday_5m = fetch_nifty_intraday_tvdatafeed(start, end)

    yf_5m = fetch_nifty_intraday_5m(start, end)
    if not yf_5m.empty:
        if intraday_5m.empty:
            intraday_5m = yf_5m
        else:
            yf_dates = set(yf_5m.index.date)
            existing_dates = set(intraday_5m.index.date)
            new_dates = yf_dates - existing_dates
            if new_dates:
                new_data = yf_5m[yf_5m.index.normalize().isin(
                    [pd.Timestamp(d, tz="Asia/Kolkata") for d in new_dates]
                )]
                if not new_data.empty:
                    intraday_5m = pd.concat([intraday_5m, new_data]).sort_index()

    intraday_1h = fetch_nifty_intraday_1h(start, end)

    dax_intraday = fetch_dax_intraday(start, end)

    map_5m  = build_intraday_map(intraday_5m)
    map_1h  = build_intraday_map(intraday_1h)
    map_dax = build_intraday_map(dax_intraday)

    # -- Step 2: Compute DAX signals -------------------------------------------
    signal_df = compute_dax_gap(dax_daily)

    # -- Step 3: Align trading days --------------------------------------------
    nifty_daily_aligned = nifty_daily.reindex(signal_df.index, method=None)
    common_idx = signal_df.index.intersection(nifty_daily_aligned.dropna(how="all").index)
    signal_df  = signal_df.loc[common_idx]

    n_5m  = sum(1 for d in common_idx if d.date() in map_5m)
    n_1h  = sum(1 for d in common_idx if d.date() in map_1h and d.date() not in map_5m)
    n_daily = len(common_idx) - n_5m - n_1h

    log.info(f"Total trading days analysed : {len(signal_df)}")
    log.info(f"  Days with 5-min data      : {n_5m}")
    log.info(f"  Days with 1-hour data     : {n_1h}")
    log.info(f"  Days using daily fallback : {n_daily}")

    # -- Step 4: Trade loop with per-scenario option handling ------------------
    all_rows = []
    confirmed_count = 0
    rejected_count  = 0
    filters = CFG.get("filters", {})

    for trade_date, sig_row in signal_df.iterrows():
        date_obj = trade_date.date()
        signal   = sig_row["Signal"]
        gap_pct  = sig_row["DAX_Gap_Pct"]
        dax_prev = sig_row["DAX_Prev_Close"]
        dax_open = sig_row["DAX_Open"]

        # -- DAX signal confirmation -------------------------------------------
        signal_confirmed = True
        if signal != "No Trade" and CFG.get("dax_confirmation_enabled", True):
            dax_day = map_dax.get(date_obj)
            signal_confirmed = confirm_dax_signal(signal, dax_day, float(dax_open))
            if signal_confirmed:
                confirmed_count += 1
            else:
                rejected_count += 1

        # -- Per-day data (shared across scenarios) ----------------------------
        entry_data = None
        data_source_used = "-"
        sigma_used = CFG["implied_vol"]
        nifty_entry = None
        atm_strike = None
        bhav_df = None

        if signal != "No Trade" and signal_confirmed:
            # Determine intraday data source
            if date_obj in map_5m:
                entry_data = get_entry_data_from_intraday(
                    map_5m[date_obj], CFG["entry_time_ist"], signal,
                    bar_interval_min=5,
                )
                data_source_used = "5min"
            elif date_obj in map_1h:
                entry_data = get_entry_data_from_intraday(
                    map_1h[date_obj], CFG["entry_time_ist"], signal,
                    bar_interval_min=60,
                )
                data_source_used = "1hour"
            else:
                entry_data = None
                data_source_used = "Daily"

            if entry_data is None:
                if trade_date in nifty_daily.index:
                    entry_data = get_entry_data_from_daily(
                        nifty_daily.loc[trade_date], signal
                    )
                    data_source_used = "Daily"
                else:
                    continue

            nifty_entry = entry_data["entry_price"]
            atm_strike  = get_atm_strike(nifty_entry, CFG["nifty_strike_step"])

            # Dynamic IV estimation
            sigma_used = estimate_implied_volatility(nifty_daily, trade_date)

            # Fetch bhavcopy once per day (cached)
            if CFG["data_source"] in ("auto", "nse_bhav"):
                bhav_df = fetch_nse_bhavcopy_option(date_obj)

        # -- Per-scenario processing -------------------------------------------
        for scenario in CFG["scenarios"]:
            trade_mode    = scenario.get("trade_mode", "BUY")
            strike_offset = scenario.get("strike_offset", 0)

            row = {
                "Date":           trade_date.strftime("%Y-%m-%d"),
                "DAX_Prev_Close": round(dax_prev, 2),
                "DAX_Open":       round(dax_open, 2),
                "DAX_Gap_Pct":    round(gap_pct, 3),
                "Signal":         signal,
                "Scenario":       scenario["name"],
                "Trade_Mode":     trade_mode,
                "TP_%":           scenario["tp_pct"],
                "SL_Type":        scenario["sl_type"],
                "SL_%":           scenario["sl_pct"] if scenario["sl_pct"] else "Candle",
            }

            if signal == "No Trade" or not signal_confirmed or entry_data is None:
                row.update({
                    "Nifty_Entry": None, "Strike_Used": None,
                    "Trade_Type": "-", "Option_Price": None,
                    "BS_Entry_Price": None, "Bhav_Ref_Price": None,
                    "IV_Used": None,
                    "Expiry_Date": None, "Days_To_Expiry": None,
                    "Is_Expiry_Day": None,
                    "Data_Source": "-", "Data_Quality": "-",
                    "Signal_Confirmed": signal_confirmed if signal != "No Trade" else None,
                    "Max_Favorable_%": None, "Max_Adverse_%": None,
                    "SL_Trigger_%": None, "Transaction_Cost_%": None,
                    "Exit_Type": "No Trade" if signal == "No Trade" else "Signal_Rejected",
                    "Gross_PL_Pct": 0.0, "Net_PL_Pct": 0.0,
                })
                all_rows.append(row)
                continue

            # -- Per-scenario signal filter ------------------------------------
            signal_filter = scenario.get("signal_filter")
            if signal_filter and signal not in signal_filter:
                row.update({
                    "Nifty_Entry": round(nifty_entry, 2),
                    "Strike_Used": None, "Trade_Type": "-",
                    "Option_Price": None, "BS_Entry_Price": None,
                    "Bhav_Ref_Price": None, "IV_Used": round(sigma_used, 4),
                    "Expiry_Date": None, "Days_To_Expiry": None,
                    "Is_Expiry_Day": None,
                    "Data_Source": data_source_used, "Data_Quality": "-",
                    "Signal_Confirmed": signal_confirmed,
                    "Max_Favorable_%": 0.0, "Max_Adverse_%": 0.0,
                    "SL_Trigger_%": 0.0, "Transaction_Cost_%": 0.0,
                    "Exit_Type": "Skipped_SignalFilter",
                    "Gross_PL_Pct": 0.0, "Net_PL_Pct": 0.0,
                })
                all_rows.append(row)
                continue

            # -- Per-scenario day-of-week filter -------------------------------
            skip_days = scenario.get("skip_days")
            if skip_days and date_obj.weekday() in skip_days:
                row.update({
                    "Nifty_Entry": round(nifty_entry, 2),
                    "Strike_Used": None, "Trade_Type": "-",
                    "Option_Price": None, "BS_Entry_Price": None,
                    "Bhav_Ref_Price": None, "IV_Used": round(sigma_used, 4),
                    "Expiry_Date": None, "Days_To_Expiry": None,
                    "Is_Expiry_Day": None,
                    "Data_Source": data_source_used, "Data_Quality": "-",
                    "Signal_Confirmed": signal_confirmed,
                    "Max_Favorable_%": 0.0, "Max_Adverse_%": 0.0,
                    "SL_Trigger_%": 0.0, "Transaction_Cost_%": 0.0,
                    "Exit_Type": "Skipped_DayFilter",
                    "Gross_PL_Pct": 0.0, "Net_PL_Pct": 0.0,
                })
                all_rows.append(row)
                continue

            # -- Determine option type based on trade_mode ---------------------
            # BUY: Bullish->CE, Bearish->PE (profit from direction)
            # SELL/SPREAD: Bullish->PE, Bearish->CE (sell opposite, profit from theta)
            if trade_mode == "BUY":
                option_type = "CE" if signal == "Bullish" else "PE"
            else:  # SELL or SPREAD
                option_type = "PE" if signal == "Bullish" else "CE"

            # -- Determine strike (ATM or OTM) ---------------------------------
            if strike_offset == 0:
                strike_used = atm_strike
            else:
                strike_used = get_otm_strike(
                    nifty_entry, option_type, offset=strike_offset,
                    step=CFG["nifty_strike_step"]
                )

            # -- Get option price from bhavcopy + BS ---------------------------
            bhav_ref_price = None
            expiry_date    = None
            if bhav_df is not None:
                bhav_ref_price, expiry_date = get_option_entry_price_from_bhavcopy(
                    bhav_df, strike_used, option_type, trade_date=date_obj
                )

            if expiry_date is not None:
                days_to_expiry = (expiry_date - date_obj).days
                is_expiry_day  = (days_to_expiry == 0)
            else:
                days_to_expiry = 7
                is_expiry_day  = False

            # On expiry day, use fractional DTE representing actual
            # minutes remaining at 13:35 entry (115 min of 375 min session)
            if is_expiry_day:
                days_to_expiry = 115.0 / 375.0   # ~0.307 of a trading day

            # -- Expiry-only filter (must come after expiry detection) ----------
            if scenario.get("expiry_only") and not is_expiry_day:
                row.update({
                    "Nifty_Entry": round(nifty_entry, 2),
                    "Strike_Used": None, "Trade_Type": "-",
                    "Option_Price": None, "BS_Entry_Price": None,
                    "Bhav_Ref_Price": None, "IV_Used": round(sigma_used, 4),
                    "Expiry_Date": None, "Days_To_Expiry": None,
                    "Is_Expiry_Day": False,
                    "Data_Source": data_source_used, "Data_Quality": "-",
                    "Signal_Confirmed": signal_confirmed,
                    "Max_Favorable_%": 0.0, "Max_Adverse_%": 0.0,
                    "SL_Trigger_%": 0.0, "Transaction_Cost_%": 0.0,
                    "Exit_Type": "Skipped_NonExpiry",
                    "Gross_PL_Pct": 0.0, "Net_PL_Pct": 0.0,
                })
                all_rows.append(row)
                continue

            T_entry = max(days_to_expiry / 252.0, 0.0001)
            option_entry_price = estimate_option_price_at_entry(
                nifty_entry, strike_used, T_entry, sigma_used,
                CFG["risk_free_rate"], option_type,
                bhav_ref_price=bhav_ref_price,
            )

            # -- Apply trade filters -------------------------------------------
            skip_reason = None
            iv_max = filters.get("iv_max")
            if iv_max and sigma_used > iv_max:
                skip_reason = "Skipped_HighIV"
            gap_max = filters.get("gap_max_pct")
            if gap_max and abs(gap_pct) > gap_max:
                skip_reason = "Skipped_GapTooLarge"

            if skip_reason:
                row.update({
                    "Nifty_Entry": round(nifty_entry, 2),
                    "Strike_Used": strike_used,
                    "Trade_Type": option_type,
                    "Option_Price": round(option_entry_price, 2),
                    "BS_Entry_Price": round(option_entry_price, 2),
                    "Bhav_Ref_Price": round(bhav_ref_price, 2) if bhav_ref_price else None,
                    "IV_Used": round(sigma_used, 4),
                    "Expiry_Date": expiry_date.strftime("%Y-%m-%d") if expiry_date else None,
                    "Days_To_Expiry": days_to_expiry,
                    "Is_Expiry_Day": is_expiry_day,
                    "Data_Source": data_source_used,
                    "Data_Quality": "-",
                    "Signal_Confirmed": signal_confirmed,
                    "Max_Favorable_%": 0.0, "Max_Adverse_%": 0.0,
                    "SL_Trigger_%": 0.0, "Transaction_Cost_%": 0.0,
                    "Exit_Type": skip_reason,
                    "Gross_PL_Pct": 0.0, "Net_PL_Pct": 0.0,
                })
                all_rows.append(row)
                continue

            # -- Simulate trade ------------------------------------------------
            if trade_mode == "SPREAD":
                # Credit spread: sell near-strike, buy far-strike
                spread_w = scenario.get("spread_width_strikes", 1)
                sold_strike = strike_used
                step = CFG["nifty_strike_step"]
                if option_type == "CE":
                    bought_strike = sold_strike + spread_w * step
                else:  # PE
                    bought_strike = sold_strike - spread_w * step

                # Price both legs
                sold_prem = option_entry_price  # already computed above
                bought_bhav, _ = (None, None)
                if bhav_df is not None:
                    bought_bhav, _ = get_option_entry_price_from_bhavcopy(
                        bhav_df, bought_strike, option_type, trade_date=date_obj
                    )
                bought_prem = estimate_option_price_at_entry(
                    nifty_entry, bought_strike, T_entry, sigma_used,
                    CFG["risk_free_rate"], option_type,
                    bhav_ref_price=bought_bhav,
                )

                result = simulate_spread_trade(
                    entry_data, signal, sold_prem, bought_prem,
                    sold_strike, bought_strike, scenario,
                    option_type=option_type,
                    days_to_expiry=float(days_to_expiry),
                    sigma=sigma_used,
                )
                display_premium = result.get("net_credit_rs", sold_prem - bought_prem)
            else:
                result = simulate_trade(
                    entry_data, signal, option_entry_price, scenario,
                    option_type=option_type,
                    days_to_expiry=float(days_to_expiry),
                    sigma=sigma_used,
                    trade_mode=trade_mode,
                )
                display_premium = option_entry_price

            # Data quality label
            is_approx = entry_data.get("is_daily_approx", False)
            bar_int   = entry_data.get("bar_interval", 5)
            has_real_option = (bhav_ref_price is not None)

            if not is_approx and bar_int <= 5:
                src_label = "5min"
            elif not is_approx and bar_int <= 60:
                src_label = "1hour"
            else:
                src_label = "Daily"

            opt_label = "RealOption" if has_real_option else "EstimatedOption"
            quality = f"{src_label}+{opt_label}"

            row.update({
                "Nifty_Entry":       round(nifty_entry, 2),
                "Strike_Used":       strike_used,
                "Trade_Type":        option_type,
                "Option_Price":      round(display_premium, 2) if display_premium else 0,
                "BS_Entry_Price":    round(option_entry_price, 2),
                "Bhav_Ref_Price":    round(bhav_ref_price, 2) if bhav_ref_price else None,
                "IV_Used":           round(sigma_used, 4),
                "Expiry_Date":       expiry_date.strftime("%Y-%m-%d") if expiry_date else None,
                "Days_To_Expiry":    days_to_expiry,
                "Is_Expiry_Day":     is_expiry_day,
                "Data_Source":       data_source_used,
                "Data_Quality":      quality,
                "Signal_Confirmed":  signal_confirmed,
                "Max_Favorable_%":   result["max_favorable_pct"],
                "Max_Adverse_%":     result["max_adverse_pct"],
                "SL_Trigger_%":      result["sl_trigger_pct"],
                "Transaction_Cost_%": result["transaction_cost_pct"],
                "Exit_Type":         result["exit_type"],
                "Gross_PL_Pct":      result["gross_pl_pct"],
                "Net_PL_Pct":        result["net_pl_pct"],
            })
            if trade_mode == "SPREAD":
                row["Spread_Capital_Rs"] = result.get("spread_capital_rs", 0)
                row["Net_Credit_Rs"] = result.get("net_credit_rs", 0)

            all_rows.append(row)

    trades_df = pd.DataFrame(all_rows)

    if CFG.get("dax_confirmation_enabled", True):
        log.info(f"DAX signal confirmation: {confirmed_count} confirmed, "
                 f"{rejected_count} rejected")

    return trades_df


# =============================================================================
#  ENTRY POINT
# =============================================================================

def plot_equity_curves(trades_df: pd.DataFrame, output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed -- skipping chart.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    colors  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#795548", "#607D8B", "#E91E63", "#FF9800"]
    scenario_names = [s["name"] for s in CFG["scenarios"]]

    for scen, color in zip(scenario_names, colors):
        df_scen = trades_df[
            (trades_df["Scenario"] == scen) &
            (trades_df["Signal"] != "No Trade") &
            (~trades_df["Exit_Type"].isin(["No Trade", "Signal_Rejected"])) &
            (~trades_df["Exit_Type"].str.startswith("Skipped", na=False))
        ].copy()
        if df_scen.empty:
            continue
        eq_curve = df_scen["Net_PL_Pct"].cumsum().reset_index(drop=True)
        ax.plot(eq_curve.index, eq_curve.values, label=scen, color=color, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"DAX Gap -> Nifty Options Strategy  |  "
        f"Equity Curves  ({CFG['start_date']} -> {CFG['end_date']})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Trade #", fontsize=11)
    ax.set_ylabel("Cumulative Net P&L (%)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"Equity curve chart saved to: {output_path}")


def _export_excel_report(trades_df: pd.DataFrame, xlsx_path: str, summaries: list) -> None:
    """Export a detailed Excel report with one sheet per scenario + summary."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        log.warning("openpyxl not installed -- skipping Excel export. "
                    "Install with: pip install openpyxl")
        return

    INITIAL_CAPITAL = 20_000.0
    lot_size = CFG["lot_size"]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # -- Summary sheet ---------------------------------------------------
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary", index=False)

        # -- Per-scenario sheets ---------------------------------------------
        for scenario in CFG["scenarios"]:
            scen_name = scenario["name"]
            df = trades_df[trades_df["Scenario"] == scen_name].copy()
            active = df[
                (~df["Exit_Type"].isin(["No Trade", "Signal_Rejected"])) &
                (~df["Exit_Type"].str.startswith("Skipped", na=False))
            ].copy()

            if active.empty:
                continue

            # Determine how many lots fit in the capital
            has_spread = "Spread_Capital_Rs" in active.columns
            if has_spread:
                avg_cap = active["Spread_Capital_Rs"].dropna().mean()
                if avg_cap > 0:
                    lots_per_trade = max(1, int(INITIAL_CAPITAL / avg_cap))
                else:
                    lots_per_trade = 1
            else:
                lots_per_trade = 1

            # Build detailed sheet
            rows = []
            portfolio = INITIAL_CAPITAL
            for _, r in active.iterrows():
                cap_per_lot = r.get("Spread_Capital_Rs", 0) if has_spread else (
                    r.get("Option_Price", 50) * lot_size)
                total_capital_used = cap_per_lot * lots_per_trade
                net_credit = r.get("Net_Credit_Rs", 0) * lots_per_trade if has_spread else 0
                pl_pct = r["Net_PL_Pct"]
                pl_rs = total_capital_used * pl_pct / 100.0
                portfolio += pl_rs

                # Determine sold/bought strikes
                strike = r.get("Strike_Used", "")
                opt_type = r.get("Trade_Type", "")
                spread_w = scenario.get("spread_width_strikes", 0)
                step = CFG["nifty_strike_step"]
                if has_spread and spread_w > 0 and pd.notna(strike) and strike != "":
                    sold_strike = int(float(strike))
                    if opt_type == "CE":
                        bought_strike = sold_strike + spread_w * step
                    else:
                        bought_strike = sold_strike - spread_w * step
                    action_desc = (f"SELL {opt_type} {sold_strike} + "
                                   f"BUY {opt_type} {bought_strike}")
                else:
                    sold_strike = strike
                    bought_strike = ""
                    action_desc = f"{scenario.get('trade_mode','BUY')} {opt_type} {strike}"

                rows.append({
                    "Date":              r["Date"],
                    "Day":               pd.Timestamp(r["Date"]).strftime("%A"),
                    "DAX_Gap_%":         round(r.get("DAX_Gap_Pct", 0), 3),
                    "Signal":            r["Signal"],
                    "Nifty_Entry":       r.get("Nifty_Entry", ""),
                    "Action":            action_desc,
                    "Sold_Strike":       sold_strike if has_spread else "",
                    "Bought_Strike":     bought_strike if has_spread else "",
                    "Option_Type":       opt_type,
                    "IV_Used":           r.get("IV_Used", ""),
                    "Days_To_Expiry":    r.get("Days_To_Expiry", ""),
                    "Data_Quality":      r.get("Data_Quality", ""),
                    "Capital_Per_Lot":   round(cap_per_lot, 2),
                    "Lots":              lots_per_trade,
                    "Total_Capital":     round(total_capital_used, 2),
                    "Net_Credit_Rs":     round(net_credit, 2) if has_spread else "",
                    "Exit_Type":         r["Exit_Type"],
                    "Gross_PL_%":        r.get("Gross_PL_Pct", 0),
                    "Net_PL_%":          round(pl_pct, 2),
                    "P&L_Amount_Rs":     round(pl_rs, 2),
                    "Portfolio_Rs":      round(portfolio, 2),
                })

            out = pd.DataFrame(rows)
            # Truncate sheet name to 31 chars (Excel limit)
            sheet_name = scen_name[:31]
            out.to_excel(writer, sheet_name=sheet_name, index=False)

    log.info(f"Excel report saved to: {xlsx_path}")
    print(f"Excel report saved to: {xlsx_path}")


def main():
    print("\n" + "=" * 110)
    print("  DAX GAP -> NIFTY OPTIONS INTRADAY BACKTESTER  (v7 - Expiry-Day Credit Spreads)")
    print(f"  Period       : {CFG['start_date']}  ->  {CFG['end_date']}")
    print(f"  Gap threshold: +/-{CFG['gap_threshold_pct']}%")
    print(f"  Entry time   : {CFG['entry_time_ist']} IST  |  Force exit: {CFG['exit_time_ist']} IST")
    print(f"  Lot size     : {CFG['lot_size']}  |  Brokerage: Rs {CFG['brokerage_per_side_rs']}/side")
    print(f"  DAX confirm  : {'Enabled' if CFG.get('dax_confirmation_enabled') else 'Disabled'}")
    n_spread = sum(1 for s in CFG['scenarios'] if s.get('trade_mode') == 'SPREAD')
    print(f"  Scenarios    : {n_spread} SPREAD")
    flt = CFG.get("filters", {})
    if flt:
        print(f"  Filters      : IV<={flt.get('iv_max','')}, "
              f"|gap|<={flt.get('gap_max_pct','')}%")
    print("=" * 110 + "\n")

    # -- Run backtest ----------------------------------------------------------
    trades_df = run_backtest()

    # -- Print trade log (limited to active trades) ----------------------------
    print("\n--- TRADE LOG (active trades only) ---")
    display_cols = [
        "Date", "DAX_Gap_Pct", "Signal", "Scenario", "Trade_Mode",
        "Strike_Used", "Trade_Type", "IV_Used", "Option_Price",
        "Days_To_Expiry", "Data_Source",
        "Max_Favorable_%", "Max_Adverse_%",
        "Exit_Type", "Gross_PL_Pct", "Net_PL_Pct",
    ]
    available_cols = [c for c in display_cols if c in trades_df.columns]
    active_mask = (
        (trades_df["Signal"] != "No Trade") &
        (~trades_df["Exit_Type"].isin(["No Trade", "Signal_Rejected"])) &
        (~trades_df["Exit_Type"].str.startswith("Skipped", na=False))
    )
    print(trades_df.loc[active_mask, available_cols].to_string(index=False))

    # -- Per-scenario summary --------------------------------------------------
    summaries = []
    scenario_names = [s["name"] for s in CFG["scenarios"]]
    for scenario in CFG["scenarios"]:
        summaries.append(performance_summary(trades_df, scenario["name"]))
    print_summary_table(summaries)

    # -- Monthly returns (first sell + first buy) ------------------------------
    sell_names = [s["name"] for s in CFG["scenarios"] if s.get("trade_mode") == "SELL"]
    buy_names  = [s["name"] for s in CFG["scenarios"] if s.get("trade_mode") != "SELL"]
    for scen in (sell_names[:1] + buy_names[:1]):
        monthly = compute_monthly_returns(trades_df, scen)
        print_monthly_table(monthly, scen)

    # -- Expiry vs non-expiry --------------------------------------------------
    print_expiry_stats(trades_df, scenario_names)

    # -- Data quality breakdown ------------------------------------------------
    if "Data_Quality" in trades_df.columns:
        print("--- DATA QUALITY BREAKDOWN ---")
        trade_rows = trades_df[
            (trades_df["Signal"] != "No Trade") &
            (~trades_df["Exit_Type"].isin(["No Trade", "Signal_Rejected"])) &
            (~trades_df["Exit_Type"].str.startswith("Skipped", na=False))
        ]
        dq = trade_rows.drop_duplicates(subset=["Date"])["Data_Quality"].value_counts()
        for label, count in dq.items():
            print(f"  {label:<30} : {count} trading days")
        print()

    # -- Export ----------------------------------------------------------------
    from datetime import datetime as _dt
    ts         = _dt.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = f"backtest_results_{ts}.csv"
    chart_path = f"equity_curves_{ts}.png"
    xlsx_path  = f"backtest_full_report_{ts}.xlsx"
    trades_df.to_csv(csv_path, index=False)
    print(f"Results exported to  : {csv_path}")

    plot_equity_curves(trades_df, chart_path)

    # -- Excel export with full details per scenario -------------------------
    _export_excel_report(trades_df, xlsx_path, summaries)

    return trades_df, summaries


if __name__ == "__main__":
    trades_df, summaries = main()
