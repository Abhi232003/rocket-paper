#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTEST WITH PARAMETER OPTIMIZATION
Test all combinations to find the best strategy
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
STRIKE_STEP    = 50
LOT_SIZE       = 65
SPREAD_WIDTH   = 200  # 4 strikes
SESSION_MINUTES = 375
IV_DEFAULT      = 0.15
RISK_FREE       = 0.065

# Costs
BROKERAGE_PER_SIDE = 20.0
STT_PCT            = 0.05 / 100
GST_PCT            = 0.18
SLIPPAGE_PCT       = 0.005

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

# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════
def load_nifty_data(start_date=None):
    """Load all cached Nifty data"""
    all_data = {}
    
    if not CACHE_DIR.exists():
        print(f"Cache dir not found: {CACHE_DIR}")
        return all_data
    
    for csv_file in sorted(CACHE_DIR.glob("nifty_*.csv")):
        date_str = csv_file.stem.replace("nifty_", "")
        try:
            df = pd.read_csv(csv_file)
            df = df.sort_values("timestamp", ignore_index=True)
            all_data[date_str] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    return all_data

def get_dax_direction(trade_date):
    """Get DAX direction for a specific date"""
    try:
        dax = yf.download("^GDAXI", period="5d", progress=False, auto_adjust=True)
        if isinstance(dax.columns, pd.MultiIndex):
            dax.columns = [c[0] for c in dax.columns]
        
        today_open = float(dax["Open"].iloc[-1])
        prev_close = float(dax["Close"].iloc[-2])
        gap_pct = ((today_open - prev_close) / prev_close) * 100.0
        direction = "Bullish" if gap_pct >= 0 else "Bearish"
        return direction, gap_pct
    except:
        return "Bullish", 0.0

def compute_costs(sold_premium, bought_premium):
    brokerage = BROKERAGE_PER_SIDE * 4
    gst = brokerage * GST_PCT
    stt = (sold_premium + bought_premium) * LOT_SIZE * STT_PCT
    slippage = (sold_premium + bought_premium) * LOT_SIZE * SLIPPAGE_PCT * 2
    return brokerage + gst + stt + slippage

# ═══════════════════════════════════════════════════════════════════
#  SIMULATE TRADE
# ═══════════════════════════════════════════════════════════════════
def simulate_trade(nifty_prices, entry_nifty, signal, tp_pct, sl_pct, use_trend_filter=False):
    """Simulate a single trade with given TP/SL thresholds"""
    atm = round(entry_nifty / STRIKE_STEP) * STRIKE_STEP

    # ✅ FIXED: Correct strike selection for both directions
    if signal == "Bullish":
        opt_type = "PE"
        sold_strike   = atm
        bought_strike = atm - SPREAD_WIDTH
    else:
        opt_type = "CE"
        sold_strike   = atm
        bought_strike = atm + SPREAD_WIDTH  # ✅ HIGHER for Bearish

    # Entry pricing
    T_entry = 115.0 / (SESSION_MINUTES * 252)
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

    # Path-dependent monitoring
    exit_type = "Time Exit"
    exit_nifty = entry_nifty
    exit_pl_pct = 0.0
    
    for i, nifty_now in enumerate(nifty_prices):
        T_now = max(0, (115.0 - i) / (SESSION_MINUTES * 252))
        
        sold_now   = _bs_price(nifty_now, sold_strike,   T_now, RISK_FREE, IV_DEFAULT, opt_type)
        bought_now = _bs_price(nifty_now, bought_strike, T_now, RISK_FREE, IV_DEFAULT, opt_type)
        
        spread_value = max(0, min(sold_now - bought_now, SPREAD_WIDTH))
        pl_per_unit = net_credit - spread_value
        pl_pct = (pl_per_unit * LOT_SIZE / max_risk_rs) * 100.0

        # TP check
        if pl_pct > 0 and pl_pct >= tp_pct:
            exit_type = "TP"
            exit_pl_pct = pl_pct
            exit_nifty = nifty_now
            break

        # SL check
        if pl_pct < 0 and abs(pl_pct) >= sl_pct:
            exit_type = "SL"
            exit_pl_pct = -sl_pct
            exit_nifty = nifty_now
            break

        # Last bar
        if i == len(nifty_prices) - 1:
            exit_type = "Time Exit"
            exit_pl_pct = pl_pct
            exit_nifty = nifty_now

    # Costs
    cost_rs = compute_costs(sold_premium, bought_premium)
    cost_pct = (cost_rs / max_risk_rs) * 100.0

    gross_pl_pct = exit_pl_pct
    net_pl_pct = exit_pl_pct - cost_pct
    net_pl_pct = max(net_pl_pct, -100.0)
    net_pl_rs = net_pl_pct / 100.0 * max_risk_rs

    return {
        "entry_nifty": entry_nifty,
        "exit_nifty": exit_nifty,
        "net_credit_rs": net_credit_rs,
        "max_risk_rs": max_risk_rs,
        "exit_type": exit_type,
        "gross_pl_pct": gross_pl_pct,
        "net_pl_pct": net_pl_pct,
        "net_pl_rs": net_pl_rs,
        "signal": signal,
    }

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def run_optimization():
    print("="*90)
    print("COMPREHENSIVE BACKTEST - PARAMETER OPTIMIZATION")
    print("="*90)
    
    nifty_data = load_nifty_data()
    print(f"\nLoaded {len(nifty_data)} days of Nifty data")
    print(f"Date range: {min(nifty_data.keys())} to {max(nifty_data.keys())}")
    
    # Parameter combinations to test
    tp_thresholds = [10, 12, 15, 18, 20, 25]
    sl_thresholds = [60, 70, 75, 80, 90]
    
    results = []
    
    # Test each combination
    for tp_pct in tp_thresholds:
        for sl_pct in sl_thresholds:
            trades = []
            
            # Simulate all available dates
            for date_str in sorted(nifty_data.keys()):
                df = nifty_data[date_str]
                
                # Get first entry price (around 13:35)
                if len(df) < 5:
                    continue
                
                entry_idx = min(5, len(df) - 1)
                entry_nifty = float(df.iloc[entry_idx]["close"])
                
                # Get remaining prices for monitoring
                nifty_prices = df.iloc[entry_idx:]["close"].values.astype(float)
                
                # Get DAX direction
                try:
                    # Parse date
                    d = datetime.strptime(date_str, "%Y%m%d").date()
                    signal, gap = get_dax_direction(d)
                except:
                    signal = "Bullish"
                
                # Simulate trade
                trade = simulate_trade(nifty_prices, entry_nifty, signal, tp_pct, sl_pct)
                if trade:
                    trades.append(trade)
            
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                n = len(trades_df)
                wins = (trades_df["net_pl_rs"] > 0).sum()
                wr = wins / n * 100
                total_pl = trades_df["net_pl_rs"].sum()
                avg_pl = trades_df["net_pl_rs"].mean()
                
                results.append({
                    "TP%": tp_pct,
                    "SL%": sl_pct,
                    "Trades": n,
                    "Wins": wins,
                    "WR%": f"{wr:.1f}",
                    "Total P&L": f"{total_pl:,.0f}",
                    "Avg P&L": f"{avg_pl:,.0f}",
                })
    
    # Show results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Total P&L", ascending=False, key=lambda x: x.str.replace(",", "").astype(float))
    
    print("\n" + "="*90)
    print("TOP 20 PARAMETER COMBINATIONS (Sorted by Total P&L)")
    print("="*90)
    print(results_df.head(20).to_string(index=False))
    
    print("\n" + "="*90)
    print("BEST STRATEGY")
    print("="*90)
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"TP Threshold: {best['TP%']}%")
        print(f"SL Threshold: {best['SL%']}%")
        print(f"Total Trades: {best['Trades']}")
        print(f"Wins: {best['Wins']}")
        print(f"Win Rate: {best['WR%']}%")
        print(f"Total P&L: Rs {best['Total P&L']}")
        print(f"Avg P&L: Rs {best['Avg P&L']}")

if __name__ == "__main__":
    run_optimization()
