#!/usr/bin/env python3
"""
FAST BACKTEST OPTIMIZER - Test Parameter Combinations
Uses existing backtest results to find optimal TP/SL
"""

import pandas as pd
import numpy as np

# Load the original backtest results
df_backtest = pd.read_csv('rocket_backtest_20260308_212638.csv')

print("="*90)
print("FAST PARAMETER OPTIMIZATION (Using existing backtest data)")
print("="*90)
print(f"\nOriginal backtest: {len(df_backtest)} trades")
print(f"Date range: {df_backtest['date'].min()} to {df_backtest['date'].max()}")

# Current settings gave us:
current_tp = 20
current_sl = 80
current_trades = df_backtest[(df_backtest['exit_type'] != 'SL')].copy()
current_wr = (current_trades['net_pl_rs'] > 0).sum() / len(current_trades) * 100 if len(current_trades) > 0 else 0
current_total = current_trades['net_pl_rs'].sum() if len(current_trades) > 0 else 0

print(f"\nCURRENT CONFIG (TP={current_tp}%, SL={current_sl}%):")
print(f"  Trades: {len(current_trades)}")
print(f"  Wins: {(current_trades['net_pl_rs'] > 0).sum()}")
print(f"  WR: {current_wr:.1f}%")
print(f"  Total P&L: Rs {current_total:,.0f}")
print(f"  Avg P&L: Rs {current_trades['net_pl_rs'].mean():,.0f}")

# Now test different TP thresholds while keeping SL at 80%
print("\n" + "="*90)
print("TEST 1: DIFFERENT TP THRESHOLDS (SL=80% fixed)")
print("="*90)

tp_results = []
for tp_pct in [10, 12, 15, 18, 20, 25]:
    # Use gross_pl_pct to check TP hit
    tp_hits = df_backtest[df_backtest['gross_pl_pct'] >= tp_pct].copy()
    sl_hits = df_backtest[df_backtest['gross_pl_pct'] <= -80].copy()
    time_exits = df_backtest[(df_backtest['gross_pl_pct'] > -80) & (df_backtest['gross_pl_pct'] < tp_pct)].copy()
    
    all_trades = pd.concat([tp_hits, sl_hits, time_exits]).drop_duplicates()
    
    if len(all_trades) > 0:
        # For time exits, keep half the profit
        tp_hits['final_pl'] = tp_hits['net_pl_rs']
        sl_hits['final_pl'] = -sl_hits['max_risk_rs'] * 0.8 / 100
        time_exits['final_pl'] = time_exits['net_pl_rs']
        
        final_trades = pd.concat([tp_hits[['final_pl']], sl_hits[['final_pl']], time_exits[['final_pl']]])
        
        wins = (final_trades['final_pl'] > 0).sum()
        wr = wins / len(final_trades) * 100
        total_pl = final_trades['final_pl'].sum()
        avg_pl = final_trades['final_pl'].mean()
        
        tp_results.append({
            'TP%': tp_pct,
            'SL%': 80,
            'Trades': len(final_trades),
            'Wins': wins,
            'WR%': f"{wr:.1f}",
            'Total P&L': f"{total_pl:,.0f}",
            'Avg P&L': f"{avg_pl:,.0f}",
        })

tp_df = pd.DataFrame(tp_results).sort_values('Total P&L', 
                                             ascending=False,
                                             key=lambda x: x.str.replace(',', '').astype(float))
print(tp_df.to_string(index=False))

# Test different SL thresholds with best TP
print("\n" + "="*90)
print("TEST 2: DIFFERENT SL THRESHOLDS (TP=15% fixed - likely best)")
print("="*90)

sl_results = []
for sl_pct in [60, 70, 75, 80, 90]:
    tp_hits = df_backtest[df_backtest['gross_pl_pct'] >= 15].copy()
    sl_hits = df_backtest[df_backtest['gross_pl_pct'] <= -sl_pct].copy()
    time_exits = df_backtest[(df_backtest['gross_pl_pct'] > -sl_pct) & (df_backtest['gross_pl_pct'] < 15)].copy()
    
    if len(tp_hits) + len(sl_hits) + len(time_exits) > 0:
        tp_hits['final_pl'] = tp_hits['net_pl_rs']
        sl_hits['final_pl'] = -sl_hits['max_risk_rs'] * sl_pct / 100
        time_exits['final_pl'] = time_exits['net_pl_rs']
        
        final_trades = pd.concat([tp_hits[['final_pl']], sl_hits[['final_pl']], time_exits[['final_pl']]])
        
        wins = (final_trades['final_pl'] > 0).sum()
        wr = wins / len(final_trades) * 100
        total_pl = final_trades['final_pl'].sum()
        avg_pl = final_trades['final_pl'].mean()
        
        sl_results.append({
            'TP%': 15,
            'SL%': sl_pct,
            'Trades': len(final_trades),
            'Wins': wins,
            'WR%': f"{wr:.1f}",
            'Total P&L': f"{total_pl:,.0f}",
            'Avg P&L': f"{avg_pl:,.0f}",
        })

sl_df = pd.DataFrame(sl_results).sort_values('Total P&L',
                                             ascending=False,
                                             key=lambda x: x.str.replace(',', '').astype(float))
print(sl_df.to_string(index=False))

# Find absolute best combination
print("\n" + "="*90)
print("TEST 3: GRID SEARCH - All TP/SL COMBINATIONS")
print("="*90)

all_results = []
for tp_pct in [10, 12, 15, 18, 20, 25]:
    for sl_pct in [60, 70, 75, 80, 90]:
        tp_hits = df_backtest[df_backtest['gross_pl_pct'] >= tp_pct]
        sl_hits = df_backtest[df_backtest['gross_pl_pct'] <= -sl_pct]
        time_exits = df_backtest[(df_backtest['gross_pl_pct'] > -sl_pct) & (df_backtest['gross_pl_pct'] < tp_pct)]
        
        if len(tp_hits) + len(sl_hits) + len(time_exits) > 0:
            tp_hits_pl = tp_hits['net_pl_rs'].sum() if len(tp_hits) > 0 else 0
            sl_hits_pl = -(sl_hits['max_risk_rs'] * sl_pct / 100).sum() if len(sl_hits) > 0 else 0
            time_exit_pl = time_exits['net_pl_rs'].sum() if len(time_exits) > 0 else 0
            
            total_trades = len(tp_hits) + len(sl_hits) + len(time_exits)
            wins = len(tp_hits) + ((time_exits['net_pl_rs'] > 0).sum() if len(time_exits) > 0 else 0)
            total_pl = tp_hits_pl + sl_hits_pl + time_exit_pl
            
            wr = wins / total_trades * 100 if total_trades > 0 else 0
            avg_pl = total_pl / total_trades if total_trades > 0 else 0
            
            all_results.append({
                'TP%': tp_pct,
                'SL%': sl_pct,
                'Trades': total_trades,
                'Wins': wins,
                'WR%': f"{wr:.1f}",
                'Total P&L': f"{total_pl:,.0f}",
                'Avg P&L': f"{avg_pl:,.0f}",
            })

all_df = pd.DataFrame(all_results).sort_values('Total P&L',
                                               ascending=False,
                                               key=lambda x: x.str.replace(',', '').astype(float))

print("\nTop 10 Parameter Combinations:")
print(all_df.head(10).to_string(index=False))

print("\n" + "="*90)
print("RECOMMENDATION")
print("="*90)

if len(all_df) > 0:
    best = all_df.iloc[0]
    print(f"\n✅ BEST STRATEGY:")
    print(f"   TP Threshold: {best['TP%']}%")
    print(f"   SL Threshold: {best['SL%']}%")
    print(f"   Total Trades: {best['Trades']}")
    print(f"   Wins: {best['Wins']}")
    print(f"   Win Rate: {best['WR%']}%")
    print(f"   Total P&L: Rs {best['Total P&L']}")
    print(f"   Avg P&L: Rs {best['Avg P&L']}") 
    print(f"\n✅ IMPROVEMENTS VS CURRENT (TP=20%, SL=80%):")
    print(f"   Previous Total P&L: Rs {current_total:,.0f}")
    print(f"   New Total P&L: Rs {best['Total P&L']}")
    print(f"   Previous WR: {current_wr:.1f}%")
    print(f"   New WR: {best['WR%']}%")
