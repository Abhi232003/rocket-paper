import pandas as pd
import numpy as np

print("="*70)
print("BACKTEST ANALYSIS (2024-06 to 2026-03)")
print("="*70)
df_backtest = pd.read_csv('rocket_backtest_20260308_212638.csv')
print(f"\nTotal trades: {len(df_backtest)}")
print(f"Wins: {(df_backtest['net_pl_rs'] > 0).sum()}")
print(f"Losses: {(df_backtest['net_pl_rs'] < 0).sum()}")
print(f"Total P&L: Rs {df_backtest['net_pl_rs'].sum():,.0f}")
print(f"Avg P&L: Rs {df_backtest['net_pl_rs'].mean():,.0f}")
print(f"Win Rate: {(df_backtest['net_pl_rs'] > 0).sum() / len(df_backtest) * 100:.1f}%")
print(f"\nBullish trades: {(df_backtest['signal'] == 'Bullish').sum()} | P&L: Rs {df_backtest[df_backtest['signal'] == 'Bullish']['net_pl_rs'].sum():,.0f}")
print(f"Bearish trades: {(df_backtest['signal'] == 'Bearish').sum()} | P&L: Rs {df_backtest[df_backtest['signal'] == 'Bearish']['net_pl_rs'].sum():,.0f}")

print("\n" + "="*70)
print("PAPER TRADING ANALYSIS (2026-03 to 2026-07)")
print("="*70)
df_paper = pd.read_csv('paper_trades.csv')
df_paper_complete = df_paper[df_paper['status'] == 'completed'].copy()
print(f"\nTotal trades: {len(df_paper_complete)}")
print(f"Wins: {(df_paper_complete['net_pl_rs'] > 0).sum()}")
print(f"Losses: {(df_paper_complete['net_pl_rs'] < 0).sum()}")
print(f"Total P&L: Rs {df_paper_complete['net_pl_rs'].sum():,.0f}")
print(f"Avg P&L: Rs {df_paper_complete['net_pl_rs'].mean():,.0f}")
print(f"Win Rate: {(df_paper_complete['net_pl_rs'] > 0).sum() / len(df_paper_complete) * 100:.1f}%")
print(f"\nBullish trades: {(df_paper_complete['direction'] == 'Bullish').sum()} | P&L: Rs {df_paper_complete[df_paper_complete['direction'] == 'Bullish']['net_pl_rs'].sum():,.0f}")
print(f"Bearish trades: {(df_paper_complete['direction'] == 'Bearish').sum()} | P&L: Rs {df_paper_complete[df_paper_complete['direction'] == 'Bearish']['net_pl_rs'].sum():,.0f}")

print("\n" + "="*70)
print("DETAILED TRADES (Paper Trading)")
print("="*70)
for _, row in df_paper_complete.iterrows():
    pl = row['net_pl_rs']
    pct = row['net_pl_pct']
    d = row['date']
    direction = row['direction'][:3]
    exit_type = str(row['exit_type'])[:4]
    icon = 'W' if pl > 0 else 'L'
    print(f"{d} | {direction:3s} | {exit_type:4s} | {icon} Rs {pl:+8,.0f} ({pct:+6.1f}%)")

print("\n" + "="*70)
print("KEY INSIGHT: BEARISH vs BULLISH")
print("="*70)

print(f"\nBACKTEST:")
bull = df_backtest[df_backtest['signal'] == 'Bullish']
bear = df_backtest[df_backtest['signal'] == 'Bearish']
print(f"  Bullish: {len(bull)} trades, {(bull['net_pl_rs'] > 0).sum()} wins, WR={len(bull[bull['net_pl_rs']>0])/len(bull)*100:.0f}%, Total={bull['net_pl_rs'].sum():,.0f}")
print(f"  Bearish: {len(bear)} trades, {(bear['net_pl_rs'] > 0).sum()} wins, WR={len(bear[bear['net_pl_rs']>0])/len(bear)*100:.0f}%, Total={bear['net_pl_rs'].sum():,.0f}")

print(f"\nPAPER TRADING:")
bull = df_paper_complete[df_paper_complete['direction'] == 'Bullish']
bear = df_paper_complete[df_paper_complete['direction'] == 'Bearish']
print(f"  Bullish: {len(bull)} trades, {(bull['net_pl_rs'] > 0).sum()} wins, WR={len(bull[bull['net_pl_rs']>0])/len(bull)*100:.0f}%, Total={bull['net_pl_rs'].sum():,.0f}")
print(f"  Bearish: {len(bear)} trades, {(bear['net_pl_rs'] > 0).sum()} wins, WR={len(bear[bear['net_pl_rs']>0])/len(bear)*100:.0f}%, Total={bear['net_pl_rs'].sum():,.0f}")
