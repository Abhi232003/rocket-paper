import pandas as pd
import numpy as np
from datetime import datetime, date

# Load paper trades
df = pd.read_csv('paper_trades.csv')
df_complete = df[df['status'] == 'completed'].copy()

print("="*80)
print("DETAILED BEARISH TRADE FAILURE ANALYSIS")
print("="*80)

# Analyze Bearish trades
bearish = df_complete[df_complete['direction'] == 'Bearish'].copy()
print(f"\nTotal Bearish trades (paper): {len(bearish)}")
print(f"Wins: {(bearish['net_pl_rs'] > 0).sum()}")
print(f"Losses: {(bearish['net_pl_rs'] < 0).sum()}")
print(f"Win Rate: {(bearish['net_pl_rs'] > 0).sum() / len(bearish) * 100:.0f}%")
print(f"Total P&L: Rs {bearish['net_pl_rs'].sum():,.0f}")
print(f"\nBEARISH TRADES BREAKDOWN:")
print("-" * 80)

for idx, (_, t) in enumerate(bearish.iterrows(), 1):
    date_str = t['date']
    nifty_entry = t.get('nifty_entry', 0)
    nifty_exit = t.get('nifty_exit', 0)
    nifty_move = nifty_exit - nifty_entry
    nifty_move_pct = (nifty_move / nifty_entry * 100) if nifty_entry else 0
    
    sold_strike = t.get('sold_strike', 0)
    bought_strike = t.get('bought_strike', 0)
    
    entry_credit = t.get('net_credit_rs', 0)
    exit_type = t.get('exit_type', 'N/A')
    entry_time = t.get('entry_time', '?')
    exit_time = t.get('exit_time', '15:15')
    
    pl = t['net_pl_rs']
    pct = t['net_pl_pct']
    gap = t.get('gap_pct', 0)
    
    icon = '✓' if pl > 0 else '✗'
    
    print(f"\n{idx}. {date_str} {icon}")
    print(f"   DAX gap: {gap:+.2f}% | Entry: {entry_time} | Exit: {exit_time} ({exit_type})")
    print(f"   Strike: Sell {sold_strike:.0f}CE / Buy {bought_strike:.0f}CE")
    print(f"   Nifty: {nifty_entry:.0f} → {nifty_exit:.0f} ({nifty_move:+.0f}pt, {nifty_move_pct:+.1f}%)")
    print(f"   Entry Credit: Rs {entry_credit:,.0f}")
    print(f"   RESULT: Rs {pl:+,.0f} ({pct:+.1f}%)")
    
    # Analyze the move direction
    if exit_type == "Time Exit" and pl < -10000:
        reason = "💥 MASSIVE LOSS - Spread went full width against position"
        if nifty_move > 0:
            reason += " (Nifty went UP, sold CE spread lost)"
    elif exit_type == "TP":
        reason = "✓ TP HIT - Spread decayed as expected"
    else:
        reason = ""
    
    if reason:
        print(f"   {reason}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Count losses by exit type
time_exit_losses = bearish[(bearish['exit_type'] == 'Time Exit') & (bearish['net_pl_rs'] < 0)]
print(f"\n✗ Time Exit Losses: {len(time_exit_losses)} trades")
print(f"   Loss amount: Rs {time_exit_losses['net_pl_rs'].sum():,.0f}")
print(f"   Avg loss per trade: Rs {time_exit_losses['net_pl_rs'].mean():,.0f}")

# Check Nifty movement on Bearish trades
bearish['nifty_move'] = bearish['nifty_exit'] - bearish['nifty_entry']
bearish['move_pct'] = (bearish['nifty_move'] / bearish['nifty_entry'] * 100)

print(f"\n📊 Nifty Movement Analysis (Bearish Trades):")
print(f"   Avg Nifty move: {bearish['move_pct'].mean():+.2f}%")
print(f"   Trades with Nifty UP: {(bearish['nifty_move'] > 0).sum()}")
print(f"   Trades with Nifty DOWN: {(bearish['nifty_move'] < 0).sum()}")

nifty_up_trades = bearish[bearish['nifty_move'] > 0]
nifty_down_trades = bearish[bearish['nifty_move'] < 0]

print(f"\n   When Nifty went UP ({len(nifty_up_trades)} trades):")
print(f"      Wins: {(nifty_up_trades['net_pl_rs'] > 0).sum()} | Losses: {(nifty_up_trades['net_pl_rs'] < 0).sum()}")
print(f"      P&L: Rs {nifty_up_trades['net_pl_rs'].sum():,.0f}")

print(f"\n   When Nifty went DOWN ({len(nifty_down_trades)} trades):")
print(f"      Wins: {(nifty_down_trades['net_pl_rs'] > 0).sum()} | Losses: {(nifty_down_trades['net_pl_rs'] < 0).sum()}")
print(f"      P&L: Rs {nifty_down_trades['net_pl_rs'].sum():,.0f}")

print(f"\n🔴 CONCLUSION:")
print(f"   Bearish trades are LOSING when Nifty goes UP (which it did {(bearish['nifty_move'] > 0).sum()}/{len(bearish)} times)")
print(f"   → The strategy is SHORT on Nifty direction")
print(f"   → Paper trading period (Mar-Jul 2026) was BULLISH biased")
print(f"   → DAX gap predicts direction INCORRECTLY for Bearish trades recently")
