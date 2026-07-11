import pandas as pd

df = pd.read_csv('paper_trades.csv')
df_complete = df[df['status'] == 'completed'].copy()

print("="*80)
print("TRADES WITH DETAILS - CHECKING FOR DATA ANOMALIES")
print("="*80)

# Check all trades
for idx, (_, t) in enumerate(df_complete.iterrows(), 1):
    direction = t['direction']
    sold_prem = t.get('sold_entry_prem', 0)
    bought_prem = t.get('bought_entry_prem', 0)
    net_credit = t.get('net_credit', 0)
    net_credit_rs = t.get('net_credit_rs', 0)
    
    calc_credit = sold_prem - bought_prem
    
    sold_strike = t.get('sold_strike', 0)
    bought_strike = t.get('bought_strike', 0)
    opt_type = t.get('option_type', '?')
    
    issue = ""
    if net_credit_rs < 0:
        issue = " ⚠️ NEGATIVE CREDIT (DEBIT SPREAD?)"
    
    if abs(calc_credit - net_credit) > 0.1:
        issue += " ⚠️ CALC MISMATCH"
    
    pl = t['net_pl_rs']
    icon = '✓' if pl > 0 else '✗'
    
    print(f"{idx:2d}. {t['date']} {direction:8s} {icon} | "
          f"Sell {sold_strike:.0f}{opt_type} ({sold_prem:.2f}) | "
          f"Buy {bought_strike:.0f}{opt_type} ({bought_prem:.2f}) | "
          f"Credit: Rs {net_credit_rs:+8.0f}{issue}")

print("\n" + "="*80)
print("ROOT CAUSE HYPOTHESIS")
print("="*80)
print("""
Looking at the NEGATIVE CREDITS:
- Bearish trades have NEGATIVE entry credits on most losses
- This means they PAID money to enter, not COLLECTED
- This is BACKWARDS for a credit spread!

Example: Trade 2 (2026-04-28)
  - Sold 24000 CE at ~18.7 (supposed to collect)
  - Bought 23800 CE at ~191.6 (supposed to pay ~0.02)
  - Net: Should be ~18.7 - 0.02 = +18.68
  - But system shows: -11,238 / 65 = -172.9 (NEGATIVE!)

This suggests:
1. BID/ASK CONFUSION: Using bid for bought leg instead of ask
2. Or using ask for sold leg instead of bid
3. Or the spread strikes are INVERTED

This explains why ALL the massive losses are Bearish:
- On Bullish (PUT spreads): When you sell OTM puts, you usually collect credit
- On Bearish (CALL spreads): When you sell OTM calls, the system is confused:
  - It's selling a HIGHER call (OTM) - should get little premium
  - It's buying a LOWER call (ITM or closer) - should pay more
  - Net result: NEGATIVE credit (DEBIT) instead of positive credit
""")
