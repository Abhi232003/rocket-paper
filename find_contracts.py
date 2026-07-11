#!/usr/bin/env python3
"""Find what Nifty contracts are ACTUALLY available today"""

import os
import sys
import time
from datetime import date, timedelta
import pyotp
from SmartApi import SmartConnect

API_KEY      = os.environ.get("ANGEL_API_KEY", "")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
MPIN         = os.environ.get("ANGEL_MPIN", "")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")

if not all([API_KEY, CLIENT_ID, MPIN, TOTP_SECRET]):
    print("❌ Credentials not set")
    sys.exit(1)

# Login
obj = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
data = obj.generateSession(CLIENT_ID, MPIN, totp)
if data.get("message") != "SUCCESS":
    print(f"❌ Login failed: {data}")
    sys.exit(1)
print("✓ Login successful")

# Try different expiry dates
today = date.today()
next_tuesday = today + timedelta(days=(1 - today.weekday()) % 7 or 7)

dates_to_try = [
    ("Today (if Tuesday)", today),
    ("Next Tuesday", next_tuesday),
]

print("\n" + "="*60)
print("SEARCHING FOR AVAILABLE NIFTY CONTRACTS")
print("="*60 + "\n")

for label, search_date in dates_to_try:
    expiry_str = search_date.strftime("%d%b%y").upper()
    query = f"NIFTY{expiry_str}"
    
    print(f"📅 {label}: {search_date} ({search_date.strftime('%A')})")
    print(f"   Query: {query}")
    
    try:
        time.sleep(0.5)
        result = obj.searchScrip("NFO", query)
        
        if result and result.get("data"):
            contracts = result.get("data", [])
            print(f"   ✓ Found {len(contracts)} contracts")
            
            # Show PE contracts only
            pe_contracts = [c for c in contracts if "PE" in c.get("tradingsymbol", "")]
            ce_contracts = [c for c in contracts if "CE" in c.get("tradingsymbol", "")]
            
            if pe_contracts:
                print(f"   PE: {len(pe_contracts)} available")
                pe_strikes = sorted(set([
                    int(c.get("tradingsymbol", "").replace("NIFTY" + expiry_str, "").replace("PE", ""))
                    for c in pe_contracts
                ]))
                print(f"       Strikes: {pe_strikes[:15]}{'...' if len(pe_strikes) > 15 else ''}")
            
            if ce_contracts:
                print(f"   CE: {len(ce_contracts)} available")
                ce_strikes = sorted(set([
                    int(c.get("tradingsymbol", "").replace("NIFTY" + expiry_str, "").replace("CE", ""))
                    for c in ce_contracts
                ]))
                print(f"       Strikes: {ce_strikes[:15]}{'...' if len(ce_strikes) > 15 else ''}")
        else:
            print(f"   ✗ No contracts found")
    
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}")
    
    print()

print("="*60)
