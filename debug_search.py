#!/usr/bin/env python3
"""Quick script to find what Nifty option contracts are actually available"""

import os
import pyotp
from SmartApi import SmartConnect
from datetime import date

API_KEY      = os.environ.get("ANGEL_API_KEY", "")
CLIENT_ID    = os.environ.get("ANGEL_CLIENT_ID", "")
MPIN         = os.environ.get("ANGEL_MPIN", "")
TOTP_SECRET  = os.environ.get("ANGEL_TOTP_SECRET", "")

# Login
obj = SmartConnect(api_key=API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
data = obj.generateSession(CLIENT_ID, MPIN, totp)
print(f"Login: {data.get('message')}")

# Try different searches to find the format
today = date.today()
expiry_str = today.strftime("%d%b%y").upper()

# Test variations
tests = [
    f"NIFTY{expiry_str}",           # e.g. NIFTY14APR26
    "NIFTY",                         # broad search
    f"NIFTY{today.strftime('%d%b%y')}", # lowercase month
]

print("\n=== SEARCHING FOR AVAILABLE NIFTY CONTRACTS ===\n")

for query in tests:
    print(f"Search: {query}")
    try:
        result = obj.searchScrip("NFO", query)
        if result and result.get("data"):
            count = len(result.get("data", []))
            print(f"  Found {count} contracts")
            if count > 0 and count <= 20:
                for item in result["data"][:10]:
                    print(f"    - {item.get('tradingsymbol', 'N/A')} (token: {item.get('symboltoken')})")
                if count > 10:
                    print(f"    ... and {count - 10} more")
        else:
            print(f"  No results")
    except Exception as e:
        print(f"  Error: {e}")
    print()
