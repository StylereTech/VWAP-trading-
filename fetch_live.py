#!/usr/bin/env python3
"""Fetch live data from TradeLocker API and output as JSON."""
import json, sys, os
from datetime import datetime, timedelta

try:
    from tradelocker import TLAPI
    
    tl = TLAPI(
        environment="https://live.tradelocker.com",
        username="ryanlawrence217@hotmail.com",
        password="l0Qka03a",
        server="GATESFX"
    )
    
    # Get all accounts
    all_accs = tl.get_all_accounts()
    print(f"Accounts found: {all_accs}", file=sys.stderr)
    
    # Try to get account info
    acc_id = None
    if hasattr(tl, 'account_id'):
        acc_id = tl.account_id
    
    # Get account state
    state = {}
    try:
        state = tl.get_state()
        print(f"State keys: {list(state.keys()) if isinstance(state, dict) else type(state)}", file=sys.stderr)
    except Exception as e:
        print(f"get_state error: {e}", file=sys.stderr)
    
    # Get positions
    positions = []
    try:
        pos = tl.get_all_positions()
        print(f"Positions type: {type(pos)}", file=sys.stderr)
        if pos is not None:
            positions = pos if isinstance(pos, list) else []
    except Exception as e:
        print(f"positions error: {e}", file=sys.stderr)
    
    # Get orders/history
    orders = []
    try:
        ords = tl.get_all_orders()
        print(f"Orders type: {type(ords)}", file=sys.stderr)
        if ords is not None:
            orders = ords if isinstance(ords, list) else []
    except Exception as e:
        print(f"orders error: {e}", file=sys.stderr)
    
    # Get order history
    history = []
    try:
        hist = tl.get_all_order_history()
        if hist is not None:
            history = hist if isinstance(hist, list) else []
            print(f"History count: {len(history)}", file=sys.stderr)
    except Exception as e:
        print(f"history error: {e}", file=sys.stderr)
    
    # Get position history
    pos_history = []
    try:
        ph = tl.get_all_position_history()
        if ph is not None:
            pos_history = ph if isinstance(ph, list) else []
            print(f"Position history count: {len(pos_history)}", file=sys.stderr)
    except Exception as e:
        print(f"pos_history error: {e}", file=sys.stderr)

    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "account_id": str(acc_id),
        "accounts": str(all_accs),
        "state": state if isinstance(state, dict) else str(state),
        "positions": [str(p) for p in positions] if positions else [],
        "orders": [str(o) for o in orders] if orders else [],
        "order_history": [str(h) for h in history[:50]] if history else [],
        "position_history": [str(h) for h in pos_history[:50]] if pos_history else [],
    }
    
    with open("/tmp/vwap-dash/live_data.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print("SUCCESS", file=sys.stderr)

except Exception as e:
    print(f"TradeLocker error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    
    # Write fallback
    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "error": str(e),
        "account": {
            "balance": 0,
            "equity": 0,
            "margin": 0,
            "free_margin": 0,
            "margin_level": 0
        },
        "positions": [],
        "orders": [],
        "order_history": [],
        "position_history": []
    }
    with open("/tmp/vwap-dash/live_data.json", "w") as f:
        json.dump(result, f, indent=2)
