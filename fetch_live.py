#!/usr/bin/env python3
"""Fetch live data from TradeLocker API and output as JSON."""
import json, sys
from datetime import datetime

try:
    from tradelocker import TLAPI

    tl = TLAPI(
        environment="https://live.tradelocker.com",
        username="ryanlawrence217@hotmail.com",
        password="!0Qka03a",
        server="GATESFX",
    )

    # Get all accounts (DataFrame)
    accs_df = tl.get_all_accounts()
    row = accs_df.iloc[0]
    account_info = {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "currency": str(row["currency"]),
        "accNum": int(row["accNum"]),
        "accountBalance": float(row["accountBalance"]),
        "status": str(row["status"]),
    }

    # Account state
    state = {}
    try:
        s = tl.get_account_state()
        if isinstance(s, dict):
            state = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in s.items()}
        elif hasattr(s, 'to_dict'):
            state = {k: str(v) for k, v in s.to_dict().items()}
        else:
            state = {"raw": str(s)}
    except Exception as e:
        state = {"error": str(e)}

    # Positions (DataFrame)
    positions = []
    try:
        pos = tl.get_all_positions()
        if pos is not None and hasattr(pos, 'to_dict'):
            positions = pos.to_dict('records') if len(pos) > 0 else []
        elif pos is not None and isinstance(pos, list):
            positions = pos
    except Exception as e:
        pass

    # Orders
    orders = []
    try:
        ords = tl.get_all_orders()
        if ords is not None and hasattr(ords, 'to_dict'):
            orders = ords.to_dict('records') if len(ords) > 0 else []
    except:
        pass

    # Executions (trade history)
    executions = []
    try:
        execs = tl.get_all_executions()
        if execs is not None and hasattr(execs, 'to_dict'):
            executions = execs.to_dict('records') if len(execs) > 0 else []
        elif execs is not None and isinstance(execs, list):
            executions = execs[:50]
    except:
        pass

    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "connected": True,
        "account": account_info,
        "state": state,
        "positions": positions,
        "orders": orders,
        "executions": executions[:50] if executions else [],
    }

    with open("/tmp/vwap-dash/live_data.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("SUCCESS", file=sys.stderr)
    print(json.dumps(result, indent=2, default=str))

except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "connected": False,
        "error": str(e),
        "account": {"id": 684012, "accountBalance": 0, "currency": "USD", "status": "UNKNOWN"},
        "state": {},
        "positions": [],
        "orders": [],
        "executions": [],
    }
    with open("/tmp/vwap-dash/live_data.json", "w") as f:
        json.dump(result, f, indent=2)
