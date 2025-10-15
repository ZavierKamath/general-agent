import os
import json
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dotenv import load_dotenv

load_dotenv()


"""
Plaid tool for fetching balances and transactions across accounts.

Environment variables required:
- PLAID_CLIENT_ID
- PLAID_SECRET
- PLAID_ENV=production  (also supports 'sandbox' or 'development')

Per-account Plaid access tokens (set the ones you use):
- PLAID_ACCESS_TOKEN_DISCOVER
- PLAID_ACCESS_TOKEN_HUNTINGTON
- PLAID_ACCESS_TOKEN_ALLY
- PLAID_ACCESS_TOKEN_PNC

How to obtain Plaid access tokens (high level):
1) Integrate Plaid Link in your app to let you select and connect your bank/credit card.
2) After a successful Link, you receive a transient `public_token` on the client.
3) Exchange the `public_token` on your backend via `/item/public_token/exchange` to get a long-lived `access_token`.
4) Store the `access_token` securely, and set it in the corresponding env var above (e.g., `PLAID_ACCESS_TOKEN_PNC`).

Quick path for development: Use Plaid's Quickstart sample or API Explorer to link test items
and obtain access tokens, then paste them into the env vars listed above.

Main entry point for the agent:
    get_financial_info(account_names, request_type, start_date=None, end_date=None)

Arguments:
- account_names: a single string (e.g., "PNC", "Discover", "all") or a list of strings.
- request_type: "balance" or "transactions" (case-insensitive).
- start_date, end_date: optional ISO dates (YYYY-MM-DD). Only used for transactions.

Behavior:
- For balances: returns a concise, plain-English summary string of balances.
- For transactions: fetches transactions across selected accounts for the date range
  (default last 90 days), writes a JSON file to `backend/tmp/`, and returns the full file path.

Notes:
- This tool expects `plaid-python` to be installed: pip install plaid-python
- It loads environment variables already present in the process; ensure your runtime loads .env.
"""


FRIENDLY_TO_ENV: Dict[str, str] = {
    "discover": "PLAID_ACCESS_TOKEN_DISCOVER",
    "huntington": "PLAID_ACCESS_TOKEN_HUNTINGTON",
    "ally": "PLAID_ACCESS_TOKEN_ALLY",
    "pnc": "PLAID_ACCESS_TOKEN_PNC",
}

NAME_SYNONYMS: Dict[str, List[str]] = {
    "discover": ["discover", "discover credit card"],
    "huntington": ["huntington", "huntington national bank"],
    "ally": ["ally", "ally bank", "ally money market", "ally money market account"],
    "pnc": ["pnc", "pnc bank", "pnc checking", "pnc savings"],
}


def _normalize_account_name(name: str) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if s == "all":
        return "all"
    for canonical, variants in NAME_SYNONYMS.items():
        if s in variants:
            return canonical
    # fallback: if user typed the canonical as-is
    if s in FRIENDLY_TO_ENV:
        return s
    return None


def _resolve_selected_accounts(account_names: Union[str, List[str]]) -> List[Tuple[str, str]]:
    """Resolve user-provided names to (friendly_name, access_token) tuples.

    Returns list of tuples. Skips accounts without tokens set. If none resolved,
    raises ValueError with a helpful message.
    """
    # Build reverse map for validation
    available: Dict[str, str] = {}
    for friendly, env_key in FRIENDLY_TO_ENV.items():
        token = os.getenv(env_key)
        if token:
            available[friendly] = token

    def resolve_all() -> List[Tuple[str, str]]:
        return [(friendly, token) for friendly, token in available.items()]

    resolved: List[Tuple[str, str]] = []
    if isinstance(account_names, str):
        norm = _normalize_account_name(account_names)
        if norm == "all":
            resolved = resolve_all()
        elif norm and norm in available:
            resolved = [(norm, available[norm])]
        else:
            resolved = []
    else:
        for n in account_names:
            norm = _normalize_account_name(str(n))
            if norm == "all":
                resolved = resolve_all()
                break
            if norm and norm in available:
                tup = (norm, available[norm])
                if tup not in resolved:
                    resolved.append(tup)

    if not resolved:
        allowed = ", ".join(sorted(FRIENDLY_TO_ENV.keys()))
        present = ", ".join(sorted(available.keys())) or "none"
        raise ValueError(
            f"No accounts resolved. Allowed names: {allowed}. "
            f"Tokens present for: {present}. Use 'all' to include all with tokens."
        )
    return resolved


def _init_plaid_client():
    try:
        from plaid import ApiClient, Configuration, Environment
        from plaid.api import plaid_api
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'plaid-python'. Install with: pip install plaid-python"
        ) from e

    client_id = os.getenv("PLAID_CLIENT_ID")
    secret = os.getenv("PLAID_SECRET")
    env = (os.getenv("PLAID_ENV") or "sandbox").strip().lower()

    if not client_id or not secret:
        raise RuntimeError("PLAID_CLIENT_ID and PLAID_SECRET must be set in environment")

    if env == "production":
        host = Environment.Production
    elif env == "development":
        host = Environment.Development
    else:
        host = Environment.Sandbox

    configuration = Configuration(host=host, api_key={"clientId": client_id, "secret": secret})
    api_client = ApiClient(configuration)
    return plaid_api.PlaidApi(api_client)


def _parse_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        raise ValueError("Dates must be in YYYY-MM-DD format")


def _format_money(amount: Optional[float], currency: Optional[str]) -> str:
    if amount is None:
        return "unknown"
    cur = currency or "USD"
    prefix = "$" if cur.upper() == "USD" else f"{cur} "
    # Show commas and two decimals
    return f"{prefix}{amount:,.2f}"


def _fetch_balances(client, access_token: str) -> Dict:
    from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest
    req = AccountsBalanceGetRequest(access_token=access_token)
    resp = client.accounts_balance_get(req)
    return resp.to_dict()


def _fetch_transactions(
    client,
    access_token: str,
    start_date: date,
    end_date: date,
) -> Dict:
    from plaid.model.transactions_get_request import TransactionsGetRequest
    from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions

    all_transactions = []
    accounts_cache = None
    count = 100
    offset = 0

    while True:
        options = TransactionsGetRequestOptions(count=count, offset=offset)
        req = TransactionsGetRequest(
            access_token=access_token,
            start_date=start_date,
            end_date=end_date,
            options=options,
        )
        resp = client.transactions_get(req)
        data = resp.to_dict()
        if accounts_cache is None:
            accounts_cache = data.get("accounts", [])
        txns = data.get("transactions", [])
        all_transactions.extend(txns)
        total = data.get("total_transactions", len(all_transactions))
        if len(all_transactions) >= total:
            break
        offset = len(all_transactions)

    return {"accounts": accounts_cache or [], "transactions": all_transactions}


def _summarize_balances(institution: str, balance_payload: Dict) -> str:
    lines: List[str] = []
    accounts = balance_payload.get("accounts", [])
    if not accounts:
        return f"{institution.title()}: No accounts found."
    lines.append(f"{institution.title()}:")
    for acct in accounts:
        name = acct.get("name") or acct.get("official_name") or "Account"
        subtype = acct.get("subtype") or acct.get("type") or ""
        balances = acct.get("balances", {})
        available = balances.get("available")
        current = balances.get("current")
        limit_amt = balances.get("limit")
        currency = balances.get("iso_currency_code") or balances.get("unofficial_currency_code")

        bal_str = f"available {_format_money(available, currency)}, current {_format_money(current, currency)}"
        if limit_amt is not None:
            bal_str += f", limit {_format_money(limit_amt, currency)}"
        if subtype:
            lines.append(f"- {name} ({subtype}): {bal_str}")
        else:
            lines.append(f"- {name}: {bal_str}")
    return "\n".join(lines)


def _write_transactions_file(
    base_dir: Path,
    institutions: List[str],
    overall_start: date,
    overall_end: date,
    collected: List[Dict],
) -> str:
    tmp_dir = base_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = tmp_dir / f"plaid_transactions_{timestamp}.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start_date": overall_start.isoformat(), "end_date": overall_end.isoformat()},
        "institutions": institutions,
        "transactions": collected,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(out_path)


def get_financial_info(
    account_names: Union[str, List[str]],
    request_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[str, Dict[str, str]]:
    """Fetch balances or transactions across selected accounts via Plaid.

    Parameters
    - account_names: "all", a single name (e.g., "PNC"), or a list of names.
      Allowed friendly names: Discover, Huntington, Ally, PNC (case-insensitive and simple synonyms).
    - request_type: "balance" or "transactions".
    - start_date: optional YYYY-MM-DD (transactions only). Defaults to today-90d.
    - end_date: optional YYYY-MM-DD (transactions only). Defaults to today.

    Returns
    - For balances: a human-readable multi-line summary string.
    - For transactions: a dict {"transactions_file": "/full/path/to/file.json"}.

    Errors are returned as {"error": "..."} for agent clarity.
    """
    try:
        client = _init_plaid_client()
        selected = _resolve_selected_accounts(account_names)
        base_dir = Path(__file__).resolve().parent.parent  # backend/

        rtype = (request_type or "").strip().lower()
        if rtype not in ("balance", "balances", "transactions", "txns", "txn"):
            return {"error": "Invalid request_type. Use 'balance' or 'transactions'."}

        if rtype in ("balance", "balances"):
            summaries: List[str] = []
            for friendly, token in selected:
                try:
                    bal = _fetch_balances(client, token)
                    summaries.append(_summarize_balances(friendly, bal))
                except Exception as e:
                    summaries.append(f"{friendly.title()}: error fetching balances: {str(e)}")
            return "\n\n".join(summaries)

        # Transactions
        start_d = _parse_date(start_date)
        end_d = _parse_date(end_date)
        if start_d and end_d and start_d > end_d:
            return {"error": "start_date must be on or before end_date"}
        if not start_d:
            start_d = date.today() - timedelta(days=90)
        if not end_d:
            end_d = date.today()

        collected_txns: List[Dict] = []
        institutions: List[str] = []

        for friendly, token in selected:
            institutions.append(friendly.title())
            try:
                data = _fetch_transactions(client, token, start_d, end_d)
                accounts = {a.get("account_id"): a for a in (data.get("accounts") or [])}
                for tx in data.get("transactions") or []:
                    acct = accounts.get(tx.get("account_id"), {})
                    acct_name = acct.get("name") or acct.get("official_name") or "Account"
                    currency = tx.get("iso_currency_code") or tx.get("unofficial_currency_code")
                    tdate = tx.get("date")
                    if isinstance(tdate, (date, datetime)):
                        tdate = tdate.isoformat()
                    collected_txns.append(
                        {
                            "institution": friendly.title(),
                            "account_id": tx.get("account_id"),
                            "account_name": acct_name,
                            "transaction_id": tx.get("transaction_id"),
                            "date": tdate,
                            "name": tx.get("name"),
                            "merchant_name": tx.get("merchant_name"),
                            "amount": tx.get("amount"),
                            "currency": currency,
                            "pending": tx.get("pending"),
                            "category": tx.get("category"),
                        }
                    )
            except Exception as e:
                collected_txns.append(
                    {
                        "institution": friendly.title(),
                        "error": f"error fetching transactions: {str(e)}",
                    }
                )

        out_file = _write_transactions_file(base_dir, institutions, start_d, end_d, collected_txns)
        return {"transactions_file": out_file}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("--- Testing Balances for Discover ---")
    balances = get_financial_info("discover", "balance")
    print(balances)

    print("\n--- Testing Transactions for Discover ---")
    transactions = get_financial_info("discover", "transactions")
    print(transactions)
