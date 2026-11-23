import json

from agent_cost_eval.tools import (
    TOOLS,
    get_invoices,
    lookup_account,
    search_help_center,
)


def test_get_invoices_count():
    assert len(get_invoices.invoke({"account_id": "acct_1"})) == 24


def test_get_invoices_is_large():
    result = get_invoices.invoke({"account_id": "acct_1"})
    assert len(json.dumps(result)) >= 36000  # ~9,000 tokens at 4 chars/token


def test_deterministic():
    a = get_invoices.invoke({"account_id": "acct_1"})
    b = get_invoices.invoke({"account_id": "acct_1"})
    assert a == b


def test_double_charge_present():
    invoices = get_invoices.invoke({"account_id": "acct_1"})
    latest_month = max(i["date"][:7] for i in invoices)
    paid = [i for i in invoices if i["date"][:7] == latest_month and i["status"] == "paid"]
    assert len(paid) == 2


def test_other_tools_small():
    acct = lookup_account.invoke({"email": "jo@example.com"})
    hits = search_help_center.invoke({"query": "double charge"})
    assert len(json.dumps(acct)) < 2000
    assert len(json.dumps(hits)) < 2000


def test_tools_list():
    assert {t.name for t in TOOLS} == {"lookup_account", "get_invoices", "search_help_center"}
