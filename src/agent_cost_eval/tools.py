"""Deterministic mock support tools. No network."""
from __future__ import annotations

from langchain_core.tools import tool

_LINE_ITEMS = [
    ("Subscription seat, Standard plan", 29.00),
    ("Cloud storage add-on, 100 GB", 10.00),
    ("Collaboration module", 6.00),
    ("Priority support entitlement", 4.00),
]


def _line_items(invoice_id: str) -> list[dict]:
    items = []
    for n, (desc, price) in enumerate(_LINE_ITEMS * 2, start=1):
        items.append(
            {
                "line": n,
                "description": f"{desc} ({invoice_id}, line {n})",
                "quantity": 1,
                "unit_price": price / 2,
                "tax_code": "SAAS-STD",
                "note": (
                    "Billed in advance for the monthly term; usage-based charges, "
                    "proration and credits are itemised on the statement of account "
                    "attached to this invoice and reflect the plan in force on the "
                    "billing date shown."
                ),
            }
        )
    return items


def _invoice(invoice_id: str, date: str) -> dict:
    return {
        "id": invoice_id,
        "date": date,
        "amount": 49.00,
        "status": "paid",
        "currency": "USD",
        "lines": _line_items(invoice_id),
    }


def _invoices() -> list[dict]:
    months = [(y, m) for y in (2023, 2024) for m in range(1, 13)][1:]  # 2023-02..2024-12 (23)
    invoices = [
        _invoice(f"inv_{y % 100}{m:02d}", f"{y}-{m:02d}-03") for y, m in months
    ]
    # The double charge: a second paid invoice in the latest month.
    invoices.append(_invoice("inv_2412b", "2024-12-04"))
    return invoices


@tool
def lookup_account(email: str) -> dict:
    """Look up a customer account by email address."""
    return {
        "account_id": "acct_1",
        "email": email,
        "plan": "Standard",
        "billing_cycle": "monthly",
        "payment_method": "card ending 4242",
    }


@tool
def get_invoices(account_id: str) -> list[dict]:
    """Get the full invoice history (two years) for an account."""
    return _invoices()


@tool
def search_help_center(query: str) -> list[dict]:
    """Search help center articles."""
    return [
        {
            "title": "Why was I charged twice?",
            "snippet": "Duplicate charges usually come from a retried payment. "
            "A duplicate is refunded automatically within 5 business days.",
        },
        {
            "title": "How refunds work",
            "snippet": "Refunds return to the original payment method.",
        },
    ]


TOOLS = [lookup_account, get_invoices, search_help_center]
