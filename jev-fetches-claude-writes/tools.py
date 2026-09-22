"""Read-only lookup tools with simulated network latency.

The records in data.json are synthetic, made up for this demo and for nothing else. Any
resemblance to real people, companies, orders or records is purely coincidental."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_LATENCY_S = 1.5
REGIONS = ("US", "EU", "UK", "IN")
ID_PATTERNS = {"customer_id": r"C-\d+", "order_id": r"O-\d+", "sku": r"SKU-\d+"}
DATA: dict[str, dict[str, Any]] = json.loads((Path(__file__).parent / "data.json").read_text())


@dataclass(frozen=True)
class ToolSpec:
    description: str
    arg: str
    table: str


TOOLS = {
    "get_customer": ToolSpec("Customer profile: region, tier and their order ids.", "customer_id", "customers"),
    "get_tickets": ToolSpec("Support tickets opened by a customer.", "customer_id", "tickets"),
    "get_order": ToolSpec("Order status, date, total and the SKUs in it.", "order_id", "orders"),
    "get_shipment": ToolSpec("Carrier tracking for an order: status, last scan, ETA.", "order_id", "shipments"),
    "get_invoice": ToolSpec("Invoice for an order: amount, paid flag and card charges.", "order_id", "invoices"),
    "get_inventory": ToolSpec("Stock on hand and restock date for a SKU.", "sku", "inventory"),
    "get_warranty": ToolSpec("Warranty length and coverage for a SKU.", "sku", "warranties"),
    "get_refund_policy": ToolSpec("Refund policy text for a region.", "region", "refund_policies"),
}


def _schema(name: str, spec: ToolSpec) -> dict[str, Any]:
    prop: dict[str, Any] = {"type": "string"}
    if spec.arg == "region":
        prop["enum"] = list(REGIONS)
    return {
        "name": name,
        "description": spec.description,
        "input_schema": {"type": "object", "properties": {spec.arg: prop}, "required": [spec.arg]},
    }


TOOL_SCHEMAS = [_schema(name, spec) for name, spec in TOOLS.items()]


async def call_tool(name: str, value: str) -> str:
    """Run one lookup. The sleep stands in for a real API round trip."""
    await asyncio.sleep(TOOL_LATENCY_S)
    record = DATA[TOOLS[name].table].get(value)
    return json.dumps(record) if record is not None else f"No record for {value}"
