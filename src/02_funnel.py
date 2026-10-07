"""Step 2: Funnel analysis (customer journey + order journey proxies)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
F = ROOT / "outputs" / "figures"
F.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(T / "transactions_clean.csv", parse_dates=["InvoiceDate"])
inv = pd.read_csv(T / "invoices.csv", parse_dates=["invoice_date"])
cust = pd.read_csv(T / "customers.csv", parse_dates=["first_purchase", "last_purchase"])

# ---------------------------------------------------------------- order journey
# Orders started = all invoices; completed = non-cancel; kept = completed and
# not subsequently refunded (matched to a later cancellation by customer+stock).
n_started = len(inv)
n_completed = int((~inv["is_cancel"]).sum())
n_cancelled = int(inv["is_cancel"].sum())

comp_lines = df[~df["is_cancel"] & (df["Quantity"] > 0) & (df["Price"] > 0)]
cancel_lines = df[df["is_cancel"] & (df["Quantity"] < 0)].copy()

# match each cancel line to latest prior completed line (same customer+stock)
comp_lines = comp_lines.sort_values("InvoiceDate")
cancel_lines = cancel_lines.sort_values("InvoiceDate")
merged = pd.merge_asof(
    cancel_lines,
    comp_lines[["Customer ID", "StockCode", "InvoiceDate", "Invoice", "revenue"]],
    on="InvoiceDate",
    by=["Customer ID", "StockCode"],
    direction="backward",
    suffixes=("_cancel", "_orig"),
)
refunded_invoices = set(merged["Invoice_orig"].dropna())
inv["is_refunded"] = inv["Invoice"].isin(refunded_invoices)
n_refunded = int((inv["is_refunded"] & ~inv["is_cancel"]).sum())
n_kept = n_completed - n_refunded

order_funnel = pd.DataFrame(
    [
        {"stage": "1. Order started (invoice created)", "orders": n_started},
        {"stage": "2. Order completed (not cancelled at creation)", "orders": n_completed},
        {"stage": "3. Order kept (no refund matched)", "orders": n_kept},
    ]
)
order_funnel["step_conversion"] = (order_funnel["orders"] / order_funnel["orders"].shift(1)).round(4)
order_funnel["overall_conversion"] = (order_funnel["orders"] / n_started).round(4)
order_funnel["step_conversion"] = order_funnel["step_conversion"].fillna(1.0)
order_funnel.to_csv(T / "funnel_orders.csv", index=False)

# ---------------------------------------------------------------- customer journey
n_all = int(df["Customer ID"].nunique())  # any engagement incl. cancel-only
n_first = int((cust["n_completed_inv"] >= 1).sum())
n_repeat = int((cust["n_completed_inv"] >= 2).sum())
n_active = int((cust["n_completed_inv"] >= 5).sum())

# customers who only ever appear in cancellations (bought-then-fully-returned look)
cancel_only = set(df[df["is_cancel"]]["Customer ID"]) - set(comp_lines["Customer ID"])

customer_funnel = pd.DataFrame(
    [
        {"stage": "1. Active customer (any engagement)", "customers": n_all},
        {"stage": "2. First purchase", "customers": n_first},
        {"stage": "3. Repeat purchase (2+ orders)", "customers": n_repeat},
        {"stage": "4. Loyal (5+ orders)", "customers": n_active},
    ]
)
customer_funnel["step_conversion"] = (customer_funnel["customers"] / customer_funnel["customers"].shift(1)).round(4)
customer_funnel["overall_conversion"] = (customer_funnel["customers"] / n_all).round(4)
customer_funnel["step_conversion"] = customer_funnel["step_conversion"].fillna(1.0)
customer_funnel.to_csv(T / "funnel_customers.csv", index=False)

# ---------------------------------------------------------------- new vs returning
# For each completed invoice: is it the customer's 1st or 2nd+ order?
completed_inv = inv[~inv["is_cancel"]].copy()
completed_inv = completed_inv.sort_values("invoice_date")
completed_inv["order_seq"] = completed_inv.groupby("Customer ID").cumcount() + 1
completed_inv["type"] = np.where(completed_inv["order_seq"] == 1, "New customer order", "Returning customer order")

seg = (
    completed_inv.groupby("type")
    .agg(orders=("Invoice", "count"), revenue=("revenue", "sum"))
    .reset_index()
)
seg["avg_order_value"] = (seg["revenue"] / seg["orders"]).round(2)
seg.to_csv(T / "segment_new_vs_returning.csv", index=False)

# cancellation rate by order sequence (do new customers cancel more?)
all_inv_seq = inv.copy().sort_values("invoice_date")
all_inv_seq["order_seq"] = all_inv_seq.groupby("Customer ID").cumcount() + 1
all_inv_seq["type"] = np.where(all_inv_seq["order_seq"] == 1, "New", "Returning")
cancel_rate = (
    all_inv_seq.groupby("type")["is_cancel"].mean().round(4).rename("cancel_rate").reset_index()
)
cancel_rate.to_csv(T / "cancel_rate_by_type.csv", index=False)

# ---------------------------------------------------------------- cart value tiers
tier_stats = (
    inv[~inv["is_cancel"]]
    .groupby("value_tier")
    .agg(orders=("Invoice", "count"), revenue=("revenue", "sum"), refund_rate=("is_refunded", "mean"))
    .reset_index()
)
tier_stats["refund_rate"] = tier_stats["refund_rate"].round(4)
tier_stats["aov"] = (tier_stats["revenue"] / tier_stats["orders"]).round(2)
tier_stats.to_csv(T / "value_tier_stats.csv", index=False)

# ---------------------------------------------------------------- drop-off focus
# Biggest single friction point in this data: cart->completed (cancellation) and
# completed->kept (refund).
dropoffs = pd.DataFrame(
    [
        {"transition": "Started -> Completed", "lost": n_started - n_completed,
         "loss_rate": round((n_started - n_completed) / n_started, 4)},
        {"transition": "Completed -> Kept", "lost": n_completed - n_kept,
         "loss_rate": round((n_completed - n_kept) / n_completed, 4)},
        {"transition": "First purchase -> Repeat", "lost": n_first - n_repeat,
         "loss_rate": round((n_first - n_repeat) / n_first, 4)},
    ]
)
dropoffs.to_csv(T / "dropoffs.csv", index=False)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for ax, tbl, title, unit in [
    (axes[0], order_funnel, "Order Journey Funnel", "orders"),
    (axes[1], customer_funnel, "Customer Journey Funnel", "customers"),
]:
    bars = ax.barh(tbl["stage"][::-1], tbl[unit][::-1], color=colors[: len(tbl)][::-1])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(unit)
    maxv = tbl[unit].max()
    for b, v, oc in zip(bars, tbl[unit][::-1], tbl["overall_conversion"][::-1]):
        ax.text(b.get_width() + maxv * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:,}  ({oc:.1%} overall)", va="center", fontsize=9)
    ax.set_xlim(0, maxv * 1.35)

fig.suptitle("E-Commerce Conversion Funnels", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(F / "funnel.png", bbox_inches="tight")
plt.close(fig)

# drop-off waterfall
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(dropoffs["transition"], dropoffs["loss_rate"] * 100, color="#C44E52", width=0.55)
for b, r, l in zip(bars, dropoffs["loss_rate"], dropoffs["lost"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
            f"{r:.1%}\n({l:,} lost)", ha="center", fontsize=10)
ax.set_ylabel("Loss rate (%)")
ax.set_title("Where Users Drop Off", fontweight="bold")
ax.set_ylim(0, dropoffs["loss_rate"].max() * 100 * 1.35)
fig.tight_layout()
fig.savefig(F / "dropoffs.png")
plt.close(fig)

summary = {
    "order_funnel": order_funnel.to_dict(orient="records"),
    "customer_funnel": customer_funnel.to_dict(orient="records"),
    "dropoffs": dropoffs.to_dict(orient="records"),
    "cancel_only_customers": len(cancel_only),
    "cancel_rate_overall": round(float(inv["is_cancel"].mean()), 4),
    "refund_rate_on_completed": round(n_refunded / n_completed, 4),
    "repeat_rate": round(n_repeat / n_first, 4),
}
(T / "funnel_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
