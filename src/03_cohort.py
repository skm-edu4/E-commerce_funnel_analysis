"""Step 3: Cohort retention analysis (monthly cohorts, new vs returning)."""
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

completed = pd.read_csv(T / "transactions_completed.csv", parse_dates=["InvoiceDate"])
completed = completed[completed["is_valid_qty_price"]]

completed["order_month"] = completed["InvoiceDate"].dt.to_period("M")
first_month = completed.groupby("Customer ID")["order_month"].min().rename("cohort")
completed = completed.join(first_month, on="Customer ID")
completed["age"] = (completed["order_month"] - completed["cohort"]).apply(lambda x: x.n)

# ---------------------------------------------------------------- retention matrix
cohort_sizes = completed.groupby("cohort")["Customer ID"].nunique()
retained = (
    completed[completed["age"] >= 0]
    .groupby(["cohort", "age"])["Customer ID"]
    .nunique()
    .unstack("age")
)
retention = retained.div(cohort_sizes, axis=0).round(4)

# keep readable slice: first 15 cohorts, ages 0..11
retention_plot = retention.iloc[:15, :12]
retention.to_csv(T / "cohort_retention.csv")

# ---------------------------------------------------------------- aggregate curve
agg_ret = retention.mean(numeric_only=True).round(4)
agg_ret.index.name = "month_age"
agg_ret.rename("retention_rate").reset_index().to_csv(T / "retention_curve.csv", index=False)

# ---------------------------------------------------------------- new vs returning monthly
monthly = (
    completed.groupby([completed["order_month"].astype(str), completed["age"].eq(0)])
    .agg(orders=("Invoice", "nunique"), revenue=("revenue", "sum"), customers=("Customer ID", "nunique"))
    .reset_index(names=["month", "is_new"])
)
monthly["cohort_type"] = np.where(monthly["is_new"], "New customers", "Returning customers")
monthly.to_csv(T / "monthly_new_vs_returning.csv", index=False)

# AOV comparison new vs returning (order level)
inv = pd.read_csv(T / "invoices.csv", parse_dates=["invoice_date"])
comp_inv = inv[~inv["is_cancel"]].sort_values("invoice_date")
comp_inv["seq"] = comp_inv.groupby("Customer ID").cumcount() + 1
aov = (
    comp_inv.groupby(comp_inv["seq"].eq(1).map({True: "New", False: "Returning"}))["revenue"]
    .agg(["count", "mean", "median"])
    .round(2)
)
aov.to_csv(T / "aov_new_vs_returning.csv")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# heatmap
fig, ax = plt.subplots(figsize=(11, 7))
data = retention_plot.values * 100
im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=60)
ax.set_xticks(range(retention_plot.shape[1]))
ax.set_xticklabels(retention_plot.columns.astype(int))
ax.set_yticks(range(retention_plot.shape[0]))
ax.set_yticklabels(retention_plot.index.astype(str))
ax.set_xlabel("Months since first purchase")
ax.set_ylabel("Cohort (first purchase month)")
ax.set_title("Monthly Cohort Retention (%)", fontweight="bold")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not np.isnan(data[i, j]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > 40 else "#333333")
fig.colorbar(im, ax=ax, label="Retention %")
fig.tight_layout()
fig.savefig(F / "cohort_heatmap.png")
plt.close(fig)

# aggregate retention curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(agg_ret.index, agg_ret.values * 100, marker="o", color="#4C72B0", linewidth=2)
for x, y in zip(agg_ret.index, agg_ret.values * 100):
    if x <= 6:
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("Months since first purchase")
ax.set_ylabel("Avg retention (%)")
ax.set_title("Average Retention Curve (all cohorts)", fontweight="bold")
ax.set_xticks(range(0, len(agg_ret)))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(F / "retention_curve.png")
plt.close(fig)

# new vs returning stacked monthly revenue
fig, ax = plt.subplots(figsize=(11, 5))
pivot = monthly.pivot(index="month", columns="cohort_type", values="revenue").fillna(0)
pivot = pivot[["New customers", "Returning customers"]]
pivot.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#55A868"], width=0.8)
ax.set_ylabel("Revenue (£)")
ax.set_title("Monthly Revenue: New vs Returning Customers", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(F / "new_vs_returning.png")
plt.close(fig)

summary = {
    "cohorts": int(len(cohort_sizes)),
    "avg_retention_m1": float(agg_ret.get(1, np.nan)),
    "avg_retention_m3": float(agg_ret.get(3, np.nan)),
    "avg_retention_m6": float(agg_ret.get(6, np.nan)),
    "new_customer_aov": float(aov.loc["New", "mean"]),
    "returning_customer_aov": float(aov.loc["Returning", "mean"]),
    "returning_share_of_revenue": round(
        float(monthly[~monthly["is_new"]]["revenue"].sum() / monthly["revenue"].sum()), 4
    ),
}
json.dump(summary, open(T / "cohort_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))
