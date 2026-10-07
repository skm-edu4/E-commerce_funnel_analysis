"""Step 5: Assemble the full analysis dashboard (single PNG)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
F = ROOT / "outputs" / "figures"

order_funnel = pd.read_csv(T / "funnel_orders.csv")
cust_funnel = pd.read_csv(T / "funnel_customers.csv")
dropoffs = pd.read_csv(T / "dropoffs.csv")
retention = pd.read_csv(T / "cohort_retention.csv", index_col=0)
retention.columns = retention.columns.astype(int)
ab = json.load(open(T / "ab_test_results.json"))
clean = json.load(open(T / "cleaning_report.json"))
cohort = json.load(open(T / "cohort_summary.json"))
funnel_sum = json.load(open(T / "funnel_summary.json"))

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

fig = plt.figure(figsize=(18, 13))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3,
              left=0.055, right=0.975, top=0.895, bottom=0.06)

fig.suptitle("E-Commerce Conversion Funnel & A/B Test Analysis",
             fontsize=20, fontweight="bold", y=0.965)
fig.text(0.5, 0.925,
         f"Online Retail II dataset  |  {clean['rows_clean']:,} transactions  |  "
         f"{clean['n_customers']:,} customers  |  {clean['date_min'][:10]} to {clean['date_max'][:10]}",
         ha="center", fontsize=11, color="#555555")

# ---------------------------------------------------------------- KPI strip
kpis = [
    ("Order completion", f"{ab['baseline_completion_rate']:.1%}"),
    ("Order cancel rate", f"{funnel_sum['cancel_rate_overall']:.1%}"),
    ("Repeat purchase", f"{funnel_sum['repeat_rate']:.1%}"),
    ("M1 retention", f"{cohort['avg_retention_m1']:.1%}"),
    ("A/B lift", f"+{ab['observed_lift_rel'] * 100:.1f}%"),
    ("p-value", f"{ab['p_value']:.1e}"),
]
kpi_ax = fig.add_subplot(gs[0, :])
kpi_ax.axis("off")
for i, (label, value) in enumerate(kpis):
    x = (i + 0.5) / len(kpis)
    kpi_ax.text(x, 0.62, value, ha="center", va="center", fontsize=17,
                fontweight="bold", color="#1a4d7a",
                transform=kpi_ax.transAxes)
    kpi_ax.text(x, 0.18, label, ha="center", va="center", fontsize=10,
                color="#666666", transform=kpi_ax.transAxes)
kpi_ax.plot([0, 1], [0, 0], color="#dddddd", transform=kpi_ax.transAxes, clip_on=False)

# ---------------------------------------------------------------- panel 1: order funnel
ax1 = fig.add_subplot(gs[1, 0])
labels = ["Started", "Completed", "Kept"]
vals = order_funnel["orders"].tolist()
bars = ax1.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"], width=0.6)
for b, v, oc in zip(bars, vals, order_funnel["overall_conversion"]):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 600,
             f"{v:,}\n{oc:.0%}", ha="center", fontsize=9)
ax1.set_title("Order Journey Funnel", fontweight="bold")
ax1.set_ylim(0, max(vals) * 1.22)
ax1.set_ylabel("Orders")

# ---------------------------------------------------------------- panel 2: customer funnel
ax2 = fig.add_subplot(gs[1, 1])
labels2 = ["Active", "1st buy", "Repeat", "Loyal"]
vals2 = cust_funnel["customers"].tolist()
bars = ax2.bar(labels2, vals2, color="#4C72B0", width=0.6, alpha=0.85)
for b, v, oc in zip(bars, vals2, cust_funnel["overall_conversion"]):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 90,
             f"{v:,}\n{oc:.0%}", ha="center", fontsize=9)
ax2.set_title("Customer Journey Funnel", fontweight="bold")
ax2.set_ylim(0, max(vals2) * 1.22)
ax2.set_ylabel("Customers")

# ---------------------------------------------------------------- panel 3: drop-offs
ax3 = fig.add_subplot(gs[1, 2])
d = dropoffs.iloc[::-1]
ax3.barh(d["transition"], d["loss_rate"] * 100, color="#C44E52", height=0.5)
for i, (r, l) in enumerate(zip(d["loss_rate"], d["lost"])):
    ax3.text(r * 100 + 0.5, i, f"{r:.1%} ({l:,})", va="center", fontsize=9)
ax3.set_xlim(0, d["loss_rate"].max() * 100 * 1.45)
ax3.set_xlabel("Loss rate (%)")
ax3.set_title("Biggest Drop-Offs", fontweight="bold")

# ---------------------------------------------------------------- panel 4: cohort heatmap
ax4 = fig.add_subplot(gs[2, 0])
plot = retention.iloc[:12, :10] * 100
im = ax4.imshow(plot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=60)
ax4.set_xticks(range(plot.shape[1]))
ax4.set_xticklabels(plot.columns, fontsize=8)
ax4.set_yticks(range(plot.shape[0]))
ax4.set_yticklabels(plot.index, fontsize=8)
ax4.set_xlabel("Months since first purchase", fontsize=9)
ax4.set_title("Cohort Retention (%)", fontweight="bold")
fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.03)

# ---------------------------------------------------------------- panel 5: A/B results
ax5 = fig.add_subplot(gs[2, 1])
rates = [ab["observed_control_rate"], ab["observed_variant_rate"]]
errs = [1.96 * np.sqrt(p * (1 - p) / (4 * ab["required_n_per_group"])) for p in rates]
bars = ax5.bar(["Control", "Variant"], rates, yerr=errs, capsize=5,
               color=["#4C72B0", "#55A868"], width=0.5)
for b, r in zip(bars, rates):
    ax5.text(b.get_x() + b.get_width() / 2, r + 0.015, f"{r:.2%}",
             ha="center", fontweight="bold", fontsize=10)
ax5.set_ylim(min(rates) - 0.05, max(rates) + 0.05)
ax5.set_ylabel("Completion rate")
sig = "significant" if ab["significant_at_005"] else "not significant"
ax5.set_title(f"A/B Test: +{ab['observed_lift_rel']*100:.1f}% lift\n"
              f"p = {ab['p_value']:.1e} ({sig})", fontweight="bold")

# ---------------------------------------------------------------- panel 6: novelty
ax6 = fig.add_subplot(gs[2, 2])
weeks = [w["week"] for w in ab["weekly_novelty"]]
lifts = [w["lift_rel"] * 100 for w in ab["weekly_novelty"]]
pvals = [w["p_value"] for w in ab["weekly_novelty"]]
colors = ["#55A868" if p < 0.05 else "#C44E52" for p in pvals]
ax6.plot(weeks, lifts, marker="o", color="#4C72B0", linewidth=2, zorder=1)
ax6.scatter(weeks, lifts, c=colors, s=90, zorder=2, edgecolors="white")
for x, y, p in zip(weeks, lifts, pvals):
    ax6.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8)
ax6.axhline(ab["assumed_relative_lift"] * 100, ls=":", color="gray")
ax6.set_xlabel("Experiment week")
ax6.set_ylabel("Relative lift (%)")
ax6.set_xticks(weeks)
ax6.set_title("Novelty Effect Check\n(green = p<0.05, red = not)", fontweight="bold")
ax6.grid(alpha=0.3)

out = F / "dashboard.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"dashboard written: {out}")
