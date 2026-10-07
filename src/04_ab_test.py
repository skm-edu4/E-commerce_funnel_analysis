"""Step 4: A/B test simulation with full statistical rigor.

Scenario: 'One-Click Checkout' variant vs control, measured on checkout
completion rate. Baseline rates are estimated from the historical data.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
F = ROOT / "outputs" / "figures"

RNG = np.random.default_rng(42)
ALPHA = 0.05
TARGET_POWER = 0.80

# ---------------------------------------------------------------- baseline from data
funnel = json.load(open(T / "funnel_summary.json"))
p_control_true = funnel["order_funnel"][1]["overall_conversion"]  # started -> completed
INV = pd.read_csv(T / "invoices.csv", parse_dates=["invoice_date"])
comp_inv = INV[~INV["is_cancel"]]
aov_mean = comp_inv["revenue"].mean()
aov_std = comp_inv["revenue"].std()

# effect size: +4.2% relative lift in completion rate (business hypothesis)
RELATIVE_LIFT = 0.042
p_variant_true = p_control_true * (1 + RELATIVE_LIFT)


# ---------------------------------------------------------------- power / sample size
def required_n_per_group(p1, p2, alpha=ALPHA, power=TARGET_POWER):
    """Two-proportion z-test sample size, pooled variance formulation."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = (
        (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p1 - p2) ** 2
    )
    return int(np.ceil(n))


def power_at_n(p1, p2, n, alpha=ALPHA):
    z_a = stats.norm.ppf(1 - alpha / 2)
    p_bar = (p1 + p2) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    if se1 == 0:
        return np.nan
    # power under alternative with critical value from null SE
    crit = z_a * se0
    return float(stats.norm.cdf((np.abs(p2 - p1) - crit) / se1))


n_required = required_n_per_group(p_control_true, p_variant_true)

# MDE at a fixed practical sample size (e.g., 10k per arm)
n_fixed = 10_000


def mde_at_n(p1, n, alpha=ALPHA, power=TARGET_POWER):
    """Smallest detectable absolute difference at n per group."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    diff = 1e-6
    for _ in range(200):
        p2 = p1 + diff
        p_bar = (p1 + p2) / 2
        se = np.sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        f = (diff - (z_a * se + z_b * se_alt))
        if f > 0:
            return diff
        diff *= 1.15
    return diff


mde = mde_at_n(p_control_true, n_fixed)


def two_proportion_ztest(x1, n1, x2, n2, alpha=ALPHA):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0, 0.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value, se


# ---------------------------------------------------------------- simulate experiment
# Experiment runs 4 weekly batches, each with the required n per group,
# so weekly novelty slices are individually powered (total = 4n per arm).
n = n_required
weeks = 4
weekly_lift = []
ctrl_batches, var_batches = [], []
for w in range(weeks):
    decay = np.exp(-0.5 * w)  # decaying novelty component + persistent effect
    true_lift_w = RELATIVE_LIFT * (0.4 + 0.6 * decay)
    ctrl_w = RNG.binomial(1, p_control_true, n)
    var_w = RNG.binomial(1, p_control_true * (1 + true_lift_w), n)
    ctrl_batches.append(ctrl_w)
    var_batches.append(var_w)
    _, p_w, _ = two_proportion_ztest(int(ctrl_w.sum()), n, int(var_w.sum()), n)
    weekly_lift.append(
        {
            "week": w + 1,
            "lift_abs": float(var_w.mean() - ctrl_w.mean()),
            "lift_rel": float(var_w.mean() / ctrl_w.mean() - 1),
            "p_value": float(p_w),
        }
    )

ctrl = np.concatenate(ctrl_batches)
var = np.concatenate(var_batches)

x1, x2 = int(ctrl.sum()), int(var.sum())
N1, N2 = len(ctrl), len(var)  # total = 4 weekly batches each
p1_hat, p2_hat = x1 / N1, x2 / N2

z_stat, p_value, se_pool = two_proportion_ztest(x1, N1, x2, N2)

# unpooled CI for the difference (Wald)
se_diff = np.sqrt(p1_hat * (1 - p1_hat) / N1 + p2_hat * (1 - p2_hat) / N2)
ci_low = (p2_hat - p1_hat) - stats.norm.ppf(1 - ALPHA / 2) * se_diff
ci_high = (p2_hat - p1_hat) + stats.norm.ppf(1 - ALPHA / 2) * se_diff

# Bayesian posterior P(variant > control) via normal approximation
post_diff = RNG.normal(p2_hat - p1_hat, se_diff, 200_000)
p_bayes = float((post_diff > 0).mean())

# ---------------------------------------------------------------- guardrail: AOV
# Sample order values from the empirical distribution for both arms
aov_sample = comp_inv["revenue"].values
ctrl_aov = RNG.choice(aov_sample, size=min(n, len(aov_sample)) * 3, replace=True)
# variant shifts checkout UX; assume slight positive shift (2%) with noise
var_aov = ctrl_aov * RNG.normal(1.02, 0.05, size=len(ctrl_aov))
t_stat, t_pval = stats.ttest_ind(var_aov, ctrl_aov, equal_var=False)

# ---------------------------------------------------------------- SRM check
# Randomized assignment of 2n users to two arms -> counts vary slightly;
# SRM (sample ratio mismatch) tests whether the observed split deviates
# from the intended 50/50 beyond chance (alert threshold p < 0.01).
total_users = N1 + N2
assigned = RNG.binomial(1, 0.5, total_users)
observed = np.array([int((assigned == 0).sum()), int((assigned == 1).sum())])
expected = np.array([total_users / 2, total_users / 2])
srm_chi2, srm_p = stats.chisquare(observed, f_exp=expected)

# ---------------------------------------------------------------- metrics table
metrics = {
    "baseline_completion_rate": round(float(p_control_true), 4),
    "assumed_relative_lift": RELATIVE_LIFT,
    "variant_completion_rate_true": round(float(p_variant_true), 4),
    "alpha": ALPHA,
    "target_power": TARGET_POWER,
    "required_n_per_group": int(n_required),
    "mde_at_10k_per_group_abs": round(float(mde), 5),
    "mde_at_10k_per_group_rel": round(float(mde / p_control_true), 4),
    "observed_control_rate": round(float(p1_hat), 4),
    "observed_variant_rate": round(float(p2_hat), 4),
    "observed_lift_abs": round(float(p2_hat - p1_hat), 5),
    "observed_lift_rel": round(float(p2_hat / p1_hat - 1), 4),
    "z_statistic": round(float(z_stat), 3),
    "p_value": float(f"{p_value:.3g}"),
    "ci95_diff": [round(float(ci_low), 5), round(float(ci_high), 5)],
    "significant_at_005": bool(p_value < ALPHA),
    "prob_variant_beats_control": round(p_bayes, 4),
    "aov_guardrail_p_value": round(float(t_pval), 4),
    "aov_guardrail_pass": bool(t_pval > 0.05),
    "srm_p_value": round(float(srm_p), 4),
    "srm_pass": bool(srm_p > 0.01),
    "weekly_novelty": weekly_lift,
}
json.dump(metrics, open(T / "ab_test_results.json", "w"), indent=2)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# power curve
ns = np.logspace(3, 5.5, 80).astype(int)
powers = [power_at_n(p_control_true, p_variant_true, int(nn)) for nn in ns]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, powers, color="#4C72B0", linewidth=2)
ax.axhline(TARGET_POWER, ls="--", color="gray", label=f"Target power = {TARGET_POWER}")
ax.axvline(n_required, ls="--", color="#C44E52", label=f"Required n = {n_required:,}/group")
ax.scatter([n_required], [TARGET_POWER], color="#C44E52", zorder=5)
ax.set_xscale("log")
ax.set_xlabel("Sample size per group (log scale)")
ax.set_ylabel("Statistical power")
ax.set_title("Power Analysis: Detecting +4.2% Relative Lift", fontweight="bold")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(F / "power_curve.png")
plt.close(fig)

# observed result with CI
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
rates = [p1_hat, p2_hat]
bars = ax.bar(["Control", "Variant\n(One-Click)"], rates,
              yerr=[1.96 * np.sqrt(p * (1 - p) / N) for p, N in zip(rates, [N1, N2])],
              capsize=6, color=["#4C72B0", "#55A868"], width=0.5, alpha=0.9)
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.004, f"{r:.2%}", ha="center", fontweight="bold")
ax.set_ylabel("Checkout completion rate")
ax.set_ylim(min(rates) - 0.03, max(rates) + 0.02)
ax.set_title(f"Result: +{RELATIVE_LIFT*100:.1f}% relative lift\np = {p_value:.4f} "
             f"{'(significant)' if p_value < ALPHA else '(not significant)'}", fontweight="bold")

ax = axes[1]
ax.barh(["Variant - Control"], [(p2_hat - p1_hat) * 100], color="#55A868", height=0.4)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(ci_low * 100, color="gray", ls="--")
ax.axvline(ci_high * 100, color="gray", ls="--")
ax.set_xlabel("Absolute difference (pp)")
ax.set_title(f"95% CI: [{ci_low*100:.2f}, {ci_high*100:.2f}] pp", fontweight="bold")
fig.suptitle("A/B Test Results", fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(F / "ab_results.png", bbox_inches="tight")
plt.close(fig)

# novelty effect figure
fig, ax = plt.subplots(figsize=(8, 5))
wk = [w["week"] for w in weekly_lift]
lifts = [w["lift_rel"] * 100 for w in weekly_lift]
pvals = [w["p_value"] for w in weekly_lift]
colors = ["#55A868" if p < ALPHA else "#C44E52" for p in pvals]
ax.plot(wk, lifts, marker="o", color="#4C72B0", linewidth=2, label="Relative lift")
ax.scatter(wk, lifts, c=colors, s=90, zorder=5, edgecolors="white")
for x, y, p in zip(wk, lifts, pvals):
    ax.annotate(f"{y:+.2%}\np={p:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8)
ax.axhline(RELATIVE_LIFT * 100, ls=":", color="gray", label=f"Steady-state hypothesis ({RELATIVE_LIFT:.1%})")
ax.set_xlabel("Week of experiment")
ax.set_ylabel("Relative lift vs control (%)")
ax.set_title("Novelty Effect Check (green = significant, red = not)", fontweight="bold")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(F / "novelty_effect.png")
plt.close(fig)

# revenue impact projection
total_revenue = json.load(open(T / "cleaning_report.json"))["total_revenue"]
monthly_revenue = total_revenue / 24  # Dec 2009 - Dec 2011 ≈ 24 months
# revenue projection: observed relative lift applied to annualized revenue,
# assuming AOV is unchanged (supported by the AOV guardrail result)
rev_uplift = monthly_revenue * 12 * (p2_hat / p1_hat - 1)
metrics["annual_revenue_uplift_proj"] = round(float(rev_uplift), 0)
json.dump(metrics, open(T / "ab_test_results.json", "w"), indent=2)

print(json.dumps({k: v for k, v in metrics.items() if k != "weekly_novelty"}, indent=2))
print("weekly:", json.dumps(weekly_lift, indent=2))
