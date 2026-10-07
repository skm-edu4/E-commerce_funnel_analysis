"""Step 1: Data cleaning & feature engineering for Online Retail II dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_retail_ii.xlsx"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

report = {}

# ---------------------------------------------------------------- load
xl = pd.ExcelFile(DATA)
frames = []
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    df["source_sheet"] = sheet
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
report["rows_raw"] = int(len(raw))
report["sheets"] = xl.sheet_names

# The two sheets overlap on the boundary day (2010-12-01): 1,088 invoices
# appear in BOTH sheets with identical rows. There are also exact duplicate
# rows within sheets. Double counting inflates revenue and order counts,
# so drop every exact duplicate (keep first occurrence).
dup_cols = [c for c in raw.columns if c != "source_sheet"]
within_sheet_dups = sum(int(f.duplicated(subset=[c for c in dup_cols if c != "source_sheet"]).sum()) for f in frames)
cross_sheet_dups = int(raw.duplicated(subset=dup_cols).sum()) - within_sheet_dups
report["dropped_duplicate_rows"] = int(raw.duplicated(subset=dup_cols).sum())
report["dup_within_sheets"] = within_sheet_dups
report["dup_cross_sheets_boundary_overlap"] = cross_sheet_dups
raw = raw.drop_duplicates(subset=dup_cols, keep="first").reset_index(drop=True)
report["rows_after_dedup"] = int(len(raw))

df = raw.copy()
df["Invoice"] = df["Invoice"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

# ---------------------------------------------------------------- flags
df["is_cancel"] = df["Invoice"].str.startswith("C")

# Administrative / non-merandise stock codes (shipping, fees, samples...)
ADMIN_CODES = {"POST", "M", "BANK CHARGES", "C2", "DOT", "CRUK", "PADS", "S", "AMAZONFEE", "D"}
df["is_admin"] = df["StockCode"].isin(ADMIN_CODES)

df["is_free"] = df["Price"] <= 0
df["is_bad_qty"] = df["Quantity"] <= 0
df["is_valid_qty_price"] = (~df["is_bad_qty"]) & (~df["is_free"]) & (~df["is_admin"])

# ---------------------------------------------------------------- clean
no_customer = df["Customer ID"].isna().sum()
report["dropped_null_customer"] = int(no_customer)
df = df[df["Customer ID"].notna()].copy()

report["dropped_zero_neg_price"] = int((df["Price"] <= 0).sum())
report["dropped_admin_rows"] = int(df["is_admin"].sum())

# Keep cancels (needed for abandonment analysis) but mark them.
# Completed purchase lines: positive qty/price, non-cancel, non-admin.
df["revenue"] = df["Quantity"] * df["Price"]

# ---------------------------------------------------------------- features
df["Customer ID"] = df["Customer ID"].astype("int64").astype(str)
df["date"] = df["InvoiceDate"].dt.normalize()
df["year_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

completed = df[~df["is_cancel"] & df["is_valid_qty_price"]].copy()

# invoice level
inv = (
    df.groupby(["Invoice", "Customer ID"])
    .agg(
        invoice_date=("InvoiceDate", "min"),
        is_cancel=("is_cancel", "max"),
        revenue=("revenue", "sum"),
        n_lines=("StockCode", "size"),
        n_items=("Quantity", "sum"),
    )
    .reset_index()
)
inv["revenue"] = inv["revenue"].round(2)
inv = inv[inv["revenue"] != 0]  # drop pure administrative invoices (keep cancels)

# cart value tiers (on completed invoices only)
comp_inv = inv[~inv["is_cancel"]]
q25, q75 = comp_inv["revenue"].quantile([0.25, 0.75])


def tier(r):
    if r <= q25:
        return "Low"
    if r <= q75:
        return "Mid"
    return "High"


inv["value_tier"] = np.where(inv["is_cancel"], "Cancelled", comp_inv["revenue"].reindex(inv.index).apply(tier))
report["cart_tier_bounds"] = {"q25": round(float(q25), 2), "q75": round(float(q75), 2)}

# customer level
first_purchase = (
    completed.groupby("Customer ID")["InvoiceDate"].min().rename("first_purchase_dt")
)
cust = (
    completed.groupby("Customer ID")
    .agg(
        n_completed_inv=("Invoice", "nunique"),
        total_revenue=("revenue", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        n_countries=("Country", "nunique"),
    )
    .reset_index()
)
cust["country"] = (
    completed.sort_values("InvoiceDate").groupby("Customer ID")["Country"].last().values
)
cust["total_revenue"] = cust["total_revenue"].round(2)
cust["first_cohort_month"] = cust["first_purchase"].dt.to_period("M").astype(str)
cust["lifetime_days"] = (cust["last_purchase"] - cust["first_purchase"]).dt.days

# days from first-ever activity (incl. cancels) to first completed purchase
first_seen = df.groupby("Customer ID")["InvoiceDate"].min().rename("first_seen")
cust = cust.merge(first_seen, on="Customer ID", how="left")
cust["time_to_purchase_days"] = (
    cust["first_purchase"] - cust["first_seen"]
).dt.total_seconds() / 86400

# cancel behavior per customer
cancel_cust = (
    df[df["is_cancel"]].groupby("Customer ID")["Invoice"].nunique().rename("n_cancel_inv")
)
cust = cust.merge(cancel_cust, on="Customer ID", how="left")
cust["n_cancel_inv"] = cust["n_cancel_inv"].fillna(0).astype(int)

# ---------------------------------------------------------------- save
df.to_csv(OUT / "transactions_clean.csv", index=False)
completed.to_csv(OUT / "transactions_completed.csv", index=False)
inv.to_csv(OUT / "invoices.csv", index=False)
cust.to_csv(OUT / "customers.csv", index=False)

report["rows_clean"] = int(len(df))
report["rows_completed"] = int(len(completed))
report["date_min"] = str(df["InvoiceDate"].min())
report["date_max"] = str(df["InvoiceDate"].max())
report["n_customers"] = int(cust.shape[0])
report["n_invoices"] = int(inv.shape[0])
report["n_cancellations"] = int(df["is_cancel"].sum())
report["total_revenue"] = round(float(completed["revenue"].sum()), 2)

(OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
