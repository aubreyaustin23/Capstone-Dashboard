# Topic Analysis: 2024 Federal Agency AI Use Case Inventory

### 1. Open datasets + install libraries
import seaborn as sns
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap, hdbscan
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# import ORIGINAL dataset
aiusecase = pd.read_excel('2024_consolidated_ai_inventory_raw_v2.xls')

aiusecase.shape, aiusecase.head()

# Check open ended fields vs categories
try:
    aiusecase
except NameError:
    aiusecase = pd.read_excel("ai_use_case_features.xlsx")

df = aiusecase.copy()

# ===== Heuristic thresholds  =====
MIN_ROWS            = 50               # skip tiny columns
LONG_CHAR_THRESHOLD = 100              # chars to count a cell as "long"
OPEN_AVG_LEN_MIN    = 80               # avg char length likely open-ended
OPEN_LONG_PCT_MIN   = 0.30             # ≥30% long cells → open-ended
CAT_CARD_MAX        = 0.05             # ≤5% unique ratio → categorical-ish
CAT_AVG_LEN_MAX     = 60               # shortish text avg length
NA_PCT_MAX_IGNORE   = 0.98             # ignore columns that are ~all NA

# ===== Helpers =====
def col_stats(s: pd.Series) -> dict:
    n = s.shape[0]
    nonnull = s.dropna()
    nn = nonnull.shape[0]
    na_pct = 1 - (nn / n) if n else 1.0

    # cast to string for length measures (preserve NaNs for counts)
    s_str = nonnull.astype(str)
    # detect dates/numeric quickly
    is_numeric = pd.api.types.is_numeric_dtype(s)
    is_datetime = pd.api.types.is_datetime64_any_dtype(s)

    avg_len = s_str.str.len().mean() if nn else 0.0
    pct_long = (s_str.str.len() > LONG_CHAR_THRESHOLD).mean() if nn else 0.0
    nunique = nonnull.nunique(dropna=True)
    card_ratio = nunique / nn if nn else 0.0

    return dict(
        n=n, nn=nn, na_pct=na_pct,
        is_numeric=is_numeric, is_datetime=is_datetime,
        nunique=nunique, card_ratio=card_ratio,
        avg_len=avg_len, pct_long=pct_long
    )

def classify_col(meta: dict) -> str:
    # ignore near-empty
    if meta["n"] < MIN_ROWS or meta["na_pct"] >= NA_PCT_MAX_IGNORE:
        return "ignored (too sparse)"

    if meta["is_numeric"] or meta["is_datetime"]:
        return "dates/ids/numeric"

    # Open-ended narrative: longer text and many long cells
    if (meta["avg_len"] >= OPEN_AVG_LEN_MIN) or (meta["pct_long"] >= OPEN_LONG_PCT_MIN):
        return "open_ended"

    # Categorical/checklist: short text and low cardinality
    if (meta["card_ratio"] <= CAT_CARD_MAX) and (meta["avg_len"] <= CAT_AVG_LEN_MAX):
        return "categorical"

    # Other text (short free text, labels, short descriptions)
    return "other_text"

# ===== Compute summary for all columns =====
rows = []
for col in df.columns:
    meta = col_stats(df[col])
    label = classify_col(meta)
    rows.append({"column": col, "class": label, **meta})

summary = pd.DataFrame(rows).sort_values(["class", "column"])

# Pretty display
open_ended = summary[summary["class"]=="open_ended"].copy()
categorical = summary[summary["class"]=="categorical"].copy()
other_text = summary[summary["class"]=="other_text"].copy()

print("\n=== Open-ended text candidates (most likely narrative) ===")
print(open_ended[["column","avg_len","pct_long","card_ratio","nunique","nn"]].head(30).to_string(index=False))

print("\n=== Categorical / checklist candidates ===")
print(categorical[["column","avg_len","card_ratio","nunique","nn"]].head(50).to_string(index=False))

print("\n=== Other short text (not quite categorical, not long narrative) ===")
print(other_text[["column","avg_len","pct_long","card_ratio","nunique","nn"]].head(50).to_string(index=False))

# Save a full audit to CSV for appendix
summary.to_csv("Topic Modeling/column_text_vs_category_audit.csv", index=False)
print("\nSaved: column_text_vs_category_audit.csv")

def column_role(s, name):
    n = len(s)
    nn = s.notna().sum()
    nunique = s.dropna().nunique()
    card_ratio = nunique / nn if nn else 0
    is_num = pd.api.types.is_numeric_dtype(s)
    is_dt  = pd.api.types.is_datetime64_any_dtype(s)
    avg_len = s.dropna().astype(str).str.len().mean() if not (is_num or is_dt) else 0

    # 1) numeric/date → numeric
    if is_num or is_dt:
        return "numeric/date"

    # 2) force-categorical if variety is low (fixes long canned text)
    if nunique <= 25 and card_ratio <= 0.05:
        return "categorical"

    # 3) narrative if long/varied
    if avg_len >= 80 or card_ratio >= 0.25:
        return "open_ended"

    return "other_text"

roles = {col: column_role(df[col], col) for col in df.columns}

# Force the two narrative fields we actually want
must_narr = {
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs."
}
for col in must_narr:
    if col in roles: roles[col] = "open_ended"

# Build final lists
open_ended_cols = [c for c,r in roles.items() if r=="open_ended"]
categorical_cols = [c for c,r in roles.items() if r=="categorical"]
numeric_date_cols = [c for c,r in roles.items() if r=="numeric/date"]
other_text_cols = [c for c,r in roles.items() if r=="other_text"]

print("Open-ended (use for topic modeling):")
for c in open_ended_cols: print(" -", c)

print("\nCategorical/checklist (use as factors):")
for c in categorical_cols[:50]: print(" -", c)

# 2. Basic EDA ---

plt.figure(figsize=(12,6))
sns.heatmap(aiusecase.isna(), cbar=False)
plt.title("Missing Data Pattern")
plt.show()

# Basic info
print("Shape:", aiusecase.shape)
print("\nColumn names:\n", aiusecase.columns.tolist())

# Data types and non-null counts
aiusecase.info()

# First few rows
aiusecase.head()

# Check missing values per column
missing = aiusecase.isna().mean().sort_values(ascending=False) * 100
print("\nMissing % by column:\n", missing)

# Distribution of agencies
if 'Agency' in aiusecase.columns:
    plt.figure(figsize=(8,5))
    aiusecase['Agency'].value_counts().head(15).plot(kind='bar')
    plt.title('Top Agencies by Use Case Count')
    plt.ylabel('Count')
    plt.show()

# Retired vs Active
# Retired vs Active
retired_col = next((c for c in aiusecase.columns if re.search(r'date\s*retired', c, re.I)), None)
if retired_col:
    aiusecase['retired_flag'] = aiusecase[retired_col].notna()
    counts = aiusecase['retired_flag'].value_counts().sort_index(ascending=True)
    total = counts.sum()

    ax = counts.plot(kind='bar', color=['#2171B5', '#9ECAE1'])
    plt.title('Retired vs Active', fontsize=13)
    plt.xticks(ticks=[0, 1], labels=['Active', 'Retired'], rotation=0)
    plt.ylabel('Count')

    # Add labels (count and %)
    for i, v in enumerate(counts):
        pct = v / total * 100
        ax.text(i, v + total * 0.01, f"{v:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


# DHS vs Non-DHS count
if 'Agency' in aiusecase.columns:
    aiusecase['is_DHS'] = aiusecase['Agency'].str.contains('Homeland', case=False, na=False)
    aiusecase['is_DHS'].value_counts().plot(kind='bar')
    plt.title('DHS vs Non-DHS')
    plt.xticks(ticks=[0,1], labels=['Non-DHS','DHS'], rotation=0)
    plt.show()

#Is the AI use case rights-impacting, safety-impacting, both, or neither? (Now, high-impact)

aiusecase['Impact_Normalized'] = aiusecase['Is the AI use case rights-impacting, safety-impacting, both, or neither?'].replace({
    'Safety-Impacting': 'Safety-impacting',
    'Rights-Impacting\n': 'Rights-impacting' # Added this line to handle the newline character
})

# Get and display counts
impact_counts = aiusecase['Impact_Normalized'].value_counts(dropna=False)
print("Counts of AI Use Case Impact:")
print(impact_counts)

plt.figure(figsize=(10, 6))
impact_counts.plot(kind='bar')
plt.title('Distribution of AI Use Case Impact on Rights and Safety')
plt.xlabel('Impact Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Bar chart of proportion of impact types per agency
impact_by_agency = aiusecase.groupby('Agency')['Impact_Normalized'].value_counts(normalize=True).unstack(fill_value=0)

# Select and reorder columns for plotting
impact_by_agency = impact_by_agency[['Rights-impacting', 'Safety-impacting', 'Both', 'Neither']]

impact_by_agency.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Proportion of AI Use Case Impact Types by Agency')
plt.xlabel('Agency')
plt.ylabel('Proportion')
plt.xticks(rotation=90)
plt.legend(title='Impact Type')
plt.tight_layout()
plt.show()

# Calculate percentage and count of cases with impact (Both, Rights-impacting, Safety-impacting)
# Make sure 'Combined_Impact' column exists before calculating
if 'Combined_Impact' in aiusecase.columns:
    impact_cases = aiusecase[aiusecase['Combined_Impact'] == 'Impact (Both/Rights/Safety)']
    total_cases = len(aiusecase)
    impact_count = len(impact_cases)
    impact_percentage = (impact_count / total_cases) * 100

    print(f"Total number of cases with impact (Both, Rights-impacting, Safety-impacting): {impact_count}")
    print(f"Percentage of total cases with impact: {impact_percentage:.2f}%")
else:
    print("'Combined_Impact' column not found. Please ensure the cell creating this column has been executed.")

# --- Cleaned Impact Category Chart ---
col = 'Is the AI use case rights-impacting, safety-impacting, both, or neither?'

if col in aiusecase.columns:
    # Clean and simplify categories
    aiusecase['Impact_Normalized'] = (
        aiusecase[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            'rights-impacting': 'Rights-impacting',
            'safety-impacting': 'Safety-impacting',
            'both': 'Both',
            'neither': 'Neither',
            'nan': np.nan,
            '': np.nan
        })
        .replace(
            {
                # Group similar/ambiguous responses
                'no, use case is too new to fully assess impacts; will be reassessed before end of initiation stage.': 'Too new to assess',
                'no, use case is too new to fully assess impacts; will be reassessed before end of acquisition and development stage.': 'Too new to assess',
                'case-by-case assessment': 'Too new to assess'
            }
        )
        .fillna('Unknown')
    )

    # Group minor categories
    main_cats = ['Neither', 'Both', 'Rights-impacting', 'Safety-impacting', 'Too new to assess', 'Unknown']
    impact_counts = aiusecase['Impact_Normalized'].value_counts()
    impact_counts = impact_counts.reindex(main_cats, fill_value=0)

    total = impact_counts.sum()
    impact_pct = (impact_counts / total * 100).round(1)

    # Print summary
    summary_df = pd.DataFrame({'Count': impact_counts, 'Percent': impact_pct})
    print("\nAI Use Case Impact Summary:\n")
    print(summary_df)

    # --- Plot ---
    plt.figure(figsize=(8,5))
    ax = impact_counts.plot(kind='bar', color='#2171B5', edgecolor='black')
    plt.title('Distribution of AI Use Case Impact on Rights and Safety', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Add labels (count + %)
    for i, (label, count) in enumerate(impact_counts.items()):
        pct = count / total * 100
        ax.text(i, count + total * 0.005, f"{count:,}\n({pct:.1f}%)",
                ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print(f"Column '{col}' not found in dataset.")

# Is there available documentation for the model training and evaluation data that demonstrates the degree to which it is appropriate to be used in analysis or for making predictions?

col = "Is there available documentation for the model training and evaluation data that demonstrates the degree to which it is appropriate to be used in analysis or for making predictions?"

if col in aiusecase.columns:
    doc_avail = (
        aiusecase[col]
        .astype(str)
        .str.strip()
        .str.title()                   # Standardize casing (Yes/No/etc.)
        .replace({"Nan": np.nan, "": np.nan})
        .fillna("Unknown")
    )

    # Optional: collapse quirky variants
    doc_avail = doc_avail.replace({
        "Not Applicable": "Not applicable",
        "N/A": "Not applicable",
        "Na": "Unknown"
    })

    counts = doc_avail.value_counts()
    total = counts.sum()
    pct = (counts / total * 100).round(1)

    # Console summary
    summary_df = pd.DataFrame({"Count": counts, "Percent": pct})
    print("\nDocumentation Availability Summary:\n")
    print(summary_df)

    # Fixed order if you prefer consistent charts:
    preferred_order = [c for c in ["Yes", "No", "Not applicable", "Unknown"] if c in counts.index]
    counts = counts.reindex(preferred_order + [i for i in counts.index if i not in preferred_order])

    # Plot
    plt.figure(figsize=(7.5, 5))
    ax = counts.plot(kind="bar", color="#2171B5", edgecolor="black")
    plt.title("Documentation for Training & Evaluation Data", fontsize=14)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Add count + % labels above bars
    for i, (label, count) in enumerate(counts.items()):
        p = (count / total * 100) if total else 0
        ax.text(i, count + max(1, total*0.01), f"{count:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()
else:
    print(f"Column not found: {col!r}")

#Checking counts:

RAW = 'Is the AI use case rights-impacting, safety-impacting, both, or neither?'
VALID = {'Rights-impacting','Safety-impacting','Both','Neither'}

# --- 1) Normalize labels robustly ---
def normalize_impact(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.lower() == 'nan':
        return np.nan

    # unify dashes/spaces/case and strip newline variants
    x = x.replace('Rights-Impacting\n', 'Rights-impacting')
    x = re.sub(r'\s*[–—-]\s*', '-', x)
    x = re.sub(r'\s+', ' ', x).strip()
    xl = x.lower()

    if re.fullmatch(r'rights[-\s]?impacting', xl):  return 'Rights-impacting'
    if re.fullmatch(r'safety[-\s]?impacting', xl):  return 'Safety-impacting'
    if re.fullmatch(r'both', xl):                   return 'Both'
    if re.fullmatch(r'neither', xl):                return 'Neither'

    # Treat “too new / reassess / case-by-case” as Neither (or switch to 'Unknown')
    if xl.startswith('no, use case is too new') or 'reassessed' in xl or 'case-by-case' in xl:
        return 'Neither'
    return x  # leave for inspection

aiusecase['Impact_Normalized'] = aiusecase[RAW].apply(normalize_impact)

# --- 2) Overall counts (spot unmapped values) ---
overall_counts = aiusecase['Impact_Normalized'].value_counts(dropna=False).sort_values(ascending=False)
print("\nOverall Impact_Normalized counts (after cleaning):")
print(overall_counts)

unexpected = [v for v in overall_counts.index if (pd.notna(v) and v not in VALID)]
if unexpected:
    print("\n⚠️ Unexpected labels still present (map or drop these):", unexpected)

# --- 3) Per-agency summary (raw counts & proportions) ---
impact_cols = ['Rights-impacting','Safety-impacting','Both']
ct = pd.crosstab(aiusecase['Agency'], aiusecase['Impact_Normalized']).reindex(columns=['Rights-impacting','Safety-impacting','Both','Neither'], fill_value=0)

summary = ct.copy()
summary['impact_count'] = summary[impact_cols].sum(axis=1)
summary['total_cases']  = summary.sum(axis=1)
summary['impact_pct']   = np.where(summary['total_cases']>0, summary['impact_count']/summary['total_cases'], np.nan)

# keep only agencies that have any impact cases
summary_plot = summary[summary['impact_count'] > 0].sort_values('impact_pct', ascending=False)

print("\nTop 10 agencies by IMPACT % (with counts):")
print(summary_plot[['impact_count','total_cases','impact_pct','Rights-impacting','Safety-impacting','Both']].head(10))

print("\nBottom 10 agencies by IMPACT % (still >0):")
print(summary_plot[['impact_count','total_cases','impact_pct','Rights-impacting','Safety-impacting','Both']].tail(10))

# --- 4) Flag very low impact counts ---
low_cut = 3  # change threshold if we want
low_impact = summary_plot[summary_plot['impact_count'] <= low_cut]
print(f"\nAgencies with impact_count <= {low_cut}: {len(low_impact)}")
print(low_impact[['impact_count','total_cases','impact_pct']].sort_values(['impact_count','impact_pct']).head(20))

# --- 5) Sanity checks that the plot will represent 100% of considered labels ---
# proportions including 'Neither' should sum to 1.0
prop = (ct.T / summary['total_cases']).T.replace([np.inf, -np.inf], np.nan).fillna(0)
row_sums = prop.sum(axis=1).round(6)
off = row_sums[row_sums.ne(1.0)]
if len(off):
    print("\n⚠️ These agencies do not sum to 1.0 (inspect raw rows):")
    print(off.head(20))
else:
    print("\n✅ All agencies sum to 100% across the four categories (after cleaning).")

# --- 6) Numbers used on the plot ---

impact_counts_for_plot = summary_plot['impact_count']
print("\nCounts used for n=… labels on the plot (first 15):")
print(impact_counts_for_plot.head(15))

from matplotlib.ticker import PercentFormatter

# --- Normalize fields ---
agency = (aiusecase['Agency'].astype(str).str.strip().replace({'nan': np.nan}).fillna('Unknown'))

impact_raw_col = 'Impact_Normalized'  # use your cleaned column if present
if impact_raw_col not in aiusecase.columns:
    impact_raw_col = 'Is the AI use case rights-impacting, safety-impacting, both, or neither?'

impact = (aiusecase[impact_raw_col]
          .astype(str).str.strip().str.lower()
          .replace({'nan': np.nan, '': np.nan})
          .replace({
              'rights-impacting': 'Rights-impacting',
              'safety-impacting': 'Safety-impacting',
              'both': 'Both',
              'neither': 'Neither',
              'no, use case is too new to fully assess impacts; will be reassessed before end of initiation stage.': 'Unknown',
              'no, use case is too new to fully assess impacts; will be reassessed before end of acquisition and development stage.': 'Unknown',
              'case-by-case assessment': 'Unknown'
          })
          .fillna('Unknown'))

df = aiusecase.copy()
df['Agency_clean'] = agency
df['Impact_clean'] = impact

impact_cols = ['Rights-impacting', 'Safety-impacting', 'Both']

# Denominator: ALL use cases per agency (incl. Unknown/Neither/etc.)
agency_totals_all = df.groupby('Agency_clean').size().rename('total_all')

# Numerator: counts for each impactful type per agency
ct_imp = pd.crosstab(df['Agency_clean'], df['Impact_clean']).reindex(columns=impact_cols, fill_value=0)

# ---- FILTER: only agencies with ≥1 impactful case
has_impact = ct_imp.sum(axis=1) > 0
ct_imp = ct_imp.loc[has_impact]
agency_totals_all = agency_totals_all.loc[ct_imp.index]

# Proportions: share of ALL use cases
prop = (ct_imp.T / agency_totals_all).T.fillna(0)

# Order agencies (pick one)
# order_idx = (ct_imp.sum(axis=1)).sort_values(ascending=False).index     # by impactful COUNT
order_idx = (ct_imp.sum(axis=1) / agency_totals_all).sort_values(ascending=False).index  # by impactful SHARE

prop_plot = prop.loc[order_idx, impact_cols]
ct_plot   = ct_imp.loc[order_idx, impact_cols]
tot_plot  = agency_totals_all.loc[order_idx]

# ---- Plot
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
fig, (ax, axN) = plt.subplots(
    ncols=2, figsize=(12, 7),
    gridspec_kw={'width_ratios': [14, 2]}
)

# Stacked horizontal bars: shares of ALL use cases
prop_plot.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.75)
ax.set_title("Share of Impactful AI Use Cases by Agency (Impactful agencies only)", pad=12)
ax.set_xlabel("Share of all use cases")
ax.set_ylabel("Agency")
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.invert_yaxis()
ax.legend(title="Impact Type", frameon=False, loc='lower right')
ax.set_yticklabels(prop_plot.index)

# Label segment counts if segment ≥ 4% of that agency's total
MIN_PCT_TO_LABEL = 0.04
for i, agency_name in enumerate(prop_plot.index):
    cum = 0.0
    for col in impact_cols:
        h = prop_plot.loc[agency_name, col]
        if h >= MIN_PCT_TO_LABEL:
            x = cum + h/2
            n_seg = int(ct_plot.loc[agency_name, col])
            ax.text(x, i, f"{n_seg}", ha='center', va='center', fontsize=8)
        cum += h

# Right gutter: clearly labeled total use cases (all statuses)
axN.set_title("Total use cases (all statuses)", pad=12)
axN.set_xlim(0, 1); axN.set_ylim(-0.5, len(prop_plot)-0.5)
axN.set_xticks([]); axN.set_yticks([])
for i, n in enumerate(tot_plot):
    axN.text(0.5, i, f"{int(n):,}", ha='center', va='center', fontsize=9, fontweight='semibold')

for spine in ['top','left','right','bottom']:
    axN.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()

# ----- Build a tidy table for a simple view -----
def short_label(name, max_len=30):
    if len(name) > max_len:
        # Split by spaces and join until max_len reached
        parts = name.split()
        current_len = 0
        shortened_parts = []
        for part in parts:
            if current_len + len(part) + (1 if shortened_parts else 0) <= max_len:
                shortened_parts.append(part)
                current_len += len(part) + (1 if shortened_parts else 0)
            else:
                break
        if shortened_parts:
            return ' '.join(shortened_parts).strip() + '...'
        else:
            return name[:max_len-3] + '...'
    return name

simple = summary.loc[summary_plot.index, ['impact_pct','total_cases']].copy()

# Top 12 ( all agencies)
TOP_K = 12
simple_top = simple.head(TOP_K)[::-1]  # reverse for horizontal top-to-bottom

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([short_label(a) for a in simple_top.index], simple_top['impact_pct'])

ax.set_title("Share of Impactful Use Cases (Rights/Safety/Both) — Top Agencies")
ax.set_xlabel("Impact share of all use cases")
ax.xaxis.set_major_formatter(PercentFormatter(1.0))

# annotate N on the right of each bar
for y, (pct, n) in enumerate(zip(simple_top['impact_pct'], simple_top['total_cases'])):
    ax.text(pct + 0.01, y, f"N={int(n)}", va='center', fontsize=9)

ax.set_xlim(0, max(0.12, simple_top['impact_pct'].max() + 0.05))  # pad for labels
plt.tight_layout()
plt.show()

# --- Top Agencies by Use Case Count ---
agency_counts = aiusecase['Agency'].value_counts(dropna=False)
total = agency_counts.sum()

# Summary stats
print(agency_counts.describe())
print("\nPercentage breakdown (top 15):")
print((agency_counts.head(15) / total * 100).round(2).astype(str) + '%')

# Plot
plt.figure(figsize=(8,5))
ax = agency_counts.head(15).plot(kind='bar', color='#2171B5')
plt.title('Top Agencies by Use Case Count', fontsize=13)
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Add count + % labels above bars
for i, (agency, count) in enumerate(agency_counts.head(15).items()):
    pct = count / total * 100
    ax.text(i, count + total * 0.005, f"{count:,}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

#Retirement
if 'retired_flag' not in aiusecase.columns:
    retired_col = next((c for c in aiusecase.columns if 'date retired' in c.lower()), None)
    aiusecase['retired_flag'] = aiusecase[retired_col].notna()

retire_by_agency = (
    aiusecase.groupby('Agency')['retired_flag']
    .mean().mul(100).sort_values(ascending=False)
)
print(retire_by_agency.head(10))

# --- Top Agencies by Use Case Count

# Sort and compute total
agency_counts = aiusecase['Agency'].value_counts(dropna=False)
total = agency_counts.sum()

# Summary stats
print(agency_counts.describe())
print("\nPercentage breakdown (top 15):")
print((agency_counts.head(15) / total * 100).round(2).astype(str) + '%')

# Plot with larger figure and fonts
plt.figure(figsize=(12, 7))  # Bigger chart
top_n = 20  # you can change this to 15, 25, etc.
ax = agency_counts.head(top_n).plot(
    kind='bar',
    color='#2171B5',
    edgecolor='black'
)

plt.title(f'Top {top_n} Agencies by Use Case Count', fontsize=16, pad=15)
plt.ylabel('Count', fontsize=13)
plt.xlabel('')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# Add count + % labels above bars
for i, (agency, count) in enumerate(agency_counts.head(top_n).items()):
    pct = count / total * 100
    ax.text(i, count + total * 0.003, f"{count:,}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='medium')

# Add grid for clarity
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

#Does this AI use case have an associated Authority to Operate (ATO) for an AI system?
# --- ATO (Authority to Operate) Status ---
col = "Does this AI use case have an associated Authority to Operate (ATO) for an AI system?"

if col in aiusecase.columns:
    # Clean responses
    ato = (
        aiusecase[col]
        .astype(str)
        .str.strip()
        .str.title()  # e.g., "Yes", "No", "Unknown"
        .replace({"Nan": np.nan, "": np.nan})
        .fillna("Unknown")
    )

    # Count + %
    ato_counts = ato.value_counts()
    total = ato_counts.sum()
    ato_pct = (ato_counts / total * 100).round(1)

    # Print summary
    summary_df = pd.DataFrame({
        "Count": ato_counts,
        "Percent": ato_pct
    })
    print("\nAuthority to Operate (ATO) Summary:\n")
    print(summary_df)

    # Plot
    plt.figure(figsize=(6,4))
    ax = ato_counts.plot(kind="bar", color="#2171B5", edgecolor="black")
    plt.title("Authority to Operate (ATO) Status", fontsize=14)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Add count + % labels
    for i, (label, count) in enumerate(ato_counts.items()):
        pct = count / total * 100
        ax.text(i, count + total * 0.01, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print(f"Column '{col}' not found in dataset.")

#Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?

# Clean column names to remove leading/trailing whitespace
aiusecase.columns = aiusecase.columns.str.strip()

col = 'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?'

# Confirm column name
print("Column found:", col in aiusecase.columns)

# Clean & categorize responses
dev = aiusecase[col].astype(str).str.strip().str.lower()

dev_cat = (
    dev.replace({
        'contract': 'Contract',
        'contracts': 'Contract',
        'in-house': 'In-house',
        'in house': 'In-house',
    })
    .replace(r'.*contract.*', 'Contract', regex=True)
    .replace(r'.*in[- ]?house.*', 'In-house', regex=True)
    .replace(['nan', ''], np.nan)
    .fillna('Unknown')
)

# Counts and percentages
dev_counts = dev_cat.value_counts()
total = dev_counts.sum()
dev_pct = (dev_counts / total * 100).round(1)

# Combine counts + percentages into a summary table
summary_df = pd.DataFrame({'Count': dev_counts, 'Percent': dev_pct})
print("\nAI System Development Method Summary:\n")
print(summary_df)

# Plot
plt.figure(figsize=(6,4))
ax = dev_counts.plot(kind='bar', color='#2171B5')
plt.title('AI System Development Method', fontsize=13)
plt.xlabel('Method')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Add count + % labels above bars
for i, (label, count) in enumerate(dev_counts.items()):
    pct = count / total * 100
    ax.text(i, count + total * 0.01, f"{count:,}\n({pct:.1f}%)",
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

development_method_col = 'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?'

if development_method_col in aiusecase.columns:

    dev = aiusecase[development_method_col].copy()

    # recode the long phrase into "Unknown"
    dev = dev.replace(
        'data not reported by submitter and will be updated once additional information is collected',
        'Unknown'
    )

    development_counts = dev.value_counts(dropna=False)
    print("Development Method Counts:")
    print(development_counts)

    plt.figure(figsize=(8,5))
    development_counts.plot(kind='bar')
    plt.title('AI System Development Method')
    plt.xlabel('Method')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

col = 'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?'

def dev_bucket(x: str) -> str:
    if pd.isna(x) or str(x).strip() == '':
        return 'Unknown'
    s = str(x).strip().lower()

    # explicit unknown phrase
    if 'data not reported by submitter' in s:
        return 'Unknown'

    # hybrid first (catch "both" / "combination")
    if 'both' in s or 'combination' in s:
        return 'Hybrid (Contract + In-house)'

    # in-house (catch "no contract" or explicit)
    if 'no contract' in s or 'in-house' in s or 'in house' in s or 'inhouse' in s:
        # if it also mentions contract but *not* both/combination, still treat as Hybrid
        if 'contract' in s or 'external' in s:
            return 'Hybrid (Contract + In-house)'
        return 'In-house'

    # contract-only
    if 'contract' in s or 'external' in s:
        return 'Contract'

    # fallback
    return 'Unknown'

# Apply categorization
aiusecase['Dev_Method_Bucket'] = aiusecase[col].apply(dev_bucket)

# Tidy summary (ordered)
order = ['Contract', 'In-house', 'Hybrid (Contract + In-house)', 'Unknown']
counts = aiusecase['Dev_Method_Bucket'].value_counts().reindex(order, fill_value=0)
total  = counts.sum()
summary = pd.DataFrame({
    'Count': counts.astype(int),
    'Share (%)': (counts / total * 100).round(1)
})
print(summary)

# Plot
palette = {
    'Contract': '#1f77b4',                      # blue
    'In-house': '#2ca02c',                      # green
    'Hybrid (Contract + In-house)': '#ff7f0e',  # purple
    'Unknown': '#d9d9d9'                        # light gray
}
ax = counts.reindex(order).plot(kind='bar', color=[palette[k] for k in order], figsize=(8,5))

ax.set_title('AI System Development Method (Simplified Categories)')
ax.set_xlabel('Method')
ax.set_ylabel('Count')
ax.set_xticklabels(order, rotation=0)

# labels: count + share on top of bars
for i, v in enumerate(counts.reindex(order)):
    ax.text(i, v + max(counts)*0.02, f"{v:,}\n({v/total:.1%})", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 1) Parse date columns (US-style like "7/1/2023") + coalesce
# ------------------------------------------------------------
aiusecase.columns = aiusecase.columns.str.strip()

DATE_COLS = {
    "init": "Date Initiated",
    "acq":  "Date when Acquisition and/or Development began",
    "impl": "Date Implemented",
    "ret":  "Date Retired"
}

for key, col in DATE_COLS.items():
    if col in aiusecase.columns:
        aiusecase[col] = pd.to_datetime(
            aiusecase[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}),
            errors="coerce",
            format="%m/%d/%Y"  # US format like 7/1/2023
        )

# Created date = first non-null among Implemented -> Initiated -> Acquisition began
avail_created_cols = [c for c in [DATE_COLS.get("impl"), DATE_COLS.get("init"), DATE_COLS.get("acq")] if c in aiusecase.columns]
if avail_created_cols:
    created_mat = aiusecase[avail_created_cols]
    created_date = created_mat.bfill(axis=1).iloc[:, 0]
else:
    created_date = pd.NaT

retired_date = aiusecase[DATE_COLS["ret"]] if DATE_COLS["ret"] in aiusecase.columns else pd.NaT

aiusecase["created_date"] = created_date
aiusecase["retired_date"] = retired_date
aiusecase["created_year"] = aiusecase["created_date"].dt.year
aiusecase["retired_year"] = aiusecase["retired_date"].dt.year

# Flag retired/active
aiusecase["retired_flag"] = aiusecase["retired_date"].notna()

# ------------------------------------------------------------
# 2) When were most cases retired? When were most created?
# ------------------------------------------------------------
def _bar_with_labels(series, title, xlabel):
    counts = series.value_counts(dropna=False).sort_index()
    total = counts.sum()

    plt.figure(figsize=(8,4))
    ax = counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(counts):
        pct = (v / total * 100) if total else 0
        ax.text(i, v + max(1, total*0.01), f"{int(v):,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()

# Retirements by year (drop NaT)
_bar_with_labels(aiusecase.loc[aiusecase["retired_date"].notna(), "retired_year"].dropna(),
                 "Retired AI Use Cases by Year", "Retired Year")

# Creations by year (drop NaT)
_bar_with_labels(aiusecase.loc[aiusecase["created_date"].notna(), "created_year"].dropna(),
                 "Created (Initiated/Implemented) AI Use Cases by Year", "Created Year")

# ------------------------------------------------------------
# 3) Rights/Safety/Both vs dates
#    Filter “not neither”, then examine patterns over time
# ------------------------------------------------------------
impact_col = "Is the AI use case rights-impacting, safety-impacting, both, or neither?"
impact_norm = (
    aiusecase[impact_col]
      .astype(str).str.strip().str.lower()
      .replace({"nan": np.nan, "": np.nan})
)

aiusecase["impact_class"] = (
    impact_norm
      .replace({"rights-impacting": "Rights-impacting",
                "safety-impacting": "Safety-impacting",
                "both": "Both",
                "neither": "Neither"})
)

# Impactful flag (anything except 'Neither')
aiusecase["impactful_flag"] = aiusecase["impact_class"].isin(["Rights-impacting", "Safety-impacting", "Both"])

# --- Share of impactful among CREATED by year ---
created_ok = aiusecase.loc[aiusecase["created_date"].notna()].copy()
share = (
    created_ok
      .groupby("created_year")["impactful_flag"]
      .mean()
      .mul(100)
      .rename("Impactful Share (%)")
)

plt.figure(figsize=(8,4))
ax = share.sort_index().plot(marker="o")
plt.title("Share of Rights/Safety Impacting (vs Neither) Among Created, by Year")
plt.xlabel("Created Year")
plt.ylabel("Percent of Created That Are Impactful")
plt.xticks(rotation=0)
for x, y in share.sort_index().items():
    ax.text(x, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

# --- Mix of impact classes among CREATED by year (stacked bars) ---
mix = (
    created_ok
      .pivot_table(index="created_year", columns="impact_class", values="impactful_flag",
                   aggfunc="count", fill_value=0)
      .sort_index()
)

if not mix.empty:
    mix_pct = mix.div(mix.sum(axis=1), axis=0).mul(100)

    # Stacked bars with labels per segment
    plt.figure(figsize=(9,5))
    bottom = np.zeros(len(mix_pct))
    x = np.arange(len(mix_pct.index))
    for col in ["Both", "Rights-impacting", "Safety-impacting", "Neither"]:
        if col in mix_pct.columns:
            vals = mix_pct[col].values
            plt.bar(x, vals, bottom=bottom, label=col)
            # optional: label big segments
            for i, v in enumerate(vals):
                if v >= 7:  # label only if >=7% to avoid clutter
                    plt.text(i, bottom[i] + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
            bottom += vals

    plt.title("Impact Class Mix Among Created, by Year (Percent)")
    plt.xlabel("Created Year")
    plt.ylabel("Percent")
    plt.xticks(x, mix_pct.index, rotation=0)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

# --- Are retirements disproportionately impactful? (share by retired year) ---
ret_ok = aiusecase.loc[aiusecase["retired_date"].notna()].copy()
ret_share = (
    ret_ok.groupby("retired_year")["impactful_flag"].mean().mul(100).rename("Impactful Share (%)")
)

if not ret_share.empty:
    plt.figure(figsize=(8,4))
    ax = ret_share.sort_index().plot(marker="o")
    plt.title("Share of Rights/Safety Impacting Among Retired, by Year")
    plt.xlabel("Retired Year")
    plt.ylabel("Percent of Retired That Were Impactful")
    plt.xticks(rotation=0)
    for x, y in ret_share.sort_index().items():
        ax.text(x, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 4) (Optional) Durations + simple diagnostics
# ------------------------------------------------------------
# Time from created to retired (completed lifecycles)
if "created_date" in aiusecase and "retired_date" in aiusecase:
    done = aiusecase.dropna(subset=["created_date", "retired_date"]).copy()
    if not done.empty:
        done["lifecycle_days"] = (done["retired_date"] - done["created_date"]).dt.days

        # Boxplot-like summary (text)
        desc = done["lifecycle_days"].describe(percentiles=[.25, .5, .75]).astype(int)
        print("\nLifecycle (days) summary for retired systems:")
        print(desc.to_string())

        # Histogram of lifecycle days
        plt.figure(figsize=(8,4))
        vals = done["lifecycle_days"].clip(lower=0)
        bins = max(10, int(np.sqrt(len(vals))))
        plt.hist(vals, bins=bins)
        plt.title("Lifecycle Length (Created → Retired), Days")
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

# --- Topic area cross tabs with counts & percentages ---
if 'Use Case Topic Area' in aiusecase.columns:
    topic_col = 'Use Case Topic Area'
    topic_counts = aiusecase[topic_col].value_counts(dropna=False)
    total = topic_counts.sum()
    topic_pct = (topic_counts / total * 100).round(1)

    summary_df = pd.DataFrame({
        'Count': topic_counts,
        'Percent': topic_pct
    })
    print("\nUse Case Topic Area Summary:\n")
    print(summary_df)

    # --- Plot ---
    plt.figure(figsize=(8,6))
    ax = sns.countplot(
        y=topic_col,
        data=aiusecase,
        order=topic_counts.index,
        color='#2171B5'
    )
    plt.title('Use Case Topic Areas', fontsize=13)
    plt.xlabel('Count')
    plt.ylabel('Topic Area')

    # Add count and % labels
    for i, (label, count) in enumerate(topic_counts.items()):
        pct = count / total * 100
        ax.text(count + total * 0.005, i, f"{count:,} ({pct:.1f}%)",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5) Mission Type × Dates: creation/retirement patterns
# ------------------------------------------------------------
# Try to find the mission column by name
mission_col = next((c for c in aiusecase.columns if re.search(r'\bmission\b', c, re.I)), None)

if mission_col:
    # Clean/normalize mission values
    mission = (
        aiusecase[mission_col]
        .astype(str).str.strip()
        .replace({"nan": np.nan, "": np.nan})
        .fillna("Unknown")
    )
    # Title-case common values while keeping acronyms; safe default:
    aiusecase["mission_norm"] = mission.str.replace(r"\s+", " ", regex=True).str.strip().str.title()

    # -------------------------
    # Created by year × mission
    # -------------------------
    created_by_yr_mis = (
        aiusecase.loc[aiusecase["created_date"].notna()]
        .groupby(["created_year", "mission_norm"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if not created_by_yr_mis.empty:
        totals = created_by_yr_mis.sum(axis=1)
        ax = created_by_yr_mis.plot(kind="bar", stacked=True, figsize=(10,5))
        plt.title("Created AI Use Cases by Year × Mission Type (Stacked)")
        plt.xlabel("Created Year")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        # Label total counts above each stacked bar
        for i, v in enumerate(totals.values):
            ax.text(i, v + max(1, totals.max()*0.01), f"{int(v):,}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()

    # --------------------------
    # Retired by year × mission
    # --------------------------
    retired_by_yr_mis = (
        aiusecase.loc[aiusecase["retired_date"].notna()]
        .groupby(["retired_year", "mission_norm"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if not retired_by_yr_mis.empty:
        totals_r = retired_by_yr_mis.sum(axis=1)
        ax = retired_by_yr_mis.plot(kind="bar", stacked=True, figsize=(10,5))
        plt.title("Retired AI Use Cases by Year × Mission Type (Stacked)")
        plt.xlabel("Retired Year")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        for i, v in enumerate(totals_r.values):
            ax.text(i, v + max(1, totals_r.max()*0.01), f"{int(v):,}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------
    # Heatmap: % Impactful among CREATED, by year × mission
    # ------------------------------------------------------
    created_ok = aiusecase.loc[aiusecase["created_date"].notna()].copy()
    if not created_ok.empty:
        # Denominator: total created per year × mission
        denom = (
            created_ok.groupby(["created_year", "mission_norm"]).size().rename("n")
            .reset_index()
        )
        # Numerator: impactful created per year × mission
        num = (
            created_ok[created_ok["impactful_flag"]]
            .groupby(["created_year", "mission_norm"]).size().rename("k")
            .reset_index()
        )
        imp = pd.merge(denom, num, on=["created_year", "mission_norm"], how="left").fillna({"k": 0})
        imp["pct"] = (imp["k"] / imp["n"] * 100).round(1)

        heat = imp.pivot(index="created_year", columns="mission_norm", values="pct").sort_index()
        counts_tbl = imp.pivot(index="created_year", columns="mission_norm", values="n").sort_index()

        if not heat.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(heat.values, aspect="auto")
            ax.set_title("% Impactful Among Created by Year × Mission (Rights/Safety/Both)")
            ax.set_xlabel("Mission Type")
            ax.set_ylabel("Created Year")
            ax.set_xticks(np.arange(len(heat.columns)))
            ax.set_yticks(np.arange(len(heat.index)))
            ax.set_xticklabels(heat.columns, rotation=45, ha="right")
            ax.set_yticklabels(heat.index)

            # Annotate cells with "pct% (n)"
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    val = heat.iloc[i, j]
                    n_val = counts_tbl.iloc[i, j] if counts_tbl.notna().iloc[i, j] else 0
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.0f}%\n(n={int(n_val)})",
                                ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, label="% Impactful")
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------
    # Quick summary: peak creation year per mission
    # ------------------------------------------------------
    if not created_by_yr_mis.empty:
        peak_year = created_by_yr_mis.idxmax(axis=0)      # year of max for each mission
        peak_count = created_by_yr_mis.max(axis=0)        # count at that peak
        peak_df = pd.DataFrame({
            "Peak Created Year": peak_year,
            "Peak Count": peak_count
        }).sort_values(by="Peak Count", ascending=False)
        print("\nPeak creation year by mission type:")
        print(peak_df.to_string())
else:
    print("No mission-type column detected (looked for a column containing the word 'mission').")

# ========== # Helpers # ==========

def clean_categorical(series: pd.Series) -> pd.Series:
    """
    Generic cleaner for survey-like categorical fields.
    - Strips whitespace
    - Normalizes common NA patterns
    - Collapses 'Yes – ...' / 'No – ...' to 'Yes'/'No'
    - Fills missing with 'Unknown'
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "Nan": np.nan,
                "nan": np.nan,
                "NA": np.nan,
                "N/A": "Not applicable",
                "n/a": "Not applicable",
            }
        )
    )

    # Collapse "Yes – ..." / "No – ..." variants
    s = s.str.replace(
        r'^(yes|no)\s*[-–].*$',
        lambda m: m.group(1).title(),  # Yes/No
        regex=True,
        case=False # Added case=False to handle case-insensitivity
    )

    return s.fillna("Unknown")


def clean_text(series: pd.Series) -> pd.Series:
    """
    Cleaner for free-text fields, keeping full content.
    Only trims whitespace and normalizes obvious empties.
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "Nan": np.nan,
                "nan": np.nan,
            }
        )
    )
    return s  # keep NaN as NaN for text


# ========== # Column → variable name maps # ==========

cat_cols = {
    "Is this AI use case supporting a High-Impact Service Provider (HISP) public-facing service?":
        "hisp_support_flag",
    "Does this AI use case disseminate information to the public?":
        "public_dissemination_flag",
    "Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?":
        "pii_flag",
    "Has the Senior Agency Official for Privacy (SAOP) assessed the privacy risks associated with this AI use case?":
        "saop_assessed_flag",
    "Do you have access to an enterprise data catalog or agency-wide data repository that enables you to identify whether or not the necessary datasets exist and are ready to develop your use case?":
        "data_catalog_access",
    "Is there available documentation for the model training and evaluation data that demonstrates the degree to which it is appropriate to be used in analysis or for making predictions?":
        "doc_training_eval_status",
    "Does this project include custom-developed code?":
        "custom_code_flag",
    "Does the agency have access to the code associated with the AI use case?":
        "code_access_flag",
    "Does this AI use case have an associated Authority to Operate (ATO) for an AI system?":
        "ato_flag",
    "How long have you waited for the necessary developer tools to implement the AI use case? ":
        "dev_tools_wait_time",
    "For this AI use case, is the required IT infrastructure provisioned via a centralized intake form or process inside the agency?":
        "infra_central_intake",
    "Do you have a process in place to request access to computing resources for model training and development of the AI involved in this use case?":
        "compute_access_process",
    "Has communication regarding the provisioning of your requested resources been timely?":
        "resource_comm_timely",
    "Has information regarding the AI use case, including performance metrics and intended use of the model, been made available for review and feedback within the agency?":
        "info_shared_internal",
    "Has your agency requested an extension to implement the minimum risk management practices for this AI use case?":
        "extension_requested_flag",
    "Has an AI impact assessment been conducted for this AI use case?":
        "impact_assessment_status",
    "Has the AI use case been tested in operational or real-world environments to understand the performance and impact it may have on affected individuals or communities?":
        "operational_testing_status",
    "Has an independent evaluation of the AI use case been conducted?":
        "independent_eval_flag",
    "Is there a process to monitor performance of the AI system’s functionality and changes to its impact on rights or safety as part of the post-deployment plan for the AI use case?":
        "monitoring_process_status",
    "For this particular use case, can the AI carry out a decision or action without direct human involvement that could result in a significant impact on rights or safety?":
        "autonomous_high_impact_flag",
    "How is the agency providing reasonable and timely notice regarding the use of AI when people interact with an AI-enabled service as a result of this AI use case?":
        "notice_mechanism",
    "Is the AI used to significantly influence or inform decisions or actions that could have an adverse or negative impact on specific individuals or groups?":
        "adverse_impact_influence_flag",
    "Is there an established fallback and escalation process for this AI use case in the event that an impacted individual or group would like to appeal or contest the AI system’s outcome?":
        "fallback_escalation_flag",
    "Where practicable and consistent with applicable law and governmentwide policy, is there an established mechanism for individuals to opt-out from the AI functionality in favor of a human alternative?":
        "opt_out_mechanism_flag",
}

text_cols = {
    "Which HISP is the AI use case supporting?":
        "hisp_name",
    "Which public-facing service is the AI use case supporting?":
        "public_service_name",
    "How is the agency ensuring compliance with Information Quality Act guidelines, if applicable?":
        "iqa_compliance_description",
    "Describe any agency-owned data used to train, fine-tune, and/or evaluate performance of the model(s) used in this use case.":
        "agency_owned_data_desc",
    "Which, if any, demographic variables does the AI use case explicitly use as model features?":
        "demographic_features_used",
    "How are existing data science tools, libraries, data products, and internally-developed AI infrastructure being re-used for the current AI use case?":
        "reuse_of_tools_infra",
    "What are the key risks from using the AI for this particular use case and how were they identified?":
        "key_risks_desc",
    "What steps has the agency taken to detect and mitigate significant disparities in the model’s performance across demographic groups for this AI use case?":
        "disparity_mitigation_steps",
    "What steps has the agency taken to consult and incorporate feedback from groups affected by this AI use case?":
        "stakeholder_feedback_steps",
}


# ========== # Apply cleaners: main categorical & text variables # ==========

# Categorical-style (Yes/No/Unknown/etc.)
for col, varname in cat_cols.items():
    if col in aiusecase.columns:
        aiusecase[varname] = clean_categorical(aiusecase[col])
    else:
        print(f"[WARN] Column not found (categorical): {col}")

# Free-text variables for qualitative EDA
for col, varname in text_cols.items():
    if col in aiusecase.columns:
        aiusecase[varname] = clean_text(aiusecase[col])
    else:
        print(f"[WARN] Column not found (text): {col}")


# ========== # Handle all "If Other, please explain." and "If No, please explain." fields # ==========

# This will automatically create cleaned versions for every such column,
# even if pandas has suffixes like 'If Other, please explain..1'
for col in aiusecase.columns:
    if col.startswith("If Other, please explain"):
        new_name = (
            col.lower()
               .replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
        )
        aiusecase[new_name] = clean_text(aiusecase[col])

    if col.startswith("If No, please explain"):
        new_name = (
            col.lower()
               .replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
        )
        aiusecase[new_name] = clean_text(aiusecase[col])

# ==========
# Helpers
# ==========

def clean_categorical(series: pd.Series) -> pd.Series:
    """
    Generic cleaner for survey-like categorical fields.
    - Strips whitespace
    - Normalizes common NA patterns
    - Collapses 'Yes – ...' / 'No – ...' to 'Yes'/'No'
    - Fills missing with 'Unknown'
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "Nan": np.nan,
                "nan": np.nan,
                "NA": np.nan,
                "N/A": "Not applicable",
                "n/a": "Not applicable",
            }
        )
    )

    # Collapse "Yes – ..." / "No – ..." variants
    s = s.str.replace(
        r'^(yes|no)\s*[-–].*$',
        lambda m: m.group(1).title(),  # Yes/No
        regex=True,
        case=False
    )

    return s.fillna("Unknown")


def clean_text(series: pd.Series) -> pd.Series:
    """
    Cleaner for free-text fields, keeping full content.
    Only trims whitespace and normalizes obvious empties.
    """
    s = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "Nan": np.nan,
                "nan": np.nan,
            }
        )
    )
    return s  # keep NaN as NaN for text


# ==========
# Column → variable name maps
# ==========

cat_cols = {
    "Is this AI use case supporting a High-Impact Service Provider (HISP) public-facing service?":
        "hisp_support_flag",
    "Does this AI use case disseminate information to the public?":
        "public_dissemination_flag",
    "Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?":
        "pii_flag",
    "Has the Senior Agency Official for Privacy (SAOP) assessed the privacy risks associated with this AI use case?":
        "saop_assessed_flag",
    "Do you have access to an enterprise data catalog or agency-wide data repository that enables you to identify whether or not the necessary datasets exist and are ready to develop your use case?":
        "data_catalog_access",
    "Is there available documentation for the model training and evaluation data that demonstrates the degree to which it is appropriate to be used in analysis or for making predictions?":
        "doc_training_eval_status",
    "Does this project include custom-developed code?":
        "custom_code_flag",
    "Does the agency have access to the code associated with the AI use case?":
        "code_access_flag",
    "Does this AI use case have an associated Authority to Operate (ATO) for an AI system?":
        "ato_flag",
    "How long have you waited for the necessary developer tools to implement the AI use case? ":
        "dev_tools_wait_time",
    "For this AI use case, is the required IT infrastructure provisioned via a centralized intake form or process inside the agency?":
        "infra_central_intake",
    "Do you have a process in place to request access to computing resources for model training and development of the AI involved in this use case?":
        "compute_access_process",
    "Has communication regarding the provisioning of your requested resources been timely?":
        "resource_comm_timely",
    "Has information regarding the AI use case, including performance metrics and intended use of the model, been made available for review and feedback within the agency?":
        "info_shared_internal",
    "Has your agency requested an extension to implement the minimum risk management practices for this AI use case?":
        "extension_requested_flag",
    "Has an AI impact assessment been conducted for this AI use case?":
        "impact_assessment_status",
    "Has the AI use case been tested in operational or real-world environments to understand the performance and impact it may have on affected individuals or communities?":
        "operational_testing_status",
    "Has an independent evaluation of the AI use case been conducted?":
        "independent_eval_flag",
    "Is there a process to monitor performance of the AI system’s functionality and changes to its impact on rights or safety as part of the post-deployment plan for the AI use case?":
        "monitoring_process_status",
    "For this particular use case, can the AI carry out a decision or action without direct human involvement that could result in a significant impact on rights or safety?":
        "autonomous_high_impact_flag",
    "How is the agency providing reasonable and timely notice regarding the use of AI when people interact with an AI-enabled service as a result of this AI use case?":
        "notice_mechanism",
    "Is the AI used to significantly influence or inform decisions or actions that could have an adverse or negative impact on specific individuals or groups?":
        "adverse_impact_influence_flag",
    "Is there an established fallback and escalation process for this AI use case in the event that an impacted individual or group would like to appeal or contest the AI system’s outcome?":
        "fallback_escalation_flag",
    "Where practicable and consistent with applicable law and governmentwide policy, is there an established mechanism for individuals to opt-out from the AI functionality in favor of a human alternative?":
        "opt_out_mechanism_flag",
}

text_cols = {
    "Which HISP is the AI use case supporting?":
        "hisp_name",
    "Which public-facing service is the AI use case supporting?":
        "public_service_name",
    "How is the agency ensuring compliance with Information Quality Act guidelines, if applicable?":
        "iqa_compliance_description",
    "Describe any agency-owned data used to train, fine-tune, and/or evaluate performance of the model(s) used in this use case.":
        "agency_owned_data_desc",
    "Which, if any, demographic variables does the AI use case explicitly use as model features?":
        "demographic_features_used",
    "How are existing data science tools, libraries, data products, and internally-developed AI infrastructure being re-used for the current AI use case?":
        "reuse_of_tools_infra",
    "What are the key risks from using the AI for this particular use case and how were they identified?":
        "key_risks_desc",
    "What steps has the agency taken to detect and mitigate significant disparities in the model’s performance across demographic groups for this AI use case?":
        "disparity_mitigation_steps",
    "What steps has the agency taken to consult and incorporate feedback from groups affected by this AI use case?":
        "stakeholder_feedback_steps",
}


# ==========
# Apply cleaners: main categorical & text variables
# ==========

# Categorical-style (Yes/No/Unknown/etc.)
for col, varname in cat_cols.items():
    if col in aiusecase.columns:
        aiusecase[varname] = clean_categorical(aiusecase[col])
    else:
        print(f"[WARN] Column not found (categorical): {col}")

# Free-text variables for qualitative EDA
for col, varname in text_cols.items():
    if col in aiusecase.columns:
        aiusecase[varname] = clean_text(aiusecase[col])
    else:
        print(f"[WARN] Column not found (text): {col}")


# ==========
# Handle all "If Other, please explain." and "If No, please explain." fields
# ==========

# This will automatically create cleaned versions for every such column,
# even if pandas has suffixes like 'If Other, please explain..1'
for col in aiusecase.columns:
    if col.startswith("If Other, please explain"):
        new_name = (
            col.lower()
               .replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
        )
        aiusecase[new_name] = clean_text(aiusecase[col])

    if col.startswith("If No, please explain"):
        new_name = (
            col.lower()
               .replace(" ", "_")
               .replace(",", "")
               .replace(".", "")
        )
        aiusecase[new_name] = clean_text(aiusecase[col])

# 1) Simple distributions
for col in [
    "pii_flag", "saop_assessed_flag", "ato_flag",
    "impact_assessment_status", "operational_testing_status",
    "independent_eval_flag", "monitoring_process_status",
    "extension_requested_flag"
]:
    print(f"\n=== {col} ===")
    print(aiusecase[col].value_counts(dropna=False))

# 2) Cross-tabs vs impact class (using the original impact column)
impact_col = "Is the AI use case rights-impacting, safety-impacting, both, or neither?"

pd.crosstab(aiusecase["ato_flag"], aiusecase[impact_col], normalize="index")

pd.crosstab(aiusecase["impact_assessment_status"], aiusecase[impact_col], normalize="index")

# 3) Fairness / recourse-related fields
for col in ["adverse_impact_influence_flag",
            "disparity_mitigation_steps",
            "fallback_escalation_flag"]:
    print(f"\n=== {col} ===")
    print(aiusecase[col].value_counts(dropna=False))

# --- SAOP ASSESSED SIMPLE ---
aiusecase["saop_assessed_simple"] = (
    aiusecase["saop_assessed_flag"]
      .replace({"NO": "No", "YES": "Yes"})
      .fillna("Unknown")
)

# --- ATO SIMPLE ---
def simplify_ato(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x in ["Yes", "Operated in an approved enclave", "Data.State-SBU"]:
        return "Yes"
    if x.startswith("No"):
        return "No"
    if x == "Not Applicable":
        return "Not applicable"
    return "Other"

aiusecase["ato_simple"] = aiusecase["ato_flag"].apply(simplify_ato)

# --- IMPACT ASSESSMENT SIMPLE ---
aiusecase["impact_assessment_simple"] = (
    aiusecase["impact_assessment_status"]
      .replace({"YES": "Yes", "Planned or in-progress.": "Planned"})
      .fillna("Unknown")
)

# --- INDEPENDENT EVAL SIMPLE ---
def simplify_independent_eval(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x in ["Yes", "True"]:
        return "Yes"
    if x == "Planned or in-progress":
        return "Planned"
    if "not safety or rights-impacting" in x.lower() or "does not apply" in x.lower():
        return "Not applicable"
    if "waived this minimum practice" in x:
        return "Waived"
    if x == "Not Applicable":
        return "Not applicable"
    return x  # keep other labels if any

aiusecase["independent_eval_simple"] = aiusecase["independent_eval_flag"].apply(simplify_independent_eval)

# --- OPERATIONAL TESTING SIMPLE ---
def simplify_operational_testing(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x == "Not Applicable":
        return "Not applicable"
    if x.startswith("No testing"):
        return "None"
    if x.startswith("Benchmark evaluation"):
        return "Benchmark only"
    if x.startswith("Performance evaluation in operational environment") or x == "Yes":
        return "Operational test"
    if x.startswith("Impact evaluation in operational environment"):
        return "Impact evaluation"
    if "waived this minimum practice" in x:
        return "Waived"
    return "Other"

aiusecase["operational_testing_simple"] = aiusecase["operational_testing_status"].apply(simplify_operational_testing)

# --- MONITORING SIMPLE ---
def simplify_monitoring(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x == "Not Applicable" or "not safety or rights-impacting" in x.lower():
        return "Not applicable"
    if x.startswith("No monitoring protocols have been established"):
        return "None"
    if x.startswith("Intermittent and Manually Updated"):
        return "Manual"
    if x.startswith("Automated and Regularly Scheduled Updates"):
        return "Automated"
    if x.startswith("Established Process of Machine Learning Operations"):
        return "MLOps"
    return "Other"

aiusecase["monitoring_simple"] = aiusecase["monitoring_process_status"].apply(simplify_monitoring)

simple_cols = [
    "pii_flag",
    "saop_assessed_simple",
    "ato_simple",
    "impact_assessment_simple",
    "operational_testing_simple",
    "independent_eval_simple",
    "monitoring_simple",
]

for col in simple_cols:
    print(f"\n=== {col} (proportions) ===")
    print(
        aiusecase[col]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(1)
    )

# SAOP simple
aiusecase["saop_assessed_simple"] = (
    aiusecase["saop_assessed_flag"]
      .replace({"NO": "No", "YES": "Yes"})
      .fillna("Unknown")
)

# ATO simple
def simplify_ato(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x in ["Yes", "Operated in an approved enclave", "Data.State-SBU"]:
        return "Yes"
    if x.startswith("No"):
        return "No"
    if x == "Not Applicable":
        return "Not applicable"
    return "Other"

aiusecase["ato_simple"] = aiusecase["ato_flag"].apply(simplify_ato)

# Impact assessment simple
aiusecase["impact_assessment_simple"] = (
    aiusecase["impact_assessment_status"]
      .replace({
          "YES": "Yes",
          "Planned or in-progress.": "Planned"
      })
      .fillna("Unknown")
)

# Independent evaluation simple
def simplify_independent_eval(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x in ["Yes", "True"]:
        return "Yes"
    if x == "Planned or in-progress":
        return "Planned"
    if "not safety or rights-impacting" in x.lower() or "does not apply" in x.lower():
        return "Not applicable"
    if "waived this minimum practice" in x:
        return "Waived"
    if x == "Not Applicable":
        return "Not applicable"
    return x

aiusecase["independent_eval_simple"] = aiusecase["independent_eval_flag"].apply(simplify_independent_eval)

# Operational testing simple
def simplify_operational_testing(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x == "Not Applicable":
        return "Not applicable"
    if x.startswith("No testing"):
        return "None"
    if x.startswith("Benchmark evaluation"):
        return "Benchmark only"
    if x.startswith("Performance evaluation in operational environment") or x == "Yes":
        return "Operational test"
    if x.startswith("Impact evaluation in operational environment"):
        return "Impact evaluation"
    if "waived this minimum practice" in x:
        return "Waived"
    return "Other"

aiusecase["operational_testing_simple"] = aiusecase["operational_testing_status"].apply(simplify_operational_testing)

# Monitoring simple
def simplify_monitoring(x):
    if pd.isna(x) or x == "Unknown":
        return "Unknown"
    x = str(x).strip()
    if x == "Not Applicable" or "not safety or rights-impacting" in x.lower():
        return "Not applicable"
    if x.startswith("No monitoring protocols"):
        return "None"
    if x.startswith("Intermittent and Manually Updated"):
        return "Manual"
    if x.startswith("Automated and Regularly Scheduled Updates"):
        return "Automated"
    if x.startswith("Established Process of Machine Learning Operations"):
        return "MLOps"
    return "Other"

aiusecase["monitoring_simple"] = aiusecase["monitoring_process_status"].apply(simplify_monitoring)

simple_cols = [
    "pii_flag",
    "saop_assessed_simple",
    "ato_simple",
    "impact_assessment_simple",
    "operational_testing_simple",
    "independent_eval_simple",
    "monitoring_simple",
]

for col in simple_cols:
    print(f"\n=== {col} (proportions) ===")
    print(
        aiusecase[col]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(1)
    )

impact_col = "Is the AI use case rights-impacting, safety-impacting, both, or neither?"

pd.crosstab(aiusecase["ato_simple"], aiusecase[impact_col], normalize="index").round(2)
pd.crosstab(aiusecase["impact_assessment_simple"], aiusecase[impact_col], normalize="index").round(2)
pd.crosstab(aiusecase["monitoring_simple"], aiusecase[impact_col], normalize="index").round(2)

# Text length:
# Combine relevant text columns into 'text_all'
aiusecase['text_all'] = aiusecase['Use Case Name'].fillna('') + ' ' + \
                        aiusecase['What is the intended purpose and expected benefits of the AI?'].fillna('') + ' ' + \
                        aiusecase['Describe the AI system’s outputs.'].fillna('')

aiusecase['text_len'] = aiusecase['text_all'].str.split().str.len()

sns.histplot(aiusecase['text_len'], bins=50)
plt.title('Distribution of Text Length')
plt.xlabel('Word Count')
plt.show()

# Compare DHS vs Non-DHS
sns.boxplot(x='is_DHS', y='text_len', data=aiusecase)
plt.title('Text Length: DHS vs Non-DHS')
plt.show()



#quick TF-IDF n-grams (for EDA, not for modeling)

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)
X = tfidf.fit_transform(aiusecase["text_all"])
idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
# show top 30 distinctive n-grams by lowest idf
top_terms = sorted(idf.items(), key=lambda kv: kv[1])[:30]
top_terms[:10], len(idf)

"""3. Prepare text field for topic modeling"""

# 0) Choose the base text column
BASE_TEXT_COL = "text_all"
assert BASE_TEXT_COL in aiusecase.columns, f"{BASE_TEXT_COL} not found in dataframe!"

# 1) Light pre-clean for phrases (keeps semantics for embeddings)

PHRASE_STOP = [
    r"\bmission[-\s]+enabling\b",   # mission-enabling / mission enabling
    r"\buse[-\s]*case(?:s)?\b",     # use case / use-case / usecases
    r"\bintended\s+purpose\b",
    r"\bexpected\s+benefits\b",
]

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HYPHEN_RE = re.compile(r"[-–—]+")

def light_preclean(s: str) -> str:
    t = str(s)
    t = HYPHEN_RE.sub(" ", t)            # normalize hyphens/dashes
    t = URL_RE.sub(" ", t)               # remove URLs
    t = EMAIL_RE.sub(" ", t)             # remove emails
    t = re.sub(r"\s+", " ", t).strip()   # collapse whitespace
    tl = t.lower()
    for pat in PHRASE_STOP:
        tl = re.sub(pat, " ", tl, flags=re.I)
    tl = re.sub(r"\s+", " ", tl).strip()
    return tl

# Build the cleaned text column from base text
aiusecase["text_clean"] = aiusecase[BASE_TEXT_COL].fillna("").apply(light_preclean)

#Lemmatize

def lemmatize_text(text: str) -> str:
    doc = nlp(str(text))
    out = []
    for tok in doc:
        # keep alphabetic lemmas (skip numbers/punct), drop spaCy stopwords (we still keep domain words)
        if tok.is_stop or not tok.is_alpha:
            continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) > 2:
            out.append(lemma)
    return " ".join(out)

# Build lemmatized column
aiusecase["text_lemma"] = aiusecase["text_clean"].apply(lemmatize_text)

# 1) Build doc-frequency counts from cleaned text
TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9\-]+\b")
def doc_tokens(s: str) -> set:
    return set(TOKEN_RE.findall(str(s).lower()))

docs_sets = aiusecase["text_clean"].map(doc_tokens)
df_counter = Counter()
for s in docs_sets:
    df_counter.update(s)
N = len(aiusecase)

# 2) Base admin/boilerplate to always hide in LABELS (NOT in embeddings)
always_hide_for_labels = {
    "use","case","cases","project","projects",
    "platform","platforms","service","services",
    "solution","solutions","tool","tools",
    "implementation","assessment","intended",
    "benefits","purpose"
    # NOTE: "mission" and "capability/capabilities" intentionally NOT hidden
}

# 3) Very common tokens (overall DF > 85%) are likely boilerplate
DF_THRESH = 0.85
very_common = {w for (w, c) in df_counter.items() if c / N > DF_THRESH}

# 4) Protect domain-critical tokens from being hidden in labels
whitelist = {
    "data","model","models","system","systems","support",
    "capability","capabilities",
    "privacy","security","safety","risk","fraud","cyber","inspection","screening",
    "asylum","border","maritime","aviation","medical","health","threat","detection",
    "text","imagery","image","video","vision","nlp","speech","biometric","cargo","port",
    "foia","records","satellite","maps","geospatial","legal","law","justice"
}

# 5) Final label-only stopwords = English stopwords + always-hide + very_common − whitelist
label_stopwords = (set(ENGLISH_STOP_WORDS) | always_hide_for_labels | very_common) - whitelist

print(f"Label stopwords (count): {len(label_stopwords)}")
print("Sample:", sorted(list(label_stopwords))[:50])

"""4. Topic Modeling"""

# === 3) Vectorizer for c-TF-IDF (labels) + tuned UMAP/HDBSCAN ===

# refined vectorizer (replaces old CountVectorizer)
vectorizer_model_refined = CountVectorizer(
    stop_words=list(label_stopwords),
    ngram_range=(1,3),
    min_df=5,
    max_df=0.95
)

umap_model = umap.UMAP(
    n_neighbors=8,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
docs = aiusecase["text_clean"].astype(str).tolist()

topic_model = BERTopic(
    embedding_model=embedder,
    vectorizer_model=vectorizer_model_refined,  # ← use refined vectorizer here
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=True,
    top_n_words=10,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)
aiusecase["topic_id"] = topics

"""4. Model Evaluation/ Analysis"""

# ===== Topic Modeling Metrics: Auto vs K Sweep =====
from collections import defaultdict
from sklearn.metrics import silhouette_score

# ------------- Inputs -------------
docs_text = aiusecase["text_clean"].astype(str).tolist()

# Fresh ST embedder (avoid BERTopic backend object)
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------- Helpers -------------
def topic_diversity(model, top_n=10):
    info = model.get_topic_info()
    topic_ids = [t for t in info["Topic"] if t >= 0]
    if not topic_ids: return np.nan
    terms = []
    for t in topic_ids:
        words = model.get_topic(t) or []
        terms.extend([w for w,_ in words[:top_n]])
    uniq = len(set(terms))
    return uniq / (len(topic_ids) * top_n) if topic_ids else np.nan

def cohesion_sep_silhouette(labels, embeddings):
    m = labels >= 0
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return np.nan, np.nan, np.nan
    X = embeddings[m]
    y = labels[m]

    # per-topic centroids + cohesion
    idxs = defaultdict(list)
    for i, t in enumerate(y): idxs[t].append(i)
    cents = []
    sims  = []
    for ii in idxs.values():
        Xi = X[ii]
        c  = Xi.mean(axis=0, keepdims=True)
        cents.append(c.squeeze())
        sims.append(cosine_similarity(Xi, c).mean())
    cohesion = float(np.mean(sims)) if sims else np.nan

    # separation
    sep = np.nan
    if len(cents) >= 2:
        C = np.vstack(cents)
        psim = cosine_similarity(C)
        tril = psim[np.tril_indices_from(psim, k=-1)]
        sep  = float(1 - tril.mean()) if tril.size else np.nan

    # silhouette on embeddings (cosine)
    sil = silhouette_score(X, y, metric="cosine")
    return cohesion, sep, sil

def evaluate_variant(model, labels, embeddings, tag):
    info = model.get_topic_info()
    k_topics = int((info["Topic"] >= 0).sum())
    out_rate = float((labels < 0).mean())
    div = topic_diversity(model, top_n=10)
    coh, sep, sil = cohesion_sep_silhouette(labels, embeddings)
    return {
        "Variant": tag,
        "Topics": k_topics,
        "OutlierRate": round(out_rate, 4),
        "Diversity@10": round(div, 4) if div==div else np.nan,
        "Cohesion": round(coh, 4) if coh==coh else np.nan,
        "Separation": round(sep, 4) if sep==sep else np.nan,
        "Silhouette": round(sil, 4) if sil==sil else np.nan
    }

# ------------- Build embeddings once -------------
# normalize to stabilize cosine metrics
doc_emb = emb_model.encode(docs_text, show_progress_bar=True, normalize_embeddings=True)

# ------------- Evaluate: as-fit / auto / K-sweep -------------
results = []

# 1) as-fit
labels_asfit = np.array(topics)
results.append(evaluate_variant(topic_model, labels_asfit, doc_emb, tag="as_fit"))

# 2) auto reduction
m_auto = deepcopy(topic_model).reduce_topics(docs_text, nr_topics="auto")
labels_auto, _ = m_auto.transform(docs_text)
results.append(evaluate_variant(m_auto, np.array(labels_auto), doc_emb, tag="auto"))

# 3) K sweep (we can tune the range)
K_CANDIDATES = [20, 25, 30, 35, 40, 50]
for K in K_CANDIDATES:
    mK = deepcopy(topic_model).reduce_topics(docs_text, nr_topics=K)
    labelsK, _ = mK.transform(docs_text)
    results.append(evaluate_variant(mK, np.array(labelsK), doc_emb, tag=f"K={K}"))

metrics_df = pd.DataFrame(results)
metrics_df

#Elbow method:

# Filter only K-sweep rows and extract numeric K
k_df = metrics_df[metrics_df["Variant"].str.startswith("K=")].copy()
k_df["K_target"] = k_df["Variant"].str.extract(r"K=(\d+)").astype(int)

fig, ax = plt.subplots(1, 3, figsize=(12, 3))

ax[0].plot(k_df["K_target"], k_df["Silhouette"], marker="o")
ax[0].set_title("Silhouette (cosine)")
ax[0].set_xlabel("K target")
ax[0].set_ylabel("score")


ax[1].plot(k_df["K_target"], k_df["OutlierRate"], marker="o")
ax[1].set_title("Outlier rate")
ax[1].set_xlabel("K target")
ax[1].set_ylabel("score")

ax[2].plot(k_df["K_target"], k_df["Cohesion"], marker="o")
ax[2].set_title("Cohesion")
ax[2].set_xlabel("K target")
ax[2].set_ylabel("score")


plt.tight_layout()
plt.show()

"""K=25–30 → maximizes semantic distinctness (high diversity, good interpretability).

K=35 → slightly better silhouette (separation) but a bit less diversity.

K=50-> better cohesion and silhouette, but risk topic fragmentation (splitting semantically similar clusters into several small topics)
"""

#Best model:
# prefer low outliers, then high silhouette, then cohesion, then diversity
cands = metrics_df[metrics_df["Variant"] != "as_fit"].copy()
best = cands.sort_values(
    by=["OutlierRate","Silhouette","Cohesion","Diversity@10"],
    ascending=[True, False, False, False]
).head(1)
best

"""Despite being the best model, we shouldn’t pick K=50 because it produces too many overlapping topics, making the results harder to interpret and less useful as one-hot features despite slightly higher silhouette.

Comparing Results: K25-35
"""

# ===== Compare topics: K=25 vs K=30 =====

# 0) Inputs (same docs we trained on)
docs_text = aiusecase["text_lemma"].astype(str).tolist()

# 1) Build reduced models and labels
def reduce_and_label(base_model, docs, K):
    mK = deepcopy(base_model).reduce_topics(docs, nr_topics=K)
    labelsK, _ = mK.transform(docs)
    info = mK.get_topic_info().query("Topic >= 0").copy().sort_values("Count", ascending=False)
    return mK, np.array(labelsK), info

model25, lab25, info25 = reduce_and_label(topic_model, docs_text, K=25)
model30, lab30, info30 = reduce_and_label(topic_model, docs_text, K=30)

print(info25.head(10))
print(info30.head(10))

# 2) Utility: get top-n words per topic as a set
def top_words(model, topic_id, n=12):
    return {w for (w,_) in (model.get_topic(topic_id) or [])[:n]}

# 3) Jaccard similarity between K=30 and K=25 topics (based on top words)
def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

T30 = info30["Topic"].tolist()
T25 = info25["Topic"].tolist()

J = pd.DataFrame(0.0, index=T30, columns=T25)
for t30 in T30:
    w30 = top_words(model30, t30, n=12)
    for t25 in T25:
        w25 = top_words(model25, t25, n=12)
        J.loc[t30, t25] = jaccard(w30, w25)

# 4) For each K=30 topic, find best K=25 match
best_map = (J.stack()
              .rename("Jaccard")
              .reset_index()
              .rename(columns={"level_0":"K30","level_1":"K25"}))
best_map = best_map.sort_values(["K30","Jaccard"], ascending=[True, False])\
                   .groupby("K30", as_index=False).first()

# 5) Add counts and top words for readability
def topic_signature(model, tid, n=8):
    return ", ".join([w for (w,_) in (model.get_topic(tid) or [])[:n]])

cnt30 = info30.set_index("Topic")["Count"].to_dict()
cnt25 = info25.set_index("Topic")["Count"].to_dict()

best_map["K30_Count"] = best_map["K30"].map(cnt30)
best_map["K25_Count"] = best_map["K25"].map(cnt25)
best_map["K30_Words"] = best_map["K30"].map(lambda t: topic_signature(model30, t))
best_map["K25_Words"] = best_map["K25"].map(lambda t: topic_signature(model25, t))

# 6) Also show how documents flow: contingency table (K30 -> K25)
flow = pd.crosstab(lab30, lab25)  # rows: K30 topics, cols: K25 topics

# 7) Summary table: top matches (sort by K30 size, then Jaccard)
summary = best_map.sort_values(["K30_Count","Jaccard"], ascending=[False, False])
summary = summary[["K30","K30_Count","K30_Words","K25","K25_Count","K25_Words","Jaccard"]]
print("K=30 → best matching K=25 topic (by Jaccard of top words):")
print(summary.head(20))

# 8) Quick helper to inspect a pair
def compare_pair(k30, k25, n_words=12, n_docs=2):
    print(f"\n=== K30 Topic {k30} (n={cnt30.get(k30,0)}) ===")
    for w,_ in (model30.get_topic(k30) or [])[:n_words]:
        print(" ", w)
    try:
        reps = model30.get_representative_docs(k30)[:n_docs]
        for i, d in enumerate(reps, 1): print(f"  • Doc {i}: {d[:180]}…")
    except Exception: pass

    print(f"\n=== K25 Topic {k25} (n={cnt25.get(k25,0)}) ===")
    for w,_ in (model25.get_topic(k25) or [])[:n_words]:
        print(" ", w)
    try:
        reps = model25.get_representative_docs(k25)[:n_docs]
        for i, d in enumerate(reps, 1): print(f"  • Doc {i}: {d[:180]}…")
    except Exception: pass

# Example: inspect the top 5 biggest K30 topics and their best K25 match
for k30 in summary["K30"].head(5):
    k25 = int(summary.loc[summary["K30"]==k30, "K25"].iloc[0])
    compare_pair(int(k30), k25)

"""**stable topics:**
Chatbots (K25 topic 3 ↔ K30 topic 3)→ consistent keywords (chatbot, question, answer, taxpayer, cdc, response).

Code generation / generative usage (K25 topic 4 ↔ K30 topic 4).

Video/audio transcription (K25 topic 5 ↔ K30 topic 5).

FDA/medical devices (K25 topic 9 ↔ K30 topic 10).

Water/DOI models (K25 topic 2 ↔ K30 topic 1).


**merged topics**
Merged topics at K=25 (splits at K=30):

“Application/review/search/user” is huge at K=25 (topic 0, n=361).
At K=30, it splits into:

Topic 0: search, user, document, form, metadata…

Topic 2: comment, grant, review, application… → Suggests that K=30 is teasing apart grants/public comments vs. generic application/search forms.

Risk & security topics:

At K=25: topic 1 = threat, detection, security, anomaly…

At K=30:

Topic 6 = CBP, detection, safety, image, inspection…

Topic 8 = threat, security, detection, network…
→ K=30 creates more fine-grained clusters (CBP/inspection vs. network security).

Patients/clinical risk:

At K=25: topic 6 = patient, risk, health, clinical, veteran, care.

At K=30: topic 9 is almost identical.→ Stable.



**Takeaway: K=25 -> fewer, bigger clusters; easier to interpret at the very high level (10-12 buckets); BUT merges distinct use cases**

**K=30 -> splits some of those buckets, so it's better for feature engineering to capture nuance given the purporses of our project. Eg, shows grants/public comments vs. search/applications. More redudancy risk but more interpretable**

Select K=30 and Analyze:
"""

#Lock in K=30 and re-label docs

docs_text = aiusecase["text_lemma"].astype(str).tolist()
topic_model_K30 = deepcopy(topic_model).reduce_topics(docs_text, nr_topics=30)

# get labels/probabilities for K=30
labels_K30, probs_K30 = topic_model_K30.transform(docs_text)
aiusecase["topic_id"] = labels_K30

def topic_table(model, top_n=30):
    info = model.get_topic_info().copy()
    info = info[info["Topic"] >= 0].sort_values("Count", ascending=False)
    print(info.head(top_n))
    return info

topic_info_K30 = topic_table(topic_model_K30, top_n=30)

# 0) Ensure labels for K=30
docs_text = aiusecase["text_lemma"].astype(str).tolist()
labels_K30, probs_K30 = topic_model_K30.transform(docs_text)

# 1) Build representative docs WITHOUT set_representative_docs

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(docs_text, show_progress_bar=False, normalize_embeddings=True)

def compute_representatives(labels, X, texts, topn=5, min_docs=2):
    reps = {}
    idxs = defaultdict(list)
    for i, t in enumerate(labels):
        if t >= 0:
            idxs[int(t)].append(i)
    for t, ii in idxs.items():
        if len(ii) < min_docs:
            reps[t] = [texts[i] for i in ii]
            continue
        Xi = X[ii]
        c  = Xi.mean(axis=0, keepdims=True)
        scores = cosine_similarity(Xi, c).ravel()
        order  = np.argsort(-scores)[:topn]
        reps[t] = [texts[ii[j]] for j in order]
    return reps

reps_K30 = compute_representatives(labels_K30, emb, docs_text, topn=5)

# 2) Single helper to print a topic (top words + rep docs)
def show_topic(model, topic_id, n_words=12, n_docs=2, info_df=None, reps_dict=None):
    if info_df is None:
        info_df = model.get_topic_info()
    cnt = int(info_df.loc[info_df["Topic"]==topic_id, "Count"].values[0])
    words = model.get_topic(topic_id) or []
    print(f"\nTopic {topic_id}  (n={cnt})")
    for term, w in words[:n_words]:
        print(f"  {term:28s} {w:.4f}")

    # Prefer our precomputed reps
    printed = False
    if reps_dict is not None and topic_id in reps_dict:
        for i, txt in enumerate(reps_dict[topic_id][:n_docs], 1):
            print(f"   • Doc {i}: {txt[:220]}…")
        printed = True
    if not printed:
        try:
            for i, txt in enumerate((model.get_representative_docs(topic_id) or [])[:n_docs], 1):
                print(f"   • Doc {i}: {txt[:220]}…")
            printed = True
        except Exception:
            pass
    if not printed:
        print("   (no representative docs available)")

# 3) Call it on the largest topics
topic_info_K30 = topic_model_K30.get_topic_info()
topic_info_K30 = topic_info_K30[topic_info_K30["Topic"]>=0].sort_values("Count", ascending=False)

for t in topic_info_K30["Topic"].head(8):
    show_topic(topic_model_K30, int(t), n_words=12, n_docs=2,
               info_df=topic_info_K30, reps_dict=reps_K30)

#Topic bar chart
fig_bar = topic_model_K30.visualize_barchart(top_n_topics=30, n_words=10, height=600)
fig_bar.show()

#Intertopic Distance Map
fig_topics = topic_model_K30.visualize_topics(height=650, width=900)
fig_topics.show()

#Hierarchy: dendogram of topic similarity
fig_hier = topic_model_K30.visualize_hierarchy(height=700, width=900)
fig_hier.show()

# ----- Build a label map for K=30 -----
def auto_label(model, tid, n=4):
    words = model.get_topic(tid) or []
    return f"{tid}: " + ", ".join([w for w,_ in words[:n]])

name_overrides = {
    0:  "Search & Forms (Metadata)",
    1:  "Water/DOI Outputs & Updates",
    2:  "Grants & Public Comments Review",
    3:  "Enterprise Chatbots (IRS/CDC)",
    4:  "Code Generation & Recommendations",
    5:  "Audio/Video Transcription",
    6:  "CBP Imaging & Safety Inspection",
    7:  "Financial Risk & Fraud (Bank/Portfolio)",
    8:  "Network Threat Detection",
    9:  "Clinical Risk & Patient Care",
    10: "FDA-Cleared Medical Devices",
    11: "Social Media / Humanitarian Monitoring",
    12: "Forest & Landcover Mapping",
    13: "Identity Verification & Face Matching",
    14: "Legal Research & Investigations",
    15: "Wildlife Surveys & Species Counts",
    16: "Classification/Coding (Supply Chain)",
    17: "Translation & Multilingual NLP",
    18: "DEA/Public Data Analytics",
    19: "Computer Vision Measurement",
    20: "Risk Scoring & Classification",
    21: "Microsoft Copilot / Productivity",
    22: "USCIS/Census Identity Matching",
    23: "OCR on PDFs (Text Extraction)",
    24: "Training & Workforce Upskilling",
    25: "NLM Biomedical Indexing/Citations",
    26: "Patient Monitoring (Home/Clinical)",
    27: "Foodborne/Outbreak Investigation (CDC)",
    28: "Scientific Control & Real-time Optimization",
}

# Build final label map from current model topics
infoK30 = topic_model_K30.get_topic_info().query("Topic >= 0").copy()
topic_ids = infoK30["Topic"].astype(int).tolist()
label_map = {tid: name_overrides.get(tid, auto_label(topic_model_K30, tid))
             for tid in topic_ids}

# Attach names to rows
aiusecase["topic_name"] = aiusecase["topic_id"].map(label_map)

# Hard one-hot by topic name
topic_ohe = pd.get_dummies(aiusecase["topic_name"], prefix="topic", prefix_sep="=")
aiusecase_features = pd.concat([aiusecase, topic_ohe], axis=1)

# Columns in the same order as infoK30["Topic"]
topic_order = infoK30["Topic"].astype(int).tolist()
idx_map = {tid:i for i, tid in enumerate(topic_order)}

P = np.zeros((len(aiusecase), len(topic_order)), dtype=np.float32)
valid = aiusecase["topic_id"].values >= 0
for i in range(len(aiusecase)):
    if not valid[i]:
        continue
    try:
        P[i, :] = probs_K30[i][:len(topic_order)]
    except Exception:
        pass

proba_cols = [f"topicP_{label_map[tid]}" for tid in topic_order]
proba_df   = pd.DataFrame(P, columns=proba_cols, index=aiusecase.index)

aiusecase_features_soft = pd.concat([aiusecase_features, proba_df], axis=1)

# Peek
aiusecase_features_soft.filter(regex=r"^topic(=|P_)").head(3)

# Prep DFS
is_dhs    = aiusecase["Agency Abbreviation"].eq("DHS")
dhs_df    = aiusecase[is_dhs].copy()
non_df    = aiusecase[~is_dhs].copy()

# Counts by topic id
dhs_counts = dhs_df["topic_id"].value_counts().rename_axis("Topic").reset_index(name="DHS_Count")
non_counts = non_df["topic_id"].value_counts().rename_axis("Topic").reset_index(name="NonDHS_Count")

topic_counts = infoK30[["Topic","Count"]].merge(dhs_counts, on="Topic", how="left") \
                                         .merge(non_counts, on="Topic", how="left")
topic_counts.fillna(0, inplace=True)
topic_counts["DHS_Count"]    = topic_counts["DHS_Count"].astype(int)
topic_counts["NonDHS_Count"] = topic_counts["NonDHS_Count"].astype(int)
topic_counts["TopicName"]    = topic_counts["Topic"].astype(int).map(label_map)

# --- A) Top DHS topics ---
tc_sorted = topic_counts.sort_values("DHS_Count", ascending=False)
fig = px.bar(
    tc_sorted.head(15),
    x="DHS_Count", y="TopicName",
    orientation="h",
    title="Top DHS Topics (K=30)",
    text="DHS_Count"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
fig.show()

# --- B) Treemap of DHS topics ---
fig = px.treemap(
    tc_sorted[tc_sorted["DHS_Count"]>0],
    path=["TopicName"], values="DHS_Count",
    title="DHS Topics Treemap (K=30)"
)
fig.show()

# --- C) Normalized comparison DHS vs Non-DHS ---
tc = topic_counts.copy()
tc["DHS_Share"]    = tc["DHS_Count"]    / max(1, dhs_df.shape[0])
tc["NonDHS_Share"] = tc["NonDHS_Count"] / max(1, non_df.shape[0])
melt = tc.melt(id_vars=["Topic","TopicName"], value_vars=["DHS_Share","NonDHS_Share"],
               var_name="Group", value_name="Share")

top_by_dhs = tc.sort_values("DHS_Share", ascending=False).head(15)["Topic"].tolist()
melt_top   = melt[melt["Topic"].isin(top_by_dhs)]

fig = px.bar(
    melt_top,
    x="Share", y="TopicName", color="Group",
    barmode="group",
    title="Topic Share: DHS vs Non-DHS (normalized by group size)"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=650)
fig.show()

# --- D) DHS-only interactive scatter (documents colored by topic) ---
docs_text = aiusecase["text_lemma"].astype(str).tolist()
labels_K30_list = aiusecase["topic_id"].astype(int).tolist() # Get labels as a list

# Generate the full document visualization
fig = topic_model_K30.visualize_documents(
    docs=docs_text, # Pass full documents
    topics=labels_K30_list, # Pass full labels
    width=950, height=700,
    hide_annotations=False
)

# Filter the figure data to show only DHS documents
dhs_indices = aiusecase.index[is_dhs].tolist()
filtered_data = [trace for trace in fig.data if isinstance(trace, go.Scattergl)] # Get scatter traces

new_traces = []
for trace in filtered_data:
    # Ensure text and customdata are not None before indexing
    trace_text = trace.text if trace.text is not None else []
    trace_customdata = trace.customdata if trace.customdata is not None else []

    dhs_x = [trace.x[i] for i in dhs_indices if i < len(trace.x)] # Add index check
    dhs_y = [trace.y[i] for i in dhs_indices if i < len(trace.y)] # Add index check
    dhs_text = [trace_text[i] for i in dhs_indices if i < len(trace_text)] # Use checked text
    dhs_customdata = [trace_customdata[i] for i in dhs_indices if i < len(trace_customdata)] # Use checked customdata


    # Only create a new trace if there are data points for DHS
    if dhs_x and dhs_y:
        new_trace = go.Scattergl(
            x=dhs_x,
            y=dhs_y,
            mode='markers',
            marker=trace.marker,
            text=dhs_text,
            name=trace.name,
            customdata=dhs_customdata if dhs_customdata else None # Use checked customdata
        )
        new_traces.append(new_trace)


# Create a new figure with only the filtered DHS traces
fig_dhs_only = go.Figure(data=new_traces, layout=fig.layout)
fig_dhs_only.update_layout(title="DHS-only Document Map (colored by topic)")
fig_dhs_only.show()


# --- E) Over-representation (lift) ---
tc["Overall_Share"] = tc["Count"] / aiusecase.shape[0]
tc["DHS_Lift"]      = tc["DHS_Share"] / tc["Overall_Share"].replace(0, np.nan)

lift_tab = tc.sort_values("DHS_Lift", ascending=False)[
    ["Topic","TopicName","DHS_Count","Count","DHS_Share","Overall_Share","DHS_Lift"]
]
print(lift_tab.head(15))

fig = px.bar(
    lift_tab.head(15),
    x="DHS_Lift", y="TopicName",
    orientation="h",
    title="Topics Most Over-represented in DHS (Lift = DHS_share / Overall_share)",
    text="DHS_Lift"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=650)
fig.show()

# Hard one-hot by topic name
topic_ohe = pd.get_dummies(aiusecase["topic_name"], prefix="topic", prefix_sep="=")
aiusecase_features = pd.concat([aiusecase, topic_ohe], axis=1)

# In case needed: soft features from probabilities (probs_K30)
# Columns in the same order as infoK30["Topic"]
topic_order = infoK30["Topic"].astype(int).tolist()
idx_map = {tid:i for i, tid in enumerate(topic_order)}

P = np.zeros((len(aiusecase), len(topic_order)), dtype=np.float32)
valid = aiusecase["topic_id"].values >= 0
for i in range(len(aiusecase)):
    if not valid[i]:
        continue
    try:
        P[i, :] = probs_K30[i][:len(topic_order)]
    except Exception:
        pass

proba_cols = [f"topicP_{label_map[tid]}" for tid in topic_order]
proba_df   = pd.DataFrame(P, columns=proba_cols, index=aiusecase.index)

aiusecase_features_soft = pd.concat([aiusecase_features, proba_df], axis=1)

# Peek
aiusecase_features_soft.filter(regex=r"^topic(=|P_)").head(3)

# Define path to save to Topic Modeling
output_path = 'Topic Modeling/ai_use_case_features.csv'

# Save df to a CSV file
try:
    aiusecase_features_soft.to_csv(output_path, index=False)
    print(f"DataFrame successfully saved to {output_path}")
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")

# assumes: aiusecase, topics, probs, topic_model

aiusecase["is_outlier"] = (np.array(topics) < 0)

# Which texts are short / boilerplate?
aiusecase["text_len"] = aiusecase["text_all"].str.split().str.len()
print(aiusecase["is_outlier"].mean())  # ~0.23

# Outliers by agency / bureau / stage
for col in ["Agency", "Bureau", "Stage of Development"]:
    if col in aiusecase.columns:
        print(f"\nOutlier rate by {col}:")
        print(aiusecase.groupby(col)["is_outlier"].mean().sort_values(ascending=False).head(15))

# Text length vs outliers
print("\nMedian text length — outliers vs non-outliers:")
print(aiusecase.groupby("is_outlier")["text_len"].median())

# Spot duplicates/near-empties
dups = aiusecase["text_all"].str.lower().str.strip().duplicated(keep=False)
print("\nShare of exact-duplicate texts among outliers:",
      aiusecase.loc[aiusecase.is_outlier & dups].shape[0] / max(1, aiusecase.is_outlier.sum()))

df = aiusecase.copy()
by_agency = (df.groupby("Agency")
               .agg(n=("Use Case Name","size"), out_rate=("is_outlier","mean"))
               .sort_values("out_rate", ascending=False))
by_bureau = (df.groupby("Bureau")
               .agg(n=("Use Case Name","size"), out_rate=("is_outlier","mean"))
               .sort_values("out_rate", ascending=False))

print("Agencies with out_rate ≥ 0.5 but n ≤ 5")
print(by_agency[(by_agency["out_rate"]>=0.5)&(by_agency["n"]<=5)])

print("Bureaus with out_rate = 1.0 but n ≤ 3")
print(by_bureau[(by_bureau["out_rate"]>=0.99)&(by_bureau["n"]<=3)])

# === OUTLIER AUDIT


# 0) Shorthand
df = aiusecase.copy()
feat = aiusecase_features.copy()

# 1) Ensure a topic label exists and build outlier flag
if "topic_id" not in df.columns:
    # Fall back to features if topic_id lives there
    if "topic_id" in feat.columns:
        # Join topic_id back to aiusecase
        join_keys = ["Use Case Name"] + (["Agency"] if "Agency" in df.columns and "Agency" in feat.columns else [])
        df = df.merge(feat[join_keys + ["topic_id"]].drop_duplicates(), on=join_keys, how="left", suffixes=("","_feat"))
    else:
        raise ValueError("No 'topic_id' column found in aiusecase or aiusecase_features.")
df["is_outlier"] = df["topic_id"].lt(0)

# 2) Pick TEXT column
CAND_TEXT_COLS = [
    "text_lemma", "text_clean", "text_all",
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs.", "Use Case Name"
]
TEXT_COL = next((c for c in CAND_TEXT_COLS if c in df.columns), None)
if TEXT_COL is None:
    raise ValueError("Could not find modeling text column. Add it to CAND_TEXT_COLS.")

# 3) Attach probabilities from features for “near-topic” checks
topic_p_cols = [c for c in feat.columns if c.startswith("topicP_")]
if topic_p_cols:
    join_keys = ["Use Case Name"] + (["Agency"] if "Agency" in df.columns and "Agency" in feat.columns else [])
    df = df.merge(feat[join_keys + topic_p_cols].drop_duplicates(), on=join_keys, how="left")

# 4) Basic tally
n_total = len(df)
n_out = int(df["is_outlier"].sum())
print(f"Total rows: {n_total} | Outliers: {n_out} ({n_out/n_total:.2%})")

# 5) Text metrics
WORD_RE  = re.compile(r"\b[a-zA-Z0-9-]+\b")
DIGIT_RE = re.compile(r"\d")
PHRASE_STOP = [
    r"\bmission[-\s]+enabling\b",
    r"\buse[-\s]*case(?:s)?\b",
    r"\bintended\s+purpose\b",
    r"\bexpected\s+benefits\b",
    r"\bthis\s+(project|system|tool)\b",
    r"\bprovide(s|d)?\s+(capability|capabilities)\b",
]

def text_metrics(s: str) -> dict:
    s = "" if pd.isna(s) else str(s)
    words = WORD_RE.findall(s)
    uniq  = set(w.lower() for w in words)
    char_len = len(s)
    punct_ratio = sum(ch in ".,;:!?()[]{}\"'/" for ch in s) / max(1, char_len)
    digit_count = len(DIGIT_RE.findall(s))
    has_boiler  = any(re.search(p, s, flags=re.I) for p in PHRASE_STOP)
    return {
        "word_count": len(words),
        "uniq_tokens": len(uniq),
        "lex_diversity": (len(uniq) / max(1, len(words))),
        "char_len": char_len,
        "punct_ratio": punct_ratio,
        "digit_count": digit_count,
        "has_boilerplate": has_boiler
    }

metrics_df = df[TEXT_COL].apply(text_metrics).apply(pd.Series)
dfm = pd.concat([df, metrics_df], axis=1)

# 6) Look directly at the OUTLIER entries
cols_show = [
    "Agency","Bureau","Stage of Development","Use Case Name", TEXT_COL,
    "word_count","uniq_tokens","lex_diversity","punct_ratio","digit_count","has_boilerplate"
]
pd.set_option("display.max_colwidth", 220)
outliers_view = (dfm[dfm["is_outlier"]]
                 .sort_values(["word_count","lex_diversity","punct_ratio"], ascending=[True, True, False]))
print(outliers_view[cols_show].head(50))  # 👈 read first 50 to spot patterns

# Export full audit for manual review
audit_path = "Topic Modeling/aiusecase_outlier_audit.csv"
outliers_view[cols_show].to_csv(audit_path, index=False, encoding="utf-8")
print(f"Saved detailed outlier audit → {audit_path}")

# 7) Duplicate & near-empty checks among OUTLIERS
base_text_norm = dfm[TEXT_COL].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
dup_mask = base_text_norm.duplicated(keep=False)
print("Exact-duplicate share among OUTLIERS:",
      (dup_mask & dfm["is_outlier"]).mean())

empties_share = (dfm["is_outlier"] & (dfm["word_count"] <= 3)).mean()
print("Near-empty (≤3 words) share among OUTLIERS:", empties_share)

# 8) “Near-topic” outliers using max topic probability
if topic_p_cols:
    dfm["p_max"] = dfm[topic_p_cols].max(axis=1)
    print("\nMax topic probability (p_max) — OUTLIERS vs NON-outliers")
    print("OUTLIERS:\n", dfm.loc[dfm["is_outlier"], "p_max"].describe())
    print("NON-OUTLIERS:\n", dfm.loc[~dfm["is_outlier"], "p_max"].describe())

    near = dfm[dfm["is_outlier"] & (dfm["p_max"] >= 0.25)]
    print(f"\nOutliers with high nearest-topic probability (n={len(near)}):")
    print(near.sort_values("p_max", ascending=False)[["Use Case Name", TEXT_COL, "p_max"]].head(25))

# 9) Group profiles (NO standardization of names)
def rate_with_n(frame, by):
    g = (frame.groupby(by)["is_outlier"]
                .agg(rate="mean", n="size")
                .sort_values(["rate","n"], ascending=[False, False]))
    return g

if "Agency" in df.columns:
    print("\nOutlier rate by Agency (with n):")
    print(rate_with_n(dfm, "Agency").head(20))

if "Bureau" in df.columns:
    print("\nOutlier rate by Bureau (with n):")
    print(rate_with_n(dfm, "Bureau").head(20))

if "Stage of Development" in df.columns:
    print("\nOutlier rate by Stage of Development (row-normalized):")
    stage_tab = pd.crosstab(dfm["Stage of Development"], dfm["is_outlier"],
                            normalize='index').rename(columns={False:"non_outlier", True:"outlier"})
    print(stage_tab.sort_values("outlier", ascending=False))

# 10) Visuals
plt.figure(figsize=(8,4))
dfm.boxplot(column="word_count", by="is_outlier")
plt.suptitle(""); plt.title("Word Count by Outlier Flag"); plt.xlabel("is_outlier"); plt.ylabel("words"); plt.show()

plt.figure(figsize=(8,4))
dfm.boxplot(column="lex_diversity", by="is_outlier")
plt.suptitle(""); plt.title("Lexical Diversity by Outlier Flag"); plt.xlabel("is_outlier"); plt.ylabel("unique/words"); plt.show()

# 11) Print the modeling knobs used earlier
try:
    print("\nUMAP params:", topic_model.umap_model.get_params())
    print("HDBSCAN params:", topic_model.hdbscan_model.get_params())
except Exception:
    pass

"""#Reasons for needing further cleaning (outliers):

UMAP is very “sharp”: n_neighbors=8 (tiny local neighborhoods) + min_dist=0.0 (squeeze points into tight blobs) + n_components=5 (aggressive compression). That combination over-separates the space.

HDBSCAN is still fairly strict: min_cluster_size=15, min_samples=5, epsilon=0.0. With UMAP’s sharp separation, many points never reach core density ⇒ labeled -1.

Stage table shows early/transition stages have higher outlier rates (heterogeneous prose), and small-n bureaus report 1.0 just because n≤2—not a modeling signal.

Boxplots: outliers are not shorter; they’re often longer and lower in lexical diversity → multi-theme/boilerplate text sits between clusters, which our strict UMAP+HDBSCAN combo treats as noise.

"""

TEXT_COL = "text_lemma" if "text_lemma" in aiusecase else (
           "text_clean" if "text_clean" in aiusecase else
           "What is the intended purpose and expected benefits of the AI?")

new_topics = topic_model.reduce_outliers(
    aiusecase[TEXT_COL].astype(str).tolist(),
    topics=aiusecase["topic_id"].to_numpy(),
    strategy="c-tf-idf",   # try "embeddings" if we want semantic nearest neighbor
    threshold=0.35         # 0.30–0.50; lower = more aggressive reassignment
)

aiusecase["topic_id"] = new_topics
aiusecase["is_outlier"] = (aiusecase["topic_id"] < 0)
print("Outlier rate AFTER reassignment:", aiusecase["is_outlier"].mean())

TEXT_COL = "text_lemma" if "text_lemma" in aiusecase else (
           "text_clean" if "text_clean" in aiusecase else
           "What is the intended purpose and expected benefits of the AI?")
docs = aiusecase[TEXT_COL].fillna("").astype(str).tolist()

# Stronger sentence model helps multi-theme/boilerplate text
embedder = SentenceTransformer("all-mpnet-base-v2")

# >>> DENSER UMAP <<<
umap_model = umap.UMAP(
    n_neighbors=30,      # was 8  → use 25–40
    n_components=10,     # was 5  → keep more structure
    min_dist=0.10,       # was 0.0 → reduces over-separation
    metric="cosine",
    random_state=42
)

# >>> FRIENDLIER HDBSCAN <<<
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,           # was 15
    min_samples=1,                 # was 5  → fewer “noise” points
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.05,# was 0.0 → merges thin rims
    metric="euclidean",
    prediction_data=True
)

topic_model2 = BERTopic(
    embedding_model=embedder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=getattr(topic_model, "vectorizer_model", None),
    calculate_probabilities=True,
    top_n_words=10,
    verbose=True
)

topics2, probs2 = topic_model2.fit_transform(docs)
out_rate2 = (np.array(topics2) < 0).mean()
print("Outlier rate (denser refit):", out_rate2)

# Keep results
aiusecase["topic_id"] = topics2
aiusecase["is_outlier"] = (aiusecase["topic_id"] < 0)

# Update one-hot/probabilities
onehot = (pd.get_dummies(aiusecase["topic_id"].astype(int), prefix="topic")
          .add_prefix("") )  # already prefixed
aiusecase_features = aiusecase[["Use Case Name","Agency"]].join(onehot)

# Add probability matrix as topicP_#
if probs2 is not None:
    prob_df = pd.DataFrame(probs2, index=aiusecase.index)
    prob_df.columns = [f"topicP_{i}" for i in range(prob_df.shape[1])]
    aiusecase_features = aiusecase_features.join(prob_df)

"""What each code block does

Sanity/transition check

Rebuilds docs from aiusecase, compares w/ current labels to reduce_outliers

Prints: baseline outlier rate, “−1→topic”

If any topic→−1 > 0, docs were misaligned; fix this before trusting any results.

reduce_outliers sweep

Tries both strategies ("c-tf-idf" and "embeddings") and thresholds (0.15–0.55).

Picks the combo that minimizes the outlier rate without refitting the model.

This only reassigns noise points; topics themselves don’t change.

Probability-based patch (optional)

Uses aiusecase_features.topicP_*.

For rows still at -1, if max(topicP_*) ≥ cutoff (e.g., 0.30), force-assign to the argmax topic.

Non-destructive; affects only confident outliers.

Denser refit (the “second chunk”)

Rebuilds the geometry with:

UMAP: more neighbors (30), more dims (10), nonzero min_dist (0.10) → smoother, denser manifold.

HDBSCAN: smaller min_cluster_size (10), min_samples=1, epsilon=0.05 → keeps thin cluster rims.

Stronger embedder (all-mpnet-base-v2) for multi-theme, boilerplatey text.

This is the proper fix; it changes topics and probabilities and typically drops outliers by 8–15 points.
"""

# ==== Micro-sweep to push outliers ====

TEXT_COL = ("text_lemma" if "text_lemma" in aiusecase
            else "text_clean" if "text_clean" in aiusecase
            else "What is the intended purpose and expected benefits of the AI?")
docs = aiusecase[TEXT_COL].fillna("").astype(str).tolist()

# ---- Precompute embeddings once (normalize for cosine) ----
embedder = SentenceTransformer("all-mpnet-base-v2")
doc_emb = embedder.encode(docs, show_progress_bar=True, normalize_embeddings=True)

# ---- Small grid ----
NN_LIST  = [25, 35]          # neighbors
MD_LIST  = [0.08, 0.12]      # min_dist
MCS_LIST = [8, 10]           # min_cluster_size
MS_LIST  = [1, 2]            # min_samples
EPS_LIST = [0.05, 0.10]      # epsilon
NCOMP    = 12                # keep a bit more structure than 5/10

results, models = [], []

def tiny_topic_share(labels, min_size=6):
    labels = np.array(labels)
    uniq = [t for t in np.unique(labels) if t != -1]
    if not uniq: return 1.0
    sizes = np.array([(labels==t).sum() for t in uniq])
    return (sizes < min_size).mean()

for nn in NN_LIST:
    for md in MD_LIST:
        for mcs in MCS_LIST:
            for ms in MS_LIST:
                for eps in EPS_LIST:
                    umap_model = umap.UMAP(
                        n_neighbors=nn, n_components=NCOMP, min_dist=md,
                        metric="cosine", random_state=42
                    )
                    hdbscan_model = hdbscan.HDBSCAN(
                        min_cluster_size=mcs, min_samples=ms,
                        cluster_selection_method="eom",
                        cluster_selection_epsilon=eps,
                        metric="euclidean", prediction_data=True
                    )
                    tm = BERTopic(
                        embedding_model=None,                 # we pass precomputed embeddings
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        vectorizer_model=getattr(topic_model, "vectorizer_model", None),
                        calculate_probabilities=False,        # speed: off for sweep
                        top_n_words=10, verbose=False
                    )
                    labels, _ = tm.fit_transform(docs, embeddings=doc_emb)
                    labels = np.array(labels)
                    out_rate = (labels < 0).mean()
                    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
                    tiny_share = tiny_topic_share(labels, min_size=6)

                    results.append({
                        "n_neighbors": nn, "min_dist": md,
                        "min_cluster_size": mcs, "min_samples": ms, "epsilon": eps,
                        "n_components": NCOMP,
                        "outlier_rate": out_rate,
                        "n_topics": n_topics,
                        "tiny_topic_share": tiny_share
                    })
                    models.append((tm, labels))

res = pd.DataFrame(results).sort_values(["outlier_rate","tiny_topic_share","n_topics"])
print("Top candidates:"); print(res.head(10))

# ---- Choose best under reasonable constraints ----
cands = res[
    (res["n_topics"].between(20, 120)) &
    (res["tiny_topic_share"] <= 0.35)
].sort_values(["outlier_rate","tiny_topic_share","n_topics"])

best_idx = res.index[0] if cands.empty else cands.index[0]
best_row = res.loc[best_idx]
print("\nSelected config:", best_row.to_dict())

# ---- Refit best with probabilities + post-fit outlier reduction ----
umap_best = umap.UMAP(
    n_neighbors=int(best_row["n_neighbors"]),
    n_components=int(best_row["n_components"]),
    min_dist=float(best_row["min_dist"]),
    metric="cosine", random_state=42
)
hdbscan_best = hdbscan.HDBSCAN(
    min_cluster_size=int(best_row["min_cluster_size"]),
    min_samples=int(best_row["min_samples"]),
    cluster_selection_method="eom",
    cluster_selection_epsilon=float(best_row["epsilon"]),
    metric="euclidean", prediction_data=True
)

tm_best = BERTopic(
    embedding_model=None,
    umap_model=umap_best,
    hdbscan_model=hdbscan_best,
    vectorizer_model=getattr(topic_model, "vectorizer_model", None),
    calculate_probabilities=True,        # turn probs back on
    top_n_words=10, verbose=True
)

labels_best, probs_best = tm_best.fit_transform(docs, embeddings=doc_emb)
rate_before = (np.array(labels_best) < 0).mean()

labels_reduced = tm_best.reduce_outliers(
    docs, topics=labels_best, strategy="c-tf-idf", threshold=0.35
)
rate_after = (np.array(labels_reduced) < 0).mean()
print(f"\nOutlier rate BEFORE reduce_outliers: {rate_before:.2%}")
print(f"Outlier rate AFTER  reduce_outliers: {rate_after:.2%}")

# ---- Commit to aiusecase and rebuild features ----
aiusecase["topic_id"]   = labels_reduced
aiusecase["is_outlier"] = (aiusecase["topic_id"] < 0)
print("\nFINAL outlier rate:", aiusecase["is_outlier"].mean())
print("Number of topics:",
      len(set(aiusecase["topic_id"])) - (1 if -1 in set(aiusecase["topic_id"]) else 0))

# Rebuild aiusecase_features for fresh one-hots + probabilities
REGENERATE_FEATURES = True
if REGENERATE_FEATURES:
    oh = pd.get_dummies(aiusecase["topic_id"].astype(int), prefix="topic")
    feat_new = aiusecase[["Use Case Name","Agency"]].join(oh)
    if probs_best is not None and len(probs_best) == len(aiusecase):
        prob_df = pd.DataFrame(probs_best, index=aiusecase.index)
        prob_df.columns = [f"topicP_{i}" for i in range(prob_df.shape[1])]
        feat_new = feat_new.join(prob_df)
    aiusecase_features = feat_new

labels = aiusecase["topic_id"].to_numpy()
out_rate = (labels < 0).mean()

# tiny-topic share
lbl = labels[labels >= 0]
uniq = np.unique(lbl)
sizes = np.array([(lbl==t).sum() for t in uniq])
tiny_share = (sizes < 6).mean() if len(sizes) else 1.0

# nearest-topic confidence among outliers
prob_cols = [c for c in aiusecase_features.columns if c.startswith("topicP_")]
recoverable = np.nan
if prob_cols:
    P = aiusecase_features[prob_cols].to_numpy()
    pmax = P.max(axis=1)
    recoverable = ( (labels < 0) & (pmax >= 0.30) ).mean()  # share of outliers with p_max ≥ 0.30

print(f"Outlier rate: {out_rate:.2%}")
print(f"Tiny-topic share (<6 docs): {tiny_share:.2%}")
if prob_cols:
    print(f"Recoverable outliers (p_max ≥ 0.30): {recoverable:.2%}")

"""Stratefy to reduce outliers worked, but solution implied using 60-80 topics, which is too big for us.

We also tested this reassignment technique but it did not yield better results (failed to address outliers at a higher rate)

In BERTopic, any doc labeled -1 by HDBSCAN is an outlier (not dense enough to be assigned to a topic). Code doen't recluster; it only tries to reassign some of those -1s to existing 30 topics when there’s strong evidence.

How the code rescues outliers

Probability-based rescue (most conservative)

MODEL30.transform(docs) computes, for each doc, a probability vector P over the 30 topics (the -1 class is not a column in P).

It maps argmax(P) to the true topic ID using get_topic_info() (important because BERTopic’s column order ≠ topic IDs).

For any doc currently labeled -1, if its max topic probability p_max ≥ 0.30, it’s reassigned to that topic.

Effects: only turns confident -1s into topics; leaves all inliers alone; topic set stays 30.

Centroid-similarity rescue (geometry check)

Builds topic centroids in the same embedding space the model used (via MODEL30._extract_embeddings or the model’s embedding_model).

For remaining -1s, computes cosine similarity to every centroid and assigns only if similarity ≥ 0.35 (tunable).

Effects: “snaps” borderline points into the nearest existing topic only when geometry also agrees. Still no new topics created.

Commit + rebuild features

Writes rescued labels to topic_id_30 (and is_outlier_30), and rebuilds one-hots + probabilities in the 30-topic space so everything downstream aligns with the 30-topic taxonomy validated.

Why it’s “no harm”

Topics are preserved: we don’t refit, merge, or split; just relabel some -1s.

Conservative gates: both probability and similarity thresholds ensure we only pull in outliers when there’s clear evidence.

Model-consistent: probabilities and embeddings come from the same 30-topic model, avoiding mismatches.

What we could tune (and trade-offs)

p_max cutoff (default 0.30):
Lower → fewer outliers, higher risk of misassignment; Higher → purer topics, more outliers remain.

similarity THRESH (default 0.35):
Lower → more rescues; Higher → stricter.

Optional guardrails: skip centroid assignment into tiny topics (e.g., size < 6), or require both p_max and similarity to be above thresholds.
"""

# ==== Lock K=30 (version-safe) + no-harm outlier rescue in 30-topic space ====
from inspect import signature
from collections.abc import Sequence

# 0) Text column used to train data
TEXT_COL = ("text_lemma" if "text_lemma" in aiusecase
            else "text_clean" if "text_clean" in aiusecase
            else "What is the intended purpose and expected benefits of the AI?")
docs = aiusecase[TEXT_COL].astype(str).tolist()

# 1) Reduce CURRENT model to 30 topics (handles both BERTopic API variants)
base_labels = aiusecase["topic_id"].to_numpy()
sig = signature(topic_model.reduce_topics)

if "topics" in sig.parameters:
    reduced = deepcopy(topic_model).reduce_topics(docs, topics=base_labels, nr_topics=30)
    if isinstance(reduced, Sequence) and len(reduced) == 2:
        topic_model_K30, labels_K30 = reduced
    else:
        topic_model_K30 = reduced
        labels_K30 = None
else:
    topic_model_K30 = deepcopy(topic_model).reduce_topics(docs, nr_topics=30)
    labels_K30 = None

# 2) Always get labels/probabilities from the reduced model to be safe
labels_chk, probs_K30 = topic_model_K30.transform(docs)
if labels_K30 is None:
    labels_K30 = labels_chk
else:
    try:
        np.testing.assert_array_equal(labels_K30, labels_chk)
    except AssertionError:
        labels_K30 = labels_chk

# Coerce to arrays early to avoid list/ndarray type errors later
labels = np.asarray(labels_K30, dtype=int)

# Map probability columns -> true topic IDs (columns ≠ topic IDs)
info30 = topic_model_K30.get_topic_info()  # includes -1 in row 0
topic_id_order_30 = info30.loc[info30["Topic"] != -1, "Topic"].tolist()

# 3) No-harm outlier rescue in 30-topic space
# 3a) Probability-based rescue (conservative)
P = None
if probs_K30 is None:
    print("Warning: model was not fit with calculate_probabilities=True; skipping prob-based rescue.")
else:
    P = np.asarray(probs_K30, dtype=np.float32)
    assert P.shape[1] == len(topic_id_order_30), "Prob matrix width != #non-(-1) topics in reduced model."
    pmax = P.max(axis=1)
    arg  = P.argmax(axis=1)
    assign_ids = np.array(topic_id_order_30, dtype=int)[arg]
    mask_conf = (labels < 0) & (pmax >= 0.30)   # tune 0.30–0.40
    labels[mask_conf] = assign_ids[mask_conf]
    print("After prob rescue (K=30) — outlier rate:", (labels < 0).mean())

# 3b) Centroid-similarity rescue (same embedding space as the reduced model)
try:
    E = topic_model_K30._extract_embeddings(docs)   # best match to training embeddings
except Exception:
    if getattr(topic_model_K30, "embedding_model", None) is None:
        raise ValueError("topic_model_K30 has no embedding_model and _extract_embeddings failed.")
    E = topic_model_K30.embedding_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
E = np.asarray(E, dtype=np.float32)

inlier = labels >= 0
topic_ids_inliers = np.unique(labels[inlier])
if topic_ids_inliers.size > 0:
    centroids = np.vstack([E[labels == t].mean(axis=0) for t in topic_ids_inliers])
    out_idx = np.where(labels < 0)[0]
    if out_idx.size:
        sims   = cosine_similarity(E[out_idx], centroids)
        simmax = sims.max(axis=1)
        assign = topic_ids_inliers[sims.argmax(axis=1)]
        THRESH = 0.35                                     # tune 0.35–0.45
        take   = simmax >= THRESH
        labels[out_idx[take]] = assign[take]

print("After centroid rescue (K=30) — outlier rate:", (labels < 0).mean())

# 4) Commit to 30-topic columns (don’t overwrite micro-topic work)
aiusecase["topic_id_30"]   = labels
aiusecase["is_outlier_30"] = aiusecase["topic_id_30"] < 0

# 5) Build 30-topic features (one-hots + probs) with clear prefixes
oh30 = pd.get_dummies(aiusecase["topic_id_30"].astype(int), prefix="topic30")
aiusecase_features_30 = aiusecase[["Use Case Name","Agency"]].join(oh30)

if P is not None:
    prob_df_30 = pd.DataFrame(P, index=aiusecase.index,
                              columns=[f"topicP30_{tid}" for tid in topic_id_order_30])
    aiusecase_features_30 = aiusecase_features_30.join(prob_df_30)

# 6) Summary
print("Final outlier rate (K=30):", aiusecase["is_outlier_30"].mean())
print("Preserved topics (K=30):",
      aiusecase.loc[aiusecase["topic_id_30"] >= 0, "topic_id_30"].nunique())

# ----- Gather baseline artifacts -----
labels_before = np.asarray(labels_K30, dtype=int)                  # pre-rescue labels
labels_final  = np.asarray(aiusecase["topic_id_30"], dtype=int)    # post-rescue labels
P = None if probs_K30 is None else np.asarray(probs_K30, dtype=np.float32)

# Identify rescue method per doc
rescue = np.full(len(labels_final), "none", dtype=object)
became_topic = (labels_before < 0) & (labels_final >= 0)
rescue[became_topic] = "unknown"  # will refine below if we can detect prob vs centroid

# infer via p_max threshold (approx):
if P is not None:
    pmax = P.max(axis=1)
    rescue[(rescue=="unknown") & (pmax >= 0.30)] = "prob"
    rescue[(rescue=="unknown") & (pmax < 0.30)]  = "centroid"

# ----- Report transitions -----
print("Outlier rate before rescue:", (labels_before < 0).mean())
print("Outlier rate after  rescue:", (labels_final  < 0).mean())
print("Docs rescued total:", int(((labels_before < 0) & (labels_final >= 0)).sum()))
print("Rescue breakdown:",
      pd.Series(rescue[rescue!="none"]).value_counts(dropna=False).to_dict())

# ----- Centroid similarity diagnostics for centroid-rescues -----
try:
    E = topic_model_K30._extract_embeddings(docs)
except Exception:
    E = topic_model_K30.embedding_model.encode(docs, show_progress_bar=False, normalize_embeddings=True)
E = np.asarray(E, dtype=np.float32)

centroid_mask = (rescue == "centroid")
if centroid_mask.any():
    inlier_mask = labels_final >= 0
    topic_ids   = np.unique(labels_final[inlier_mask])
    centroids   = np.vstack([E[labels_final==t].mean(axis=0) for t in topic_ids])
    # map topic_id -> centroid row
    id_to_row = {tid:i for i,tid in enumerate(topic_ids)}
    # compute similarity to assigned centroid
    idxs = np.where(centroid_mask)[0]
    assigned = labels_final[centroid_mask]
    sims = np.array([cosine_similarity(E[i:i+1], centroids[id_to_row[t]:id_to_row[t]+1])[0,0]
                     for i,t in zip(idxs, assigned)])
    print("\nCentroid-rescues — cosine similarity to assigned centroid")
    print(pd.Series(sims).describe(percentiles=[.1,.25,.5,.75,.9]).round(3))

    if P is not None:
        print("\nCentroid-rescues — p_max distribution")
        print(pd.Series(pmax[centroid_mask]).describe(percentiles=[.1,.25,.5,.75,.9]).round(3))

# ----- Which topics gained the most via rescue? -----
gain = pd.DataFrame({
    "topic_id": labels_final[became_topic],
    "method": rescue[became_topic]
}).value_counts().reset_index(name="count").sort_values("count", ascending=False)
print("\nTop topics by rescue inflow:")
print(gain.head(15))

# ----- Topic size table (post-rescue) -----
sizes = pd.Series(labels_final[labels_final>=0]).value_counts().sort_values(ascending=False)
topic_sizes = pd.DataFrame({"topic_id": sizes.index, "size": sizes.values})
print("\nTopic sizes after rescue (top 20):")
print(topic_sizes.head(20))
print("Empty topics (K=30 labels present but size==0):",
      [tid for tid in range(-1, max(topic_sizes.topic_id.max(),0)+1)
       if (tid>=0 and tid not in set(topic_sizes.topic_id))])

# ===== STRONGER GUARDS (dual threshold + avoid tiny topics) =====
# Re-run the centroid step more strictly:
APPLY_STRICT = False
if APPLY_STRICT:
    labels_strict = labels_before.copy()
    # probability gate: allow low p_max but not extremely low
    p_gate = (P.max(axis=1) >= 0.20) if P is not None else np.ones(len(labels_strict), dtype=bool)

    # build centroids from inliers of labels_before (pre-rescue)
    inlier0 = labels_before >= 0
    topic_ids0 = np.unique(labels_before[inlier0])
    centroids0 = np.vstack([E[labels_before==t].mean(axis=0) for t in topic_ids0])
    id_to_row0 = {tid:i for i,tid in enumerate(topic_ids0)}

    out0 = np.where(labels_before < 0)[0]
    if out0.size:
        sims0 = cosine_similarity(E[out0], centroids0)
        simmax0 = sims0.max(axis=1)
        assign0 = topic_ids0[sims0.argmax(axis=1)]

        # do not assign into tiny topics
        sizes0 = pd.Series(labels_before[labels_before>=0]).value_counts()
        big_topic = sizes0[sizes0 >= 6].index.to_numpy() if not sizes0.empty else np.array([], dtype=int)
        keep_topic = np.isin(assign0, big_topic)

        take0 = (simmax0 >= 0.40) & p_gate[out0] & keep_topic
        labels_strict[out0[take0]] = assign0[take0]

    print("\n[STRICT] Outlier rate after dual-gate centroid rescue:", (labels_strict < 0).mean())

"""started with ~30.0% outliers.

Prob-rescue (p_max ≥ 0.30) pulled that down to 22.6% → conservative, as intended.

Centroid-rescue (sim ≥ 0.35) did the heavy lift to 0.56%, with median sim ≈ 0.635 (strong) and a long tail (min 0.384).

Inflow is concentrated (Topic 0 +106, Topic 3 +78, etc.)—that’s expected; those are likely broad themes

We decided to try out LDA it this point since we we'ren't getting good results in reducing outliers. We decided to move forward with LDA instead. But here are some things one could do to further reduce outliers:

Keep the prob-rescue as-is (p_max ≥ 0.30).

Make centroid-rescue slightly stricter and topic-aware:

require sim ≥ 0.50 (cuts roughly the lowest ~10% by percentiles),

require p_max ≥ 0.12 (centroid-rescue median),

avoid assigning into topics that are getting swamped (e.g., inflow > 35% of their pre-rescue size), and

require similarity ≥ that topic’s inlier Q20 (dynamic per-topic threshold).

# **# 2. Topic modeling - LDA**

To validate our findings from BERT, where k=30
"""

# Commented out IPython magic to ensure Python compatibility.
# Fix gensim ←→ NumPy ABI mismatch
# %pip install -U --force-reinstall "numpy==2.0.2" "scipy==1.14.1" "gensim==4.3.3"

import IPython

import platform, numpy as np, gensim
from gensim.models import LdaModel, CoherenceModel
print("Python:", platform.python_version(),
      "| NumPy:", np.__version__,
      "| gensim:", gensim.__version__)

from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

import os
from pathlib import Path

import os
# ---------- CONFIG ----------
TEXT_COL = "text_clean"
BASE     = "Topic Modeling"
OUTDIR   = f"{BASE}/lda_outputs"
CONSOLIDATED_PATH = f"{BASE}/ai_2024_consolidated.csv"

STANDARD_K = 30                          # quick baseline K (to mirror BERTopic K)
K_GRID     = [15, 20, 25, 30, 35, 40]
RANDOM_SEED = 42
PASSES      = 6                          # passes/iterations balance speed/quality
ITERATIONS  = 200
TOPN_WORDS  = 15                         # terms per topic in tables/plots
REP_DOCS_PER_TOPIC = 5                   # representative docs per topic

os.makedirs(OUTDIR, exist_ok=True)

# ---------- 0) INPUTS ----------
assert TEXT_COL in aiusecase.columns, f"{TEXT_COL} not in aiusecase!"
texts_raw = aiusecase[TEXT_COL].fillna("").astype(str).tolist()

from gensim.utils import simple_preprocess

# ---------- 1) TOKENIZE ----------
stopwords = set(ENGLISH_STOP_WORDS) | {
    "ai","use","case","cases","model","models","data","dataset","agency","agencies",
    "system","systems","information","service","services","project","projects",
    "u","s","us","gov","federal"
}
texts = [simple_preprocess(x, deacc=True, min_len=2, max_len=30) for x in texts_raw]
texts = [[w for w in doc if w not in stopwords] for doc in texts]

# dictionary & corpus
id2word = corpora.Dictionary(texts)
id2word.filter_extremes(no_below=5, no_above=0.90, keep_n=50000)
corpus = [id2word.doc2bow(t) for t in texts]

# ---------- 2) STANDARD-K MODEL (K=30) ----------
lda_std = models.LdaModel(
    corpus=corpus, id2word=id2word, num_topics=STANDARD_K,
    passes=PASSES, iterations=ITERATIONS, random_state=RANDOM_SEED,
    alpha="auto", eta="auto", eval_every=None
)
coh_std = CoherenceModel(model=lda_std, texts=texts, dictionary=id2word, coherence='c_v').get_coherence()
perp_std = lda_std.log_perplexity(corpus)
print(f"[STD] K={STANDARD_K}  c_v={coh_std:.4f}  log_perplexity={perp_std:.2f}")
lda_std.save(f"{OUTDIR}/lda_k{STANDARD_K}.model")

# ---------- 3) K SWEEP ----------
k_scores = []
for k in K_GRID:
    lda_k = models.LdaModel(
        corpus=corpus, id2word=id2word, num_topics=k,
        passes=PASSES, iterations=ITERATIONS, random_state=RANDOM_SEED,
        alpha="auto", eta="auto", eval_every=None
    )
    coh_cv = CoherenceModel(model=lda_k, texts=texts, dictionary=id2word, coherence='c_v').get_coherence()
    perp   = lda_k.log_perplexity(corpus)
    k_scores.append({"k": k, "coh_c_v": coh_cv, "log_perplexity": perp})
    print(f"[GRID] K={k:>2}  c_v={coh_cv:.4f}  log_perplexity={perp:.2f}")

scores_df = pd.DataFrame(k_scores).sort_values(["coh_c_v","k"], ascending=[False, True])
scores_path = f"{OUTDIR}/lda_kgrid_scores.csv"
scores_df.to_csv(scores_path, index=False)
best_k = int(scores_df.iloc[0]["k"])
print(f"\n[SELECT] best_k by c_v -> {best_k}")

# plot coherence curve
plt.figure(figsize=(6,4))
plt.plot([d["k"] for d in k_scores], [d["coh_c_v"] for d in k_scores], marker="o")
plt.xlabel("K (num_topics)"); plt.ylabel("c_v coherence"); plt.title("LDA coherence across K")
plt.grid(True, alpha=.3)
cohplot_path = f"{OUTDIR}/lda_coherence_curve.png"
plt.savefig(cohplot_path, bbox_inches="tight"); plt.close()

# === Show LDA performance across K ===

# 1) Collect the K-sweep results
if "scores_df" in globals():
    df = scores_df.copy()
elif "metrics_df" in globals():
     df = metrics_df.copy()
elif "k_scores" in globals():
    df = pd.DataFrame(k_scores)
else:
     raise RuntimeError("No sweep metrics found.")

# 2) Clean/sort and validate
if "k" not in df.columns:
    raise ValueError("Input table 'df' must have a 'k' column for plotting.")
df = df.sort_values("k").reset_index(drop=True)

# 3) Choose best K by c_v
best_metric_col = "coh_c_v" if "coh_c_v" in df.columns else ( "c_v" if "c_v" in df.columns else None )
best_k = int(df.loc[df[best_metric_col].idxmax(), "k"]) if best_metric_col else None

# 4) Helper to plot any available metric column
def plot_metric(df, col, ylabel=None, mark_best=True):
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in the DataFrame. Skipping plot.")
        return
    ks = df["k"].to_numpy()
    ys = df[col].to_numpy()
    plt.figure(figsize=(6,4))
    plt.plot(ks, ys, marker="o")
    if mark_best and best_k is not None:
        yb = float(df.loc[df["k"] == best_k, col].values[0])
        plt.scatter([best_k], [yb], s=80)
        plt.annotate(f"best K={best_k}\n{col}={yb:.3f}" if np.isfinite(yb) else f"best K={best_k}",
                     (best_k, yb), xytext=(6,8), textcoords="offset points")
    plt.xticks(ks)
    plt.xlabel("K (num_topics)")
    plt.ylabel(ylabel or col)
    plt.title(col)
    plt.grid(True, alpha=.3)
    plt.show()

# 5) Plot metrics
for col, ylab in [
    ("coh_c_v",        "c_v (↑ better)"),
    ("coh_c_npmi",     "c_npmi (↑ better)"),
    ("coh_u_mass",     "u_mass (→0 better)"),
    ("log_perplexity", "log perplexity (→0 better)"),
    ("topic_diversity","Topic diversity (↑ better)"),
    ("doc_concentration","Mean max doc-topic prob"),
    ("doc_density_>1pct","Mean #topics/doc with ≥1% prob"),
    ("c_v",            "c_v (↑ better)")
]:
    if col in df.columns:
        plot_metric(df, col, ylab)

# 6) Show table
print(df.round(4))
if best_k is not None:
    print(f"Best K by {best_metric_col}: {best_k}")

# ---------- 4) FINAL LDA @ best_k ----------
lda = models.LdaModel(
    corpus=corpus, id2word=id2word, num_topics=best_k,
    passes=PASSES, iterations=ITERATIONS, random_state=RANDOM_SEED,
    alpha="auto", eta="auto", eval_every=None
)
coh_best = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v').get_coherence()
print(f"[FINAL] K={best_k}  c_v={coh_best:.4f}")
lda_path = f"{OUTDIR}/lda_k{best_k}.model"
lda.save(lda_path)

# ---------- 5) TOPIC TABLES ----------
def topic_terms_table(model, topn=TOPN_WORDS):
    rows=[]
    for t in range(model.num_topics):
        pairs = model.show_topic(t, topn=topn)  # [(word, weight), ...]
        words = [w for w,_ in pairs]
        weights = [wt for _,wt in pairs]
        rows.append({
            "Topic": t,
            "Keywords": ", ".join(words),
            "TopWordsWeighted": "; ".join([f"{w}:{wt:.3f}" for w,wt in pairs]),
        })
    return pd.DataFrame(rows)

topic_terms_df = topic_terms_table(lda, topn=TOPN_WORDS)

# doc-topic gamma
doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
gamma = np.zeros((len(corpus), lda.num_topics))
for i, dt in enumerate(doc_topics):
    for t, p in dt:
        gamma[i, t] = p
dominant = gamma.argmax(axis=1)
dom_prob = gamma.max(axis=1)

# counts per topic
topic_counts = pd.Series(dominant).value_counts().sort_index()
topic_sizes_df = pd.DataFrame({"Topic": topic_counts.index, "Count": topic_counts.values})
topic_sizes_df["Pct"] = (topic_sizes_df["Count"] / len(corpus) * 100).round(2)

# merge sizes ↔ terms
topics_lda_df = topic_sizes_df.merge(topic_terms_df, on="Topic", how="left").sort_values("Count", ascending=False)
topics_csv = f"{OUTDIR}/topics_lda_k{best_k}.csv"
topics_lda_df.to_csv(topics_csv, index=False)

# ---------- 6) REPRESENTATIVE DOCS ----------
rep_rows=[]
use_names = aiusecase["Use Case Name"] if "Use Case Name" in aiusecase.columns else pd.Series([f"doc_{i}" for i in range(len(aiusecase))])
for t in range(lda.num_topics):
    top_ix = np.argsort(-gamma[:, t])[:REP_DOCS_PER_TOPIC]
    for rank, i_doc in enumerate(top_ix, start=1):
        rep_rows.append({
            "Topic": t,
            "Rank": rank,
            "DocIndex": int(i_doc),
            "Use Case Name": str(use_names.iloc[i_doc]),
            "Score": round(float(gamma[i_doc, t]), 4),
            "Snippet": texts_raw[i_doc][:300].replace("\n"," ")
        })
rep_docs_df = pd.DataFrame(rep_rows)
rep_csv = f"{OUTDIR}/rep_docs_lda_k{best_k}.csv"
rep_docs_df.to_csv(rep_csv, index=False)

# -------- 6) REPRESENTATIVE DOCS --------

from IPython.display import display
pd.set_option("display.max_colwidth", 200)

REP_DOCS_PER_TOPIC = 5

# Use the name column if present, else a synthetic label
use_names = aiusecase["Use Case Name"] if "Use Case Name" in aiusecase.columns \
           else pd.Series([f"doc_{i}" for i in range(len(aiusecase))])

# Build gamma (doc-topic probs)
if "gamma" not in globals():
    doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    gamma = np.zeros((len(corpus), lda.num_topics), dtype=float)
    for i, row in enumerate(doc_topics):
        for t, p in row:
            gamma[i, t] = p

# Build the representative-doc table
rep_rows = []
for t in range(lda.num_topics):
    top_ix = np.argsort(-gamma[:, t])[:REP_DOCS_PER_TOPIC]
    for rank, i_doc in enumerate(top_ix, start=1):
        rep_rows.append({
            "Topic": t,
            "Rank": rank,
            "DocIndex": int(i_doc),
            "Use Case Name": str(use_names.iloc[i_doc]),
            "Score": round(float(gamma[i_doc, t]), 4),
            "Snippet": aiusecase[TEXT_COL].iloc[i_doc][:300].replace("\n", " ")
        })
rep_docs_df = pd.DataFrame(rep_rows)

# Show the whole table inline
print(rep_docs_df.head(20))

# highlight top 6 largest topics neatly
if "topics_lda_df" in globals():
    top_topics = topics_lda_df.sort_values("Count", ascending=False).head(6)["Topic"].tolist()
else:
    # simple fallback if topics_lda_df isn't defined
    top_topics = np.argsort((-gamma).max(axis=0))[:6].tolist()

for t in top_topics:
    print(f"\n=== Topic {t} — top {REP_DOCS_PER_TOPIC} docs ===")
    print(rep_docs_df.loc[rep_docs_df["Topic"] == t, ["Rank","Use Case Name","Score","Snippet"]])

# ---------- 7) VISUALIZATIONS ----------

def show_topic_top_terms(model, topic_id, topn=10):
    pairs = model.show_topic(topic_id, topn=topn)     # [(word, weight), ...]
    words = [w for w, _ in pairs][::-1]
    weights = [wt for _, wt in pairs][::-1]
    plt.figure(figsize=(6,4))
    plt.barh(words, weights)
    plt.xlabel("Weight")
    plt.title(f"Top {topn} terms — Topic {topic_id}")
    plt.tight_layout()
    plt.show()

# Display for the 6 biggest topics
if "topics_lda_df" in globals():
    top_topics = topics_lda_df.sort_values("Count", ascending=False).head(6)["Topic"].tolist()
else:
    # fallback: pick the 6 topics with highest total prob mass
    top_topics = np.argsort(gamma.sum(axis=0))[::-1][:6].tolist()

for t in top_topics:
    show_topic_top_terms(lda, t, topn=10)

# === Table: top terms + 1–2 representative comments per topic==
from IPython.display import display

pd.set_option("display.max_colwidth", 220)

TOPN_WORDS = 10     # terms to list per topic
N_REPS     = 2      # number of representative comments per topic
SNIP       = 280    # max chars shown for each snippet

# "Use Case Name"
names = aiusecase["Use Case Name"] if "Use Case Name" in aiusecase.columns \
        else pd.Series([f"doc_{i}" for i in range(len(aiusecase))])

# Build gamma (doc-topic probabilities)
if "gamma" not in globals() or getattr(lda, "num_topics", None) != getattr(gamma, "shape", [None, None])[1]:
    doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    gamma = np.zeros((len(corpus), lda.num_topics), dtype=float)
    for i, row in enumerate(doc_topics):
        for t, p in row:
            gamma[i, t] = p

# For topic sizes: dominant topic per doc
dominant = gamma.argmax(axis=1)

rows = []
for t in range(lda.num_topics):
    # Top terms
    terms = [w for w, _ in lda.show_topic(t, topn=TOPN_WORDS)]
    # Representative docs: pick highest-prob docs for this topic
    top_ix = np.argsort(-gamma[:, t])[:N_REPS]
    reps = []
    for i_doc in top_ix:
        reps.append({
            "name":   str(names.iloc[i_doc]),
            "score":  float(gamma[i_doc, t]),
            "snippet": aiusecase[TEXT_COL].iloc[i_doc][:SNIP].replace("\n", " ")
        })
    # Pad if fewer than N_REPS (edge cases)
    while len(reps) < N_REPS:
        reps.append({"name":"", "score":np.nan, "snippet":""})

    # Assemble row
    row = {
        "Topic": t,
        "Count": int((dominant == t).sum()),
        "TopTerms": ", ".join(terms),
        "Rep1_Name": reps[0]["name"],
        "Rep1_Score": round(reps[0]["score"], 4) if np.isfinite(reps[0]["score"]) else np.nan,
        "Rep1_Snippet": reps[0]["snippet"],
    }
    if N_REPS >= 2:
        row.update({
            "Rep2_Name": reps[1]["name"],
            "Rep2_Score": round(reps[1]["score"], 4) if np.isfinite(reps[1]["score"]) else np.nan,
            "Rep2_Snippet": reps[1]["snippet"],
        })
    rows.append(row)

top_terms_reps = pd.DataFrame(rows).sort_values(["Count","Topic"], ascending=[False, True]).reset_index(drop=True)
print(top_terms_reps)

# === Auto-name LDA topics with multi-word phrases; attach to docs  ===

# 0) Build gamma/dominant
if "gamma" not in globals() or getattr(gamma, "shape", [0,0])[1] != lda.num_topics:
    doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    gamma = np.zeros((len(corpus), lda.num_topics), dtype=float)
    for i, row in enumerate(doc_topics):
        for t, p in row:
            gamma[i, t] = p
    dominant = gamma.argmax(axis=1)

# 1) Get top terms per topic
def topic_top_terms(model, topn=12):
    out = {}
    for t in range(model.num_topics):
        out[t] = [w for w,_ in model.show_topic(t, topn=topn)]
    return out

top_terms = topic_top_terms(lda, topn=12)

# 2) Build data per topic to mine phrases from dominant docs
texts_raw = aiusecase[TEXT_COL].fillna("").astype(str).tolist()

# Stopwords
extra_stop = {
    "ai","use","case","cases","model","models","data","dataset","system","systems",
    "project","projects","service","services","information","agency","agencies",
    "u","s","us","gov","federal","provide","provided","using","use","based"
}
stop_words = set(ENGLISH_STOP_WORDS) | extra_stop

def best_ngram_name(docs, ngram_range=(2,3), min_df=2, max_df=0.8, topk=1):
    if len(docs) == 0:
        return []
    vec = CountVectorizer(stop_words=stop_words,
                          ngram_range=ngram_range,
                          min_df=min_df, max_df=max_df)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return []
    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = freqs.argsort()[::-1]
    phrases = terms[order][:topk].tolist()
    return phrases

def nice_title(text):
    # Title-case while preserving common acronyms
    keep_upper = {"AI","LLM","NLP","FOIA","FAA","IRS","VA","ML","UAS","GIS","EE","BE"}
    titled = " ".join([w.upper() if w.upper() in keep_upper else w.capitalize() for w in text.split()])
    return titled

topic_name_map = {}
topic_phrase_map = {}

for t in range(lda.num_topics):
    idx = np.where(dominant == t)[0]
    docs_t = [texts_raw[i] for i in idx]
    phrases = best_ngram_name(docs_t, ngram_range=(3,3), topk=2)  # try trigrams first
    if not phrases:
        phrases = best_ngram_name(docs_t, ngram_range=(2,3), topk=2)  # then bi/tri
    # Fallback to top LDA terms if no phrases
    if not phrases:
        fallback = " ".join(top_terms[t][:3])  # 3 most salient words
        phrases = [fallback]
    # Build a readable name
    topic_name = nice_title(phrases[0])
    topic_name_map[t] = topic_name
    topic_phrase_map[t] = [nice_title(p) for p in phrases]

# 3) manual overrides
manual_overrides = {}
topic_name_map.update(manual_overrides)

# 4) Preview names
names_df = pd.DataFrame({
    "Topic": list(topic_name_map.keys()),
    "Topic_Name": [topic_name_map[t] for t in topic_name_map],
    "Also_Consider": [", ".join(topic_phrase_map[t]) for t in topic_phrase_map],
    "Doc_Count": [int((dominant==t).sum()) for t in topic_name_map]
}).sort_values(["Doc_Count","Topic"], ascending=[False, True]).reset_index(drop=True)

from IPython.display import display
print(names_df.head(20))

# 5) Attach names to per-doc table
if "aiusecase_lda" not in globals():
    aiusecase_lda = aiusecase.copy()
    aiusecase_lda["lda_topic"] = dominant
    aiusecase_lda["lda_topic_prob"] = gamma.max(axis=1)

aiusecase_lda["lda_topic_name"] = aiusecase_lda["lda_topic"].map(topic_name_map)
print(aiusecase_lda[["Use Case Name","lda_topic","lda_topic_name","lda_topic_prob"]].head(10))

# === Better LDA topic names via class-based TF-IDF (phrases) — FIXED STOPWORDS + DE-DUP ===

pd.set_option("display.max_colwidth", 220)

# Build gamma/dominant if missing or mismatched
if "gamma" not in globals() or getattr(gamma, "shape", (0,0))[1] != lda.num_topics:
    doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    gamma = np.zeros((len(corpus), lda.num_topics), dtype=float)
    for i, row in enumerate(doc_topics):
        for t, p in row:
            gamma[i, t] = p
    dominant = gamma.argmax(axis=1)

DOCS = aiusecase[TEXT_COL].fillna("").astype(str).tolist()
n_docs = len(DOCS)

# ----- STOPWORDS -----
domain_stop = {
    "ai","use","uses","using","used","case","cases","model","models","data","dataset",
    "system","systems","service","services","project","projects","application",
    "information","process","processing","tool","tools","based","approach","development",
    "research","analysis","analytics","text","document","documents","output","outputs",
    "time","user","users","staff","public","internal","external","help","support","provide",
    "provided","including","include","across","within","related","various","different"
}
stop_words = list(set(ENGLISH_STOP_WORDS) | domain_stop)   # ← FIX: list(), not set

# === NEXT STEP: Topic summary + attach names + features ===

pd.set_option("display.max_colwidth", 220)

TOPN_WORDS = 10   # top terms per topic in the table
N_REPS     = 2    # number of representative docs/comments to show

# 0) Ensure gamma/dominant exist
if "gamma" not in globals() or getattr(gamma, "shape", (0,0))[1] != lda.num_topics:
    doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    gamma = np.zeros((len(corpus), lda.num_topics), dtype=float)
    for i, row in enumerate(doc_topics):
        for t, p in row:
            gamma[i, t] = p
    dominant = gamma.argmax(axis=1)

# 1) Build the summary table (names from topic_name_map)
def topic_terms(model, t, topn=TOPN_WORDS):
    return ", ".join([w for w, _ in model.show_topic(t, topn=topn)])

names_series = aiusecase["Use Case Name"] if "Use Case Name" in aiusecase.columns \
               else pd.Series([f"doc_{i}" for i in range(len(aiusecase))])

rows = []
for t in range(lda.num_topics):
    # representative docs for topic t
    top_ix = np.argsort(-gamma[:, t])[:N_REPS]
    reps = []
    for i_doc in top_ix:
        reps.append({
            "name":   str(names_series.iloc[i_doc]),
            "score":  float(gamma[i_doc, t]),
            "snippet": aiusecase[TEXT_COL].iloc[i_doc][:280].replace("\n", " ")
        })

    row = {
        "Topic": t,
        "Topic_Name": topic_name_map.get(t, f"Topic {t}"),
        "Doc_Count": int((dominant == t).sum()),
        "TopTerms": topic_terms(lda, t, TOPN_WORDS),
    }
    # add 1–2 representative comments
    for k, rep in enumerate(reps, start=1):
        row[f"Rep{k}_Name"] = rep["name"]
        row[f"Rep{k}_Score"] = round(rep["score"], 4)
        row[f"Rep{k}_Snippet"] = rep["snippet"]
    rows.append(row)

summary_df = pd.DataFrame(rows).sort_values(["Doc_Count", "Topic"], ascending=[False, True]).reset_index(drop=True)
print(summary_df.head(20))  # show first 20; remove .head(...) to see all

# 2) Attach names to per-doc table (for EDA)
if "aiusecase_lda" not in globals():
    aiusecase_lda = aiusecase.copy()
    aiusecase_lda["lda_topic"] = dominant
    aiusecase_lda["lda_topic_prob"] = gamma.max(axis=1)

aiusecase_lda["lda_topic_name"] = aiusecase_lda["lda_topic"].map(topic_name_map)
print(aiusecase_lda[["Use Case Name", "lda_topic", "lda_topic_name", "lda_topic_prob"]].head(10))

# 3) Create LDA features for modeling/EDA
prob_cols = [f"lda_t{t}" for t in range(lda.num_topics)]
prob_df = pd.DataFrame(gamma, columns=prob_cols)
dom_ohe = pd.get_dummies(aiusecase_lda["lda_topic"], prefix="lda_dom_t")

id_col_df = (aiusecase[["Use Case Name"]]
             if "Use Case Name" in aiusecase.columns
             else pd.DataFrame({"DocIndex": np.arange(len(aiusecase))}))
features_df = pd.concat([id_col_df.reset_index(drop=True), prob_df.reset_index(drop=True), dom_ohe.reset_index(drop=True)], axis=1)

print(features_df.head(10))

#Mvrsquared:
from gensim import matutils

# 1. DTM (document-term matrix)
dtm = matutils.corpus2csc(corpus, num_terms=len(id2word)).T  # docs × terms

# 2. θ (doc-topic matrix)
theta = np.array([lda.get_document_topics(doc, minimum_probability=0) for doc in corpus])
theta = np.vstack([np.array([p for _, p in row]) for row in theta])  # docs × K

# 3. φ (topic-word matrix)
phi = lda.get_topics()  # K × terms

# confirm dimensions
print(dtm.shape, theta.shape, phi.shape)

# === LDA performance metrics for model ===
from gensim.models import CoherenceModel

# lda, corpus, id2word, texts  (+ optionally gamma)
def eval_lda(lda, corpus, id2word, texts, gamma=None, topn_div=20):
    # Build gamma if not provided
    if gamma is None:
        dt = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
        k = lda.num_topics
        gamma = np.zeros((len(corpus), k), dtype=float)
        for i, row in enumerate(dt):
            for t, p in row:
                gamma[i, t] = p

    # Coherence & perplexity
    c_v     = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence="c_v").get_coherence()
    c_npmi  = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence="c_npmi").get_coherence()
    u_mass  = CoherenceModel(model=lda, corpus=corpus, dictionary=id2word, coherence="u_mass").get_coherence()
    log_ppx = float(lda.log_perplexity(corpus))

    # Topic diversity (overlap of top words)
    words = []
    for t in range(lda.num_topics):
        words += [w for w, _ in lda.show_topic(t, topn=topn_div)]
    topic_div = len(set(words)) / (lda.num_topics * topn_div)

    # Doc concentration / density
    doc_concentration = float(gamma.max(axis=1).mean())          # mean dominant prob per doc
    doc_density       = float((gamma >= 0.01).sum(axis=1).mean())# avg #topics/doc >=1%

    # Topic separation (avg pairwise JSD across topic-word distributions)
    beta = lda.get_topics()  # shape (K, V)
    eps = 1e-12
    def jsd(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (np.sum(p * np.log((p + eps) / (m + eps)))) + \
               0.5 * (np.sum(q * np.log((q + eps) / (m + eps))))
    dists = []
    for i in range(beta.shape[0]):
        for j in range(i+1, beta.shape[0]):
            dists.append(jsd(beta[i], beta[j]))
    topic_sep_jsd = float(np.mean(dists)) if dists else np.nan

    return {
        "coh_c_v": c_v,
        "coh_c_npmi": c_npmi,
        "coh_u_mass": u_mass,           # → 0 (less negative) is better
        "log_perplexity": log_ppx,      # → 0 (less negative) is better
        "topic_diversity": topic_div,   # ↑ better
        "doc_concentration": doc_concentration,
        "doc_density_>1pct": doc_density,
        "topic_separation_jsd": topic_sep_jsd # ↑ better
    }

metrics_now = eval_lda(lda, corpus, id2word, texts, gamma if "gamma" in globals() else None)
print(pd.Series(metrics_now).round(4))

import numpy as np
import scipy.sparse as sp

def mv_r2(Y, Yhat):
    if sp.issparse(Y):
        ybar = np.array(Y.mean(axis=0)).ravel()
        E = Y - Yhat if sp.issparse(Yhat) else Y - sp.csr_matrix(Yhat)
        sse = (E.multiply(E)).sum()
        Ybar = sp.csr_matrix(np.tile(ybar, (Y.shape[0], 1)))
        sst = ((Y - Ybar).multiply(Y - Ybar)).sum()
    else:
        ybar = Y.mean(axis=0, keepdims=True)
        sse = np.square(Y - Yhat).sum()
        sst = np.square(Y - ybar).sum()
    return 1 - sse / sst

# normalize to proportions
row_sums = np.asarray(dtm.sum(axis=1)).ravel()
inv_rows = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums>0)
Y_prob = sp.diags(inv_rows).dot(dtm)        # docs × terms (probabilities)
Yhat = theta.dot(phi)                       # reconstruction
r2_prob = mv_r2(Y_prob, Yhat)
print("mvR² (probabilities):", r2_prob)

import scipy.sparse as sp

def mv_r2(Y, Yhat):
    if sp.issparse(Y):
        ybar = np.array(Y.mean(axis=0)).ravel()
        sse = (Y - sp.csr_matrix(Yhat)).multiply(Y - sp.csr_matrix(Yhat)).sum()
        sst = (Y - sp.csr_matrix(np.tile(ybar, (Y.shape[0], 1)))).multiply(
              Y - sp.csr_matrix(np.tile(ybar, (Y.shape[0], 1)))).sum()
    else:
        ybar = Y.mean(axis=0, keepdims=True)
        sse = np.square(Y - Yhat).sum()
        sst = np.square(Y - ybar).sum()
    return 1 - sse/sst if sst > 0 else np.nan

def corpus_to_csr(corpus, id2word):
    from gensim import matutils
    return matutils.corpus2csc(corpus, num_terms=len(id2word)).T  # docs×terms

def lda_k_sweep(k_list, corpus, id2word, texts, passes=10, iterations=400, seed=42):
    Y_csr = corpus_to_csr(corpus, id2word)
    # probability (row-normalized) target for mvR²
    row_sums = np.asarray(Y_csr.sum(axis=1)).ravel()
    inv_rows = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums>0)
    Y_prob = sp.diags(inv_rows).dot(Y_csr)

    rows = []
    for k in k_list:
        lda_k = models.LdaModel(
            corpus=corpus, id2word=id2word, num_topics=k,
            passes=passes, iterations=iterations, random_state=seed,
            alpha="auto", eta="auto", eval_every=None
        )
        # coherence (c_v)
        coh = CoherenceModel(model=lda_k, texts=texts, dictionary=id2word, coherence='c_v').get_coherence()
        # reconstruction
        theta = np.vstack([
            np.array([p for _, p in lda_k.get_document_topics(doc, minimum_probability=0)])
            for doc in corpus
        ])                        # docs×k
        phi = lda_k.get_topics()  # k×terms
        r2 = mv_r2(Y_prob, theta.dot(phi))
        rows.append({"k": k, "coh_c_v": coh, "mvR2_prob": r2})
    return pd.DataFrame(rows).sort_values("k")


k_grid = [15, 20, 25, 30, 35, 40]
results = lda_k_sweep(k_grid, corpus, id2word, texts)
print(results)

# === mvR² (proportions) across K ===

# 1) If mvR2 column is missing, compute it for each K in df["k"]
if "mvR2_prob" not in df.columns:
    # Build Y (docs×terms) once from existing corpus/id2word
    Y_csr = matutils.corpus2csc(corpus, num_terms=len(id2word)).T  # docs×terms

    # Row-normalize to probabilities
    row_sums = np.asarray(Y_csr.sum(axis=1)).ravel()
    inv_rows = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums > 0)
    Y_prob = sp.diags(inv_rows).dot(Y_csr)

    def mv_r2(Y, Yhat):
        # Frobenius 1 - SSE/SST (works with sparse Y and dense Yhat)
        ybar = np.array(Y.mean(axis=0)).ravel()
        Yhat_csr = sp.csr_matrix(Yhat) if not sp.issparse(Yhat) else Yhat
        E = Y - Yhat_csr
        sse = (E.multiply(E)).sum()
        Ybar = sp.csr_matrix(np.tile(ybar, (Y.shape[0], 1)))
        T = Y - Ybar
        sst = (T.multiply(T)).sum()
        return 1 - (sse / sst if sst > 0 else np.nan)

    mv_list = []
    for k in df["k"].astype(int).tolist():
        lda_k = models.LdaModel(
            corpus=corpus, id2word=id2word, num_topics=k,
            passes=PASSES, iterations=ITERATIONS, random_state=RANDOM_SEED,
            alpha="auto", eta="auto", eval_every=None
        )
        # Theta (docs×K)
        theta = np.vstack([
            np.array([p for _, p in lda_k.get_document_topics(doc, minimum_probability=0)])
            for doc in corpus
        ])
        # Phi (K×terms)
        phi = lda_k.get_topics()
        # Reconstruction and mvR²
        Yhat = theta.dot(phi)
        mv_list.append(mv_r2(Y_prob, Yhat))

    df["mvR2_prob"] = mv_list

# 2) Plot mvR²
plot_metric(df, "mvR2_prob", ylabel="mvR² (proportions, ↑ better)", mark_best=False)

# 3) Also print the best K by mvR² for reference
best_k_mv = int(df.loc[df["mvR2_prob"].idxmax(), "k"])
best_mv = float(df.loc[df["mvR2_prob"].idxmax(), "mvR2_prob"])
print(f"Best K by mvR2_prob: {best_k_mv}  (mvR²={best_mv:.4f})")

#With this assesment, we validated that our best k = 40.

# Overall NPMI (averaged across topics)
cm = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_npmi')
npmi_overall = cm.get_coherence()
print("NPMI (overall):", npmi_overall)

# Per-topic NPMI
npmi_per_topic = cm.get_coherence_per_topic()
for t, score in enumerate(npmi_per_topic):
    print(f"Topic {t}: NPMI={score:.4f}")

# === NPMI (c_npmi) across K, plus plot ===

# 1) If c_npmi isn't already in df, compute it for each K
if "coh_c_npmi" not in df.columns:
    npmi_overall = []
    npmi_min = []      # weakest topic per model
    npmi_median = []   # median topic quality per model

    for k in df["k"].astype(int).tolist():
        lda_k = models.LdaModel(
            corpus=corpus, id2word=id2word, num_topics=k,
            passes=PASSES, iterations=ITERATIONS, random_state=RANDOM_SEED,
            alpha="auto", eta="auto", eval_every=None
        )
        cm = CoherenceModel(model=lda_k, texts=texts, dictionary=id2word,
                            coherence='c_npmi')  # window_size can be set (default=10)
        overall = cm.get_coherence()
        per_topic = cm.get_coherence_per_topic()

        npmi_overall.append(overall)
        npmi_min.append(float(np.min(per_topic)))
        npmi_median.append(float(np.median(per_topic)))

    df["coh_c_npmi"]   = npmi_overall
    df["npmi_min"]     = npmi_min
    df["npmi_median"]  = npmi_median

# 2) Plot overall NPMI
plot_metric(df, "coh_c_npmi", ylabel="c_npmi (↑ better)", mark_best=False)

# 3) Plot median and min per-topic NPMI to see stability by K
if "npmi_median" in df.columns:
    plot_metric(df, "npmi_median", ylabel="median per-topic NPMI (↑ better)", mark_best=False)
if "npmi_min" in df.columns:
    plot_metric(df, "npmi_min", ylabel="min per-topic NPMI (↑ better)", mark_best=False)

# 4) Show table and print the best K by c_npmi
print(df.round(4))
best_k_npmi = int(df.loc[df["coh_c_npmi"].idxmax(), "k"])
print(f"Best K by c_npmi: {best_k_npmi}")

# ---------- 8) DOC-LEVEL ASSIGNMENTS — with names ----------

# Build gamma/dominant/dom_prob
doc_topics = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
K = lda.num_topics
gamma = np.zeros((len(corpus), K), dtype=float)
for i, row in enumerate(doc_topics):
    for t, p in row:
        gamma[i, t] = p
dominant = gamma.argmax(axis=1)
dom_prob = gamma.max(axis=1)

# Wide doc-level summary
assign_df = pd.DataFrame({
    "DocIndex": np.arange(len(corpus)),
    "lda_topic": dominant,
    "lda_topic_prob": dom_prob
})
# Add topic name. If missing, fall back to "Topic {id}"
assign_df["lda_topic_name"] = assign_df["lda_topic"].map(
    (topic_name_map if "topic_name_map" in globals() else {})
).fillna(assign_df["lda_topic"].map(lambda x: f"Topic {x}"))

# Long doc-topic table
assign_long_rows = []
for i in range(len(corpus)):
    for t in range(K):
        assign_long_rows.append({
            "DocIndex": i,
            "Topic": t,
            "Topic_Name": (topic_name_map[t] if "topic_name_map" in globals() and t in topic_name_map else f"Topic {t}"),
            "Prob": gamma[i, t]
        })
assign_long_df = pd.DataFrame(assign_long_rows)

# Attach to dataset for EDA
aiusecase_lda = aiusecase.copy()
aiusecase_lda["lda_topic"] = dominant
aiusecase_lda["lda_topic_prob"] = dom_prob
aiusecase_lda["lda_topic_name"] = aiusecase_lda["lda_topic"].map(
    (topic_name_map if "topic_name_map" in globals() else {})
).fillna(aiusecase_lda["lda_topic"].map(lambda x: f"Topic {x}"))

# Quick peek inline
from IPython.display import display
print(assign_df.head(10))
print(assign_long_df.head(10))
print(aiusecase_lda[["Use Case Name","lda_topic","lda_topic_name","lda_topic_prob"]].head(10))

# === LDA "outliers": weak/ambiguous docs + topic clarity  ===

# 0) Ensure gamma / dom labels exist
if "gamma" not in globals() or getattr(gamma, "shape", (0,0))[1] != lda.num_topics:
    dt = [lda.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    K = lda.num_topics
    gamma = np.zeros((len(corpus), K), dtype=float)
    for i, row in enumerate(dt):
        for t, p in row:
            gamma[i, t] = p
dominant = gamma.argmax(axis=1)
dom_prob = gamma.max(axis=1)

# 1) Assignment-strength bins (how many topics get a doc "strongly")
bins = pd.cut(dom_prob, [0, 0.20, 0.35, 0.50, 1.01],
              labels=["<0.20", "0.20–0.35", "0.35–0.50", "≥0.50"], right=False)
bin_counts = bins.value_counts().sort_index()
bin_table = pd.DataFrame({"count": bin_counts, "pct": (bin_counts / len(dom_prob) * 100).round(2)})
print("Dominant-topic probability bins:")
print(bin_table)

# 2) Entropy of each doc's topic distribution (normalized 0..1; higher = more diffuse)
eps = 1e-12
K = lda.num_topics
entropy = -(gamma * np.log(gamma + eps)).sum(axis=1) / np.log(K)
entile = pd.qcut(entropy, 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
summary_entropy = entile.value_counts().sort_index()
print("Entropy quartiles (Q4 = most diffuse):")
print(summary_entropy.to_frame("docs"))

# Mark weak/ambiguous docs
WEAK_THR = 0.20
HIGH_ENT_THR = 0.70
weak_mask = dom_prob < WEAK_THR
highent_mask = entropy >= HIGH_ENT_THR
ambig_mask = weak_mask | highent_mask

print(f"Weak (dom_prob < {WEAK_THR:.2f}): {weak_mask.sum()} / {len(dom_prob)} "
      f"({100*weak_mask.mean():.1f}%)")
print(f"High-entropy (≥ {HIGH_ENT_THR:.2f}): {highent_mask.sum()} "
      f"({100*highent_mask.mean():.1f}%)")
print(f"Ambiguous (weak or high-entropy): {ambig_mask.sum()} "
      f"({100*ambig_mask.mean():.1f}%)")

# 3) Show a few ambiguous docs
name_col = "Use Case Name" if "Use Case Name" in aiusecase.columns else None
text_col = TEXT_COL
sample = np.where(ambig_mask)[0][:10]
rows = []
for i in sample:
    rows.append({
        "DocIndex": int(i),
        "Name": aiusecase[name_col].iloc[i] if name_col else f"doc_{i}",
        "dom_topic": int(dominant[i]),
        "dom_prob": round(float(dom_prob[i]), 4),
        "entropy": round(float(entropy[i]), 3),
        "snippet": aiusecase[text_col].iloc[i][:220].replace("\n"," ")
    })
print("\nSample ambiguous docs:")
print(pd.DataFrame(rows))

# 4) Topic-level clarity: within-topic median confidence and low-confidence rate
if "topic_name_map" in globals():
    tnames = {t: topic_name_map.get(t, f"Topic {t}") for t in range(K)}
else:
    tnames = {t: f"Topic {t}" for t in range(K)}

topic_rows = []
for t in range(K):
    idx = np.where(dominant == t)[0]
    if len(idx) == 0:
        med = np.nan; lowpct = np.nan
    else:
        probs_t = dom_prob[idx]
        med = float(np.median(probs_t))
        lowpct = float((probs_t < 0.35).mean() * 100.0)  # share of docs < 0.35
    topic_rows.append({
        "Topic": t,
        "Topic_Name": tnames[t],
        "Doc_Count": int(len(idx)),
        "Median_dom_prob": round(med, 3) if not np.isnan(med) else np.nan,
        "LowConf_%(<0.35)": round(lowpct, 1) if not np.isnan(lowpct) else np.nan
    })

topic_clarity = pd.DataFrame(topic_rows).sort_values(["LowConf_%(<0.35)", "Doc_Count"],
                                                     ascending=[False, False])
print("\nTopics with most low-confidence assignments:")
print(topic_clarity.head(10))

# ---------- 9) FEATURES: topic probabilities + dominant-topic one-hots ----------
prob_cols = [f"lda_t{t}" for t in range(lda.num_topics)]
prob_df = pd.DataFrame(gamma, columns=prob_cols)
dom_ohe = pd.get_dummies(dominant, prefix="lda_dom_t")
features_df = pd.concat([aiusecase[["Use Case Name"]] if "Use Case Name" in aiusecase.columns else pd.DataFrame({"DocIndex": np.arange(len(corpus))}),
                         prob_df, dom_ohe],
                        axis=1)

assign_csv = f"{OUTDIR}/doc_topics_lda_k{best_k}.csv"
features_csv = f"{OUTDIR}/lda_features_k{best_k}.csv"
aiusecase_lda_csv = f"{OUTDIR}/aiusecase_with_lda_k{best_k}.csv"
assign_long_csv = f"{OUTDIR}/doc_topic_long_lda_k{best_k}.csv"

assign_df.to_csv(assign_csv, index=False)
features_df.to_csv(features_csv, index=False)
aiusecase_lda.to_csv(aiusecase_lda_csv, index=False)
assign_long_df.to_csv(assign_long_csv, index=False)

# ---------- 10) MERGE WITH 2024 CONSOLIDATED ----------
if os.path.exists(CONSOLIDATED_PATH):
    cons = pd.read_csv(CONSOLIDATED_PATH)
    key = "Use Case Name" if "Use Case Name" in cons.columns and "Use Case Name" in features_df.columns else ("DocIndex" if "DocIndex" in features_df.columns else None)
    if key is None:
        print("Could not find a common join key; skipping merge.")
        merged = None
    else:
        merged = cons.merge(features_df, on=key, how="left")
        merged_path = f"{OUTDIR}/ai_2024_consolidated_with_lda_k{best_k}.csv"
        merged.to_csv(merged_path, index=False)
        print(f"[MERGED] → {merged_path}  (shape={merged.shape})")
else:
    merged = None
    print(f" Consolidated dataset not found at {CONSOLIDATED_PATH}; skipped merge.")

# ---------- 11) SUMMARY ----------
print("\nSaved:")
print(" - Coherence curve:", cohplot_path)
print(" - LDA model:", lda_path)
print(" - Topic table:", topics_csv)
print(" - Representative docs:", rep_csv)
print(" - Doc assignments (wide):", assign_csv)
print(" - Doc-topic long:", assign_long_csv)
print(" - AI use case + LDA columns:", aiusecase_lda_csv)
print(" - LDA features:", features_csv)
if merged is not None:
    print(f" - Consolidated + LDA features: {merged_path}")
print(f"\nBest K by c_v: {best_k} (c_v≈{coh_best:.4f})")