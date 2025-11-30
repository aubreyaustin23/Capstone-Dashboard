# import all necessary packages
# essentials --
import numpy as np
import pandas as pd
import re
from pathlib import Path

# modeling --
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, classification_report, 
    precision_recall_curve, average_precision_score
)
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# visualizations --
import matplotlib.pyplot as plt
import plotly.express as px

# dash app --
import dash
from dash import dcc, html, Input, Output, State
import dash_table

# adjust display settings
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

# set file path
DATA_PATH = Path('combined_data_final.xlsx')

# exact feature and target names
FEATURE_COLS = [f'lda_t{str(i).zfill(2)}' for i in range(40)]
M1_TARGET = 'rights_impacting_bin'
M2_TARGET = 'complexity_score'
M3_TARGET = 'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?' 


# load dataset
df = pd.read_excel(DATA_PATH)
print(df.shape)
df.head(2)

# add fields for use in dashboard
df["Use Case Status"] = np.where(
    df["Stage of Development"] == "Retired",
    "Retired",
    "Active"
)

def normalize_resourcing(val):
    if not isinstance(val, str) or val.strip() == "":
        return "Unknown"    
    s = val.strip().lower()
    # Contracting
    if "contract" in s or "external" in s:
        if "in-house" in s or "in house" in s or "combination" in s or "both" in s:
            return "Both"
        return "Contracting Resources"
    # In-house only
    if "in-house" in s or "in house" in s:
        return "In-House"
    # Neither
    if "no contract" in s or "neither" in s:
        return "Neither"
    # Unknown
    if (
        "not reported" in s
        or "n/a" in s
        or "na" == s
        or "data not reported" in s
    ):
        return "Unknown"
    # Fallback
    return "Unknown"

df["clean_resourcing"] = df["Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house?"] \
                            .apply(normalize_resourcing)

def normalize_pii(val):
    if not isinstance(val, str) or val.strip() == "":
        return "Unknown"  
    s = val.strip().lower()
    # Yes
    if s in ["yes", "y", "true"]:
        return "Yes"
    # No
    if s in ["no", "n", "false"]:
        return "No"
    # Unknown (includes variations)
    if (
        "n/a" in s
        or "not reported" in s
        or "na" == s
        or "unknown" in s
    ):
        return "Unknown"
    # Everything else defaults to Unknown just to be safe
    return "Unknown"

df["clean_pii"] = df[
    "Does this AI use case involve personally identifiable information (PII) that is maintained by the agency?"
].apply(normalize_pii)

def normalize_demographics(val):
    # Treat blanks, None, or non-strings as No
    if not isinstance(val, str) or val.strip() == "":
        return "No"
    s = val.strip().lower()
    # Values that mean explicitly no demographic features
    no_values = [
        "none",
        "n/a",
        "na",
        "not applicable",
        "no demographic variables",
    ]
    if s in no_values:
        return "No"
    # Otherwise, if there's *anything* else reported →
    # We interpret as Yes
    return "Yes"

df["clean_demographics"] = df[
    "Which, if any, demographic variables does the AI use case explicitly use as model features?"
].apply(normalize_demographics)

CLEAN_TOPIC_AREAS = {
    "law & justice": "Law & Justice",
    "mission-enabling": "Mission-Enabling",
    "mission-enabling (internal agency support)": "Mission-Enabling",
    "mission-enabling (internal agency support) ": "Mission-Enabling",
    "mission-enabling ": "Mission-Enabling",
    "government services (includes benefits and service delivery)": "Government Services",
    "health & medical": "Health & Medical",
    "diplomacy & trade": "Diplomacy & Trade",
    "education & workforce": "Education & Workforce",
    "emergency management": "Emergency Management",
    "transportation": "Transportation",
    "ai used in transportation operations": "Transportation",
    "internal dot research project": "Transportation",
    "dot sponsored external research": "Transportation",
    "other": "Other",
    "other; mission-enabling (internal agency support)": "Other",
    "natural language processing": "Natural Language Processing",
    "nlp": "Natural Language Processing",
    "deep learning": "Deep Learning",
    "statistical methods": "Statistical Methods",
    "classification": "Classification",
    "aiml platform/environment": "AIML Platform/Environment",
    "administration of ai governance, processes, and procedures": "AI Governance",
    "department-level ai capabilities and capacity": "AI Capability Development",
}

def clean_topic_area(value):
    if pd.isna(value):
        return "Other"
    
    # Normalize casing and remove whitespace
    raw = str(value).strip().lower()

    # Some entries have multiple categories separated by semicolon
    parts = [p.strip() for p in raw.split(';')]

    cleaned = []
    for p in parts:
        if p in CLEAN_TOPIC_AREAS:
            cleaned.append(CLEAN_TOPIC_AREAS[p])
        else:
            # Unknown category — assign to Other
            cleaned.append("Other")

    # If multiple cleaned categories appear, pick the FIRST as primary
    return cleaned[0]

df["clean_use_case_topic_area"] = df["Use Case Topic Area"].apply(clean_topic_area)

# logic for technical complexity score used in M2

# utilizes:
    # LDA primary topic (1-5 based on technical complexity)
    # purpose/benefit text (1-5 based on keyword signals)
        # level 1 - basic/non-AI (simple automation, static data entry)
        # level 2 - moderate (basic ML, predictive modeling, structured analytics)
        # level 3 - advanced (complex ML pipelines, multi-modal analytics)
        # level 4 - high (specialized deep learning, domain specific AI)
        # level 5 - cutting edge/emerging (foundation-model, gen-AI, autonomous decision-making)
    # government provided use case topic area 

# topic model baseline complexity
TOPIC_COMPLEXITY = {
    0: 3, # general ML analysis & optimization
    1: 5, # air traffic & national safety analytics
    2: 2, # research program development & expertise
    3: 3, # software, content, and text-based research applications
    4: 3, # image & media creation and editing
    5: 4, # risk & issue detection / classification
    6: 2, # regulatory notices & text-based transcription processes
    7: 3, # decision intelligence & automation insights
    8: 3, # enterprise document & collaboration capability
    9: 3, # compliance, security oversight & request tracking
    10: 3, # financial document, copilot & Q&A Assistance
    11: 2, # research infrastructure & certification processes 
    12: 2, # technical training & workforce operational support
    13: 4, # aircraft labeling, testing, & quality monitoring
    14: 5, # healthcare patient risk & predictive support
    15: 3, # code generation & generative text assistance
    16: 3, # event reporting & narrative classification
    17: 4, # FAA generative deployment & tools adoption
    18: 3, # cost saving, policy & redaction process topics
    19: 4, # standards-based multi-modal performance & screening
    20: 3, # enterprise NLP, feedback & dashboard analytics
    21: 2, # agency operational coordination & acquisition processes
    22: 3, # supervisory risk assessments & document support
    23: 3, # LLM search, knowledge QA & library retrieval
    24: 2, # human response review & regulatory text processing
    25: 3, # management plans, compliance & AI-driven maintenance
    26: 3, # event recognition & automated communications processing
    27: 4, # FOIA request routing & decision classification
    28: 2, # public generative text retrieval & extraction tools
    29: 4, # aircraft safety, legal, and joint training support
    30: 3, # procurement impact analysis & scenario modeling
    31: 3, # operational analytics & collaboration efficiency
    32: 5, # cybersecurity accuracy, audits, & response tools
    33: 3, # policy tracking, document sections & predictive drafting
    34: 4, # machine learning prediction & data updates
    35: 4, # advanced ML matching & structured accuracy systems
    36: 4, # technology trend prediction & edge integration
    37: 5, # neural translation & anomaly detection science
    38: 4, # medical device writing assistance & public documentation
    39: 3  # synthetic testing, industry data & speech supervision
}

COMPLEXITY_KEYWORDS = {
    1: [
        r"\bform\b", r"\bdashboard\b", r"report(ing)?", r"faq\b",
        r"caption", r"\bemail\b", r"data entry", r"voicebot"
    ],
    2: [
        r"classif(y|ier)", r"lookup", r"keyword", r"translation",
        r"text extraction", r"matching", r"search", r"sentiment",
        r"autocoder", r"prediction", r"forecast", r"regression", r"risk model",
        r"topic model", r"clustering", r"entity (resolution|linkage)",
        r"normalization", r"data integration", r"pattern detection"
    ],
    3: [
        r"simulation", r"synthetic data", r"anomaly detection",
        r"timeseries", r"fuzzy matching", r"phenotyping",
        r"network analysis", r"data fusion"
    ],
    4: [
        r"deep learning", r"neural network", r"computer vision",
        r"speech recognition", r"transformer", r"geospatial",
        r"bioacoustic", r"autonomous", r"forecasting"
    ],
    5: [
        r"generative", r"LLM", r"foundation model", r"digital twin",
        r"RAG", r"simulation of the nation", r"VR persona",
        r"reinforcement learning", r"ChatGPT"
    ]
}

# function to score purpose text
def score_use_case(value) -> int:
    """
    Returns complexity level 1-5 for a given value.
    Handles NaN, None, numbers, lists, dicts, etc.
    """
    # treat NaN/None as neutral
    if pd.isna(value):
        return 2

    # If it's not a string, convert to string safely
    # For lists/dicts, join keys/elements to text
    if not isinstance(value, str):
        try:
            if isinstance(value, (list, tuple, set)):
                value = " ".join(map(str, value))
            elif isinstance(value, dict):
                # combine keys and values into a single string
                value = " ".join([f"{k} {v}" for k, v in value.items()])
            else:
                value = str(value)
        except Exception:
            # fallback if conversion fails
            return 2

    text = value.lower()

    # Check highest complexity first
    for level in sorted(COMPLEXITY_KEYWORDS.keys(), reverse=True):
        for pattern in COMPLEXITY_KEYWORDS[level]:
            # use case-insensitive regex matching
            if re.search(pattern, text, flags=re.IGNORECASE):
                return level

    # default neutral
    return 2

USE_CASE_COMPLEXITY = {
    "Health & Medical": 5,
    "AI used in Transportation Operations": 5,
    "Transportation": 5,
    "Law & Justice": 4,
    "Emergency Management": 4,
    "Energy & the Environment": 4,
    "Science & Space": 3,
    "Government Services (includes Benefits and Service Delivery)": 3,
    "Education & Workforce": 3,
    "Mission-Enabling (internal agency support)": 3,
    "Administration of AI Governance, Processes, and Procedures": 2,
    "Department-Level AI Capabilities and Capacity": 2,
    "Natural Language Processing": 3,
    "Deep Learning": 4,
    "Statistical Methods": 3,
    "Classification": 3,
    "AIML Platform/Environment": 3,
    "Other": 2,
    "": 2  # blank fallback
}

def score_use_case_area(value):
    return USE_CASE_COMPLEXITY.get(value, 2)

# compute total score
def compute_complexity(row):
    # T1: Topic model complexity
    t1 = TOPIC_COMPLEXITY.get(row["primary_topic_k40_id"], 2)
    
    # T2: NLP scoring of narrative
    t2 = score_use_case(row["What is the intended purpose and expected benefits of the AI?"])
    
    # T3: Gov category score
    t3 = score_use_case_area(row["clean_use_case_topic_area"])
    
    # Raw total (max theoretical score = 5 + 5 + 5 = 15)
    raw_score = t1 + t2 + t3
    
    # Convert to 0–100
    final_score = (raw_score / 15) * 100
    
    return final_score

df["complexity_score"] = df.apply(compute_complexity, axis=1)

print(df['complexity_score'].value_counts())

## Model 1: High-Impact Predictor
# High-Impact Type Predictor (Neither / Rights only / Safety only / Both)

# Find Column L (target) robustly ----
# Preferred: name contains both 'rights-impacting' and 'safety-impacting'
candidates = [c for c in df.columns if ("rights" in c.lower() and "impact" in c.lower() and "safety" in c.lower())]
if candidates:
    TARGET_COL1 = candidates[0]
else:
    # Fallback to K = 12th column (0-based index 11)
    TARGET_COL1 = df.columns[11]

print(f"M1 target column: {TARGET_COL1}")

# Clean & encode target to four classes ----
# Map to: 0=Neither, 1=Rights only, 2=Safety only, 3=Both
def map_impact(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    # canonical detection
    has_rights  = ("right" in s)     # captures 'rights-impacting'
    has_safety  = ("safety" in s)    # captures 'safety-impacting'
    has_neither = ("neither" in s)
    has_both    = ("both" in s)

    if has_both or (has_rights and has_safety):
        return 3
    if has_neither:
        return 0
    if has_rights and not has_safety:
        return 1
    if has_safety and not has_rights:
        return 2
    return np.nan  # anything unclear, drop

y_mult1 = df[TARGET_COL1].apply(map_impact)
mask1 = y_mult1.isin([0,1,2,3])
dfm1 = df.loc[mask1].copy()
y1 = y_mult1.loc[mask1].astype(int)

class_names = {0: "Neither", 1: "Rights only", 2: "Safety only", 3: "Both"}
print("M1 — class counts (cleaned):")
print(pd.Series(y1).map(class_names).value_counts())

TEXT_COLS1 = [
    "clean_use_case_topic_area",
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs.",
    "Stage of Development",
    "Agency",
    "Bureau",
    "Agency Abbreviation"
]

# Keep only columns that actually exist
TEXT_COLS1 = [c for c in TEXT_COLS1 if c in dfm1.columns]

def row_to_text(row):
    parts = []
    for c in TEXT_COLS1:
        v = row.get(c, "")
        if pd.notna(v):
            parts.append(str(v))
    return " | ".join(parts)

X_text1 = dfm1.apply(row_to_text, axis=1)

# Train/test split (stratified by multiclass y) ----
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_text1, y1, test_size=0.25, random_state=42, stratify=y1
)

# TF-IDF + Multinomial Logistic Regression ----
pipe_m1 = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
    ))
])

pipe_m1.fit(X_train1, y_train1)

# Evaluation ----
y_pred1 = pipe_m1.predict(X_test1)
acc = accuracy_score(y_test1, y_pred1)
print(f"\nM1 — Multiclass Accuracy: {acc:.4f}")

present = np.unique(np.concatenate([y_test1, y_pred1]))
labels_used = sorted(present.astype(int))
names_used  = [class_names[i] for i in labels_used]

print("\nClassification report (present classes only):")
print(classification_report(y_test1, y_pred1, labels=labels_used, target_names=names_used, zero_division=0))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test1, y_pred1, labels=labels_used)
print(cm)

# Visuals: ROC and Precision–Recall (one-vs-rest) ----
# Binarize test labels for one-vs-rest metrics
Y_test_bin1 = label_binarize(y_test1, classes=labels_used)
# predict_proba returns shape (n_samples, n_classes_present)
probs = pipe_m1.predict_proba(X_test1)

# ROC curves
plt.figure(figsize=(6,5))
for i, cname in enumerate(names_used):
    if Y_test_bin1[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(Y_test_bin1[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("M1 — ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Precision–Recall curves
plt.figure(figsize=(6,5))
for i, cname in enumerate(names_used):
    if Y_test_bin1[:, i].sum() == 0:
        continue
    precision, recall, _ = precision_recall_curve(Y_test_bin1[:, i], probs[:, i])
    ap = average_precision_score(Y_test_bin1[:, i], probs[:, i])
    plt.plot(recall, precision, lw=2, label=f"{cname} (AP = {ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("M1 — Precision–Recall Curves (One-vs-Rest)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# Top terms per class (interpretability) ----
vec = pipe_m1.named_steps["tfidf"]
clf = pipe_m1.named_steps["clf"]
feature_names = np.array(vec.get_feature_names_out())

# clf.classes_ gives the classes seen in training for coef_ row order
trained_class_labels = clf.classes_.astype(int)
trained_class_names  = [class_names[i] for i in trained_class_labels]

coef_arr = clf.coef_  # shape = (n_classes_present, n_features)
for row_idx, cname in enumerate(trained_class_names):
    coefs = coef_arr[row_idx]
    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]

    top_pos = pd.DataFrame({
        "term": feature_names[top_pos_idx],
        "coef": coefs[top_pos_idx]
    })
    top_neg = pd.DataFrame({
        "term": feature_names[top_neg_idx],
        "coef": coefs[top_neg_idx]
    })

    print(f"\nTop terms → {cname} (positive coefficients):")
    print(top_pos)
    print(f"Top terms → NOT {cname} (negative coefficients):")
    print(top_neg)

# Probability histograms for each class ----
plt.figure(figsize=(7,4))
for i, cname in enumerate(names_used):
    plt.hist(probs[:, i], bins=30, alpha=0.5, label=cname)
plt.title("M1 — Predicted Probability Distributions (Test Set)")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# Model 2: Complexity Score (Linear Regression)

# Define binary target: 1 = above median, 0 = below or equal
y_bin2 = (df['complexity_score'] > df['complexity_score'].median()).astype(int)
X2 = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# Split & scale
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y_bin2, test_size=0.25, random_state=42, stratify=y_bin2)
scaler = StandardScaler(with_mean=False)
X_train_sc = scaler.fit_transform(X_train2)
X_test_sc  = scaler.transform(X_test2)

# Fit logistic model
logit_m2 = LogisticRegression(max_iter=1000, solver='lbfgs')
logit_m2.fit(X_train_sc, y_train2)

# Predict and evaluate
y_pred2 = logit_m2.predict(X_test_sc)
y_prob2 = logit_m2.predict_proba(X_test_sc)[:,1]

acc2 = accuracy_score(y_test2, y_pred2)
prec2, rec2, f12, _ = precision_recall_fscore_support(y_test2, y_pred2, average='binary')
auc2 = roc_auc_score(y_test2, y_prob2)

print(f"Accuracy: {acc2:.4f} | Precision: {prec2:.4f} | Recall: {rec2:.4f} | F1: {f12:.4f} | AUC: {auc2:.4f}")
print(confusion_matrix(y_test2, y_pred2))

# Top coefficients
coef_m2_logit = pd.Series(logit_m2.coef_[0], index=FEATURE_COLS).sort_values(key=lambda s: s.abs(), ascending=False)
top_m2_logit = coef_m2_logit.head(15).to_frame('coef').reset_index().rename(columns={'index':'feature'})
# Map topic IDs to primary_topic_keywords from your dataset
topic_keyword_map = df.set_index('primary_topic_k40_id')['primary_topic_keywords'].to_dict()
top_m2_logit['topic_words'] = top_m2_logit['feature'].map(topic_keyword_map)
# Label effect
top_m2_logit['effect'] = np.where(top_m2_logit['coef'] > 0, '↑ high_complexity', '↓ high_complexity')
print(top_m2_logit)

# H2 Logistic Model ROC Curve
fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_prob2)
roc_auc2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr2, tpr2, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc2:.3f})')
plt.plot([0, 1], [0, 1], color='orange', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('M2 Logistic Model ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Model 3: Resourcing Strategy Model
# Prepare target (strict mapping, no made-up names) ----
def map_dev_type(x: str):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    # Keep explicit phrases first (as observed)
    if s == "developed with contracting resources." or "contract" in s:
        return "contract"
    if s == "developed in-house." or "in-house" in s or "in house" in s:
        return "in-house"
    return np.nan  # anything unclear gets dropped

y_text3 = df[M3_TARGET].apply(map_dev_type)
mask = y_text3.isin(["contract", "in-house"])

dfm3 = df.loc[mask].copy()
y3 = y_text3.loc[mask].map({"contract": 1, "in-house": 0}).astype(int)  # 1=contract, 0=in-house

print("M3 — label distribution:")
print(pd.Series(y3).map({1:"contract", 0:"in-house"}).value_counts())

# Build a text field from existing columns (no synthetic fields) ----
TEXT_COLS3 = [
    "clean_use_case_topic_area",
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs.",
    "Stage of Development",
    "Agency",
    "Bureau",
    "Agency Abbreviation"
]
# Keep only columns that actually exist (defensive)
TEXT_COLS3 = [c for c in TEXT_COLS3 if c in dfm3.columns]

def row_to_text(row):
    parts = []
    for c in TEXT_COLS3:
        val = row.get(c, "")
        if pd.notna(val):
            parts.append(str(val))
    return " | ".join(parts)

X_text3 = dfm3.apply(row_to_text, axis=1)

# Train/test split ----
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_text3, y3, test_size=0.25, random_state=42, stratify=y3
)

# Pipeline: TF-IDF -> Logistic Regression ----
pipe_m3 = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,            # ignore ultra-rare terms
        max_df=0.9,          # drop near-stopwords
        strip_accents="unicode",
        lowercase=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced"  # handle imbalance
    ))
])

pipe_m3.fit(X_train3, y_train3)

# Metrics ----
y_prob3 = pipe_m3.predict_proba(X_test3)[:, 1]
y_pred3 = (y_prob3 >= 0.5).astype(int)

acc3 = accuracy_score(y_test3, y_pred3)
prec3, rec3, f13, _ = precision_recall_fscore_support(y_test3, y_pred3, average="binary", zero_division=0)
auc3 = roc_auc_score(y_test3, y_prob3)

print(f"\nM3 (Contract vs In-House) — Metrics")
print(f"Accuracy: {acc3:.4f} | Precision: {prec3:.4f} | Recall: {rec3:.4f} | F1: {f13:.4f} | AUC: {auc3:.4f}")
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test3, y_pred3, labels=[0,1]))  # [[TN,FP],[FN,TP]]

print("\nClassification report:")
print(classification_report(y_test3, y_pred3, target_names=["in-house (0)","contract (1)"], zero_division=0))

# ROC curve ----
fpr3, tpr3, _ = roc_curve(y_test3, y_prob3)
roc_auc3 = auc3
plt.figure()
plt.plot(fpr3, tpr3, lw=2, label=f'ROC (AUC = {roc_auc3:.3f})')
plt.plot([0,1],[0,1],'--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("M3 — ROC Curve (Contract=1 vs In-House=0)")
plt.legend(loc="lower right")
plt.show()

# Precision–Recall curve ----
precision3, recall3, _ = precision_recall_curve(y_test3, y_prob3)
ap3 = average_precision_score(y_test3, y_prob3)
plt.figure()
plt.plot(recall3, precision3, lw=2, label=f'AP = {ap3:.3f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("M3 — Precision–Recall Curve (Contract=1)")
plt.legend(loc="lower left")
plt.show()

# Top terms driving predictions (interpretability) ----
# Positive coefficients -> push toward "contract"
# Negative coefficients -> push toward "in-house"
vec3 = pipe_m3.named_steps["tfidf"]
clf3 = pipe_m3.named_steps["clf"]

feature_names3 = np.array(vec3.get_feature_names_out())
coefs3 = clf3.coef_[0]

top_pos_idx3 = np.argsort(coefs3)[-20:][::-1]  # top +weights
top_neg_idx3 = np.argsort(coefs3)[:20]         # top -weights

top_pos3 = pd.DataFrame({
    "term": feature_names[top_pos_idx3],
    "coef": coefs[top_pos_idx3]
})
top_neg3 = pd.DataFrame({
    "term": feature_names[top_neg_idx3],
    "coef": coefs[top_neg_idx3]
})

print("\nTop terms → Contract (positive coefficients):")
print(top_pos3)

print("\nTop terms → In-House (negative coefficients):")
print(top_neg3)

# probability histograms by true class ----
plt.figure()
plt.hist(y_prob3[y_test3==0], bins=30, alpha=0.6, label="in-house (0)")
plt.hist(y_prob3[y_test3==1], bins=30, alpha=0.6, label="contract (1)")
plt.xlabel("Predicted probability of Contract")
plt.ylabel("Count")
plt.title("M3 — Predicted Probabilities by True Class")
plt.legend()
plt.show()