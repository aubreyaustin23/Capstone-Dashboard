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
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# visualizations --
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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
M3_TARGET = 'Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? ' 


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
        return "External Contracting"
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

df["clean_resourcing"] = df["Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? "] \
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

# ----------------------------------------------------
# MODEL 1 – High Impact: Multiclass Logistic Model
# ----------------------------------------------------

# Robustly detect the target column
candidates = [
    c for c in df.columns
    if ("rights" in c.lower() and "impact" in c.lower() and "safety" in c.lower())
]
if candidates:
    TARGET_COL1 = candidates[0]
else:
    TARGET_COL1 = df.columns[11]

# Map raw text → 4 classes
def map_impact(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    has_rights  = "right" in s
    has_safety  = "safety" in s
    has_neither = "neither" in s
    has_both    = "both" in s

    if has_both or (has_rights and has_safety): return 3
    if has_neither: return 0
    if has_rights: return 1
    if has_safety: return 2
    return np.nan

y_mult1 = df[TARGET_COL1].apply(map_impact)
mask1 = y_mult1.isin([0,1,2,3])
dfm1 = df.loc[mask1].copy()
y1 = y_mult1.loc[mask1].astype(int)

TEXT_COLS1 = [
    "clean_use_case_topic_area",
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs.",
    "Stage of Development",
    "Agency",
    "Bureau",
    "Agency Abbreviation"
]
TEXT_COLS1 = [c for c in TEXT_COLS1 if c in dfm1.columns]

def row_to_text(row):
    parts = []
    for c in TEXT_COLS1:
        v = row.get(c, "")
        if pd.notna(v):
            parts.append(str(v))
    return " | ".join(parts)

X_text1 = dfm1.apply(row_to_text, axis=1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_text1, y1, test_size=0.25, random_state=42, stratify=y1
)

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
y_pred1 = pipe_m1.predict(X_test1)
m1_accuracy = accuracy_score(y_test1, y_pred1)
print(f"M1 High-Impact Accuracy: {m1_accuracy:.4f}")

# ----------------------------------------------
# PRECOMPUTE TF-IDF EMBEDDINGS FOR NEIGHBOR SEARCH
# ----------------------------------------------
# Use the same text-building function used for Model 1
df["usecase_text"] = df.apply(row_to_text, axis=1)

# Embed all existing use cases
tfidf_matrix = pipe_m1.named_steps["tfidf"].transform(df["usecase_text"].fillna(""))

# ----------------------------------------------------
# MODEL 2 – Complexity Score (Binary Logistic Model)
# ----------------------------------------------------

FEATURE_COLS = [
    c for c in df.columns
    if str(c).startswith("topic_") or str(c).startswith("primary_topic")
]

y_bin2 = (df['complexity_score'] > df['complexity_score'].median()).astype(int)
X2 = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y_bin2, test_size=0.25, random_state=42, stratify=y_bin2
)

scaler = StandardScaler(with_mean=False)
X_train2_sc = scaler.fit_transform(X_train2)
X_test2_sc  = scaler.transform(X_test2)

logit_m2 = LogisticRegression(max_iter=1000)
logit_m2.fit(X_train2_sc, y_train2)
y_pred2 = logit_m2.predict(X_test2_sc)

m2_accuracy = accuracy_score(y_test2, y_pred2)
print(f"M2 Complexity Binary Accuracy: {m2_accuracy:.4f}")

# ----------------------------------------------------
# MODEL 3 – Resourcing Strategy (Contract vs In-House)
# ----------------------------------------------------

def map_dev_type(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    if "contract" in s: return "contract"
    if "in-house" in s or "in house" in s: return "in-house"
    return np.nan

M3_TARGET = "Was the AI system involved in this use case developed (or is it to be developed) under contract(s) or in-house? "
y_text3 = df[M3_TARGET].apply(map_dev_type)
mask3 = y_text3.isin(["contract", "in-house"])

dfm3 = df.loc[mask3].copy()
y3 = y_text3.loc[mask3].map({"contract":1, "in-house":0})

TEXT_COLS3 = [
    "clean_use_case_topic_area",
    "What is the intended purpose and expected benefits of the AI?",
    "Describe the AI system’s outputs.",
    "Stage of Development",
    "Agency",
    "Bureau",
    "Agency Abbreviation"
]
TEXT_COLS3 = [c for c in TEXT_COLS3 if c in dfm3.columns]

def row_text_3(row):
    return " | ".join(str(row[c]) for c in TEXT_COLS3 if pd.notna(row.get(c)))

X_text3 = dfm3.apply(row_text_3, axis=1)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_text3, y3, test_size=0.25, random_state=42, stratify=y3
)

pipe_m3 = Pipeline([
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
        class_weight="balanced"
    ))
])

pipe_m3.fit(X_train3, y_train3)
y_pred3 = pipe_m3.predict(X_test3)

m3_accuracy = accuracy_score(y_test3, y_pred3)
print(f"M3 Resourcing Accuracy: {m3_accuracy:.4f}")

# ---------------------------------------------
# Helper functions for Proposal Tab of Dash
# ---------------------------------------------

COLORS = {
    "green": "#86BC25",
    "dark_green": "#046A38",
    "teal": "#0097A9",
    "navy": "#003B49",
    "gray": "#5A5A5A",
    "light_gray": "#E6E6E6",
    "black": "#000000"
}

def proposal_to_text(form):
    parts = []
    for c in TEXT_COLS1:
        v = form.get(c, "")
        if v and str(v).strip():
            parts.append(str(v))
    return " | ".join(parts)

def proposal_to_m2_features(primary_topic_id):
    X = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])

    topic_col = f"topic_{str(primary_topic_id).zfill(2)}"
    if topic_col in X.columns:
        X.loc[0, topic_col] = 1.0

    if "primary_topic_k40_id" in X.columns:
        X.loc[0, "primary_topic_k40_id"] = primary_topic_id

    return X

TOPIC_LABELS = df.set_index("primary_topic_k40_id")["primary_topic_label"].to_dict()

def build_result_card(title, main_value, confidence=None, extra_content=None):

    children = [
        # --- Card Title ---
        html.H3(title, style={"marginBottom": "8px"}),

        # --- Main Value (e.g., predicted label) ---
        html.H2(
            main_value,
            style={
                "margin": "0",
                "fontWeight": "600",
                "color": "#2c3e50"
            }
        )
    ]

    # --- Optional alert / extra content (e.g., high-impact warning flag) ---
    if extra_content is not None:
        children.append(
            html.Div(
                extra_content,
                style={
                    "marginTop": "8px"
                }
            )
        )

    # --- Confidence Bar + Description ---
    if confidence is not None:
        children.append(
            html.Div([
                # Bar container
                html.Div(
                    style={
                        "backgroundColor": "#e0e0e0",
                        "height": "10px",
                        "borderRadius": "5px",
                        "overflow": "hidden",
                        "marginTop": "10px"
                    },
                    children=[
                        html.Div(style={
                            "width": f"{confidence:.1f}%",
                            "height": "100%",
                            "backgroundColor": "#2c3e50"
                        })
                    ]
                ),

                # Percent value
                html.Div(
                    f"{confidence:.1f}%",
                    style={
                        "fontSize": "13px",
                        "color": "#333",
                        "marginTop": "6px",
                        "fontWeight": "600"
                    }
                ),

                # % Explanation
                html.Div(
                    "This reflects how confident the model is in the predicted result.",
                    style={
                        "fontSize": "12px",
                        "color": "#666",
                        "marginTop": "2px",
                        "lineHeight": "1.3"
                    }
                )
            ])
        )

    # --- Return styled card ---
    return html.Div(
        children,
        style={
            "border": "1px solid #ccc",
            "borderRadius": "6px",
            "padding": "15px",
            "backgroundColor": "white",
            "boxShadow": "0px 1px 3px rgba(0,0,0,0.08)"
        }
    )

def find_similar_usecases(proposal_text, top_k=5):
    """
    Returns top-k most similar existing federal use cases.
    """

    # Embed proposal text using the same TF-IDF model
    proposal_vec = pipe_m1.named_steps["tfidf"].transform([proposal_text])

    # Compute cosine similarity with all use case vectors
    sims = cosine_similarity(proposal_vec, tfidf_matrix).flatten()

    # Get highest similarity indices
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "Use Case Name": df.loc[i, "Use Case Name"],
            "Agency": df.loc[i, "Agency"],
            "Bureau": df.loc[i, "Bureau"],
            "primary_topic_label": df.loc[i, "primary_topic_label"],
            "SimilarityScore": float(sims[i] * 100)
        })
    return results

def build_similar_cases_panel(results):
    children = [
        html.H3("Similar Existing Federal Use Cases", style={'margin-bottom': '10px'})
    ]

    for r in results:
        children.append(
            html.Div([
                html.Strong(r["Use Case Name"]),
                html.Br(),
                html.Span(f"{r['Agency']} — {r['Bureau']}"),
                html.Br(),
                html.Span(f"Topic: {r['primary_topic_label']}"),
                html.Br(),
                html.Span(f"Similarity: {r['SimilarityScore']:.1f}%")
            ], style={
                'border': '1px solid #ddd',
                'padding': '10px',
                'marginBottom': '8px',
                'borderRadius': '4px',
                'backgroundColor': '#fafafa'
            })
        )

    return html.Div(children)

def build_highimpact_guidance():
    return html.Div(
        className="highimpact-guidance-box",
        children=[
            html.H3("High-Impact AI Considerations", className="guidance-title"),

            html.P(
                "Based on your proposal, this use case may align with the federal definition of "
                "High-Impact AI. These systems require additional risk management steps to safeguard "
                "civil rights, privacy, safety, and access to essential services."
            ),

            html.H4("Federal Definition of High-Impact AI"),
            html.Ul([
                html.Li("Used as a principal basis for decisions with legal, material, or binding effects"),
                html.Li("Impacts civil rights, civil liberties, or privacy"),
                html.Li("Affects access to essential government services or benefits"),
                html.Li("Influences health, safety, or critical infrastructure"),
                html.Li("Relates to public safety or strategic government assets"),
            ]),

            html.H4("Seven Required Minimum Risk Management Practices"),
            html.Ul([
                html.Li("Pre-deployment testing and evaluation"),
                html.Li("AI impact assessment before launch"),
                html.Li("Ongoing monitoring for performance and adverse impacts"),
                html.Li("Training for personnel responsible for oversight"),
                html.Li("Human oversight and clear accountability pathways"),
                html.Li("Mechanisms for user remedies or appeals"),
                html.Li("Stakeholder and public engagement during deployment"),
            ])
        ],
        style={
            "backgroundColor": "white",
            "border": "1px solid #ccc",
            "padding": "20px",
            "borderRadius": "6px",
            "boxShadow": "0px 2px 4px rgba(0,0,0,0.06)",
            "lineHeight": "1.5"
        }
    )

#------
# Dashboard Application
#------

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Federal AI Use Case Dashboard"

# Layout with Tabs
app.layout = html.Div(style={'font-family':'Arial, sans-serif','margin':'20px'}, children=[
    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr auto 1fr",
            "alignItems": "center",
            "marginBottom": "20px"
        },
        children=[

            # LEFT SIDE (logo + divider)
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "flex-start",
                    "gap": "12px",
                },
                children=[
                    html.Img(
                        src="assets/logo.png",
                        style={"height": "50px"}
                    )
                ]
            ),

            # CENTER TITLE (perfectly centered on page)
            html.H1(
                "Federal AI Use Case Dashboard",
                style={
                    "textAlign": "center",
                    "margin": "0",
                    "fontSize": "32px",
                    "fontWeight": "600"
                }
            ),

            # RIGHT SIDE — invisible placeholder with same width as left
            html.Div(
                style={"visibility": "hidden"},
                children=[
                    html.Img(
                        src="assets/logo.png",
                        style={"height": "50px"}
                    )
                ]
            )
        ]
    ),

    dcc.Tabs(id='tabs', value='tab-usecase', children=[
        dcc.Tab(label='AI Use Case Analytics', value='tab-usecase'),
        dcc.Tab(label='Agency Analytics', value='tab-agency'),
        dcc.Tab(label='Use Case Proposals', value='tab-propose'),
    ]),

    html.Div(id='tabs-content')
])

# ----------------------
# Callback to render tab content
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-usecase':
        # Single Use Case Analytics
        return html.Div(style={'margin-top':'20px'}, children=[

        # Row: main dropdown + filters
        html.Div(style={'display':'flex', 'gap':'20px', 'align-items':'flex-start'}, children=[
            
            # Left: Use Case Dropdown
            html.Div(style={'flex':'1'}, children=[
                html.Label("Select AI Use Case:"),
                dcc.Dropdown(
                    id='usecase-dropdown',
                    placeholder="Select a use case..."
                )
            ]),

            # Right: Filters
            html.Div(style={'flex':'1'}, children=[
                html.Label("Filter Use Cases:"),
                html.Div(style={'display':'flex', 'flex-direction':'column', 'gap':'10px'}, children=[
                    dcc.Dropdown(
                        id='filter-agency',
                        options=[{'label': a, 'value': a} for a in sorted(df['Agency'].dropna().unique())],
                        placeholder="Filter by Agency"
                    ),
                    dcc.Dropdown(
                        id='filter-bureau',
                        options=[{'label': b, 'value': b} for b in sorted(df['Bureau'].dropna().unique())],
                        placeholder="Filter by Subagency/Bureau"
                    ),
                    dcc.Dropdown(
                        id='filter-topic',
                        options=[{'label': t, 'value': t} for t in sorted(df['clean_use_case_topic_area'].dropna().unique())],
                        placeholder="Filter by Topic Area"
                    ),
                ])
            ]),
        ]),

        html.Hr(style={'margin-top':'20px', 'margin-bottom':'20px', 'border-width':'2px', 'border-color':'#ccc'}),

        # Bottom layout section (two-column)
        html.Div(style={
            'display': 'flex',
            'flex-direction': 'row',
            'gap': '20px',
            'margin-top':'20px'
        }, children=[

            # === LEFT: Main Use Case Details ===
            html.Div(id='usecase-details', style={
                'flex':'1',
                'border':'1px solid #ccc',
                'padding':'15px',
                'border-radius':'5px',
                'min-height':'200px',
                'background-color':'#f9f9f9'
            }),

            # === RIGHT COLUMN: Three stacked info panels ===
            html.Div(style={
                'flex':'1',
                'display':'flex',
                'flex-direction':'column',
                'gap':'15px'
            }, children=[

                # --- Resourcing Strategy ---
                html.Div(id='usecase-resource-panel', style={
                    'border':'1px solid #ccc',
                    'padding':'12px',
                    'border-radius':'5px',
                    'background-color':'#ffffff',
                    'min-height':'120px'
                }),

                # --- High Impact Status ---
                html.Div(id='usecase-highimpact-panel', style={
                    'border':'1px solid #ccc',
                    'padding':'12px',
                    'border-radius':'5px',
                    'background-color':'#ffffff',
                    'min-height':'120px'
                }),

                # --- Estimated Complexity ---
                html.Div(id='usecase-complexity-panel', style={
                    'border':'1px solid #ccc',
                    'padding':'12px',
                    'border-radius':'5px',
                    'background-color':'#ffffff',
                    'min-height':'120px'
                })
            ])
        ])
    ])

    elif tab == 'tab-agency':
        # Agency/Bureau Analytics
        return html.Div(
            style={'margin-top':'20px'},
            children=[

                # Row: main dropdown + filters
                html.Div(
                    style={'display':'flex', 'gap':'20px', 'align-items':'flex-start'},
                    children=[
                        
                        # Left: Agency Dropdown
                        html.Div(
                            style={'flex':'1'},
                            children=[
                                html.Label("Select Agency:"),
                                dcc.Dropdown(
                                    id='agency-dropdown',
                                    options=[{'label': a, 'value': a} for a in sorted(df['Agency'].dropna().unique())],
                                    placeholder="Select an Agency...",
                                ),
                            ]
                        ),

                        # Right: Filters
                        html.Div(
                            style={'flex':'1'},
                            children=[
                                html.Label("Filter by Subagency/Bureau"),
                                html.Div(
                                    style={'display':'flex', 'flex-direction':'column', 'gap':'10px'},
                                    children=[
                                        dcc.Dropdown(
                                            id='filter-bureau',
                                            options=[{'label': b, 'value': b} for b in sorted(df['Bureau'].dropna().unique())],
                                            placeholder="Filter by Subagency/Bureau",
                                            multi=True
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),  # ← closes inner html.Div

                html.Hr(
                    style={'margin-top':'20px', 'margin-bottom':'20px', 'border-width':'2px', 'border-color':'#ccc'}),

                # Two-column layout with headers + divider
                html.Div(
                    style={
                        'display': 'grid',
                        'gridTemplateColumns': '1fr 2px 1fr',   # middle divider
                        'gap': '20px',
                        'marginTop': '20px',
                        'overflowX': 'hidden'   # <--- prevents horizontal scrolling
                    },
                    children=[

                        # =====================================================
                        # LEFT COLUMN (normal flowing page layout)
                        # =====================================================
                        html.Div(
                            id="left-col",
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '20px',
                                'overflowX': 'hidden'     # prevent horizontal scroll
                            },
                            children=[
                                # HEADER (hidden until agency selected)
                                html.Div(
                                    id="left-header-wrapper",
                                    children=[
                                        html.H3("Volume Summary", className="agency-header-title"),
                                        html.Hr(className="agency-header-line")
                                    ],
                                    style={'display': 'none'}   # hidden initially
                                ),

                                dcc.Graph(id='graph-bureau-volume', style={'height': '350px', 'width': '100%'}),
                                dcc.Graph(id='graph-topic-volume', style={'height': '350px', 'width': '100%'}),
                                dcc.Graph(id='graph-resourcing', style={'height': '350px', 'width': '100%'}),
                            ]
                        ),

                        # =====================================================
                        # MIDDLE DIVIDER
                        # =====================================================
                        html.Div(
                            id="agency-middle-divider",
                            style={
                                "display": "none",   # hidden until agency selected
                                "alignItems": "stretch",
                                "height": "100%",     # full height of its grid row
                                "overflow": "hidden"
                            },
                            children=[
                                html.Div(className="vertical-divider")
                            ]
                        ),

                        # =====================================================
                        # RIGHT COLUMN (normal flowing layout)
                        # =====================================================
                        html.Div(
                            id="right-col",
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '20px',
                                'overflowX': 'hidden'   # prevent horizontal scroll
                            },
                            children=[
                                # HEADER (hidden until agency selected)
                                html.Div(
                                    id="right-header-wrapper",
                                    children=[
                                        html.H3("Risk, Privacy & Impact Metrics", className="agency-header-title"),
                                        html.Hr(className="agency-header-line")
                                    ],
                                    style={'display': 'none'}   # hidden initially
                                ),

                                dcc.Graph(id='graph-highimpact', style={'height':'350px', 'width':'100%'}),
                                dcc.Graph(id='graph-flagged', style={'height':'350px', 'width':'100%'}),

                                html.Div(
                                    style={'display': 'flex', 'gap': '10px', 'overflowX': 'hidden'},
                                    children=[
                                        dcc.Graph(id='graph-pii', style={'flex': '1', 'height': '300px'}),
                                        dcc.Graph(id='graph-demographics', style={'flex': '1', 'height': '300px'})
                                    ])
                            ])
                    ])
            ])

    elif tab == 'tab-propose':
        return html.Div(style={'marginTop':'20px'}, children=[
            html.H2("Propose a New AI Use Case"),

            html.Div(style={'display':'flex','gap':'20px'}, children=[

                # LEFT SIDE INPUTS
                html.Div(style={'flex':'1'}, children=[
                    html.Label("Use Case Topic Area"),
                    dcc.Dropdown(
                        id="prop-topic-area",
                        options=[{'label': t, 'value': t} for t in sorted(df["clean_use_case_topic_area"].dropna().unique())],
                        placeholder="Select topic area..."
                    ),

                    html.Br(),

                    html.Label("Primary Use Case Type"),
                    dcc.Dropdown(
                        id="prop-primary-topic-id",
                        options=[{'label': f"{i}—{TOPIC_LABELS.get(i,'')}", 'value': i} for i in range(40)],
                        placeholder="Select primary topic..."
                    ),

                    html.Br(),

                    html.Label("Intended Purpose & Expected Benefits"),
                    dcc.Textarea(
                        id="prop-purpose",
                        style={'width':'100%', 'height':'140px'}
                    ),

                    html.Br(), html.Br(),

                    html.Label("Describe AI Outputs"),
                    dcc.Textarea(
                        id="prop-outputs",
                        style={'width':'100%', 'height':'140px'}
                    ),
                ]),

                # RIGHT SIDE INPUTS
                html.Div(style={'flex':'1'}, children=[
                    html.Label("Agency (optional)"),
                    dcc.Dropdown(
                        id="prop-agency",
                        options=[{'label': a, 'value': a} for a in sorted(df["Agency"].dropna().unique())],
                        placeholder="Select agency..."
                    ),

                    html.Br(),

                    html.Label("Bureau/Subagency (optional)"),
                    dcc.Dropdown(
                        id="prop-bureau",
                        options=[],  # will fill via callback
                        placeholder="Select bureau..."
                    ),
                ]),
            ]),

            html.Br(),
            html.Button("Run Predictions", id="prop-run-btn", n_clicks=0),

            html.Hr(style={'marginTop':'20px'}),

            html.Div(
                id="prop-results",
                style={
                    "marginTop": "25px",
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px"
                }
            ),
            html.Div(
                id="prop-similar-results",
                style={'marginTop': '30px'}
            ),

            html.Div(
                id="prop-highimpact-guidance",
                style={"marginTop": "30px"}
            )
        ])

# ----------------------
# Single Use Case details callback
# Callback to filter dropdown options based on filters
@app.callback(
    Output('usecase-dropdown', 'options'),
    Input('filter-agency', 'value'),
    Input('filter-bureau', 'value'),
    Input('filter-topic', 'value')
)
def update_usecase_options(selected_agency, selected_bureau, selected_topic):
    filtered_df = df.copy()
        
    if selected_agency:
        filtered_df = filtered_df[filtered_df['Agency'] == selected_agency]
    if selected_bureau:
        filtered_df = filtered_df[filtered_df['Bureau'] == selected_bureau]
    if selected_topic:
        filtered_df = filtered_df[filtered_df['clean_use_case_topic_area'] == selected_topic]
        
    return [{'label': row['Use Case Name'], 'value': idx} for idx, row in filtered_df.iterrows()]

# Callback to display selected use case details
@app.callback(
    Output('usecase-details', 'children'),
    Output('usecase-resource-panel', 'children'),
    Output('usecase-highimpact-panel', 'children'),
    Output('usecase-complexity-panel', 'children'),
    Input('usecase-dropdown', 'value')
)
def update_details(selected_idx):

    if selected_idx is None:
        return ("Select a use case to see details.", "", "", "")

    row = df.loc[selected_idx]

    # =============== LEFT COLUMN: FULL DETAILS PANE ===============
    fields_to_show = [
        'Agency', 'Bureau', 'primary_topic_label',
        'What is the intended purpose and expected benefits of the AI?',
        'Use Case Status', 'Date Initiated', 'Date Retired'
    ]

    DISPLAY_NAMES = {
        'primary_topic_label': 'Use Case Type',
        'clean_use_case_topic_area': 'Use Case Topic Area'
    }

    # Start the detail content with one header
    detail_blocks = [
        html.H3("Use Case Summary", style={'margin-top':'0', 'margin-bottom':'10px'})
    ]

    # Add field/value rows
    for field in fields_to_show:
        if field in row and pd.notna(row[field]):
            label = DISPLAY_NAMES.get(field, field)   # fallback to original name
            detail_blocks.append(
                html.Div([
                    html.Strong(f"{label}: "),
                    html.Span(str(row[field]))
                ], style={'margin-bottom':'10px'})
            )

    # =============== RIGHT PANEL 1: RESOURCING STRATEGY ===============
    resourcing_field = 'clean_resourcing'
    piid_field = "Provide the Procurement Instrument Identifier(s) (PIID) of the contract(s) used."
    resourcing = row.get(resourcing_field, "N/A")
    # Check for PIIDs
    piid_value = None
    if piid_field in row and pd.notna(row[piid_field]):
        piid_value = row[piid_field]

    # Build panel contents
    resourcing_children = [
        html.H3("Resourcing Strategy:", style={'margin-top':'0'}),
        html.P(f"Resource Type Utilized: {resourcing}")
    ]

    # If PIIDs exist, add them below
    if piid_value:
        resourcing_children.append(
            html.P(
                f"Contract PIID(s): {piid_value}",
                style={'margin-top': '6px'}
            )
        )

    resourcing_panel = html.Div(resourcing_children)

    # =============== RIGHT PANEL 2: HIGH IMPACT STATUS ===============
    highimpact_field = 'Is the AI use case rights-impacting, safety-impacting, both, or neither?'
    
    # Get actual value
    actual_val = row.get(highimpact_field, None)

    # Predict using M1 model if the row has a text representation
    if 'pipe_m1' in globals():  # ensure the model exists
        text_input = row_to_text(row)  # re-use your row_to_text function
        pred_class_idx = pipe_m1.predict([text_input])[0]  # model predicts 0-3
        pred_label = {0: "Neither", 1: "Rights only", 2: "Safety only", 3: "Both"}[pred_class_idx]
    else:
        pred_label = "Model not available"

    # Determine mismatch flag
    mismatch_flag = ""
    if actual_val is not None and pd.notna(actual_val):
        if actual_val.strip().lower() == "neither" and pred_label in ["Rights only", "Safety only", "Both"]:
            mismatch_flag = html.Span("⚠️ Predicted High-Impact classification but labeled 'Neither'",
                                      style={
                                          "color": "#8C1D18",
                                          "fontWeight": "600",
                                          "marginLeft": "8px",
                                          "fontSize": "13px"
                                      })

    # Build display items for other fields
    risk_extension = 'Has your agency requested an extension to implement the minimum risk management practices for this AI use case?'
    impact_changes = "Is there a process to monitor performance of the AI system’s functionality and changes to its impact on rights or safety as part of the post-deployment plan for the AI use case?"
    own_risks = 'For this particular use case, can the AI carry out a decision or action without direct human involvement that could result in a significant impact on rights or safety?'

    impact_items = [
        ("Rights/Safety Impact", actual_val),
        ("Risk Extension Request", row.get(risk_extension, None)),
        ("Monitoring Process for Potential Impact Change", row.get(impact_changes, None)),
        ("Potential for AI to Act Without Human Oversight", row.get(own_risks, None))
    ]

    # Filter out empty/NaN items
    display_items = []
    for label, val in impact_items:
        if val is not None and pd.notna(val) and str(val).strip() != "":
            display_items.append(html.Li(f"{label}: {val}"))

    # Always add predicted value at the top
    display_items.insert(
        0,
        html.Li([
            f"Predicted High-Impact Class: {pred_label} ",
            mismatch_flag  # this will be "" or the styled red warning
        ])
    )

    # Placeholder if nothing else to display
    if not display_items:
        display_items = [html.Li("No available data.")]

    # Construct panel
    highimpact_panel = html.Div([
        html.H3("High-Impact Status:", style={'margin-top':'0'}),
        html.Ul(display_items)
    ])
    # =============== RIGHT PANEL 3: COMPLEXITY SCORE ===============
    complexity = row.get('complexity_score', None)

    # Convert to percent (handles scores 0–1 or 0–100)
    if complexity is not None:
        if complexity <= 1:
            pct = complexity * 100
        else:
            pct = complexity
    else:
        pct = None

    # Determine label
    if pct is None or pd.isna(pct):
        label = "No model score available."
    else:
        if pct < 40:
            label = "Low"
        elif pct < 70:
            label = "Medium"
        else:
            label = "High"

    # Build progress bar with text positioned on the filled bar edge
    if pct is not None and not pd.isna(pct):
        progress_bar = html.Div(
            style={
                'position': 'relative',
                'width': '100%',
                'background-color': '#eee',
                'height': '24px',
                'border-radius': '8px',
                'overflow': 'hidden',
                'margin-top': '6px'
            },
            children=[
                # Filled portion
                html.Div(
                    style={
                        'height': '100%',
                        'width': f'{pct:.0f}%',
                        'background-color': '#43B02A',
                        'transition': 'width 0.4s ease'
                    }
                ),
                # Text label positioned at the end of the fill
                html.Div(
                    f"{pct:.0f}%",
                    style={
                        'position': 'absolute',
                        'left': f'calc({pct:.0f}% + 5px)',  # shifts label slightly back from the edge
                        'top': '50%',
                        'transform': 'translateY(-50%)',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'color': 'black'
                    }
                )
            ]
        )
    else:
        progress_bar = html.Div("")
        

    complexity_panel = html.Div([
        html.H3("Estimated Technical Complexity:", style={'margin-top':'0'}),
        html.P(label if pct is None else f"{label}"),
        progress_bar,
        html.H6("*evaluated on a scale ranging from non-AI/basic functionality (ex. simple automation) to cutting edge/emerging technologies (ex. autonomous decision-making)", 
                style={'margin-top':'0', 'font-weight': 'lighter'})
    ])

    return detail_blocks, resourcing_panel, highimpact_panel, complexity_panel

# ----------------------
# Agency impacts Bureau filter callback
@app.callback(
    Output('filter-bureau', 'options'),
    Output('filter-bureau', 'value'),
    Input('agency-dropdown', 'value')
)
def update_bureau_options(selected_agency):
    if selected_agency is None:
        # No agency selected → show all bureaus
        bureau_options = sorted(df['Bureau'].dropna().unique())
        return (
            [{'label': b, 'value': b} for b in bureau_options],
            None
        )

    # Filter only rows that belong to the selected agency
    filtered = df[df['Agency'] == selected_agency]

    bureau_options = sorted(filtered['Bureau'].dropna().unique())

    return (
        [{'label': b, 'value': b} for b in bureau_options],
        None  # reset any old selection if the agency changed
    )
# Agency analytics callback
@app.callback(
    [
        Output('graph-bureau-volume', 'figure'),
        Output('graph-topic-volume', 'figure'),
        Output('graph-resourcing', 'figure'),
        Output('graph-highimpact', 'figure'),
        Output('graph-flagged', 'figure'),
        Output('graph-pii', 'figure'),
        Output('graph-demographics', 'figure')
    ],
    [
        Input('agency-dropdown', 'value'),
        Input('filter-bureau', 'value')
    ]
)

def update_agency_graphs(selected_agency, selected_bureaus):

    # --------------------------
    # Placeholder when nothing selected
    # --------------------------
    if not selected_agency:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            height=350,
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': " ",
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }],
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return [empty_fig] * 7

    # --------------------------
    # Filter data
    # --------------------------
    df_filtered = df[df['Agency'] == selected_agency]

    if selected_bureaus:
        df_filtered = df_filtered[df_filtered['Bureau'].isin(selected_bureaus)]

    # ============================================================
    # LEFT COLUMN
    # ============================================================

    # ---------- 1. Stacked bar: Bureau x Active/Retired ----------
    bureau_counts = (
        df_filtered.groupby(['Bureau', 'Use Case Status'])
        .size()
        .reset_index(name='count')
    )

    fig_bureau = px.bar(
        bureau_counts,
        x='Bureau',
        y='count',
        color='Use Case Status',
        title='Use Case Counts by Bureau (Active vs Retired)',
        text='count',
        color_discrete_map={
            "Active": COLORS["green"],
            "Retired": COLORS["gray"]
        }
    )
    fig_bureau.update_layout(
        barmode='stack',
        height=350,
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color=COLORS["black"])
    )

    # ---------- 2. Bar: topic area ----------
    topic_counts = df_filtered["clean_use_case_topic_area"].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']

    fig_topic = px.bar(
        topic_counts,
        x='Topic',
        y='Count',
        title='Use Case Counts by Topic Area',
        text='Count',
        color_discrete_sequence=[COLORS["dark_green"]]
    )
    fig_topic.update_layout(
        height=350,
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ---------- 3. Resourcing Composition ----------
    resourcing_counts = df_filtered['clean_resourcing'].value_counts().reset_index()
    resourcing_counts.columns = ['Resourcing', 'Count']

    fig_resourcing = px.bar(
        resourcing_counts,
        x='Resourcing',
        y='Count',
        text='Count',
        title='Resourcing Strategy Composition',
        color_discrete_sequence=[COLORS["teal"]]
    )
    fig_resourcing.update_layout(
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ============================================================
    # RIGHT COLUMN
    # ============================================================

    # ---------- 4. High Impact ----------
    highimpact_counts = df_filtered[
        'Is the AI use case rights-impacting, safety-impacting, both, or neither?'
    ].value_counts().reset_index()
    highimpact_counts.columns = ['HighImpact', 'Count']

    fig_highimpact = px.bar(
        highimpact_counts,
        x='HighImpact',
        y='Count',
        text='Count',
        title='High Impact Status',
        color_discrete_sequence=[COLORS["green"]]
    )
    fig_highimpact.update_layout(
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ---------- 5. Flagged cases (pred vs actual mismatches) ----------
    def compute_highimpact_flags(df_subset):
        if 'pipe_m1' not in globals():
            return pd.DataFrame({'Bureau': df_subset['Bureau'].unique(), 'Flagged Cases': 0})

        flags = []
        highimpact_field = 'Is the AI use case rights-impacting, safety-impacting, both, or neither?'

        for _, row in df_subset.iterrows():
            actual_val = row.get(highimpact_field, None)
            if pd.isna(actual_val):
                continue

            text_input = row_to_text(row)
            pred_class_idx = pipe_m1.predict([text_input])[0]
            pred_label = {0: "Neither", 1: "Rights only", 2: "Safety only", 3: "Both"}[pred_class_idx]

            mismatch = (
                actual_val.strip().lower() == "neither" and
                pred_label in ["Rights only", "Safety only", "Both"]
            )

            flags.append({'Bureau': row['Bureau'], 'Flagged Cases': int(mismatch)})

        return (
            pd.DataFrame(flags)
            .groupby('Bureau')['Flagged Cases']
            .sum()
            .reset_index()
        )

    flagged_counts = compute_highimpact_flags(df_filtered)

    fig_flagged = px.bar(
        flagged_counts,
        x='Bureau',
        y='Flagged Cases',
        text='Flagged Cases',
        title="Predicted High-Impact Mismatches by Bureau",
        color_discrete_sequence=[COLORS["dark_green"]]
    )
    fig_flagged.update_layout(
        height=350,
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ---------- 6. PII Pie ----------
    pii_counts = df_filtered['clean_pii'].value_counts().reset_index()
    pii_counts.columns = ['PII', 'Count']

    fig_pii = px.pie(
        pii_counts,
        names='PII',
        values='Count',
        color='PII',
        title='Use of PII',
        color_discrete_map={
            "Yes": COLORS["green"],
            "No": COLORS["gray"],
            "Unknown": COLORS["light_gray"]
        }
    )
    fig_pii.update_layout(height=300)

    # ---------- 7. Demographics Pie ----------
    demo_counts = df_filtered['clean_demographics'].value_counts().reset_index()
    demo_counts.columns = ['Demographics', 'Count']

    fig_demo = px.pie(
        demo_counts,
        names='Demographics',
        values='Count',
        color='Demographics',
        title='Use of Demographic Variables',
        color_discrete_map={
            "Yes": COLORS["dark_green"],
            "No": COLORS["gray"],
            "Unknown": COLORS["light_gray"]
        }
    )
    fig_demo.update_layout(height=300)

    # --------------------------
    # Return graphs
    # --------------------------
    return (
        fig_bureau,
        fig_topic,
        fig_resourcing,
        fig_highimpact,
        fig_flagged,
        fig_pii,
        fig_demo
    )

# Header visibility callback
@app.callback(
    [
        Output("left-header-wrapper", "style"),
        Output("right-header-wrapper", "style"),
        Output("agency-middle-divider", "style"),
    ],
    [
        Input("agency-dropdown", "value"),
        Input("filter-bureau", "value"),
    ]
)
def show_hide_headers(selected_agency, selected_bureaus):
    hidden = {"display": "none"}
    visible = {"display": "block"}

    if not selected_agency:
        return hidden, hidden, hidden

    return visible, visible, visible

# ----------------------
# Proposed use case callback
@app.callback(
    [
        Output("prop-results", "children"),
        Output("prop-similar-results", "children"),
        Output("prop-highimpact-guidance", "children")
    ],
    Input("prop-run-btn", "n_clicks"),
    [
        State("prop-agency", "value"),
        State("prop-bureau", "value"),
        State("prop-topic-area", "value"),
        State("prop-primary-topic-id", "value"),
        State("prop-purpose", "value"),
        State("prop-outputs", "value")
    ]
)
def run_proposal_predictions(n_clicks, agency, bureau, topic, topic_id, purpose, outputs):
    
    # ---- PREVENT INITIAL PAGE LOAD ERRORS ----
    if n_clicks == 0:
        return ([], None, None)

    # ----------------------------------------------------
    # BUILD INPUT ROW (NO STAGE INCLUDED)
    # ----------------------------------------------------
    row = {
        "Agency": agency,
        "Bureau": bureau,
        "clean_use_case_topic_area": topic,
        "primary_topic_k40_id": topic_id,
        "What is the intended purpose and expected benefits of the AI?": purpose,
        "Describe the AI system’s outputs.": outputs
    }

    # -------------------------
    # MODEL 1 — HIGH IMPACT
    # -------------------------
    text_m1 = row_to_text(row)

    # Find nearest neighbors
    similar_cases = find_similar_usecases(text_m1, top_k=5)
    similar_panel = build_similar_cases_panel(similar_cases)

    pred_idx = pipe_m1.predict([text_m1])[0]
    pred_prob = pipe_m1.predict_proba([text_m1])[0][pred_idx] * 100

    class_map = {0: "Neither", 1: "Rights Only", 2: "Safety Only", 3: "Both"}
    pred_label = class_map[pred_idx]

    # Determine if a high-impact alert should appear
    alert_message = None
    if pred_label != "Neither":
        alert_message = html.Div(
            "⚠️ Additional high-impact guidance is available below.",
            style={
                "marginTop": "8px",
                "color": "#8C1D18",
                "fontWeight": "600",
                "fontSize": "14px"
            }
        )

    card_m1 = build_result_card(
        "Predicted Rights/Safety Impact",
        pred_label,
        confidence=pred_prob,
        extra_content=alert_message
    )

    # -------------------------
    # MODEL 2 — COMPLEXITY
    # -------------------------
    if topic_id is not None:
        features = {c: 0 for c in FEATURE_COLS}
        for c in FEATURE_COLS:
            if str(topic_id) in str(c):
                features[c] = 1
        X_input = pd.DataFrame([features])
        X_scaled = scaler.transform(X_input)
        prob_complex = float(logit_m2.predict_proba(X_scaled)[0][1] * 100)
    else:
        prob_complex = None

    if prob_complex is not None:
        complexity_label = (
            "High Complexity" if prob_complex >= 66 else
            "Medium Complexity" if prob_complex >= 40 else
            "Low Complexity"
        )
    else:
        complexity_label = "Insufficient Inputs"

    card_m2 = build_result_card(
        "Estimated Technical Complexity",
        complexity_label,
        confidence=prob_complex
    )

    # -------------------------
    # MODEL 3 — RESOURCING
    # -------------------------
    text_m3 = row_text_3(row)  # automatically excludes Stage since not in row
    pred_r = pipe_m3.predict([text_m3])[0]
    pred_r_prob = pipe_m3.predict_proba([text_m3])[0][pred_r] * 100
    res_label = "Contract" if pred_r == 1 else "In-House"

    card_m3 = build_result_card(
        "Predicted Resourcing Strategy",
        res_label,
        confidence=pred_r_prob
    )

    # -------------------------
    # NOVELTY SCORE
    # -------------------------
    novelty = 100 - max(
        pred_prob,
        prob_complex or 0,
        pred_r_prob
    )

    card_novelty = build_result_card(
        "Novelty Score (Similarity to Existing Federal Use Cases)",
        f"{novelty:.1f}%",
        confidence=None
    )

    # --- High-Impact Guidance Section ---
    if pred_label != "Neither":
        guidance_panel = build_highimpact_guidance()
    else:
        guidance_panel = None

    return (
        [
            card_m1,
            card_m2,
            card_m3,
            card_novelty
        ],
        similar_panel,
        guidance_panel
    )

@app.callback(
    Output("prop-bureau", "options"),
    Input("prop-agency", "value")
)
def update_proposal_bureaus(selected_agency):
    if not selected_agency:
        return []
    
    sub = df[df["Agency"] == selected_agency]["Bureau"].dropna().unique()
    return [{'label': b, 'value': b} for b in sorted(sub)]

# ----------------------
if __name__ == '__main__':
    app.run(debug=False)