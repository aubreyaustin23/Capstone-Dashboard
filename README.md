# Capstone Dashboard

A full interactive analytics dashboard for evaluating government AI use cases.  
This repository includes:

- Data preprocessing and topic modeling (BERTopic, LDA, clustering)
- Feature engineering pipelines
- Consolidated data generation for analysis
- A Dash dashboard (`app.py`) for exploring results
- Optional model performance visualization tools

This README explains how to set up the environment **from scratch**, as well as how to run the application using the **pre-generated files** already in the repository.

---

## 📌 1. Project Structure
Capstone-Dashboard/
│
├── app.py                         # Main dashboard application
├── eda_and_topic_modeling.py      # Full preprocessing + BERTopic workflow
├── final_data_file_consolidated.py # Merges outputs → produces final dataset
├── final_models_w_visual.py       # Optional model performance visualizations
│
├── 2024_consolidated_ai_inventory_raw_v2.xls    # Raw input data
├── combined_data_final.csv                        # Final processed dataset (used by app.py)
│
├── Topic Modeling/                # All topic-modeling outputs & intermediate files
│   ├── lda_outputs/
│   │   ├── aiusecase_with_lda_k40.csv
│   │   ├── doc_topic_long_lda_k40.csv
│   │   ├── doc_topics_lda_k40.csv
│   │   └── lda_features_k40.csv
│   ├── ai_use_case_features.csv
│   ├── aiusecase_outlier_audit.csv
│   ├── column_text_vs_category_audit.csv
│   └── Topic Name Mapping.xlsx
│
├── Assets/
│   ├── custom_styles.css          # Dashboard CSS
│   └── logo.png                   # Dashboard logo
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file

## 📦 2. Environment Setup (Recommended)
### Step 1 — Install Python

Requires Python 3.10+

Download:
https://www.python.org/downloads/

Be sure to check:

✔ “Add Python to PATH”

### Step 2 — Create a Virtual Environment
python -m venv .venv


Activate the environment:

Windows
.\.venv\Scripts\activate

macOS/Linux
source .venv/bin/activate


You should now see something like:

(.venv) C:\path\to\project>

### Step 3 — Install Required Packages
pip install -r requirements.txt


If spaCy complains about a missing model:

python -m spacy download en_core_web_sm

## 🚀 3. Running the Dashboard

If you are using the pre-generated output files already included in the repo, you can launch immediately:

python app.py


No preprocessing required.

## 🧪 4. Full Workflow (If Starting From Raw Data)

If beginning with the raw file 2024_consolidated_ai_inventory_raw_v2.xls, follow this exact order:

### Step 1 — Run EDA + Topic Modeling
python eda_and_topic_modeling.py


This script:

Cleans and preprocesses text

Generates transformer embeddings

Runs BERTopic + LDA

Produces topic assignments

Saves multiple intermediate datasets

⚠️ This step is computationally heavy (UMAP, HDBSCAN, embeddings).
Expect it to take several minutes depending on your hardware.

### Step 2 — Run Data Consolidation Script
python final_data_file_consolidated.py


This script:

Merges topic modeling outputs

Joins probabilities, labels, and metadata

Produces the final dataset:

combined_data_final.csv


This is the file consumed by the dashboard.

### Step 3 — (Optional) Model Performance Visualizations
python final_models_w_visual.py


Run only after final_data_file_consolidated.py
because it expects combined_data_final.csv.

### Step 4 — Launch the Dashboard
python app.py

⚠️ Performance Note

Topic modeling (UMAP, HDBSCAN, embeddings) is computationally expensive.

If only exploring the dashboard, use the pre-generated files and skip:

eda_and_topic_modeling.py

final_data_file_consolidated.py

Only re-run them if you have new raw data.

