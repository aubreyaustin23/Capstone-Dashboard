import pandas as pd

# import relevant files and store as dataframes
orig_data = pd.read_excel("2024_consolidated_ai_inventory_raw_v2.xls")
topic_probs = pd.read_csv("Topic Modeling/lda_outputs/lda_features_k40.csv")
topic_ids = pd.read_csv("Topic Modeling/lda_outputs/doc_topics_lda_k40.csv")
topic_names = pd.read_excel("Topic Modeling/Topic Name Mapping.xlsx")

# insert index at beginning of original
orig_data.insert(0, "DocIndex", range(len(orig_data)))

# add topic ids and main probs to orig ---
# limit topic_probs to subset of data to be included & rename columns
topic_ids_subset = topic_ids[["DocIndex", "lda_topic", "lda_topic_prob"]].rename(
    columns={
        "lda_topic": "primary_topic_k40_id",
        "lda_topic_prob": "primary_topic_prob"
    }
)

# merge using DocIndex (left join keeps all orig_data rows)
merged1 = orig_data.merge(topic_ids_subset, on="DocIndex", how="left")

# add column to merged1 based on 'primary_topic_k40_id' --
# create the new column
merged1["primary_topic_k40"] = merged1["primary_topic_k40_id"].apply(
    lambda x: f"lda_t{x}" if pd.notna(x) else None
)

# insert it before primary_topic_k40_id
col_idx = merged1.columns.get_loc("primary_topic_k40_id")
merged1.insert(col_idx, "primary_topic_k40", merged1.pop("primary_topic_k40"))

# merge with topic_names to add topic labels next to the topic ids -- 
merged2 = merged1.merge(
    topic_names[["primary_topic_k40_id", "primary_topic_label"]],
    on="primary_topic_k40_id",
    how="left"
)

# move the new column next to the ID column
id_pos = merged2.columns.get_loc("primary_topic_k40_id")
col = merged2.pop("primary_topic_label")
merged2.insert(id_pos + 1, "primary_topic_label", col)

# merge with topic_probs to add individual topic probabilities --

# remove unnecessary columns from topic_probs
# keep all columns EXCEPT those starting with 'lda_dom'
prob_cols = [c for c in topic_probs.columns if not c.startswith("lda_dom")]
topic_probs_clean = topic_probs[prob_cols]

merged_final = merged2.merge(
    topic_probs_clean, 
    on="Use Case Name", 
    how="left"
)

print(merged_final.head())

merged_final.to_excel('combined_data_final.xlsx', index=False)