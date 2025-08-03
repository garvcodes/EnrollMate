import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import shap 
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------
# 1) LOAD DATA & CREATE TARGET
# ---------------------------------------------

df = pd.read_csv("DATASET FILE PATH GOES HERE")
df["success"] = (df["Percentage"] >= 100).astype(int)

## Now, we will create the basis of the engineered features EnrollMate will lose
# ---------------------------------------------
# 2) SET UP METRICS LISTS
# ---------------------------------------------
bands         = ["0-5", "5-10", "10-50", "50-100"]
pop_metrics    = ["total_population","white_population","black_population","asian_population","hispanic_population"]
other_metrics  = ["median_income_median","unemployment_rate_mean"]

# ---------------------------------------------
# 3) LOG-TRANSFORMS OF POPULATION COUNTS
# ---------------------------------------------
for band in bands:
    for m in pop_metrics:
        df[f"{band}_{m}_log"] = np.log1p(df[f"{band}_{m}"])

# ---------------------------------------------
# 4) POPULATION-RELATED FEATURES
# ---------------------------------------------
pop_features = []
for band in bands:
    for m in pop_metrics:
        pop_features += [f"{band}_{m}_log"]
    pop_features += [f"{band}_{m2}" for m2 in other_metrics]

# ---------------------------------------------
# 5) AGE BREAKDOWN FEATURES
# ---------------------------------------------
df["Age_clean"] = (
    df["Age"].fillna("")
             .str.upper()
             .str.replace(r"\s*,\s*", ",", regex=True)
             .str.strip(",")
)
df["Only_Children"] = (df["Age_clean"] == "CHILD").astype(int)
df["Only_Adults"]   = (df["Age_clean"] == "ADULT").astype(int)
df["Mixed_Age"]     = df["Age_clean"].str.contains(",").astype(int)
df["N_Age_Groups"]  = df["Age_clean"].str.count(",").fillna(0).astype(int) + 1
df["Age_Difficulty"] = 0
df.loc[df["Only_Children"] == 1, "Age_Difficulty"] = 2
df.loc[df["Mixed_Age"]     == 1, "Age_Difficulty"] = 1
age_features = ["Only_Children","Only_Adults","Mixed_Age","N_Age_Groups","Age_Difficulty"]

# ---------------------------------------------
# 6) PHASE ONE-HOT
# ---------------------------------------------
df["Phases_clean"] = (
    df["Phases"].fillna("")
                .str.upper()
                .str.replace(r"EARLY_PHASE1","PHASE1",regex=True)
                .str.replace(r"\s*\|\s*","|",regex=True)
                .str.strip("|")
)
phase_dummies = df["Phases_clean"].str.get_dummies(sep="|")
phase_dummies.columns = [f"Phase_{c}" for c in phase_dummies.columns]
df = pd.concat([df, phase_dummies], axis=1)
phase_features = phase_dummies.columns.tolist()

# ---------------------------------------------
# 7) OUTCOME-COUNT FEATURES
# ---------------------------------------------
df["n_primary_outcomes"]   = df["Primary Outcome Measures"].fillna("").str.count(r";\s*") + 1

outcome_features = ["n_primary_outcomes"]

# ---------------------------------------------
# 8) DURATION FEATURE
# ---------------------------------------------
df["Start Date"] = pd.to_datetime(df["Start Date"], format='mixed', errors='coerce')
df['start_month'] = df['Start Date'].dt.month
df['start_year'] = df['Start Date'].dt.year

df["Primary Completion Date"] = pd.to_datetime(df["Primary Completion Date"], format='mixed', errors='coerce')
df["Planned_Duration"]  = (df["Primary Completion Date"] - df["Start Date"]).dt.days
duration_features = ["Planned_Duration"] + ["start_month"] + ["start_year"]
# ---------------------------------------------
# 9) STUDY-DESIGN FLAGS
# ---------------------------------------------
design_flags = {
    'has_randomized':   r'randomiz',
    'has_double_blind': r'double[- ]blind',
    'has_placebo':      r'placebo',
    'has_open_label':   r'open[- ]label'
}
for fname, pat in design_flags.items():
    df[fname] = df["Study Design"].str.contains(pat, case=False, regex=True).astype(int)
design_features = list(design_flags.keys())

# ---------------------------------------------
# 10) INTERACTIONS
# ---------------------------------------------
df["Dur_x_black_0_5"]   = df["Planned_Duration"] * df["0-5_black_population_log"]
df["Dur_x_hispanic_0_5"]   = df["Planned_Duration"] * df["0-5_hispanic_population_log"]
df["Dur_x_asian_0_5"]   = df["Planned_Duration"] * df["0-5_asian_population_log"]
df["Dur_x_white_0_5"]   = df["Planned_Duration"] * df["0-5_white_population_log"]
interaction_features = ["Dur_x_black_0_5", "Dur_x_hispanic_0_5", "Dur_x_asian_0_5", "Dur_x_white_0_5" ]

# ---------------------------------------------
# 11) TEXT-BASED FEATURES
# ---------------------------------------------
df["sum_len_words"] = df["Brief Summary"].str.split().str.len().fillna(0)
df["sum_len_sents"] = df["Brief Summary"].str.count(r"[\.!?]") + 1
text_keywords = ['efficacy','safety','pilot']
for kw in text_keywords:
    df[f"has_{kw}"] = df["Brief Summary"].str.contains(kw, case=False).astype(int)
text_features = ["sum_len_words","sum_len_sents"] + [f"has_{kw}" for kw in text_keywords]


# ---------------------------------------------
# 12) FUNDER TYPE ONE-HOT
# ---------------------------------------------
df["Funder Type"] = df["Funder Type"].fillna("OTHER").str.upper()
funder_dummies   = pd.get_dummies(df["Funder Type"], prefix="Funder")
df = pd.concat([df, funder_dummies], axis=1)
funder_features  = funder_dummies.columns.tolist()

# ---------------------------------------------
# 13) MISCELLANEOUS
# ---------------------------------------------
df['n_conditions'] = df['Conditions'].fillna('').str.count(r';\s*') + 1
misc_feats = ['n_conditions']

# ---------------------------------------------
# 14) TITLE TF-IDF & KEYWORD FLAGS
# ---------------------------------------------
tfidf     = TfidfVectorizer(max_features=100, ngram_range=(1,2), stop_words="english")
X_title   = tfidf.fit_transform(df["Study Title"].fillna("")).toarray()
title_feats = pd.DataFrame(X_title, columns=[f"title_{t}" for t in tfidf.get_feature_names_out()])
df = pd.concat([df, title_feats], axis=1)
df["title_word_count"]  = df["Study Title"].str.split().str.len().fillna(0)
df["title_char_count"]  = df["Study Title"].str.len().fillna(0)
df["title_randomized"]  = df["Study Title"].str.contains("randomiz", case=False, na=False).astype(int)
df["title_placebo"]     = df["Study Title"].str.contains("placebo",   case=False, na=False).astype(int)
df["title_efficacy"]    = df["Study Title"].str.contains("efficacy",  case=False, na=False).astype(int)
df["title_has_acronym"] = df["Study Title"].str.contains(r"\([A-Z0-9\-]{2,}\)", regex=True, na=False).astype(int)
title_features = title_feats.columns.tolist() + [
    "title_word_count","title_char_count",
    "title_randomized","title_placebo",
    "title_efficacy","title_has_acronym"
]

# ---------------------------------------------
# 14.1) Conditions
# ---------------------------------------------
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words="english")
X_conditions = tfidf.fit_transform(df["Conditions"].fillna("")).toarray()
conditions_feats = pd.DataFrame(X_conditions, columns=[f"conditions_{t}" for t in tfidf.get_feature_names_out()])
df = pd.concat([df, conditions_feats], axis=1)
df["conditions_word_count"] = df["Conditions"].str.split().str.len().fillna(0)
df["conditions_char_count"] = df["Conditions"].str.len().fillna(0)
df["conditions_randomized"] = df["Conditions"].str.contains("randomiz", case=False, na=False).astype(int)
df["conditions_placebo"] = df["Conditions"].str.contains("placebo", case=False, na=False).astype(int)
df["conditions_efficacy"] = df["Conditions"].str.contains("efficacy", case=False, na=False).astype(int)
df["conditions_has_acronym"] = df["Conditions"].str.contains(r"\([A-Z0-9\-]{2,}\)", regex=True, na=False).astype(int)
conditions_features = conditions_feats.columns.tolist() + [
    "conditions_word_count", "conditions_char_count",
    "conditions_randomized", "conditions_placebo",
    "conditions_efficacy", "conditions_has_acronym"
]

# ---------------------------------------------
# 14.2) MetaData Section
# ---------------------------------------------
def parse_age(age_str):
    if pd.isna(age_str):
        return None
    return int(age_str.split()[0])

df["min_age"] = df["Minimum Age"].apply(parse_age)
df["max_age"] = df["Maximum Age"].apply(parse_age)

df["sex_all"] = df["Sex"].str.upper() == "ALL"
df["sex_female"] = df["Sex"].str.upper() == "FEMALE"
df["sex_male"] = df["Sex"].str.upper() == "MALE"
df["healthy_volunteers"] = df["Healthy Volunteers"].astype(bool)

meta_feats = ["min_age"] + ["max_age"] + ["sex_all"] + ["sex_female"] + ["sex_male"] + ["healthy_volunteers"]

# ---------------------------------------------
# 14.3) TopAcademic Section
# ---------------------------------------------

top_academic_keywords = [
    # U.S. Academic Medical Centers
    "harvard", "brigham and women's", "massachusetts general", "mgh",
    "stanford", "ucsf", "university of california san francisco",
    "johns hopkins", "hopkins", "duke", "yale", "columbia", "cornell",
    "university of pennsylvania", "upenn", "penn medicine", "penn", "mayo", "sinai",
    "university of chicago", "uchicago", "university of michigan",
    "mayo clinic", "cleveland clinic", "washington university", "barnes jewish",
    "nyu langone", "vanderbilt", "emory", "northwestern", "ucla", "ucsd",
    "university of washington", "uw medicine",
    "university of texas southwestern", "utsw", "md anderson",
    "boston children", "children’s hospital philadelphia", "chop",

]

def check_top_academic(location, keywords):
    if pd.isna(location):
        return 0
    location = location.lower()
    return int(any(keyword in location for keyword in keywords))

df["top_academic"] = df["Locations"].apply(lambda loc: check_top_academic(loc, top_academic_keywords))

top_academic = ["top_academic"]

# ---------------------------------------------
# 14.4) INTERVENTION TYPE ONE-HOT
# ---------------------------------------------
df["Interventions_clean"] = df["Interventions"].fillna("").astype(str).str.upper().str.strip()


intervention_patterns = {
    "DRUG": r"^DRUG:",
    "DEVICE": r"^DEVICE:",
    "BIOLOGICAL": r"^BIOLOGICAL:",
    "BEHAVIORAL": r"^BEHAVIORAL:",
    "PROCEDURE": r"^PROCEDURE:",
    "RADIATION": r"^RADIATION:",
    "DIETARY_SUPPLEMENT": r"^DIETARY SUPPLEMENT:",
    "OTHER": r"^OTHER:", 
    }
intervention_features = []
for type_name, pattern in intervention_patterns.items():
    col_name = f"InterventionType_{type_name}"
    # Use str.contains with regex for the pattern matching
    df[col_name] = df["Interventions_clean"].str.contains(pattern, regex=True, na=False).astype(int)
    intervention_features.append(col_name)

df["has_intervention_type"] = (df[intervention_features].sum(axis=1) > 0).astype(int)
intervention_features.append("has_intervention_type")

print(f"Created {len(intervention_features)} intervention type features.")
print(df[intervention_features].head()) # Display first few rows of new features


# ---------------------------------------------
# 15) COMBINE FEATURES
# ---------------------------------------------
features = (
    pop_features
  + age_features
  + phase_features
  + outcome_features
  + duration_features
  + design_features
  + interaction_features
  + text_features
  + funder_features
  + misc_feats
  + meta_feats
  + top_academic
  + title_features
  + conditions_features
  + intervention_features
)
missing_in_df = [feat for feat in features if feat not in df.columns]
if missing_in_df:
    print(f"Warning: The following feature columns were not found and will be filled with 0: {missing_in_df}")
    for feat in missing_in_df:
        df[feat] = 0

print(f"Total number of features created: {len(features)}")

# ---------------------------------------------
# 16) PREPARE DATA & DEFINE MODEL PARAMETERS
# ---------------------------------------------

# Prepare final modeling DataFrame
df_model = df.dropna(subset=["Percentage", "success"])
logger.info(f"Rows left for modeling: {df_model.shape[0]}")

# Define X and y. Scaling and feature selection will happen inside the CV loop.
X = df_model[features].copy()
y = df_model["success"].copy()

# Handle any potential inf/-inf values from feature engineering
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Define the model parameters here for use in CV and the final model
clf_params = {
    'boosting_type': 'goss',
    'learning_rate': 0.0037243357501073964,
    'num_leaves': 154,
    'max_depth': 11,
    'feature_fraction': 0.6526962138430541,
    'min_child_samples': 26,
    'min_split_gain': 0.005221005248433998,
    'reg_alpha': 0.13411729586865326,
    'reg_lambda': 5.1560928274929886e-08,
    'max_bin': 500,
    'random_state': 42, # Add for reproducibility
    'is_unbalance': True # Often helpful for PR AUC
}

# ---------------------------------------------
# 17) LIGHTGBM 5-FOLD CV WITH ENSEMBLE (CORRECTED LEAKAGE-FREE STRUCTURE)
# ---------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_roc_aucs = []
cv_pr_aucs = []
cv_f1_scores = []
cv_precision_scores = []
cv_recall_scores = []
cv_accuracy_scores = []

best_iters = []
fold_importances = []
ensemble_models = []  # Store trained models 

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/5 ---")

    # 1. Create train/validation sets for this fold
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # 2. Scale Data 
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 3. Select Features 
    selector = SelectFromModel(estimator=LGBMClassifier(**clf_params), threshold='median')
    X_tr_selected = selector.fit_transform(X_tr_scaled, y_tr)
    X_val_selected = selector.transform(X_val_scaled)

    selected_feature_names = X_tr.columns[selector.get_support()]
    print(f"Selected {len(selected_feature_names)} features in this fold.")

    # Train Model
    clf = LGBMClassifier(**clf_params)
    clf.fit(
        X_tr_selected, y_tr,
        eval_set=[(X_val_selected, y_val)],
        eval_metric=["auc", "aucpr"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=0)]
    )

    preds = clf.predict_proba(X_val_selected)[:,1]

    # Get binary predictions using 0.5940 as a threshold
    preds_binary = (preds >= 0.5940).astype(int)

    print(f"Fold {fold+1} Predictions: {np.unique(preds_binary, return_counts=True)}")

    cv_roc_aucs.append(roc_auc_score(y_val, preds))
    cv_pr_aucs.append(average_precision_score(y_val, preds))
    cv_f1_scores.append(f1_score(y_val, preds_binary))
    cv_precision_scores.append(precision_score(y_val, preds_binary, zero_division=0))
    cv_recall_scores.append(recall_score(y_val, preds_binary, zero_division=0))
    cv_accuracy_scores.append(accuracy_score(y_val, preds_binary))

    best_iters.append(clf.best_iteration_)

    ensemble_models.append((clf, scaler, selector))

    gain_importances = pd.DataFrame({
        'feature': selected_feature_names,
        'gain': clf.feature_importances_
    }).sort_values('gain', ascending=False)
    fold_importances.append(gain_importances)

print("\n--- Cross-Validation Results ---")
print(f"Mean ROC AUC:   {np.mean(cv_roc_aucs):.4f} ± {np.std(cv_roc_aucs):.4f}")
print(f"Mean PR AUC:    {np.mean(cv_pr_aucs):.4f} ± {np.std(cv_pr_aucs):.4f}")
print(f"Mean F1 Score:  {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
print(f"Mean Precision: {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}")
print(f"Mean Recall:    {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")
print(f"Mean Accuracy:  {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}")
print("-" * 20)
print("Best iters:", best_iters, "→ Mean:", int(np.mean(best_iters)))

# ---------------------------------------------
# 17.1) ENSEMBLE PREDICTION FUNCTION
# ---------------------------------------------
def ensemble_predict_lgbm(models_scalers_selectors, X_test):
    """
    Make ensemble predictions using multiple trained LightGBM models.
    Each model has its own scaler and feature selector.
    """
    predictions = []

    # Get predictions from each model
    for clf, scaler, selector in models_scalers_selectors:
        # Apply the same preprocessing pipeline used during training
        X_scaled = scaler.transform(X_test)
        X_selected = selector.transform(X_scaled)

        # Get probability predictions
        pred = clf.predict_proba(X_selected)[:, 1]
        predictions.append(pred)

    # Simple average ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# ---------------------------------------------
# 17.2) TEST ENSEMBLE PERFORMANCE
# ---------------------------------------------
print("\n--- Testing Ensemble Performance ---")
ensemble_val_preds = []
ensemble_val_targets = []

# Test ensemble using leave-one-out approach
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_val = X.iloc[valid_idx]
    y_val = y.iloc[valid_idx]

    other_models = [ensemble_models[i] for i in range(len(ensemble_models)) if i != fold]
    if other_models:
        fold_preds = ensemble_predict_lgbm(other_models, X_val)
        ensemble_val_preds.extend(fold_preds)
        ensemble_val_targets.extend(y_val.values)

# Calculate final ensemble performance
if ensemble_val_preds:
    ensemble_val_preds_binary = (np.array(ensemble_val_preds) >= 0.6).astype(int)

    ensemble_roc_auc = roc_auc_score(ensemble_val_targets, ensemble_val_preds)
    ensemble_pr_auc = average_precision_score(ensemble_val_targets, ensemble_val_preds)
    ensemble_f1 = f1_score(ensemble_val_targets, ensemble_val_preds_binary)
    ensemble_precision = precision_score(ensemble_val_targets, ensemble_val_preds_binary)
    ensemble_recall = recall_score(ensemble_val_targets, ensemble_val_preds_binary)
    ensemble_accuracy = accuracy_score(ensemble_val_targets, ensemble_val_preds_binary)

    print(f"Ensemble Validation ROC AUC:   {ensemble_roc_auc:.4f}")
    print(f"Ensemble Validation PR AUC:    {ensemble_pr_auc:.4f}")
    print(f"Ensemble Validation F1 Score:  {ensemble_f1:.4f}")
    print(f"Ensemble Validation Precision: {ensemble_precision:.4f}")
    print(f"Ensemble Validation Recall:    {ensemble_recall:.4f}")
    print(f"Ensemble Validation Accuracy:  {ensemble_accuracy:.4f}")
else:
    ensemble_roc_auc = np.mean(cv_roc_aucs)
    ensemble_pr_auc = np.mean(cv_pr_aucs)
    ensemble_f1 = np.mean(cv_f1_scores)
    ensemble_precision = np.mean(cv_precision_scores)
    ensemble_recall = np.mean(cv_recall_scores)
    ensemble_accuracy = np.mean(cv_accuracy_scores)

# ---------------------------------------------
# 17.3) COMPARISON SUMMARY
# ---------------------------------------------
print("\n" + "="*60)
print("="*23 + " FINAL SUMMARY " + "="*22)
print("="*60)
print(f"{'Metric':<12} | {'Individual CV (Mean ± Std)':<28} | {'Ensemble':<12}")
print("-" * 60)
print(f"{'ROC AUC':<12} | {np.mean(cv_roc_aucs):.4f} ± {np.std(cv_roc_aucs):.4f}              | {ensemble_roc_auc:<12.4f}")
print(f"{'PR AUC':<12} | {np.mean(cv_pr_aucs):.4f} ± {np.std(cv_pr_aucs):.4f}              | {ensemble_pr_auc:<12.4f}")
print(f"{'F1 Score':<12} | {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}              | {ensemble_f1:<12.4f}")
print(f"{'Precision':<12} | {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}              | {ensemble_precision:<12.4f}")
print(f"{'Recall':<12} | {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}              | {ensemble_recall:<12.4f}")
print(f"{'Accuracy':<12} | {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}              | {ensemble_accuracy:<12.4f}")
print("="*60)

    # --- Define Feature Groupings for Ablation Study ---
feature_groups = {
    "Population Features": pop_features,
    "Age Breakdown Features": age_features,
    "Phase One-Hot Features": phase_features,
    "Outcome Count Features": outcome_features,
    "Duration Features": duration_features,
    "Study Design Flags": design_features,
    "Interaction Features": interaction_features,
    "Text-Based Features (Brief Summary)": text_features,
    "Funder Type Features": funder_features,
    "Miscellaneous Features": misc_feats,
    "Metadata Features": meta_feats,
    "Top Academic Location Features": top_academic,
    "Title TF-IDF & Keyword Features": title_features,
    "Conditions TF-IDF & Keyword Features": conditions_features,
    "Intervention Type Features": intervention_features
}
print(f"Defined {len(feature_groups)} feature groups for ablation study.")

# --- Ablation Study Implementation ---
print("\n" + "="*70)
print("="*25 + " STARTING ABLATION STUDY " + "="*24)
print("="*70)

ablation_results = {}
base_features = list(X.columns) # All features in your original X

# Add baseline full model performance to results for comparison
# (You already have this from your initial CV run, but let's store it explicitly for the table)
baseline_roc_auc = np.mean(cv_roc_aucs)
ablation_results["Full Model (Baseline)"] = {
    "ROC AUC": baseline_roc_auc,
    "PR AUC": np.mean(cv_pr_aucs),
    "F1 Score": np.mean(cv_f1_scores),
    "Precision": np.mean(cv_precision_scores),
    "Recall": np.mean(cv_recall_scores),
    "Accuracy": np.mean(cv_accuracy_scores),
    "Num Features": len(features),
    "ROC AUC Decline": 0.0 # Baseline has no decline
}

for group_name, group_feats in feature_groups.items():
    logger.info(f"\n--- Ablating: {group_name} ({len(group_feats)} features) ---")

    # Create the feature set for this ablation run
    # Ensure to only remove features that actually exist in the base_features
    features_to_remove = [f for f in group_feats if f in base_features]
    if not features_to_remove:
        logger.warning(f"No features from '{group_name}' found in the base feature set. Skipping ablation for this group.")
        continue

    current_ablation_features = [f for f in base_features if f not in features_to_remove]

    if not current_ablation_features:
        logger.error(f"Ablating '{group_name}' would leave no features. Skipping this ablation to prevent errors.")
        continue

    # Prepare data for this ablation run
    X_ablated = df_model[current_ablation_features].copy()
    X_ablated = X_ablated.replace([np.inf, -np.inf], np.nan).fillna(0) # Re-handle NaNs

    # Re-run the 5-fold CV for this ablated feature set
    fold_roc_aucs_abl = []
    fold_pr_aucs_abl = []
    fold_f1_abl = []
    fold_precision_abl = []
    fold_recall_abl = []
    fold_accuracy_abl = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_ablated, y)):
        X_tr_abl, X_val_abl = X_ablated.iloc[train_idx], X_ablated.iloc[valid_idx]
        y_tr_abl, y_val_abl = y.iloc[train_idx], y.iloc[valid_idx]

        # Scale Data
        scaler_abl = StandardScaler()
        X_tr_scaled_abl = scaler_abl.fit_transform(X_tr_abl)
        X_val_scaled_abl = scaler_abl.transform(X_val_abl)

        # Select Features
        # Use a fresh selector for each ablation run
        selector_abl = SelectFromModel(estimator=LGBMClassifier(**clf_params), threshold='median')
        X_tr_selected_abl = selector_abl.fit_transform(X_tr_scaled_abl, y_tr_abl)
        X_val_selected_abl = selector_abl.transform(X_val_scaled_abl)

        # Train Model
        clf_abl = LGBMClassifier(**clf_params)
        clf_abl.fit(
            X_tr_selected_abl, y_tr_abl,
            eval_set=[(X_val_selected_abl, y_val_abl)],
            eval_metric=["auc", "aucpr"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=0)]
        )

        preds_abl = clf_abl.predict_proba(X_val_selected_abl)[:,1]
        preds_binary_abl = (preds_abl >= 0.6).astype(int)

        fold_roc_aucs_abl.append(roc_auc_score(y_val_abl, preds_abl))
        fold_pr_aucs_abl.append(average_precision_score(y_val_abl, preds_abl))
        fold_f1_abl.append(f1_score(y_val_abl, preds_binary_abl))
        fold_precision_abl.append(precision_score(y_val_abl, preds_binary_abl, zero_division=0))
        fold_recall_abl.append(recall_score(y_val_abl, preds_binary_abl, zero_division=0))
        fold_accuracy_abl.append(accuracy_score(y_val_abl, preds_binary_abl))

    # Store results for this ablation
    mean_roc_auc_abl = np.mean(fold_roc_aucs_abl)
    roc_auc_decline = baseline_roc_auc - mean_roc_auc_abl
    
    ablation_results[f"Ablated: {group_name}"] = {
        "ROC AUC": mean_roc_auc_abl,
        "PR AUC": np.mean(fold_pr_aucs_abl),
        "F1 Score": np.mean(fold_f1_abl),
        "Precision": np.mean(fold_precision_abl),
        "Recall": np.mean(fold_recall_abl),
        "Accuracy": np.mean(fold_accuracy_abl),
        "Num Features": len(current_ablation_features),
        "ROC AUC Decline": roc_auc_decline
    }
    logger.info(f"Ablated '{group_name}' Results - ROC AUC: {mean_roc_auc_abl:.4f}, PR AUC: {np.mean(fold_pr_aucs_abl):.4f}, Decline: {roc_auc_decline:.4f}")

# --- Display Ablation Results ---
print("\n" + "="*70)
print("="*23 + " ABLATION STUDY RESULTS " + "="*23)
print("="*70)

results_df = pd.DataFrame.from_dict(ablation_results, orient='index')
results_df = results_df.sort_values(by="ROC AUC Decline", ascending=False) # Sort by the new column

print(results_df.to_string())

print("\n" + "="*70)
print("="*24 + " END OF ABLATION STUDY " + "="*25)
print("="*70)

# ---------------------------------------------
# 18) TRAIN FINAL MODEL & SHOW IMPORTANCES (CORRECTED)
# ---------------------------------------------
print("\n--- Training Final Model on Full Dataset ---")

# Scale dataset
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

# Perform feature selection 
selector_final = SelectFromModel(estimator=LGBMClassifier(**clf_params), threshold='median')
X_selected_final = selector_final.fit_transform(X_scaled_final, y)

final_selected_features = X.columns[selector_final.get_support()]
print(f"Final model will be trained on {len(final_selected_features)} features.")

#Train final classifier
final_iter = int(np.mean(best_iters))
final_clf = LGBMClassifier(**clf_params, n_estimators=final_iter)
final_clf.fit(X_selected_final, y)

print("\n--- Feature Importances (Gain) from Final Model ---")
importances = final_clf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': final_selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display the top 30 features
print(importance_df.head(50))


# ---------------------------------------------
# 19) SHAP ANALYSIS
# ---------------------------------------------
print("\n--- Generating SHAP Summary Plot ---")
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(X_selected_final)

X_selected_final_df = pd.DataFrame(X_selected_final, columns=final_selected_features)

shap.summary_plot(
    shap_values,
    X_selected_final_df,
    plot_type="dot",
    show=False # We'll show the plot manually to add a title
)

plt.title("SHAP Summary: Impact on Predicting Trial Success", fontsize=16)
plt.tight_layout()
plt.show()

