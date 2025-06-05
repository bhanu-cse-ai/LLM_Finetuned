from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np
from collections import Counter

# Load prediction results
results_df = pd.read_csv("/home/bhavani/Desktop/LLM_Target_Stance/btsd/tot_2/predictions_llama31_finetuned_tot_2_vast_filtered_ex_20250605_103846.csv")

# Define valid stances
valid_stances = ['FAVOR', 'AGAINST', 'NONE']

# Filter out rows where GT Stance or Predicted Stance is invalid, NaN, or blank
filtered_df = results_df[
    (results_df["GT Stance"].isin(valid_stances)) & 
    (results_df["Predicted Stance"].isin(valid_stances)) &
    (results_df["GT Stance"].notna()) &
    (results_df["Predicted Stance"].notna()) &
    (results_df["GT Stance"].str.strip() != '') &
    (results_df["Predicted Stance"].str.strip() != '')
]

# Check if filtered DataFrame is empty
if filtered_df.empty:
    print("Error: No valid data points remain after filtering for stances: FAVOR, AGAINST, NONE")
    exit()

# Extract true and predicted values from filtered DataFrame
true_targets = filtered_df["GT Target"].tolist()
predicted_targets = filtered_df["Predicted Target"].tolist()
true_stances = filtered_df["GT Stance"].tolist()
predicted_stances = filtered_df["Predicted Stance"].tolist()

# Verify unique stance classes
stance_classes = sorted(set(true_stances) | set(predicted_stances))
print(f"Unique stance classes after filtering: {stance_classes}")
print(f"Number of unique stance classes: {len(stance_classes)}")
print(f"Number of data points after filtering: {len(filtered_df)}")

# Calculate Target Detection metrics
tg_accuracy = accuracy_score(true_targets, predicted_targets)
tg_f1_macro = f1_score(true_targets, predicted_targets, average='macro', zero_division=0)
tg_f1_micro = f1_score(true_targets, predicted_targets, average='micro', zero_division=0)
tg_f1_weighted = f1_score(true_targets, predicted_targets, average='weighted', zero_division=0)
tg_precision_macro = precision_score(true_targets, predicted_targets, average='macro', zero_division=0)
tg_recall_macro = recall_score(true_targets, predicted_targets, average='macro', zero_division=0)

# Calculate Stance Detection metrics
sd_accuracy = accuracy_score(true_stances, predicted_stances)
sd_f1_macro = f1_score(true_stances, predicted_stances, average='macro', zero_division=0)
sd_f1_micro = f1_score(true_stances, predicted_stances, average='micro', zero_division=0)
sd_f1_weighted = f1_score(true_stances, predicted_stances, average='weighted', zero_division=0)
sd_precision_macro = precision_score(true_stances, predicted_stances, average='macro', zero_division=0)
sd_recall_macro = recall_score(true_stances, predicted_stances, average='macro', zero_division=0)

# Calculate true positives and ground truth counts for each valid stance
true_positives = {}
ground_truth_counts = Counter(true_stances)
for stance in valid_stances:
    true_positives[stance] = sum((np.array(true_stances) == stance) & (np.array(predicted_stances) == stance))
    if stance not in ground_truth_counts:
        ground_truth_counts[stance] = 0

# Generate classification report for Stance Detection
try:
    sd_classification_report = classification_report(
        true_stances, 
        predicted_stances, 
        target_names=valid_stances,
        output_dict=True, 
        zero_division=0
    )
except ValueError as e:
    print(f"Error in classification_report: {e}")
    sd_classification_report = None

# Print evaluation metrics
print("\nEvaluation Metrics:")
print("\nTarget Detection:")
print(f"  Accuracy: {tg_accuracy*100:.2f}%")
print(f"  Macro F1: {tg_f1_macro:.4f}")
print(f"  Micro F1: {tg_f1_micro:.4f}")
print(f"  Weighted F1: {tg_f1_weighted:.4f}")
print(f"  Macro Precision: {tg_precision_macro:.4f}")
print(f"  Macro Recall: {tg_recall_macro:.4f}")
print("\nStance Detection:")
print(f"  Accuracy: {sd_accuracy*100:.2f}%")
print(f"  Macro F1: {sd_f1_macro:.4f}")
print(f"  Micro F1: {sd_f1_micro:.4f}")
print(f"  Weighted F1: {sd_f1_weighted:.4f}")
print(f"  Macro Precision: {sd_precision_macro:.4f}")
print(f"  Macro Recall: {sd_recall_macro:.4f}")
print("\nStance Class Statistics:")
for stance in valid_stances:
    print(f"  {stance}:")
    print(f"    True Positives: {true_positives[stance]}")
    print(f"    Ground Truth Count: {ground_truth_counts[stance]}")

# Print per-class metrics from classification report
if sd_classification_report:
    print("\nPer-Class Stance Detection Metrics:")
    for stance in valid_stances:
        metrics = sd_classification_report[stance]
        print(f"  {stance}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1-score']:.4f}")
        print(f"    Support: {metrics['support']}")

# Prepare metrics for saving
metrics = {
    "Metric": [
        "Accuracy", "Macro F1", "Micro F1", "Weighted F1", "Macro Precision", "Macro Recall"
    ],
    "Target Detection": [
        tg_accuracy, tg_f1_macro, tg_f1_micro, tg_f1_weighted, tg_precision_macro, tg_recall_macro
    ],
    "Stance Detection": [
        sd_accuracy, sd_f1_macro, sd_f1_micro, sd_f1_weighted, sd_precision_macro, sd_recall_macro
    ]
}

# Prepare stance statistics for saving
stance_stats_df = pd.DataFrame({
    "Stance Class": list(true_positives.keys()),
    "True Positives": list(true_positives.values()),
    "Ground Truth Count": [ground_truth_counts[stance] for stance in true_positives.keys()]
})

# Save to Excel
with pd.ExcelWriter("evaluation_metrics.xlsx", engine='openpyxl') as writer:
    pd.DataFrame(metrics).to_excel(writer, sheet_name="Metrics", index=False)
    stance_stats_df.to_excel(writer, sheet_name="Stance Statistics", index=False)

print("\n✅ Metrics and stance statistics saved to evaluation_metrics.xlsx")

