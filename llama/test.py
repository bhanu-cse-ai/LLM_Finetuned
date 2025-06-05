import re
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from collections import Counter
from datetime import datetime
import os

# === Define Models and Test Files ===
models = [
   
   
    {
        "name": "llama31_finetuned_tot_2",
        "path": r"llama31_finetuned_tot_comibined/checkpoint-50"
    }
]

test_files = [
    
    "vast_filtered_im.csv",
    "vast_filtered_ex.csv",
    "tse_implicit.csv",
    "tse_explicit.csv"

]

# Define valid stances
valid_stances = ['FAVOR', 'AGAINST', 'NONE']

# Base directory for output files
base_dir = "/home/bhavani/Desktop/LLM_Target_Stance/btsd/tot_2"

# Get timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Output Parser ===
def parse_model_output(output):
    try:
        json_block = re.search(r"\{.*?\}", output, re.DOTALL)
        if json_block:
            parsed = json.loads(json_block.group())
            return parsed.get("Target", "Not Found"), parsed.get("Stance", "Not Found")
        else:
            print(f"❌ No JSON block found in output: {output}")
            return "No JSON Found", output.strip()
    except Exception as e:
        print(f"❌ Error parsing output: {output}\nError: {e}")
        return "Error", output.strip()

# === Process Each Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

for model_info in models:
    model_name = model_info["name"]
    model_path = model_info["path"]
    print(f"\n=== Loading Model: {model_name} ===")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        model.to(device)
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        continue

    # === Process Each Test File ===
    for test_file in test_files:
        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n=== Processing Test File: {test_file_name} with Model: {model_name} ===")

        # Load dataset
        try:
            df = pd.read_csv(test_file)
        except Exception as e:
            print(f"❌ Error loading test file {test_file}: {e}")
            continue

        # Validate dataset
        print(f"Loaded dataset with {len(df)} tweets")
        print(f"Missing tweets: {df['tweet'].isna().sum()}")
        if df["tweet"].isna().sum() > 0:
            print(" Warning: Found missing tweets. Filling with empty string.")
            df["tweet"] = df["tweet"].fillna("")

        predicted_targets = []
        predicted_stances = []
        raw_outputs = []

        # === Run Prediction for Each Tweet ===
        for idx, tweet in enumerate(tqdm(df["tweet"], desc=f"Processing tweets ({test_file_name})")):
            prompt = f""" Given a tweet, generate multiple hypotheses for what entity, theme, or subject the tweet expresses an opinion about.
Thought 1A (Explicit Target): Identify the primary topic explicitly mentioned in the tweet (e.g., proper nouns, organizations, or named policies).


Thought 1B (Implied Target): Infer a possible implied or secondary topic based on background context, discourse patterns, or sentiment direction.


Thought 1C (Hashtag-Derived Target): Derive a target from associated hashtags or metadata cues.
Evaluate the three target candidates using the following branch-scoring criteria:
Relevance: How central is the candidate to the tweet’s semantic focus?


Specificity: Does the candidate refer to a precise topic rather than a broad category?


Centrality: Is the candidate the likely object of opinion or sentiment in the tweet?
Choose the candidate with the highest composite alignment across the above criteria.For the selected target, generate three possible stance hypotheses:
Thought 2A – FAVOR: Assume the tweet expresses positive sentiment or support toward the target.


Thought 2B – AGAINST: Assume the tweet expresses negative sentiment or opposition toward the target.


Thought 2C – NONE: Assume the tweet expresses a neutral, ambiguous, or non-opinionated statement about the target.
Evaluate each stance hypothesis using linguistic and contextual cues, including:
Tone: Emotionality or affective charge like positive or negative or neutral.


Wording: Lexical patterns indicating support or disapproval.


Contextual Framing: Pragmatic interpretation based on world knowledge or sarcasm.


Hashtag Sentiment: Hashtags that suggest polarity.

tweet:
{tweet} 

Output Format:
Respond in the following JSON format:

{{

"Target": "<target word or phrase>",
"Stance": "FAVOR | AGAINST | NONE"

}}

Response:"""
            inputs = tokenizer([prompt], return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response_part = generated_text.split("Response:")[-1].strip() if "Response:" in generated_text else generated_text.strip()
            
            if idx < 5:
                print(f"\nTweet {idx + 1}: {tweet}")
                print(f"Generated Text: {generated_text}")
                print(f"Response Part: {response_part}")
            
            target, stance = parse_model_output(response_part)
            
            if idx < 5:
                print(f"Parsed Target: {target}, Stance: {stance}")
            
            predicted_targets.append(target)
            predicted_stances.append(stance)
            raw_outputs.append(generated_text)

        # === Save Predictions ===
        output_df = df.copy()
        output_df["Predicted Target"] = predicted_targets
        output_df["Predicted Stance"] = predicted_stances
        output_df["Raw Model Output"] = raw_outputs

        # Generate filenames
        output_excel = f"{base_dir}/predictions_{model_name}_{test_file_name}_{timestamp}.xlsx"
        output_csv = f"{base_dir}/predictions_{model_name}_{test_file_name}_{timestamp}.csv"

        try:
            output_df.to_excel(output_excel, index=False)
            output_df.to_csv(output_csv, index=False)
            print(f"✅ Predictions saved to {output_excel} and {output_csv}")
        except Exception as e:
            print(f"❌ Error saving predictions to {output_excel} and {output_csv}: {e}")

        # === Calculate Metrics ===
        print(f"\n=== Calculating Metrics for {model_name} on {test_file_name} ===")
        
        # Filter out rows with invalid or missing stance values
        filtered_df = output_df[
            (output_df["GT Stance"].isin(valid_stances)) & 
            (output_df["Predicted Stance"].isin(valid_stances)) &
            (output_df["GT Stance"].notna()) &
            (output_df["Predicted Stance"].notna())
        ]

        if filtered_df.empty:
            print(f"❌ No valid data after filtering for {model_name} on {test_file_name}")
            continue

        # Extract true and predicted values
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
        try:
            tg_accuracy = accuracy_score(true_targets, predicted_targets)
            tg_f1_macro = f1_score(true_targets, predicted_targets, average='macro', zero_division=0)
            tg_f1_micro = f1_score(true_targets, predicted_targets, average='micro', zero_division=0)
            tg_f1_weighted = f1_score(true_targets, predicted_targets, average='weighted', zero_division=0)
            tg_precision_macro = precision_score(true_targets, predicted_targets, average='macro', zero_division=0)
            tg_recall_macro = recall_score(true_targets, predicted_targets, average='macro', zero_division=0)
        except Exception as e:
            print(f"❌ Error calculating target detection metrics: {e}")
            tg_accuracy = tg_f1_macro = tg_f1_micro = tg_f1_weighted = tg_precision_macro = tg_recall_macro = 0

        # Calculate Stance Detection metrics
        try:
            sd_accuracy = accuracy_score(true_stances, predicted_stances)
            sd_f1_macro = f1_score(true_stances, predicted_stances, average='macro', zero_division=0)
            sd_f1_micro = f1_score(true_stances, predicted_stances, average='micro', zero_division=0)
            sd_f1_weighted = f1_score(true_stances, predicted_stances, average='weighted', zero_division=0)
            sd_precision_macro = precision_score(true_stances, predicted_stances, average='macro', zero_division=0)
            sd_recall_macro = recall_score(true_stances, predicted_stances, average='macro', zero_division=0)
        except Exception as e:
            print(f"❌ Error calculating stance detection metrics: {e}")
            sd_accuracy = sd_f1_macro = sd_f1_micro = sd_f1_weighted = sd_precision_macro = sd_recall_macro = 0

        # Calculate true positives and ground truth counts
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
            print(f"❌ Error in classification_report for {model_name} on {test_file_name}: {e}")
            sd_classification_report = None

        # Print evaluation metrics
        print(f"\nEvaluation Metrics for {model_name} on {test_file_name}:")
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

        # Print per-class metrics
        if sd_classification_report:
            print("\nPer-Class Stance Detection Metrics:")
            for stance in valid_stances:
                metrics = sd_classification_report[stance]
                print(f"  {stance}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1: {metrics['f1-score']:.4f}")
                print(f"    Support: {metrics['support']}")

        # === Save Metrics ===
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
        metrics_df = pd.DataFrame(metrics)

        # Save true positives and ground truth counts
        stance_stats_df = pd.DataFrame({
            "Stance Class": list(true_positives.keys()),
            "True Positives": list(true_positives.values()),
            "Ground Truth Count": [ground_truth_counts[stance] for stance in true_positives.keys()]
        })

        # Generate metrics filename
        metrics_excel = f"{base_dir}/metrics_{model_name}_{test_file_name}_{timestamp}.xlsx"

        # Save to Excel
        try:
            with pd.ExcelWriter(metrics_excel, engine='openpyxl') as writer:
                metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                stance_stats_df.to_excel(writer, sheet_name="Stance Statistics", index=False)
            print(f"✅ Metrics and stance statistics saved to {metrics_excel}")
        except Exception as e:
            print(f"❌ Error saving metrics to {metrics_excel}: {e}")

    # Clear model from memory
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n=== Processing Complete ===")