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

# === Define Models, Prompts, and Test Files ===
models = [
    
    {
        "name": "qween3_finetuned_direct_detailed_combined",
        "path": r"/home/bhavani/Desktop/qween/qween3_finetuned_direct_detailed_comibined/checkpoint-50",
        "prompt": """Stance classification is the
task of determining the expressed or implied opinion, or stance, of a statement toward a certain,
specified target. Analyze the following tweet, generate the target for this tweet, and determine its
stance towards the generated target. A target
should be the topic on which the tweet is talking.
The target can be a single word or a phrase, but its
maximum length MUST be 4 words. If the stance is
in favor of the target, write FAVOR, if it is against
the target write AGAINST and if it is ambiguous,
write NONE. If the stance is in favor of the generated target, write FAVOR, if it is against the target
write AGAINST and if it is ambiguous, write NONE.
The answer only has to be one of these three words:
FAVOR, AGAINST, or NONE. Do not provide any
explanation but you MUST give an output, do not
leave any output blank. 

Output Format:
Respond in the following JSON format:

{{
"Target": "<target word or phrase>",
"Stance": "FAVOR | AGAINST | NONE"
}}

Tweet:
{tweet}
Response:"""
    }
   
]

test_files = [
    "/home/bhavani/Desktop/btsd/vast_filtered_im.csv",
    "/home/bhavani/Desktop/btsd/vast_filtered_ex.csv",
    "/home/bhavani/Desktop/btsd/tse_implicit.csv",
    "/home/bhavani/Desktop/btsd/tse_explicit.csv"
]

# Define valid stances
valid_stances = ['FAVOR', 'AGAINST', 'NONE']

# Base directory for output files
base_dir = "/home/bhavani/Desktop/qween"

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
    model_prompt = model_info["prompt"]
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
            prompt = model_prompt.format(tweet=tweet)
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