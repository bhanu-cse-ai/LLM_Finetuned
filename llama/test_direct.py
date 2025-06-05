import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import json
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

def predict_from_model(tweet, model, tokenizer, max_new_tokens=150):
    prompt = f"""Analyse the text and generate stance for the given tweet by predicting its target.
Output Format:
Respond in the following JSON format:

{{
  "tweet": "<original tweet text>",
  "Target": "<target word or phrase>",
  "Stance": "FAVOR | AGAINST | NONE"
}}

Tweet: {tweet}  
Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )

    raw_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        response = raw_result.split("Response:")[-1].strip()
        json_output = json.loads(response)
        target = json_output.get("Target", "").strip()
        stance = json_output.get("Stance", "").strip().upper()
        if stance not in ["FAVOR", "AGAINST", "NONE"]:
            stance = "NONE"
    except (json.JSONDecodeError, IndexError):
        target, stance = "", "NONE"

    return target, stance, raw_result

def evaluate_file(file_path, model, tokenizer, model_name):
    # Load dataset
    test_df = pd.read_csv(file_path)
    
    # Store predictions and actuals
    pred_targets = []
    pred_stances = []
    actual_targets = []
    actual_stances = []
    stored_outputs = []

    # Run predictions
    for i in tqdm(range(len(test_df)), desc=f"Processing {os.path.basename(file_path)}"):
        tweet = test_df.loc[i, 'tweet']
        true_target = test_df.loc[i, 'GT Target']
        true_stance = test_df.loc[i, 'GT Stance'].upper()

        pred_target, pred_stance, raw_output = predict_from_model(tweet, model, tokenizer)

        actual_targets.append(true_target)
        actual_stances.append(true_stance)
        pred_targets.append(pred_target)
        pred_stances.append(pred_stance)

        stored_outputs.append({
            'Tweet': tweet,
            'True Target': true_target,
            'Predicted Target': pred_target,
            'True Stance': true_stance,
            'Predicted Stance': pred_stance,
            'Raw Model Output': raw_output
        })

        if (i + 1) % 10 == 0:
            print(f"\n--- Predictions for Rows {i - 9} to {i} in {os.path.basename(file_path)} ---")
            for row in stored_outputs[-10:]:
                print(row)

        if true_stance != pred_stance:
            print(f"[Mismatch in {os.path.basename(file_path)}] Tweet: {tweet}\n  GT: {true_stance}, Pred: {pred_stance}\n")

    # Evaluation metrics
    print(f"\n--- Evaluation Metrics for Stance Detection ({os.path.basename(file_path)}) ---")
    accuracy = accuracy_score(actual_stances, pred_stances)
    precision_macro = precision_score(actual_stances, pred_stances, average='macro', zero_division=0)
    recall_macro = recall_score(actual_stances, pred_stances, average='macro', zero_division=0)
    f1_macro = f1_score(actual_stances, pred_stances, average='macro', zero_division=0)
    precision_micro = precision_score(actual_stances, pred_stances, average='micro', zero_division=0)
    recall_micro = recall_score(actual_stances, pred_stances, average='micro', zero_division=0)
    f1_micro = f1_score(actual_stances, pred_stances, average='micro', zero_division=0)
    precision_weighted = precision_score(actual_stances, pred_stances, average='weighted', zero_division=0)
    recall_weighted = recall_score(actual_stances, pred_stances, average='weighted', zero_division=0)
    f1_weighted = f1_score(actual_stances, pred_stances, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")

    # Per-class metrics
    print("\n--- Per-Class Metrics ---")
    class_report = classification_report(actual_stances, pred_stances, labels=["FAVOR", "NONE", "AGAINST"], zero_division=0, output_dict=True)
    for label in ["FAVOR", "NONE", "AGAINST"]:
        print(f"\nClass: {label}")
        print(f"Precision: {class_report[label]['precision']:.4f}")
        print(f"Recall: {class_report[label]['recall']:.4f}")
        print(f"F1 Score: {class_report[label]['f1-score']:.4f}")

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.basename(file_path).split('.')[0]
    output_file = f"{model_name}_{file_name}_predictions_{timestamp}.csv"
    output_df = pd.DataFrame(stored_outputs)
    output_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

def main(models):
    input_files = [
        "tse_implicit.csv",
        "tse_explicit.csv",
        "vast_filtered_ex.csv",
        "vast_filtered_im.csv"
    ]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

        # Evaluate each file with the loaded model
        for file_path in input_files:
            if os.path.exists(file_path):
                evaluate_file(file_path, model, tokenizer, model_name)
            else:
                print(f"File not found: {file_path}")
        
        # Clean up to free memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example model list
    models = [
        {"name": "lama_direcct_plane", "path": "llama31_finetuned_direct_plane/checkpoint-75"}
    ]
    main(models)