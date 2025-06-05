import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json
import re
import wordninja
import os
import logging
try:
    import preprocessor as p
    TWEET_PREPROCESSOR_AVAILABLE = True
except ImportError:
    TWEET_PREPROCESSOR_AVAILABLE = False
    print("Error: 'tweet-preprocessor' not installed. Install it using 'pip install tweet-preprocessor'.")

# Set up logging
logging.basicConfig(filename='btsd_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Fallback preprocessing function
def fallback_clean(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002700-\U000027BF]', '', text)
    return text

# Preprocessing function
def data_clean(strings, norm_dict):
    try:
        if TWEET_PREPROCESSOR_AVAILABLE:
            clean_data = p.clean(strings)
            logging.info("Using tweet-preprocessor for cleaning.")
        else:
            clean_data = fallback_clean(strings)
            logging.warning("Using fallback cleaning due to missing tweet-preprocessor.")
    except Exception as e:
        clean_data = fallback_clean(strings)
        logging.error(f"Error in preprocessor.clean: {e}. Using fallback cleaning.")

    clean_data = re.sub(r"#SemST", "", clean_data, flags=re.IGNORECASE)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])
    clean_data = [j for i in clean_data for j in i]
    return clean_data

# Function to predict stance
def predict_stance(text, target, model, tokenizer, norm_dict, max_len=128, device=None):
    try:
        cleaned_text = ' '.join(data_clean(text, norm_dict))
        input_text = f"{target} [SEP] {cleaned_text}"

        encoding = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]

        return predicted_class
    except Exception as e:
        logging.error(f"Error predicting stance: {e}")
        return None

# Function to process a single test file and return results
def process_single_file(test_file, model, tokenizer, norm_dict, label_encoder, device, max_len=128, batch_size=32):
    # Load test dataset
    try:
        df = pd.read_csv(test_file, encoding='ISO-8859-1')
        logging.info(f"Test dataset {test_file} loaded with {len(df)} rows.")
    except UnicodeDecodeError:
        df = pd.read_csv(test_file, encoding='latin1')
        logging.info(f"Test dataset {test_file} loaded with {len(df)} rows using latin1 encoding.")
    except Exception as e:
        print(f"Error reading test file {test_file}: {e}")
        logging.error(f"Error reading test file {test_file}: {e}")
        return None, None

    # Verify required columns
    required_columns = ['tweet', 'Predicted Target', 'GT Stance']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Test dataset {test_file} must contain columns: {required_columns}")
        logging.error(f"Missing required columns in {test_file}: {required_columns}")
        return None, None

    # Filter valid stance labels
    valid_stances = ['FAVOR', 'AGAINST', 'NONE']
    original_len = len(df)
    df = df[df['GT Stance'].isin(valid_stances)].copy()
    if len(df) < original_len:
        print(f"Warning: {original_len - len(df)} rows with invalid stance labels were removed from {test_file}. Remaining rows: {len(df)}")
        logging.warning(f"Removed {original_len - len(df)} rows with invalid stance labels from {test_file}.")
    if df.empty:
        print(f"Error: No valid data after filtering stance labels in {test_file}.")
        logging.error(f"No valid data after filtering stance labels in {test_file}.")
        return None, None

    # Encode ground truth labels
    df['GT Stance'] = label_encoder.transform(df['GT Stance'])

    # Prepare output DataFrame
    results = []
    predicted_labels = []
    true_labels = []

    # Process in batches
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        logging.info(f"Processing batch {start_idx//batch_size + 1} for {test_file}: rows {start_idx} to {end_idx}")

        for idx, row in batch_df.iterrows():
            tweet = str(row['tweet'])
            target = row['Predicted Target']
            ground_truth = row['GT Stance']

            # Handle missing or invalid targets
            if pd.isna(target) or not isinstance(target, str):
                print(f"Warning: Skipping tweet {idx} in {test_file} due to invalid target: {target}")
                logging.warning(f"Skipping tweet {idx} in {test_file}: invalid target {target}")
                results.append({'idx': idx, 'tweet': tweet, 'target': target, 'predicted': 'N/A', 'ground_truth': label_encoder.inverse_transform([ground_truth])[0]})
                continue

            # Predict stance
            predicted_class = predict_stance(tweet, target, model, tokenizer, norm_dict, max_len, device)
            if predicted_class is None:
                print(f"Error predicting stance for tweet {idx} in {test_file}")
                results.append({'idx': idx, 'tweet': tweet, 'target': target, 'predicted': 'N/A', 'ground_truth': label_encoder.inverse_transform([ground_truth])[0]})
                continue

            predicted_stance = label_encoder.inverse_transform([predicted_class])[0]
            predicted_labels.append(predicted_class)
            true_labels.append(ground_truth)

            print(f"Tweet: {tweet} | Target: {target} | Predicted: {predicted_stance} | Ground Truth: {label_encoder.inverse_transform([ground_truth])[0]}")
            results.append({'idx': idx, 'tweet': tweet, 'target': target, 'predicted': predicted_stance, 'ground_truth': label_encoder.inverse_transform([ground_truth])[0]})

    # Calculate F1 macro score
    btsd_score = 0.0
    if predicted_labels and true_labels:
        btsd_score = f1_score(true_labels, predicted_labels, average='macro') * 100  # Scale to 0-100
        print(f"\nBTSD Score (F1 Macro) for {test_file}: {btsd_score:.2f}")
        logging.info(f"BTSD Score (F1 Macro) for {test_file}: {btsd_score:.2f}")
    else:
        print(f"Error: No valid predictions to compute F1 score for {test_file}.")
        logging.error(f"No valid predictions to compute F1 score for {test_file}.")

    return results, btsd_score

# Function to calculate BTSD score for multiple files
def calculate_btsd_score(test_files, model_dir, norm_dict_files, max_len=128, batch_size=32):
    # Verify model directory
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        logging.error(f"Model directory '{model_dir}' does not exist.")
        return None

    # Verify required model files
    required_files = ['config.json', 'model.safetensors', 'vocab.txt', 'bpe.codes', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer_config.json']
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
    if missing_files:
        print(f"Error: Missing files in model directory: {missing_files}")
        logging.error(f"Missing files in model directory: {missing_files}")
        return None

    # Check model.safetensors size
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    use_safetensors = True
    if os.path.isfile(safetensors_path):
        file_size = os.path.getsize(safetensors_path) / (1024 * 1024)  # Size in MB
        if file_size < 100:
            print(f"Error: 'model.safetensors' size is {file_size:.2f} MB, which is too small. It may be corrupted.")
            logging.error(f"'model.safetensors' size is {file_size:.2f} MB, likely corrupted.")
            return None
    else:
        pytorch_model_path = os.path.join(model_dir, 'pytorch_model.bin')
        if os.path.isfile(pytorch_model_path):
            print("Warning: 'model.safetensors' not found, using 'pytorch_model.bin'.")
            logging.warning("Using 'pytorch_model.bin' instead of 'model.safetensors'.")
            use_safetensors = False
        else:
            print(f"Error: Neither 'model.safetensors' nor 'pytorch_model.bin' found in '{model_dir}'.")
            logging.error(f"No valid model weights found in '{model_dir}'.")
            return None

    # Load normalization dictionary
    try:
        with open(norm_dict_files[0], "r") as f:
            data1 = json.load(f)
        data2 = {}
        with open(norm_dict_files[1], "r") as f:
            lines = f.readlines()
            for line in lines:
                row = line.split('\t')
                data2[row[0]] = row[1].rstrip()
        norm_dict = {**data1, **data2}
        logging.info("Normalization dictionary loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Normalization dictionary file not found: {e}")
        logging.error(f"Normalization dictionary file not found: {e}")
        return None

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, use_safetensors=use_safetensors)
        logging.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer from '{model_dir}': {e}")
        logging.error(f"Error loading model or tokenizer: {e}")
        return None

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Initialize label encoder
    valid_stances = ['FAVOR', 'AGAINST', 'NONE']
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_stances)
    logging.info("Label encoder initialized with labels: FAVOR, AGAINST, NONE")

    # Process each test file
    summary = []
    for test_file in test_files:
        print(f"\nProcessing file: {test_file}")
        logging.info(f"Starting processing for file: {test_file}")

        results, btsd_score = process_single_file(test_file, model, tokenizer, norm_dict, label_encoder, device, max_len, batch_size)

        if results is None:
            print(f"Skipping {test_file} due to errors.")
            summary.append({'file': test_file, 'btsd_score': None, 'status': 'Error'})
            continue

        # Save results to CSV
        output_file = f"/home/bhavani/Desktop/btsd/llama_simple_380_btsd/btsd_results_{os.path.basename(test_file)}"
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results for {test_file} saved to '{output_file}'")
        logging.info(f"Results for {test_file} saved to '{output_file}'")

        # Append to summary
        summary.append({'file': test_file, 'btsd_score': btsd_score, 'status': 'Success'})

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('/home/bhavani/Desktop/btsd/llama_simple_380_btsd/btsd_summary_1.csv', index=False)
    print("\nSummary of BTSD scores saved to '/home/bhavani/Desktop/qweem/btsd_summary.csv'")
    logging.info("Summary of BTSD scores saved to '/home/bhavani/Desktop/qween/btsd_summary.csv'")

    return summary

# Main execution
if __name__ == '__main__':
    # Define paths
    model_dir = '/home/bhavani/Desktop/eval/bertweet_stance_model-20250531T075637Z-1-001/bertweet_stance_model'
    test_files = [
          '/home/bhavani/Desktop/btsd/llama_3.1_cot_results/predictions_llama31_finetuned_cot_vast_filtered_im_20250529_171856.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_cot_results/predictions_llama31_finetuned_cot_vast_filtered_ex_20250529_171856.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_cot_results/predictions_llama31_finetuned_cot_tse_implicit_20250529_171856.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_cot_results/predictions_llama31_finetuned_cot_tse_explicit_20250529_171856.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_direct_results/predictions_llama31_finetuned_direct_vast_filtered_im_20250529_104740.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_direct_results/predictions_llama31_finetuned_direct_vast_filtered_ex_20250529_104740.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_direct_results/predictions_llama31_finetuned_direct_tse_implicit_20250529_044649.csv',
          '/home/bhavani/Desktop/btsd/llama_3.1_direct_results/predictions_llama31_finetuned_direct_tse_explicit_20250529_044649.csv',
          '/home/bhavani/Desktop/btsd/tot_results/predictions_llama31_finetuned_tot_vast_filtered_im_20250530_082612.csv',
          '/home/bhavani/Desktop/btsd/tot_results/predictions_llama31_finetuned_tot_vast_filtered_ex_20250530_082612.csv',
          '/home/bhavani/Desktop/btsd/tot_results/predictions_llama31_finetuned_tot_tse_implicit_20250530_082612.csv',
          '/home/bhavani/Desktop/btsd/tot_results/predictions_llama31_finetuned_tot_tse_explicit_20250530_082612.csv'
    ]
    norm_dict_files = [
        'noslang_data.json',
        'emnlp_dict.txt'
    ]

    # Install required packages
    os.system('pip install transformers==4.38.2 safetensors==0.4.2 pandas numpy scikit-learn wordninja tweet-preprocessor emoji==0.6.0')

    # Calculate BTSD scores
    summary = calculate_btsd_score(test_files, model_dir, norm_dict_files)