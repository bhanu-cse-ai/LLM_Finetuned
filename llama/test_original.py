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
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True
# Define models and test files
models = [
    {
        "name": "llama31_finetuned_cot",
        "path": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    }
]
test_files = ["tse_explicit.csv"]
valid_stances = ['FAVOR', 'AGAINST', 'NONE']
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output parser
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

# Process each model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

for model_info in models:
    model_name = model_info["name"]
    model_path = model_info["path"]
    
    print(f"\n=== Loading Model: {model_name} ===")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
        model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
        
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        continue

    # Process each test file
    for test_file in test_files:
        test_file_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n=== Processing Test File: {test_file_name} with Model: {model_name} ===")

        try:
            df = pd.read_csv(test_file)
            print(f"Loaded dataset with {len(df)} tweets")
            print(f"Missing tweets: {df['tweet'].isna().sum()}")
            if df["tweet"].isna().sum() > 0:
                df["tweet"] = df["tweet"].fillna("")
        except Exception as e:
            print(f"❌ Error loading test file {test_file}: {e}")
            continue

        predicted_targets = []
        predicted_stances = []
        raw_outputs = []

        for idx, tweet in enumerate(tqdm(df["tweet"], desc=f"Processing tweets ({test_file_name})")):
            prompt = f"""
Stance classification is the task of determining the expressed or implied opinion, or stance, of a statement toward a certain, specified target. Analyze the following tweet, generate the target for this tweet, and determine its stance towards the generated target. A target should be the topic on which the tweet is talking. The target can be a single word or a phrase, but its maximum length MUST be 4 words. If the stance is in favor of the target, write FAVOR, if it is against the target write AGAINST and if it is ambiguous, write NONE. The answer only has to be one of these three words: FAVOR, AGAINST, or NONE. Do not provide any explanation but you MUST give an output, do not leave any output blank.
Output Format:
Respond in the following JSON format:

{{
"Target": "<target word or phrase>",
"Stance": "FAVOR | AGAINST | NONE"
}}
Tweet:
{tweet}
Response:"""
            inputs = {k: v.to(device) for k, v in tokenizer([prompt], return_tensors="pt").items()}
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

        # Save predictions
        output_df = df.copy()
        output_df["Predicted Target"] = predicted_targets
        output_df["Predicted Stance"] = predicted_stances
        output_df["Raw Model Output"] = raw_outputs
        output_excel = f"predictions_{model_name}{test_file_name}{timestamp}.xlsx"
        output_csv = f"predictions_{model_name}{test_file_name}{timestamp}.csv"
        try:
            output_df.to_excel(output_excel, index=False)
            output_df.to_csv(output_csv, index=False)
            print(f"✅ Predictions saved to {output_excel} and {output_csv}")
        except Exception as e:
            print(f"❌ Error saving predictions to {output_excel} and {output_csv}: {e}")

        # Calculate and save metrics (unchanged from your original code)
        # ... [Your metrics calculation and saving code here] ...

        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        del model
        del tokenizer

print("\n=== Processing Complete ===")