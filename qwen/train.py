from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit",
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
custom_prompt = """Analyse the given tweet  {tweet}  and and generate stance for the given tweet by predicting its target 
Output Format:
Respond in the following JSON format:

{{
"Target": "<target word or phrase>",
"Stance": "FAVOR | AGAINST | NONE"
}}

Tweet:


Response: 
{{'Target': "{target}",'Stance': "{stance}"}}

"""
EOS_TOKEN = tokenizer.eos_token

def format_prompt_stance_task(examples):
    tweets  = examples["tweet"]
    targets = examples["target"]
    stances = examples["stance"]
    texts = []

    for tweet, target, stance in zip(tweets, targets, stances):
        prompt = custom_prompt.format(tweet = tweet, target = target, stance = stance) + EOS_TOKEN
        texts.append(prompt)

    return { "text": texts }
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("csv", data_files="new_combined_dataset.csv", split="train")

# Apply the formatting function
dataset = dataset.map(format_prompt_stance_task, batched=True)
print("\n✅ Sample formatted training examples:")
for i in range(min(3, len(dataset))):
    print(f"\nExample {i+1}:\n{dataset[i]['text']}\n")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


training_args = TrainingArguments(
    output_dir="./qween3_fined_direct_comibined",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    max_steps=50,  # reduced
    learning_rate=2e-4,  # lower LR
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    overwrite_output_dir=True,
    num_train_epochs=5
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args
)


trainer.train()

# ✅ Log final training loss
final_loss = trainer.state.log_history[-1].get("loss", None)
print(f"\n✅ Final training loss: {final_loss:.4f}" if final_loss else "⚠ Final loss not found.")