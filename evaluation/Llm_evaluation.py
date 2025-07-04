# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vIhBW7ovEnKEm2qRBKXlXMurDwbLgF4L
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List
from concurrent.futures import ThreadPoolExecutor
import torch
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import time
from requests.exceptions import ConnectionError
from collections import defaultdict  # Added import

# Suppress Hugging Face and TensorFlow warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_TOKEN"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Configure logging
logging.basicConfig(
    filename="evaluation_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configure two Gemini API keys
API_KEY_1 = "AIzaSyD01UOVTs7A8NthHR3krBLAcYlUL251wJk"
API_KEY_2 = "AIzaSyAek7G4ndj8RG-npSK8eL1U7i18s8QCsKw"  # Replace with your second API key
genai.configure(api_key=API_KEY_1)
model_1 = genai.GenerativeModel('gemini-1.5-pro-001')
genai.configure(api_key=API_KEY_2)
model_2 = genai.GenerativeModel('gemini-1.5-pro-001')

# Load SBERT model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_cache = defaultdict(lambda: None)

def sanitize_text(text: str) -> str:
    replacements = {
        "persecuting": "criticizing",
        "discrimination": "bias",
        "traitors": "opponents",
        "death": "harm",
        "segregated": "divided"
    }
    for sensitive, neutral in replacements.items():
        text = text.replace(sensitive, neutral)
    return text

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((GoogleAPIError, ConnectionError))
)
def batch_expand_ground_truth(gt_targets: List[str], contexts: List[str], model) -> List[List[str]]:
    sanitized_contexts = [sanitize_text(ctx) for ctx in contexts]
    prompt = "Generate 2 alternative expressions for each of the following targets with their contexts:\n"
    for i, (gt, ctx) in enumerate(zip(gt_targets, sanitized_contexts)):
        prompt += f"{i+1}. Target: '{gt}', Context: '{ctx}'\n"
    try:
        response = model.generate_content(prompt)
        time.sleep(1)
        if not response.candidates:
            logging.info(f"[Batch GT Expansion Blocked] {len(gt_targets)} targets | Model: {model._model_name}")
            return [[gt] for gt in gt_targets]
        outputs = response.text.split('\n\n')
        expanded = []
        for i, output in enumerate(outputs[:len(gt_targets)]):
            phrases = output.strip().split('\n')
            expanded.append([p.strip() for p in phrases if p.strip()] or [gt_targets[i]])
        expanded += [[gt] for gt in gt_targets[len(expanded):]]
        if len(expanded) != len(gt_targets):
            logging.error(f"[Batch GT Expansion Length Mismatch] Expected: {len(gt_targets)}, Got: {len(expanded)} | Response: {response.text[:500]}...")
        return expanded
    except (GoogleAPIError, ConnectionError) as e:
        logging.error(f"[Batch GT Expansion Failed] Error: {str(e)} | Model: {model._model_name}")
        return [[gt] for gt in gt_targets]
    except Exception as e:
        logging.error(f"[Batch GT Expansion Unexpected Error] Error: {str(e)} | Model: {model._model_name} | Response: {response.text[:500] if 'response' in locals() else 'No response'}...")
        return [[gt] for gt in gt_targets]

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((GoogleAPIError, ConnectionError))
)
def batch_llm_boolean_relevance(tweets: List[str], predicted_targets: List[str], model) -> List[bool]:
    sanitized_tweets = [sanitize_text(tweet) for tweet in tweets]
    prompt = "Answer Yes/No for the relevance of each predicted target to its tweet:\n"
    for i, (tweet, pred) in enumerate(zip(sanitized_tweets, predicted_targets)):
        prompt += f"{i+1}. Tweet: '{tweet}', Predicted Target: '{pred}'\n"
    try:
        response = model.generate_content(prompt)
        time.sleep(1)
        if not response.candidates:
            logging.info(f"[Batch LLM Relevance Blocked] {len(tweets)} tweets | Model: {model._model_name}")
            return [False] * len(tweets)
        answers = response.text.strip().split('\n')
        answers = answers[:len(tweets)]
        answers += ["No"] * (len(tweets) - len(answers))
        return ["yes" in ans.lower() for ans in answers]
    except (GoogleAPIError, ConnectionError) as e:
        logging.error(f"[Batch LLM Relevance Failed] Error: {str(e)} | Model: {model._model_name}")
        return [False] * len(tweets)
    except Exception as e:
        logging.error(f"[Batch LLM Relevance Unexpected Error] Error: {str(e)} | Model: {model._model_name} | Response: {response.text[:500] if 'response' in locals() else 'No response'}...")
        return [False] * len(tweets)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((GoogleAPIError, ConnectionError))
)
def batch_llm_likert_score(tweets: List[str], gt_targets: List[str], predicted_targets: List[str], model) -> List[int]:
    sanitized_tweets = [sanitize_text(tweet) for tweet in tweets]
    prompt = "Score 1-5 for how well each predicted target captures the tweet compared to ground truth:\n"
    for i, (tweet, gt, pred) in enumerate(zip(sanitized_tweets, gt_targets, predicted_targets)):
        prompt += f"{i+1}. Tweet: '{tweet}', GT: '{gt}', Predicted: '{pred}'\n"
    try:
        response = model.generate_content(prompt)
        time.sleep(1)
        if not response.candidates:
            logging.info(f"[Batch LLM Likert Blocked] {len(tweets)} tweets | Model: {model._model_name}")
            return [3] * len(tweets)
        scores = response.text.strip().split('\n')
        scores = scores[:len(tweets)]
        parsed_scores = []
        for score in scores:
            digits = [s for s in score if s.isdigit()]
            parsed_scores.append(int(digits[0]) if digits else 3)
        parsed_scores += [3] * (len(tweets) - len(parsed_scores))
        return parsed_scores
    except (GoogleAPIError, ConnectionError) as e:
        logging.error(f"[Batch LLM Likert Failed] Error: {str(e)} | Model: {model._model_name}")
        return [3] * len(tweets)
    except Exception as e:
        logging.error(f"[Batch LLM Likert Unexpected Error] Error: {str(e)} | Model: {model._model_name} | Response: {response.text[:500] if 'response' in locals() else 'No response'}...")
        return [3] * len(tweets)

def semantic_similarity(predicted: str, expanded_gt_list: List[str]) -> float:
    if not expanded_gt_list:
        return 0.0
    if embedding_cache[predicted] is None:
        embedding_cache[predicted] = embedding_model.encode(predicted, convert_to_tensor=True)
    pred_emb = embedding_cache[predicted]
    gt_embs = []
    for gt in expanded_gt_list:
        if embedding_cache[gt] is None:
            embedding_cache[gt] = embedding_model.encode(gt, convert_to_tensor=True)
        gt_embs.append(embedding_cache[gt])
    gt_embs = torch.stack(gt_embs)
    similarities = util.pytorch_cos_sim(pred_emb, gt_embs)[0].cpu().numpy()
    return float(np.max(similarities))

def process_chunk(chunk, model, model_name):
    chunk_results = []
    tweets = chunk['tweet'].tolist()
    gt_targets = chunk['GT Target'].tolist()
    predicted_targets = chunk['Predicted Target'].tolist()
    contexts = tweets
    chunk_len = len(tweets)

    try:
        expanded_gts = batch_expand_ground_truth(gt_targets, contexts, model)
        relevances = batch_llm_boolean_relevance(tweets, predicted_targets, model)
        likerts = batch_llm_likert_score(tweets, gt_targets, predicted_targets, model)

        if not (len(expanded_gts) == len(relevances) == len(likerts) == chunk_len):
            logging.error(f"[Chunk Length Mismatch] Tweets: {chunk_len}, Expanded GTs: {len(expanded_gts)}, Relevances: {len(relevances)}, Likerts: {len(likerts)}")
            expanded_gts = expanded_gts[:chunk_len] + [[gt] for gt in gt_targets[len(expanded_gts):chunk_len]]
            relevances = relevances[:chunk_len] + [False] * (chunk_len - len(relevances))
            likerts = likerts[:chunk_len] + [3] * (chunk_len - len(likerts))

        for i, (tweet, gt, pred, expanded_gt, relevance, likert) in enumerate(zip(tweets, gt_targets, predicted_targets, expanded_gts, relevances, likerts)):
            try:
                sim_score = semantic_similarity(pred, expanded_gt)
                sim_scaled = sim_score * 50
                if sim_score >= 0.9:
                    final_score = sim_score * 100
                    chunk_results.append({
                        "Similarity Score": round(sim_score, 3),
                        "LLM Relevance": True,
                        "Likert Score": 5,
                        "Final Score": round(final_score, 2),
                        "Decision Path": 'High Similarity (Direct Scaling)',
                        "Model Used": model_name
                    })
                else:
                    final_score = sim_scaled
                    decision_type = 'Similarity Only'
                    if relevance:
                        final_score += 20
                        final_score += (likert / 5) * 30
                        decision_type = 'LLM Evaluated'
                    else:
                        decision_type = 'Low Similarity, Irrelevant'
                    chunk_results.append({
                        "Similarity Score": round(sim_score, 6),
                        "LLM Relevance": relevance,
                        "Likert Score": likert,
                        "Final Score": round(final_score, 6),
                        "Decision Path": decision_type,
                        "Model Used": model_name
                    })
            except Exception as e:
                logging.error(f"[Row Evaluation Error] Tweet: {tweet} | GT: {gt} | Predicted: {pred} | Error: {str(e)} | Model: {model_name}")
                chunk_results.append({
                    "Similarity Score": 0,
                    "LLM Relevance": False,
                    "Likert Score": 0,
                    "Final Score": 0,
                    "Decision Path": f"Error: {str(e)}",
                    "Model Used": model_name
                })
    except (GoogleAPIError, ConnectionError) as e:
        logging.error(f"[Batch Evaluation Failed] Chunk | Error: {str(e)} | Model: {model_name}")
        for tweet, gt, pred in zip(tweets, gt_targets, predicted_targets):
            chunk_results.append({
                "Similarity Score": 0,
                "LLM Relevance": False,
                "Likert Score": 0,
                "Final Score": 0,
                "Decision Path": f"Error: Google API Error - {str(e)}",
                "Model Used": model_name
            })
    return chunk_results

def evaluate_batch_to_csv(input_files: List[str], summary_file: str = "summary_results.csv", chunk_size: int = 3):
    summaries = []

    for input_file in input_files:
        logging.info(f"Processing file: {input_file}")
        try:
            # Read the input CSV
            df = pd.read_csv(input_file)
            df = df.dropna(subset=['tweet', 'GT Target', 'Predicted Target'])
            df = df[df['tweet'].str.strip() != '']
            original_len = len(df)

            # Check if evaluation columns already exist
            if 'Similarity Score' in df.columns:
                logging.info(f"Skipping {input_file}: Evaluation columns already present")
                avg_sim_score = df['Similarity Score'].mean()
                avg_relevance = df['LLM Relevance'].astype(int).mean()
                avg_likert = df['Likert Score'].mean()
                avg_final = df['Final Score'].mean()
                summaries.append({
                    "File Name": input_file,
                    "Avg Similarity Score": round(avg_sim_score, 3),
                    "Avg LLM Relevance": round(avg_relevance, 3),
                    "Avg Likert Score": round(avg_likert, 3),
                    "Avg Final Score": round(avg_final, 3)
                })
                continue

            # Initialize lists for new columns
            sim_scores = []
            llm_relevances = []
            likert_scores = []
            final_scores = []
            decision_paths = []
            models_used = []

            # Split DataFrame into two parts
            half = len(df) // 2
            df1 = df.iloc[:half]
            df2 = df.iloc[half:]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future1 = executor.submit(process_chunk, df1, model_1, "Gemini-Key1")
                results1 = future1.result()
                time.sleep(2)
                future2 = executor.submit(process_chunk, df2, model_2, "Gemini-Key2")
                results2 = future2.result()

            # Combine results
            results = results1 + results2

            # Validate result length
            if len(results) != original_len:
                logging.error(f"[Result Length Mismatch] File: {input_file} | Expected: {original_len}, Got: {len(results)}")
                results += [{
                    "Similarity Score": 0,
                    "LLM Relevance": False,
                    "Likert Score": 0,
                    "Final Score": 0,
                    "Decision Path": "Error: Missing result due to processing failure",
                    "Model Used": "None"
                }] * (original_len - len(results))

            # Add results to DataFrame
            for result in results:
                sim_scores.append(result["Similarity Score"])
                llm_relevances.append(result["LLM Relevance"])
                likert_scores.append(result["Likert Score"])
                final_scores.append(result["Final Score"])
                decision_paths.append(result["Decision Path"])
                models_used.append(result["Model Used"])

            # Append new columns to DataFrame
            df['Similarity Score'] = sim_scores
            df['LLM Relevance'] = llm_relevances
            df['Likert Score'] = likert_scores
            df['Final Score'] = final_scores
            df['Decision Path'] = decision_paths
            df['Model Used'] = models_used

            # Save back to the input file
            try:
                df.to_csv(input_file, index=False)
                logging.info(f"Results appended to {input_file}")
            except Exception as e:
                logging.error(f"[File Save Error] Failed to save to {input_file} | Error: {str(e)}")
                raise

            # Compute averages for summary
            avg_sim_score = df['Similarity Score'].mean()
            avg_relevance = df['LLM Relevance'].astype(int).mean()
            avg_likert = df['Likert Score'].mean()
            avg_final = df['Final Score'].mean()

            # Add to summaries
            summaries.append({
                "File Name": input_file,
                "Avg Similarity Score": round(avg_sim_score, 3),
                "Avg LLM Relevance": round(avg_relevance, 3),
                "Avg Likert Score": round(avg_likert, 3),
                "Avg Final Score": round(avg_final, 3)
            })

        except Exception as e:
            logging.error(f"[File Processing Error] Failed to process {input_file} | Error: {str(e)}")
            summaries.append({
                "File Name": input_file,
                "Avg Similarity Score": np.nan,
                "Avg LLM Relevance": np.nan,
                "Avg Likert Score": np.nan,
                "Avg Final Score": np.nan
            })
            continue

    # Save summary to a separate CSV
    try:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Summary saved to {summary_file}")
    except Exception as e:
        logging.error(f"[Summary Save Error] Failed to save to {summary_file} | Error: {str(e)}")
        raise

    return True

# Example Usage
input_files = [
    "/content/predictions_qween3_finetuned_direct_detailed_combined_tse_explicit_20250530_233338.csv",
    "/content/predictions_qween3_finetuned_direct_detailed_combined_tse_implicit_20250530_233338.csv",
    "/content/predictions_qween3_finetuned_direct_detailed_combined_vast_filtered_ex_20250530_233338.csv",
    "/content/predictions_qween3_finetuned_direct_detailed_combined_vast_filtered_im_20250530_233338.csv"
]
evaluate_batch_to_csv(input_files, summary_file="summary_results_llama_tot_tse_im-&_ex.csv", chunk_size=3)