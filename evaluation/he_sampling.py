import pandas as pd
import random
import os

def load_prediction_files(file_paths):
    """Load prediction files and return a dictionary of DataFrames."""
    dataframes = {}
    for key, path in file_paths.items():
        if os.path.exists(path):
            dataframes[key] = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"File {path} not found.")
    return dataframes

def get_common_indices(dfs):
    """Get common indices across a list of DataFrames."""
    common_indices = set(dfs[0].index)
    for df in dfs[1:]:
        common_indices.intersection_update(df.index)
    return list(common_indices)

def stratified_sample(df, common_indices, n_samples, label_column='GT Stance'):
    """Sample rows to balance the specified label column."""
    # Filter DataFrame to only include rows with common indices
    df = df.loc[df.index.isin(common_indices)]
    
    # Group by label
    grouped = df.groupby(label_column)
    labels = ['FAVOR', 'AGAINST', 'NONE']
    n_per_label = n_samples // len(labels)  # Target samples per label
    remainder = n_samples % len(labels)     # Handle remainder if n_samples not divisible by 3
    
    sampled_dfs = []
    for label in labels:
        if label in grouped.groups:
            group = grouped.get_group(label)
            # Sample min(n_per_label, available rows)
            sample_size = min(n_per_label + (1 if remainder > 0 else 0), len(group))
            if sample_size > 0:
                sampled_dfs.append(group.sample(n=sample_size, random_state=42))
            remainder = max(0, remainder - 1)  # Decrease remainder after assigning
        else:
            print(f"Warning: Label {label} not found in dataset.")
    
    # Concatenate sampled DataFrames
    result = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
    
    # If fewer than n_samples due to limited rows, adjust by sampling more from available labels
    if len(result) < n_samples and len(result) > 0:
        remaining = n_samples - len(result)
        available_indices = list(set(common_indices) - set(result.index))
        if available_indices:
            extra_sample = df.loc[available_indices].sample(n=min(remaining, len(available_indices)), random_state=42)
            result = pd.concat([result, extra_sample])
    
    return result

def main():
    # Define file paths for each model's predictions on each dataset
    file_paths = {
        'tse_explicit_qwen': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_qween3_finetuned_cot_combined_tse_explicit_20250530_233338.csv',
        'tse_explicit_llama': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_llama31_finetuned_cot_combined_tse_explicit_20250529_171856.csv',
        'tse_explicit_gemini': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/gemini_cot_tse_ex_with_similarity.csv',
        'tse_implicit_qwen': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_qween3_finetuned_cot_combined_tse_implicit_20250530_233338.csv',
        'tse_implicit_llama': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_llama31_finetuned_cot_combined_tse_implicit_20250529_171856.csv',
        'tse_implicit_gemini': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/gemini_cot_tse_im_with_similarity.csv',
        'vast_explicit_qwen': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_qween3_finetuned_cot_combined_vast_filtered_ex_20250530_233338.csv',
        'vast_explicit_llama': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_llama31_finetuned_cot_combined_vast_filtered_ex_20250529_171856.csv',
        'vast_explicit_gemini': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/gemini_cot_vast_ex_with_similarity.csv',
        'vast_implicit_qwen': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_qween3_finetuned_cot_combined_vast_filtered_im_20250530_233338.csv',
        'vast_implicit_llama': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/predictions_llama31_finetuned_cot_combined_vast_filtered_im_20250529_171856.csv',
        'vast_implicit_gemini': '/home/bhavani/Desktop/LLM_Target_Stance/Human_Evalution/gemini_cot_vast_im_with_similarity.csv'
    }

    # Load all prediction files
    dataframes = load_prediction_files(file_paths)

    # Sample 150 from explicit datasets and 100 from implicit datasets for each model
    samples = []
    for dataset in ['tse_explicit', 'vast_explicit']:
        # Get DataFrames for the current dataset
        dfs = [
            dataframes[f'{dataset}_qwen'],
            dataframes[f'{dataset}_llama'],
            dataframes[f'{dataset}_gemini']
        ]
        # Find common indices across all DataFrames
        common_indices = get_common_indices(dfs)
        if not common_indices:
            print(f"Warning: No common indices for {dataset}. Skipping.")
            continue

        # Stratified sampling based on GT Stance from Qwen DataFrame
        df_qwen = stratified_sample(dataframes[f'{dataset}_qwen'], common_indices, 150)
        sampled_indices = df_qwen.index.tolist()

        # Extract rows using sampled indices for other models
        df_llama = dataframes[f'{dataset}_llama'].loc[sampled_indices]
        df_gemini = dataframes[f'{dataset}_gemini'].loc[sampled_indices]

        # Combine data
        combined = pd.DataFrame({
            'Dataset': [dataset] * len(df_qwen),  # Add dataset label
            'Tweet': df_qwen['tweet'],
            'GT_Target': df_qwen['GT Target'],
            'Qwen_Prediction': df_qwen['Predicted Target'],
            'Llama_Prediction': df_llama['Predicted Target'],
            'Gemini_Prediction': df_gemini['predicted_target'],
            'GT_Stance': df_qwen['GT Stance']  # Include GT Stance for final balancing
        })
        samples.append(combined)

    for dataset in ['tse_implicit', 'vast_implicit']:
        # Get DataFrames for the current dataset
        dfs = [
            dataframes[f'{dataset}_qwen'],
            dataframes[f'{dataset}_llama'],
            dataframes[f'{dataset}_gemini']
        ]
        # Find common indices across all DataFrames
        common_indices = get_common_indices(dfs)
        if not common_indices:
            print(f"Warning: No common indices for {dataset}. Skipping.")
            continue

        # Stratified sampling based on GT Stance from Qwen DataFrame
        df_qwen = stratified_sample(dataframes[f'{dataset}_qwen'], common_indices, 100)
        sampled_indices = df_qwen.index.tolist()

        # Extract rows using sampled indices for other models
        df_llama = dataframes[f'{dataset}_llama'].loc[sampled_indices]
        df_gemini = dataframes[f'{dataset}_gemini'].loc[sampled_indices]

        # Combine data
        combined = pd.DataFrame({
            'Dataset': [dataset] * len(df_qwen),  # Add dataset label
            'Tweet': df_qwen['tweet'],
            'GT_Target': df_qwen['GT Target'],
            'Qwen_Prediction': df_qwen['Predicted Target'],
            'Llama_Prediction': df_llama['Predicted Target'],
            'Gemini_Prediction': df_gemini['predicted_target'],
            'GT_Stance': df_qwen['GT Stance']  # Include GT Stance for final balancing
        })
        samples.append(combined)

    # Concatenate all samples
    final_samples = pd.concat(samples, ignore_index=True)

    # Ensure exactly 500 samples with equal ratio of GT Stance labels
    if len(final_samples) > 0:
        grouped = final_samples.groupby('GT_Stance')
        labels = ['FAVOR', 'AGAINST', 'NONE']
        n_per_label = 500 // len(labels)  # ~166 or 167 per label
        remainder = 500 % len(labels)
        
        balanced_samples = []
        for label in labels:
            if label in grouped.groups:
                group = grouped.get_group(label)
                sample_size = min(n_per_label + (1 if remainder > 0 else 0), len(group))
                if sample_size > 0:
                    balanced_samples.append(group.sample(n=sample_size, random_state=42))
                remainder = max(0, remainder - 1)
            else:
                print(f"Warning: Label {label} not found in final samples.")
        
        final_samples = pd.concat(balanced_samples, ignore_index=True)
        
        # If fewer than 500 samples, fill with random samples
        if len(final_samples) < 500:
            remaining = 500 - len(final_samples)
            available = final_samples.sample(n=min(remaining, len(final_samples)), random_state=42)
            final_samples = pd.concat([final_samples, available], ignore_index=True)

    # Save to CSV
    output_path = 'combined_samples_500_balanced.csv'
    final_samples.to_csv(output_path, index=False)
    print(f"Combined samples saved to {output_path}")
    print(f"Final sample size: {len(final_samples)}")
    print(f"Label distribution:\n{final_samples['GT_Stance'].value_counts()}")
    print(f"Dataset distribution:\n{final_samples['Dataset'].value_counts()}")

if __name__ == "__main__":
    main()