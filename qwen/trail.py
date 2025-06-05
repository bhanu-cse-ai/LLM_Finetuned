import pandas as pd
import json

# Define the extract_target_stance function
def extract_target_stance(text):
    try:
        # Check if input is a string and not empty
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string")
        
        target = ""
        stance = ""
        
        # Try parsing as JSON-like string
        if text.startswith("{") and text.endswith("}"):
            try:
                # Remove single quotes and replace with double quotes for valid JSON
                json_text = text.replace("'", "\"")
                data = json.loads(json_text)
                target = data.get("Target", "").strip()
                stance = data.get("Stance", "").strip()
            except json.JSONDecodeError:
                pass  # If JSON parsing fails, try comma-separated format
        
        # If JSON parsing didn't work or didn't yield results, try comma-separated format
        if not target or not stance:
            parts = text.split(', ')
            for part in parts:
                if part.startswith('Target: '):
                    target = part.replace('Target: ', '').strip()
                elif part.startswith('Stance: '):
                    stance = part.replace('Stance: ', '').strip()
        
        # Check if both values were found
        if target == "" or stance == "":
            raise ValueError("Text must contain both 'Target' and 'Stance' values")
        
        return target, stance
    except (ValueError, AttributeError):
        return None, None

# Load the dataset
df = pd.read_excel("/home/bhavani/Desktop/btsd/simple_380/predictions_llama31_finetuned_directed_simple_380_vast_filtered_ex_20250531_175652.xlsx")

# Create a mask for valid prediction format (non-empty strings)
valid_mask = df["Predicted Stance"].astype(str).apply(lambda x: isinstance(x, str) and x.strip() != "")

# Apply extraction only to valid rows
df_valid = df[valid_mask].copy()
df_valid[["Predicted Target", "Predicted Stance"]] = df_valid["Predicted Stance"].apply(
    lambda x: pd.Series(extract_target_stance(x))
)

# Identify rows where Predicted Target is "error" and reapply extraction using Raw Model Output
error_mask = df_valid["Predicted Target"] == "error"
df_valid.loc[error_mask, ["Predicted Target", "Predicted Stance"]] = df_valid.loc[error_mask, "Raw Model Output"].apply(
    lambda x: pd.Series(extract_target_stance(x))
)

# Save the updated dataframe
df_valid.to_csv("/home/bhavani/Desktop/btsd/simple_380/predictions_llama31_finetuned_directed_simple_380_vast_filtered_ex_20250531_175652_updated.csv", index=False)

# Show the extracted predictions
print(df_valid[["Raw Model Output", "Predicted Target", "Predicted Stance"]].head())