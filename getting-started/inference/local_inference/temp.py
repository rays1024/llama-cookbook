import json
import pandas as pd
from collections import defaultdict

# --- Configuration ---
FILE_PATH = "/p/liverobotics/Rui/llama-cookbook/getting-started/inference/local_inference/token_ranking_record.jsonl"

def calculate_mean_probability_by_rank(file_path):
    """
    Reads a JSONL file, extracts rank and probability data, and calculates
    the mean probability for each unique rank found across all records.
    """
    # Use defaultdict to store a list of all probabilities for each rank
    rank_probabilities = defaultdict(list)
    
    # Process the file line by line
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 1. Parse the JSON line
                try:
                    row_dict = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping corrupted JSON line: {e}")
                    continue

                # 2. Extract the 'record' list
                records = row_dict.get("record", [])

                # 3. Iterate through the records to collect data
                for item in records:
                    rank = item.get("rank")
                    probability = item.get("probability")

                    if rank is not None and probability is not None:
                        # Append the probability to the list for the corresponding rank
                        rank_probabilities[rank].append(probability)
                        
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return

    # --- Calculation and Output ---
    
    if not rank_probabilities:
        print("No valid rank/probability data found in the file.")
        return

    # 4. Convert the collected data to a DataFrame for easy calculation
    # First, flatten the data into a list of (rank, probability) tuples
    data = []
    for rank, probs in rank_probabilities.items():
        for prob in probs:
            data.append({'rank': rank, 'probability': prob})

    df = pd.DataFrame(data)

    # 5. Calculate the mean probability for each rank
    mean_probs = df.groupby('rank')['probability'].mean().reset_index()
    
    # Optionally, rename the mean column for clarity
    mean_probs.columns = ['Rank', 'Mean Probability']

    # print total by rank
    print("\n--- Percent by Rank ---")
    total_counts = df['rank'].value_counts().sort_index()
    total_sum = total_counts.sum()
    percent_by_rank = (total_counts / total_sum * 100).reset_index()
    percent_by_rank.columns = ['Rank', 'Percent']
    print(percent_by_rank.to_string(index=False))
    
    # 6. Print the results
    print("\n--- Mean Probability by Rank ---")
    print(mean_probs.to_string(index=False))


if __name__ == "__main__":
    calculate_mean_probability_by_rank(FILE_PATH)