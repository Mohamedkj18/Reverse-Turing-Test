
import pandas as pd
import os

# === Config ===
DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "labeled_dataset.csv")

# Expected sources and their filenames
SOURCES = {
    "gpt4": "gpt-4_responses.csv",
    "gpt3.5": "gpt-35_responses.csv",
    "human": "human_responses.csv",

}

def safe_read_csv(path, label):
    try:
        df = pd.read_csv(path)
        df["source"] = label
        return df[["question", "response", "source"]]
    except FileNotFoundError:
        print(f"File not found: {path}")
        return pd.DataFrame(columns=["question", "response", "source"])

def main():
    all_data = []
    for label, filename in SOURCES.items():
        path = os.path.join(DATA_DIR, filename)
        df = safe_read_csv(path, label)
        all_data.append(df)

    merged = pd.concat(all_data, ignore_index=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"Full labeled dataset saved to {OUT_FILE}")
    print(f"Total responses: {len(merged)}")
    print(merged["source"].value_counts())

if __name__ == "__main__":
    main()
