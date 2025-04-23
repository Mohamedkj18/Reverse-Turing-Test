
import pandas as pd
from time import sleep
import os
import argparse
from openai import OpenAI

# === CLI Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="e.g., gpt-4 or gpt-3.5-turbo")
args = parser.parse_args()

# === Configuration ===
client = OpenAI()
INPUT_FILE = "data/labeled_dataset.csv"
OUTPUT_FILE = f"data/llm_judgments_{args.model.replace('.', '')}.csv"
MODEL_NAME = args.model

# === Load Dataset ===
df = pd.read_csv(INPUT_FILE)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # optional shuffle

# === Define the judgment prompt ===
def ask_llm_judge(question, response):
    system_msg = (
        "You are an expert AI analyst. Given a prompt and a response, "
        "determine whether the response was written by a human or an AI. "
        "Be thoughtful but concise."
    )
    user_msg = f"""
Prompt:
{question}

Response:
{response}

Was this response written by a human or an AI?
Reply in this format:
Answer: <human or AI>
Explanation: <short reason why>
    """.strip()

    try:
        reply = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=150,
            temperature=0.3,
            top_p=1.0
        )
        result = reply.choices[0].message.content.strip()
        if "Answer:" in result:
            parts = result.split("Explanation:")
            guess = parts[0].replace("Answer:", "").strip().lower()
            explanation = parts[1].strip() if len(parts) > 1 else ""
            return guess, explanation
        else:
            return "unknown", result
    except Exception as e:
        print("API error:", e)
        return "error", str(e)

# === Run Evaluation ===
judged = []
for i, row in df.iterrows():
    print(f"Evaluating {i+1}/{len(df)}")
    guess, explanation = ask_llm_judge(row["question"], row["response"])
    judged.append({
        "question": row["question"],
        "response": row["response"],
        "source": row["source"],
        "llm_guess": guess,
        "explanation": explanation,
        "judge_model": MODEL_NAME
    })
    sleep(1)

# === Save Output ===
out_df = pd.DataFrame(judged)
out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Judgments saved to {OUTPUT_FILE}")
