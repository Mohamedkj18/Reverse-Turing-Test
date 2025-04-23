
import os
import argparse
import random
import numpy as np
import pandas as pd
from time import sleep
from typing import List
from difflib import SequenceMatcher
import openai

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")
PROMPTS_PATH = "data/prompts.txt"
OUTPUT_DIR = "data"
N_RESPONSES = 5
MAX_ATTEMPTS = 30
SIMILARITY_THRESHOLD = 0.8

# === Prompt Variants ===
PROMPT_VARIANTS = [
    "Question: {}",
    "Can you share your answer to this question? {}",
    "Think about this and respond honestly: {}",
    "Write what you'd say to a friend: {}",
    "Just a casual reply to: {}"
]

# === Style Instructions ===
STYLES = [
    "Be humorous and casual.",
    "Answer like you're tired but trying.",
    "Be quirky or weird.",
    "Keep it very dry and factual.",
    "Pretend you're texting a friend.",
    "Respond like you're in a hurry.",
    "Answer as if you're feeling nostalgic."
]

# === SIMILARITY ===

def is_too_similar(a: str, b: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold

# === GENERATION ===

def generate_response(prompt: str, model="gpt-4") -> str:
    style_instruction = random.choice(STYLES)
    prompt_variant = random.choice(PROMPT_VARIANTS).format(prompt)

    system_msg = (
        "You are simulating a human respondent in a survey. "
        f"{style_instruction} Try not to repeat yourself. 1–4 sentences max."
    )

    full_prompt = f"{system_msg}\n\n{prompt_variant}"
    model_name = "gpt-3.5-turbo" if model == "gpt-3.5" else "gpt-4"

    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=100,
        temperature=0.85,
        top_p=0.9
    )
    return response.choices[0].message.content.strip()

# === DIVERSE GENERATION ===

def generate_diverse_responses(prompt: str, model: str) -> List[str]:
    responses = []
    attempts = 0

    while len(responses) < N_RESPONSES and attempts < MAX_ATTEMPTS:
        try:
            resp = generate_response(prompt, model)
            if not any(is_too_similar(resp, existing) for existing in responses):
                responses.append(resp)
                print(f"{len(responses)}/{N_RESPONSES} - {resp}")
            else:
                print("Too similar, retrying...")
            sleep(1)
        except Exception as e:
            print("Error:", e)
            sleep(5)
        attempts += 1

    if len(responses) < N_RESPONSES:
        print(f"Only collected {len(responses)} for: {prompt}")
    return responses

# === MAIN EXECUTION ===

def main(model):
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    out_file = os.path.join(OUTPUT_DIR, f"{model.replace('.', '')}_responses_styled.csv")
    all_data = []

    for q in questions:
        print(f"\nGenerating for: {q}")
        responses = generate_diverse_responses(q, model)
        for resp in responses:
            all_data.append({"question": q, "response": resp, "source": model})
        pd.DataFrame(all_data).to_csv(out_file, index=False)  # Auto-save after each prompt

    print(f"\nAll responses saved to {out_file}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gpt-4 or gpt-3.5")
    args = parser.parse_args()
    main(args.model)
