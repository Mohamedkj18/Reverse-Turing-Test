# Reverse Turing Test Project

## Project Summary

This project investigates whether advanced language models (LLMs) such as GPT-4 can accurately distinguish between human-written and AI-generated responses. Instead of testing whether an AI can pass as a human (as in the traditional Turing Test), this project reverses the role: can an AI detect whether a response is written by a human or another AI?

We evaluated this by:

- Generating AI responses using GPT-4 and GPT-3.5
- Collecting human responses through a Google Form
- Using GPT-4 as a judge to label responses as either human or AI
- Analyzing GPT-4's accuracy and performance across different conversation topics

## Project Structure

```
reverse-turing-test/
├── analysis/
│   ├── metrics.py
│   ├── results_table.csv
├── data/
│   ├──backup/
│   ├── prompts.txt
│   ├── human_responses.csv
│   ├── gpt-4_responses.csv
│   ├── gpt-35_responses.csv
│   ├── labeled_dataset.csv
│   └── llm_judgments_gpt-4.csv
│
├── scripts/
│   ├── aggregate_data.py
│   ├── generate_responses.py
│   ├── label_responses_all_sources.py
│   └── llm_judge_generic_v1api.py
│
├── plots/
├── requirements.txt
├── metrics_topic.ipynb
└── README.md
```

## How to Run the Experiment

### 1. Generate AI Responses

```bash
python scripts/generate_responses.py --model gpt-4
python scripts/generate_responses.py --model gpt-3.5
```

### 2. Label All Sources

```bash
python scripts/label_responses.py
```

This creates `labeled_dataset.csv` containing all question-response pairs with source labels.

### 3. Run the LLM Judge

```bash
python scripts/llm_judge.py --model gpt-4
```

This saves GPT-4's predictions in `data/llm_judgments_gpt-4.csv`.

### 4. Analyze Results

```bash
python scripts/metrics_topic_advanced.py
```

This script provides detailed topic-wise metrics and confusion matrices, and saves all plots to the `plots/` directory.

## Metrics Collected

- Overall accuracy, precision, recall, and F1 score
- Confusion matrices
- Accuracy per topic (casual, philosophical, advice, etc.)
- Visualization of model performance across topics

## Observations

- GPT-4 showed reduced accuracy in identifying AI-written responses in casual and philosophical topics.
- The model performed better on writing-related tasks.
- The system is extensible and supports evaluating other language models such as Claude.
