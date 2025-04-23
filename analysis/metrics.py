
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# === Load and Prepare Data ===
df = pd.read_csv("data\llm_judgments_gpt-4.csv")  # Change filename if needed
OUTPUT_DIR = "figures"

# === Map questions to topics ===
topic_map = {
    "What’s something fun you did last weekend?": "casual",
    "What kind of music do you like to listen to when you're relaxing?": "casual",
    "How would you define multiplication using addition?": "educational",
    "Explain the concept of a binary search algorithm.": "educational",
    "Is it better to be happy or to know the truth?": "philosophical",
    "If no one remembers your actions, do they still matter?": "philosophical",
    "Rewrite this sentence to sound more formal: “I messed up the report.”": "writing",
    "Write a short email requesting an extension for a project deadline.": "writing",
    "I’m torn between a high-paying job I dislike and one I enjoy. What should I do?": "advice",
    "I had a fight with a close friend — should I apologize even if I wasn’t wrong?": "advice",
}
df["topic"] = df["question"].map(topic_map).fillna("unknown")

# === Convert labels ===
df["true_label"] = df["source"].apply(lambda x: 1 if x == "human" else 0)
df["pred_label"] = df["llm_guess"].apply(lambda x: 1 if x == "human" else 0)

# === Overview Metrics ===
print("\n=== Overall Performance ===")
print(f"Accuracy: {accuracy_score(df['true_label'], df['pred_label']):.2%}")
print(classification_report(df['true_label'], df['pred_label'], target_names=['AI', 'Human']))

# === Confusion Matrix ===
cm = confusion_matrix(df["true_label"], df["pred_label"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.show()


# === Per-topic Metrics ===
print("\n=== Metrics by Topic ===")
topics = df["topic"].unique()
rows = []
for topic in topics:
    subset = df[df["topic"] == topic]
    acc = accuracy_score(subset["true_label"], subset["pred_label"])
    prec = precision_score(subset["true_label"], subset["pred_label"], zero_division=0)
    rec = recall_score(subset["true_label"], subset["pred_label"], zero_division=0)
    f1 = f1_score(subset["true_label"], subset["pred_label"], zero_division=0)
    rows.append([topic, acc, prec, rec, f1])

metrics_df = pd.DataFrame(rows, columns=["Topic", "Accuracy", "Precision", "Recall", "F1"])
print(metrics_df)

metrics_df.set_index("Topic").sort_values("Accuracy").plot(kind="barh", figsize=(10, 6), title="Performance Metrics by Topic")
plt.xlabel("Score")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_by_topic.png")
plt.show()

# === Source distribution per topic ===
print("\n=== Human vs AI Distribution by Topic ===")
src_dist = df.groupby(["topic", "source"]).size().unstack(fill_value=0)
print(src_dist)

src_dist.plot(kind="bar", stacked=True, figsize=(10, 5), title="Source Distribution by Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/source_distribution_by_topic.png")
plt.show()

# === Per-topic Confusion Matrices ===
for topic in topics:
    sub = df[df["topic"] == topic]
    cm = confusion_matrix(sub["true_label"], sub["pred_label"])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
    plt.title(f"Confusion Matrix - {topic}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_{topic}.png")
    plt.show()
