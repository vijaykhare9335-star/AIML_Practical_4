import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Create models
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Naive Bayes": GaussianNB()
}

results = []

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred) * 100,
        "Recall": recall_score(y_test, y_pred) * 100,
        "F1-Score": f1_score(y_test, y_pred) * 100,
        "Log Loss": log_loss(y_test, y_prob)
    })

# Create results table
df = pd.DataFrame(results)

print("\nCLASSIFICATION MODEL COMPARISON")
print(df.round(4).to_string(index=False))

# Find best model by accuracy
best = df.loc[df["Accuracy"].idxmax()]

print("\nBEST MODEL:", best["Algorithm"])
print("Accuracy:", round(best["Accuracy"], 2), "%")
print("Precision:", round(best["Precision"], 2), "%")
print("Recall:", round(best["Recall"], 2), "%")
print("F1-Score:", round(best["F1-Score"], 2), "%")
print("Log Loss:", round(best["Log Loss"], 4))

# Graph 1: Accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(df["Algorithm"], df["Accuracy"])
plt.title("Accuracy Comparison")
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# Graph 2: Precision, Recall and F1-score
df.plot(
    x="Algorithm",
    y=["Precision", "Recall", "F1-Score"],
    kind="bar",
    figsize=(9, 5)
)
plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Classification Algorithm")
plt.ylabel("Score (%)")
plt.ylim(0, 105)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Graph 3: Log Loss comparison
plt.figure(figsize=(8, 5))
plt.bar(df["Algorithm"], df["Log Loss"])
plt.title("Log Loss Comparison")
plt.xlabel("Classification Algorithm")
plt.ylabel("Log Loss (lower is better)")
plt.tight_layout()
plt.show()

# Textual comparative analysis
print("\nCOMPARATIVE ANALYSIS")

for _, row in df.iterrows():
    print(
        f"\n{row['Algorithm']}: "
        f"Accuracy = {row['Accuracy']:.2f}%, "
        f"Precision = {row['Precision']:.2f}%, "
        f"Recall = {row['Recall']:.2f}%, "
        f"F1-Score = {row['F1-Score']:.2f}%, "
        f"Log Loss = {row['Log Loss']:.4f}"
    )

print("\nCONCLUSION")
print(
    f"{best['Algorithm']} achieved the highest accuracy "
    f"of {best['Accuracy']:.2f}% on the test dataset."
)
print("SVM is useful for complex decision boundaries.")
print("Decision Tree is useful when interpretability is important.")
print("KNN is useful when distance between samples is meaningful.")
print("Naive Bayes is useful for text classification and spam detection.")
