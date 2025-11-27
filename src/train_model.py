# src/train_model.py

import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def main():

    data_path = Path("outputs/cleaned_df_fast.csv")
    print("Loading cleaned data:", data_path)

    df = pd.read_csv(data_path)

    # Features and labels
    X = df["content"]
    y = df["label"]

    print("Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_vec = tfidf.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------
    # Logistic Regression
    # -----------------------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    print("\n=== Logistic Regression Results ===")
    print("Accuracy:", accuracy_score(y_test, pred_lr))
    print(classification_report(y_test, pred_lr))

    # -----------------------------------
    # Naive Bayes
    # -----------------------------------
    print("\nTraining Multinomial Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    pred_nb = nb.predict(X_test)
    print("\n=== Naive Bayes Results ===")
    print("Accuracy:", accuracy_score(y_test, pred_nb))
    print(classification_report(y_test, pred_nb))

    # Save small report
    report_path = Path("outputs/model_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Logistic Regression ===\n")
        f.write(classification_report(y_test, pred_lr))
        f.write("\n\n=== Naive Bayes ===\n")
        f.write(classification_report(y_test, pred_nb))

    print(f"\nSaved model report to: {report_path}")


if __name__ == "__main__":
    main()
