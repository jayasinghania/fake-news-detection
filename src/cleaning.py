# src/cleaning.py   (FAST VERSION — NO LEMMATIZATION)

import re
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# Regex patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHANUM = re.compile(r"[^A-Za-z0-9\s']")


def clean_basic(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = NON_ALPHANUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in STOPWORDS])


def main():
    merged_csv = Path("data/merged_clean.csv")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(merged_csv)

    print("Applying fast cleaning...")
    df["clean_text"] = df["text"].apply(clean_basic)
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)

    # Combine title + cleaned text
    df["content"] = (df["title"].fillna("") + " " + df["clean_text"]).str.strip()

    print("\nSample clean result:")
    print(df[["title", "clean_text"]].head())

    # Save output
    out_csv = out_dir / "cleaned_df_fast.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved FAST cleaned dataset to {out_csv}")


if __name__ == "__main__":
    main()
