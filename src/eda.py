# src/eda.py
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
from textblob import TextBlob

# Paths
in_csv = Path("outputs/cleaned_df_fast.csv")
out_fig_dir = Path("outputs/figures")
out_fig_dir.mkdir(parents=True, exist_ok=True)

def top_n_words(text_series, n=25):
    words = " ".join(text_series.dropna().astype(str)).split()
    counts = Counter(words)
    return counts.most_common(n)

def textblob_sentiment(series):
    polarities = series.apply(lambda t: TextBlob(t).sentiment.polarity if isinstance(t, str) and t.strip() else 0.0)
    return polarities

def main():
    print("Loading cleaned data:", in_csv)
    df = pd.read_csv(in_csv)

    # Basic info
    print("\nShape:", df.shape)
    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # Text lengths
    df['word_count'] = df['content'].astype(str).apply(lambda t: len(t.split()))
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='word_count', hue='label', bins=50, element='step', stat='count')
    plt.title('Word count distribution by label')
    plt.savefig(out_fig_dir / "wordcount_by_label.png", bbox_inches='tight')
    plt.close()

    # Top words per label
    for lab in df['label'].unique():
        print(f"\nTop words for label = {lab}:")
        top = top_n_words(df.loc[df['label']==lab, 'content'], n=20)
        print(top[:20])

        # WordCloud
        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df.loc[df['label']==lab, 'content'].astype(str)))
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud - {lab}")
        plt.savefig(out_fig_dir / f"wordcloud_{lab}.png", bbox_inches='tight')
        plt.close()

    # Example sample tweets/articles
    print("\nSample rows (label, word_count, first 120 chars):")
    for i, row in df.sample(6, random_state=42).iterrows():
        print(row['label'], row['word_count'], str(row['content'])[:120].replace('\n',' '), '...')

    # Sentiment (TextBlob) quick baseline
    print("\nComputing TextBlob polarity (quick baseline)...")
    df['polarity'] = textblob_sentiment(df['content'])
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='label', y='polarity')
    plt.title('TextBlob polarity by label')
    plt.savefig(out_fig_dir / "polarity_by_label.png", bbox_inches='tight')
    plt.close()

    # Save a small CSV sample for inspection
    df_sample = df.sample(2000, random_state=42)
    df_sample.to_csv("outputs/eda_sample.csv", index=False)
    print("\nSaved sample for inspection: outputs/eda_sample.csv")
    print("Figures saved to:", out_fig_dir)

if __name__ == "__main__":
    main()
