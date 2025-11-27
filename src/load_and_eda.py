import pandas as pd

def load_data():
    # Load both datasets
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    # Add labels
    fake["label"] = "fake"
    true["label"] = "real"

    # Combine them
    df = pd.concat([fake, true], ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def initial_eda(df):
    print("\n===== Shape of dataset =====")
    print(df.shape)

    print("\n===== Sample rows =====")
    print(df.head())

    print("\n===== Label distribution =====")
    print(df['label'].value_counts())

    print("\n===== Columns =====")
    print(df.columns)

    print("\n===== Missing values =====")
    print(df.isnull().sum())


if __name__ == "__main__":
    df = load_data()
    initial_eda(df)

    # Optionally save the merged dataset
    df.to_csv("data/merged_clean.csv", index=False)
    print("\nMerged dataset saved to data/merged_clean.csv")
