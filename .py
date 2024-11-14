import pandas as pd

df = pd.read_csv('classification_big.csv')

medium_dataset = df.sample(n=len(df) // 2, random_state=42)
small_dataset = df.sample(n=len(df) // 10, random_state=42)

medium_dataset.to_csv('classification_medium.csv', index=False)
small_dataset.to_csv('classification_small.csv', index=False)