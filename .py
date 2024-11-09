import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('big_data.csv')

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Encode object columns
df['Month'] = encoder.fit_transform(df['Month'])
df['VisitorType'] = encoder.fit_transform(df['VisitorType'])

# Convert boolean columns to integers
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Save the modified dataset
df.to_csv('encoded_big_data.csv', index=False)

print("Object and bool columns have been encoded and saved as 'encoded_big_data.csv'.")
