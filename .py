import pandas as pd

# 1. Đọc file a.csv
file_path = 'a.csv'
df = pd.read_csv(file_path)

# 2. Đưa cột HasDiabetes ra thành cột cuối cùng
target_col = 'HasDiabetes'
columns = [col for col in df.columns if col != target_col] + [target_col]
df = df[columns]

# 3. Loại bỏ các cột NoDocbcCost và HvyAlcoholConsump
df = df.drop(columns=['NoDocbcCost', 'HvyAlcoholConsump'])

# 4. Tạo file classification_big.csv
df_big = df.sample(frac=1 / 2.53, random_state=42)  # Giữ thứ tự ngẫu nhiên
df_big.to_csv('classification_big.csv', index=False)

# 5. Tạo file classification_medium.csv
df_medium = df_big.sample(frac=1 / 2, random_state=42)
df_medium.to_csv('classification_medium.csv', index=False)

# 6. Tạo file classification_small.csv
df_small = df_big.sample(frac=1 / 10, random_state=42)
df_small.to_csv('classification_small.csv', index=False)

print("Các file đã được tạo: classification_big.csv, classification_medium.csv, classification_small.csv")
