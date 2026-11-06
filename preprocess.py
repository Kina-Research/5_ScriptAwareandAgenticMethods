import pandas as pd
import unicodedata
import re
import os
from sklearn.model_selection import train_test_split

# Amharic specific normalization mapping
AMHARIC_MAP = {
    'ሃ': 'ሀ', 'ኃ': 'ሀ', 'ሓ': 'ሀ', 'ሐ': 'ሀ',
    'ሡ': 'ሱ', 'ሠ': 'ሰ', 'ኅ': 'ህ', 'ዐ': 'አ',
    'ፀ': 'ጸ', 'ጹ': 'ጹ', 'ፁ': 'ጹ'
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Unicode Normalization
    text = unicodedata.normalize('NFKC', text)
    # 2. Amharic specific character mapping
    for k, v in AMHARIC_MAP.items():
        text = text.replace(k, v)
    # 3. Noise removal (punctuation, extraterrestrial symbols)
    # Keep Ge'ez characters (U+1200 to U+137F) and common punctuation
    text = re.sub(r'[^ \u1200-\u137F0-9a-zA-Z?,.!፦፣፤፥፦]', ' ', text)
    # 4. Remove extra whitespace
    text = " ".join(text.split())
    return text

def preprocess_dataset(file_path):
    df = pd.read_csv(file_path)
    
    # Preprocess Amharic text
    df['Amharic_Text_Normalized'] = df['Amharic_Text'].apply(normalize_text)
    
    # Split dataset: 70% Train, 15% Val, 15% Test
    # Using a fixed random state for reproducibility as requested (same splits across experiments)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['Type'] if len(df) > 10 else None)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['Type'] if len(temp_df) > 10 else None)
    
    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create artifacts directory if not exists
    os.makedirs('processed_data', exist_ok=True)
    
    train_df.to_csv('processed_data/train.csv', index=False)
    val_df.to_csv('processed_data/val.csv', index=False)
    test_df.to_csv('processed_data/test.csv', index=False)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    dataset_path = "dataset.csv"
    preprocess_dataset(dataset_path)
    print("Preprocessing complete. Data saved to processed_data/")
