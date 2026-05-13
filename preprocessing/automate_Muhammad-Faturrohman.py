import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_path, output_path):
    """
    Fungsi utama untuk membaca data mentah, membersihkan, 
    mengubah kategori menjadi angka, dan menyimpan hasilnya.
    """
    print(f"Memuat data dari: {input_path}")
    
    # Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"File tidak ditemukan di {input_path}")
        return

    # Data Cleaning 
    df = df.replace('?', np.nan)
    
    # Mengisi missing values dengan Modus (Nilai terbanyak)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            modus = df[col].mode()[0]
            df[col] = df[col].fillna(modus)
            print(f"Mengisi missing value di kolom '{col}' dengan modus: {modus}")

    # Encoding 
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    print("Encoding selesai.")

    # Saving Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = 'mushroom_raw/mushrooms.csv' 
    OUTPUT_FILE = 'preprocessing/mushrooms_preprocessing/mushroom_clean.csv'
    preprocess_data(INPUT_FILE, OUTPUT_FILE)