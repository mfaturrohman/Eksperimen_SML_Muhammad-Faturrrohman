import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def load_data(path):
    return pd.read_csv(path)


def remove_duplicates(df):
    return df.drop_duplicates()


def encode_categorical(df):
    encoder_dict = {}

    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le

    return df, encoder_dict


def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    input_path = '../mushrooms_raw/mushrooms.csv'
    output_path = 'mushrooms_preprocessing/mushrooms_clean.csv'

    # Load data
    df = load_data(input_path)

    # Remove duplicate
    df = remove_duplicates(df)

    # Encoding
    df, encoders = encode_categorical(df)

    # Save processed data
    save_data(df, output_path)

    print('Preprocessing selesai')