import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------

def load_dataframe(data_file):
    df = pd.read_csv(data_file)

    return df


# ------------------------------------------------------------
# 2. Clean + convert numeric columns
# ------------------------------------------------------------

def convert_numeric(df):
    """
    Convert all columns representing numeric values
    (except categorical labels) into float32.
    """
    col_names = df.columns

    for col in col_names:
        if 'survey' in col:
            df[col] = df[col].astype(np.int32)
        elif 'joint' in col:
            df[col] = df[col].astype(np.float32)

    return df


def split_dataset(df, index_column, n_val_users, n_test_users, seed):
    unique_users = df[index_column].unique()
    random.seed(seed) # Ensure reproducibility of shuffling
    random.shuffle(unique_users)

    # Calculate the number of users for the training set
    n_train_users = len(unique_users) - n_val_users - n_test_users

    # Split the shuffled user IDs into training, validation, and test sets
    train_users = unique_users[:n_train_users]
    val_users = unique_users[n_train_users:n_train_users + n_val_users]
    test_users = unique_users[n_train_users + n_val_users:]

    # Split the dataset into training, validation, and test sets based on user IDs
    df_train = df[df[index_column].isin(train_users)]
    df_val = df[df[index_column].isin(val_users)]
    df_test = df[df[index_column].isin(test_users)]

    return df_train, df_val, df_test


def label_mapping(df):
    # Define a mapping of label names to integer labels
    label_mapping = {
        'no_pain': 0,
        'low_pain': 1,
        'high_pain': 2
    }

    # Map label names to integers in the dataframe
    df['label'] = df['label'].map(label_mapping)

    return df

def feature_mapping(df):
    n_legs_mapping = {
        'two': 1,
        'one+peg_leg': 0
    }

    n_hands_mapping = {
        'two': 1,
        'one+hook_hand': 0
    }

    n_eyes_mapping = {
        'two': 1,
        'one+eye_patch': 0
    }


    # Map n_legs, n_hands and n_eyes names to integers in the train set
    df['n_legs'] = df['n_legs'].map(n_legs_mapping)
    df['n_hands'] = df['n_hands'].map(n_hands_mapping)
    df['n_eyes'] = df['n_eyes'].map(n_eyes_mapping)

    return df



# ------------------------------------------------------------
# 3. Normalize/standardize
# ------------------------------------------------------------

def min_max_normalize(df_ref, df, scale_columns):

    # Calculate the minimum and maximum values from the training data only
    mins = df_ref[scale_columns].min()
    maxs = df_ref[scale_columns].max()

    # Apply normalisation to the specified columns in all datasets
    for column in scale_columns:
        # Normalise the training set
        df[column] = (df[column] - mins[column]) / (maxs[column] - mins[column])

    return df


# ------------------------------------------------------------
# 4. Build sliding-window sequences
# ------------------------------------------------------------
def build_sequences(df_data, df_labels, window=200, stride=200):
    # Sanity check to ensure the window is divisible by the stride
    assert window % stride == 0

    # Define feature columns for the pirate dataset
    # These include pain surveys, n_legs, n_hands, n_eyes, and joint data
    feature_columns = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4',
                       'n_legs', 'n_hands', 'n_eyes'] + [f'joint_{i:02d}' for i in range(30)] # joint_30 was dropped earlier

    # Initialise lists to store sequences and their corresponding labels
    dataset = []
    labels = []

    # Iterate over unique sample_index in the DataFrame
    for sample_idx in df_data['sample_index'].unique():
        # Extract sensor data for the current sample_idx
        # Convert to float32 for consistency
        temp = df_data[df_data['sample_index'] == sample_idx][feature_columns].values.astype(np.float32)

        # Retrieve the pain label for the current sample_idx from df_labels
        label = df_labels[df_labels['sample_index'] == sample_idx]['label'].values[0]

        # Calculate padding length to ensure full windows
        num_features = temp.shape[1]
        padding_len = window - (len(temp) % window)


        # Create zero padding and concatenate with the data if necessary
        padding = np.zeros((padding_len, num_features), dtype='float32')
        temp = np.concatenate((temp, padding))

        # Build feature windows and associate them with labels
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            labels.append(label)
            idx += stride

    # Convert lists to numpy arrays for further processing
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels


def build_sequences_inference(df_data, window=200, stride=200):
    # Sanity check to ensure the window is divisible by the stride
    assert window % stride == 0

    # Define feature columns for the pirate dataset
    # These include pain surveys, n_legs, n_hands, n_eyes, and joint data
    feature_columns = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4',
                       'n_legs', 'n_hands', 'n_eyes'] + [f'joint_{i:02d}' for i in range(30)] # joint_30 was dropped earlier

    # Initialise lists to store sequences and their corresponding labels
    dataset = []
    list_sample_idx = []

    # Iterate over unique sample_index in the DataFrame
    for sample_idx in df_data['sample_index'].unique():
        # Extract sensor data for the current sample_idx
        # Convert to float32 for consistency
        temp = df_data[df_data['sample_index'] == sample_idx][feature_columns].values.astype(np.float32)


        # Calculate padding length to ensure full windows
        num_features = temp.shape[1]
        padding_len = window - (len(temp) % window)


        # Create zero padding and concatenate with the data if necessary
        padding = np.zeros((padding_len, num_features), dtype='float32')
        temp = np.concatenate((temp, padding))

        # Build feature windows and associate them with labels
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            list_sample_idx.append(sample_idx)
            idx += stride

    # Convert lists to numpy arrays for further processing
    dataset = np.array(dataset)
    list_sample_idx = np.array(list_sample_idx)

    return dataset, list_sample_idx

# ------------------------------------------------------------
# 5. Save dataset
# ------------------------------------------------------------

def save_pt(output_path, **kwargs):
    """
    Save tensors into a .pt file.
    """
    to_save = {}
    for key, val in kwargs.items():
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        to_save[key] = val

    torch.save(to_save, output_path)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 6. Main preprocessing pipeline
# ------------------------------------------------------------

def preprocess_pipeline(
    train_file,
    labels_file,
    inference_file,
    window_size=50,
    stride=10,
    n_val_users=60,
    n_test_users=60,
    output_dir="."
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    df_training = load_dataframe(train_file)
    df_labels = load_dataframe(labels_file)
    df_inference = load_dataframe(inference_file)

    # ---- Convert numeric ----
    df_training = convert_numeric(df_training)
    df_inference = convert_numeric(df_inference)

    df_training.drop(columns=['joint_30'], inplace=True)
    df_inference.drop(columns=['joint_30'], inplace=True)

    df_train, df_val, df_test = split_dataset(df_training, "sample_index", n_val_users, n_test_users, SEED)

    # ---- Standardize using train fit ----
    df_labels = label_mapping(df_labels)
    
    df_train = feature_mapping(df_train)
    df_val = feature_mapping(df_val)
    df_test = feature_mapping(df_test)
    df_inference = feature_mapping(df_inference)

    scale_columns = list(df_train.columns[2:5]) + list(df_train.columns[9:])

    df_train = min_max_normalize(df_train, df_train, scale_columns)
    df_val = min_max_normalize(df_train, df_val, scale_columns)
    df_test = min_max_normalize(df_train, df_test, scale_columns)
    df_inference = min_max_normalize(df_train, df_inference, scale_columns)




    # ---- Sequence construction ----
    X_train, y_train = build_sequences(
        df_train, df_labels, window_size, stride)
    
    X_val, y_val = build_sequences(
        df_val, df_labels, window_size, stride)

    X_test, y_test = build_sequences(
        df_test, df_labels, window_size, stride)

    X_inf, idx_inf = build_sequences_inference(
        df_inference, window_size, stride)

    # ---- Save outputs ----
    save_pt(f"{output_dir}/dataset.pt",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test)


    save_pt(f"{output_dir}/dataset_inference.pt",
            X_inference=X_inf,
            index_inference=idx_inf)



    print("Preprocessing completed!")


# ------------------------------------------------------------
# Run standalone
# ------------------------------------------------------------
if __name__ == "__main__":
    preprocess_pipeline(
        train_file="pirate_pain_train.csv",
        labels_file="pirate_pain_train_labels.csv",
        test_file="pirate_pain_test.csv",
        window_size=50,
        stride=10,
        output_dir="."
    )
