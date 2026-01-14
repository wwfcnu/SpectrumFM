#!/usr/bin/env python3
"""
Script to process RML2016.10a_dict.pkl data according to the method in process_data() function
and save the split data into parquet format with columns: dataset_name, infer_class(label), snr, iq
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def iq2ap(IQ_data):
    """
    Convert IQ 2 AP and conduct log scaling
    """
    # 提取I和Q分量
    I = IQ_data[:, :, 0]
    Q = IQ_data[:, :, 1]
    
    # 计算幅度
    amplitude = np.sqrt(I**2 + Q**2)
    amplitude = np.log(amplitude)
    
    # 计算相位
    phase = np.arctan2(Q, I)
    
    # 将幅度和相位组合成一个新的三维数组
    AP_data = np.stack((amplitude, phase), axis=-1)
    
    return AP_data


def min_max_normalize(data, min_value=None, max_value=None):
    min_val = np.min(data) if min_value is None else min_value
    max_val = np.max(data) if max_value is None else max_value
    return (data - min_val) / (max_val - min_val)


def normalize(sample):
    normalized_sample = np.zeros_like(sample)
    # max_a = -10000
    # max_p = -10000
    # min_a = 10000
    # min_p = 10000
    for i in range(sample.shape[0]):
        # 提取 I 和 Q
        I = sample[i, :, 0]
        Q = sample[i, :, 1]
        
        # 归一化 I 和 Q
        normalized_I = min_max_normalize(I)
        normalized_Q = min_max_normalize(Q)
        # normalized_I = min_max_normalize(I)
        # normalized_Q = min_max_normalize(Q)
        # 将归一化后的 I 和 Q 放回原位置
        normalized_sample[i, :, 0] = normalized_I
        normalized_sample[i, :, 1] = normalized_Q
    #     if np.max(Q) > max_p:
    #     #         max_p = np.max(Q)
    #     #     if np.min(Q) < min_p:
    #     #         min_p = np.min(Q)
    #
    # # # 更新最大和最小幅度
    #     #     if np.max(I) > max_a:
    #     #         max_a = np.max(I)
    #     #     if np.min(I) < min_a:
    #     #         min_a = np.min(I)
    #     # print('max_p:', max_p, 'min_p:', min_p, 'max_a:', max_a, 'min_a:', min_a)
    return normalized_sample


def process_data():
    """Process RML2016.10a data according to the method in amc.py"""
    with open("Data/RML2016.10a_dict.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    amc_label_dict = {}
    for k in data.keys():
        amc, snr = k
        if amc not in amc_label_dict:
            amc_label_dict[amc] = len(amc_label_dict)
    
    # Define storage dictionaries
    train_dict = {"value": [], "label": []}
    test_dict = {"value": [], "label": [], "snr": []}

    # Process the dataset
    for k, v in data.items():
        total_len = len(v)
        
        # Use sklearn's train_test_split to split the data
        train_indices, test_indices = train_test_split(
            np.arange(total_len), test_size=0.1, random_state=42
        )
        
        # Get training and test sets
        train_values = v[train_indices]
        test_values = v[test_indices]
        
        # Get corresponding labels and SNR
        train_labels = [amc_label_dict[k[0]]] * len(train_indices)
        test_labels = [amc_label_dict[k[0]]] * len(test_indices)
        test_snrs = [k[1]] * len(test_indices)
        
        # Add data to dictionaries
        train_dict["value"].append(train_values)
        train_dict["label"].append(train_labels)
        
        test_dict["value"].append(test_values)
        test_dict["label"].append(test_labels)
        test_dict["snr"].append(test_snrs)

    # Concatenate all data to generate final numpy arrays
    train_dict["value"] = np.concatenate(train_dict["value"], axis=0)
    train_dict["value"] = np.transpose(train_dict["value"], (0, 2, 1))

    # Apply iq2ap transformation
    train_dict["value"] = iq2ap(train_dict["value"])
    # Apply normalization
    train_dict["value"] = normalize(train_dict["value"])

    train_dict["label"] = np.concatenate(train_dict["label"], axis=0)
    test_dict["value"] = np.concatenate(test_dict["value"], axis=0)
    test_dict["value"] = np.transpose(test_dict["value"], (0, 2, 1))
    test_dict["value"] = iq2ap(test_dict["value"])
    test_dict["value"] = normalize(test_dict["value"])
    
    test_dict["label"] = np.concatenate(test_dict["label"], axis=0)
    test_dict["snr"] = np.concatenate(test_dict["snr"], axis=0)
    
    return train_dict, test_dict


def convert_to_parquet_format(train_dict, test_dict):
    """
    Convert the processed data to parquet format with the required columns
    """
    # Process training data
    print("Processing training data...")
    train_rows = []
    for idx in range(len(train_dict["value"])):
        iq_data = train_dict["value"][idx].tolist()  # Convert to list format
        label = int(train_dict["label"][idx])
        # Training data doesn't have explicit SNR info in the dict, we'll need to extract from original data
        # For now, we'll assign a placeholder since train_dict doesn't contain SNR
        row = {
            "dataset_name": "RML2016.10a_train",
            "infer_class": label,
            "snr": 0,  # Placeholder for training data
            "iq": iq_data
        }
        train_rows.append(row)
    
    # Create DataFrame for training data
    train_df = pd.DataFrame(train_rows, columns=["dataset_name", "infer_class", "snr", "iq"])
    
    # Process test data
    print("Processing test data...")
    test_rows = []
    for idx in range(len(test_dict["value"])):
        iq_data = test_dict["value"][idx].tolist()  # Convert to list format
        label = int(test_dict["label"][idx])
        snr = int(test_dict["snr"][idx])
        row = {
            "dataset_name": "RML2016.10a_test",
            "infer_class": label,
            "snr": snr,
            "iq": iq_data
        }
        test_rows.append(row)
    
    # Create DataFrame for test data
    test_df = pd.DataFrame(test_rows, columns=["dataset_name", "infer_class", "snr", "iq"])
    
    return train_df, test_df


def main():
    print("Loading and processing RML2016.10a data...")
    
    # Process the data
    train_dict, test_dict = process_data()
    
    print(f"Train data shape: {train_dict['value'].shape}")
    print(f"Train labels shape: {train_dict['label'].shape}")
    print(f"Test data shape: {test_dict['value'].shape}")
    print(f"Test labels shape: {test_dict['label'].shape}")
    print(f"Test SNR shape: {test_dict['snr'].shape}")
    
    # Convert to parquet format
    train_df, test_df = convert_to_parquet_format(train_dict, test_dict)
    
    # Save to parquet files
    print("Saving to parquet files...")
    train_df.to_parquet("RML2016_10a_train.parquet", engine='pyarrow')
    test_df.to_parquet("RML2016_10a_test.parquet", engine='pyarrow')
    
    print(f"Training data saved to RML2016_10a_train.parquet with {len(train_df)} rows")
    print(f"Test data saved to RML2016_10a_test.parquet with {len(test_df)} rows")
    
    # Print some statistics
    print("\nTraining dataset statistics:")
    print(f"- Unique classes: {train_df['infer_class'].nunique()}")
    print(f"- Class distribution:\n{train_df['infer_class'].value_counts().sort_index()}")
    
    print("\nTest dataset statistics:")
    print(f"- Unique classes: {test_df['infer_class'].nunique()}")
    print(f"- Class distribution:\n{test_df['infer_class'].value_counts().sort_index()}")
    print(f"- Unique SNRs: {sorted(test_df['snr'].unique())}")


if __name__ == "__main__":
    main()